"""Durable, typed lifecycle outbox backed by one SQLite database.

The repository owns durable lifecycle work only.  It never reads Taskwarrior
and never mutates a plan after enqueue; execution progress and failure
evidence live in separate columns so recovery can be reasoned about directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
import json
import hashlib
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterator, Sequence
import uuid

from nautical_core.lifecycle_models import ExecutionStage, LifecycleContractError, LifecyclePlan


OUTBOX_SCHEMA_VERSION = 2
OUTBOX_LEGACY_SCHEMA_VERSION = 1
OUTBOX_ACK_RETENTION_SECONDS = 90.0 * 24.0 * 60.0 * 60.0
OUTBOX_HOUSEKEEPING_INTERVAL_SECONDS = 24.0 * 60.0 * 60.0
OUTBOX_HOUSEKEEPING_SIZE_THRESHOLD_BYTES = 8 * 1024 * 1024
OUTBOX_HOUSEKEEPING_ROW_LIMIT = 1000
_INIT_RETRIES = 8
_INIT_BACKOFF_S = 0.025
_MAX_INIT_BACKOFF_S = 0.25


class LifecycleOutboxError(RuntimeError):
    """Raised when the durable lifecycle outbox cannot preserve its contract."""


class OutboxProcessingState(str, Enum):
    READY = "ready"
    CLAIMED = "claimed"
    RETRY = "retry"
    MANUAL_REVIEW = "manual_review"
    QUARANTINED = "quarantined"
    ACKNOWLEDGED = "acknowledged"


class OutboxResultKind(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    RETRYABLE = "retryable"
    CONFLICT = "conflict"
    REJECTED = "rejected"


_ACTIVE_STATES = (OutboxProcessingState.READY.value, OutboxProcessingState.RETRY.value)
_EXECUTABLE_STAGES = frozenset(
    {
        ExecutionStage.PLANNED,
        ExecutionStage.PERSISTED,
        ExecutionStage.CHILD_PRESENT,
        ExecutionStage.PARENT_LINKED,
        ExecutionStage.VERIFIED,
    }
)
_TERMINAL_STATES = frozenset(
    {
        OutboxProcessingState.MANUAL_REVIEW,
        OutboxProcessingState.QUARANTINED,
        OutboxProcessingState.ACKNOWLEDGED,
    }
)
_STAGE_ORDER = {
    ExecutionStage.PLANNED: 0,
    ExecutionStage.PERSISTED: 0,
    ExecutionStage.CHILD_PRESENT: 1,
    ExecutionStage.PARENT_LINKED: 2,
    ExecutionStage.VERIFIED: 3,
    ExecutionStage.FINALIZED: 4,
}


@dataclass(frozen=True, slots=True)
class OutboxFailure:
    code: str
    message: str
    evidence: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        code = str(self.code or "").strip()
        message = str(self.message or "").strip()
        if not code or not message:
            raise LifecycleOutboxError("outbox failure requires a code and message")
        evidence = None if self.evidence is None else dict(self.evidence)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "evidence", evidence)

    def to_json(self) -> str:
        return json.dumps(
            {"code": self.code, "message": self.message, "evidence": self.evidence},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, value: str | None) -> "OutboxFailure | None":
        if not value:
            return None
        try:
            raw = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LifecycleOutboxError(f"invalid outbox failure JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise LifecycleOutboxError("invalid outbox failure JSON: expected object")
        evidence = raw.get("evidence")
        if evidence is not None and not isinstance(evidence, dict):
            raise LifecycleOutboxError("invalid outbox failure evidence")
        return cls(str(raw.get("code") or ""), str(raw.get("message") or ""), evidence)


@dataclass(frozen=True, slots=True)
class LifecycleOutboxRecord:
    intent_id: str
    plan: LifecyclePlan
    configuration_fingerprint: str
    schedule_fingerprint: str
    state: OutboxProcessingState
    stage: ExecutionStage
    lease_owner: str = ""
    lease_expires_at: float = 0.0
    attempts: int = 0
    failure: OutboxFailure | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        intent_id = str(self.intent_id or "").strip()
        if not intent_id:
            raise LifecycleOutboxError("outbox record requires an intent id")
        if not isinstance(self.plan, LifecyclePlan):
            raise LifecycleOutboxError("outbox record requires a lifecycle plan")
        if intent_id != self.plan.identity.idempotency_key:
            raise LifecycleOutboxError("outbox intent id differs from lifecycle identity")
        for name in ("configuration_fingerprint", "schedule_fingerprint"):
            if not str(getattr(self, name) or "").strip():
                raise LifecycleOutboxError(f"outbox record requires {name}")
        try:
            state = OutboxProcessingState(self.state)
            stage = ExecutionStage(self.stage)
        except (TypeError, ValueError) as exc:
            raise LifecycleOutboxError("outbox record has invalid state or stage") from exc
        if isinstance(self.attempts, bool) or int(self.attempts) < 0:
            raise LifecycleOutboxError("outbox attempts must be non-negative")
        lease_owner = str(self.lease_owner or "").strip()
        if state is OutboxProcessingState.CLAIMED and (not lease_owner or float(self.lease_expires_at) <= 0):
            raise LifecycleOutboxError("claimed outbox record requires a lease")
        if state is not OutboxProcessingState.CLAIMED and lease_owner:
            raise LifecycleOutboxError("only claimed outbox records may retain a lease")
        if state in {
            OutboxProcessingState.READY,
            OutboxProcessingState.RETRY,
            OutboxProcessingState.CLAIMED,
        } and stage not in _EXECUTABLE_STAGES:
            raise LifecycleOutboxError(
                f"active outbox state {state.value} cannot use lifecycle stage {stage.value}"
            )
        if state in _TERMINAL_STATES and stage is not ExecutionStage.FINALIZED:
            if state is not OutboxProcessingState.QUARANTINED:
                raise LifecycleOutboxError("terminal outbox state requires a finalized lifecycle stage")
        object.__setattr__(self, "intent_id", intent_id)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "lease_owner", lease_owner)
        object.__setattr__(self, "attempts", int(self.attempts))


@dataclass(frozen=True, slots=True)
class OutboxResult:
    kind: OutboxResultKind
    record: LifecycleOutboxRecord | None = None
    reason: str = ""
    lock_busy: bool = False

    def __post_init__(self) -> None:
        try:
            kind = OutboxResultKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise LifecycleOutboxError("invalid outbox result") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "reason", str(self.reason or "").strip())

    @property
    def ok(self) -> bool:
        return self.kind in {OutboxResultKind.APPLIED, OutboxResultKind.ALREADY_APPLIED}


@dataclass(frozen=True, slots=True)
class OutboxMaintenanceResult:
    """Structured result from an explicit, operator-triggered cleanup."""

    kind: OutboxResultKind
    removed: int = 0
    cutoff: float = 0.0
    retention_seconds: float = OUTBOX_ACK_RETENTION_SECONDS
    checkpoint: str = "not_requested"
    skipped: bool = False
    reason: str = ""
    lock_busy: bool = False

    def __post_init__(self) -> None:
        try:
            kind = OutboxResultKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise LifecycleOutboxError("invalid outbox maintenance result") from exc
        if isinstance(self.removed, bool) or int(self.removed) < 0:
            raise LifecycleOutboxError("outbox maintenance removed count must be non-negative")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "removed", int(self.removed))
        object.__setattr__(self, "skipped", bool(self.skipped))
        object.__setattr__(self, "reason", str(self.reason or "").strip())

    @property
    def ok(self) -> bool:
        return self.kind is OutboxResultKind.APPLIED


def lifecycle_outbox_path(taskdata: Path) -> Path:
    return Path(taskdata) / ".nautical-state" / ".nautical_lifecycle_outbox.db"


def _busy(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def _plan_json(plan: LifecyclePlan) -> str:
    return json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_plan(value: Any) -> LifecyclePlan:
    try:
        raw = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LifecycleOutboxError(f"invalid lifecycle plan JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise LifecycleOutboxError("invalid lifecycle plan JSON: expected object")
    try:
        return LifecyclePlan.from_dict(raw)
    except (LifecycleContractError, TypeError, ValueError) as exc:
        raise LifecycleOutboxError(f"invalid lifecycle plan: {exc}") from exc


def _canonical_object_json(value: Any, *, field: str) -> str:
    try:
        decoded = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LifecycleOutboxError(f"invalid outbox {field} JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise LifecycleOutboxError(f"invalid outbox {field} JSON: expected object")
    return json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _transition_allowed(current: ExecutionStage, target: ExecutionStage) -> bool:
    current_order = _STAGE_ORDER.get(current)
    target_order = _STAGE_ORDER.get(target)
    if current_order is None or target_order is None:
        return False
    return target_order == current_order or target_order == current_order + 1


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


class LifecycleOutboxRepository:
    """Single durable store for lifecycle plans and their verified progress."""

    def __init__(
        self,
        taskdata: Path,
        *,
        connect_timeout: float = 2.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.taskdata = Path(taskdata).resolve()
        self.path = lifecycle_outbox_path(self.taskdata)
        self.connect_timeout = max(0.1, float(connect_timeout))
        self._clock = clock
        self._schema_identity: tuple[int, int, int] | None = None
        self._benchmark_metrics: dict[str, float] | None = (
            {} if os.environ.get("NAUTICAL_BENCH_STATS_FILE") else None
        )

    def _metric(self, key: str, value: float = 1.0) -> None:
        if self._benchmark_metrics is not None:
            self._benchmark_metrics[key] = self._benchmark_metrics.get(key, 0.0) + float(value)

    def _connect(self) -> sqlite3.Connection:
        self._metric("outbox_connections")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        conn = sqlite3.connect(str(self.path), timeout=self.connect_timeout)
        os.chmod(self.path, 0o600)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute(f"PRAGMA busy_timeout={int(self.connect_timeout * 2000)}")
        return conn

    def _secure_state_files(self) -> None:
        """Keep the outbox and SQLite sidecars private to the Taskwarrior user."""
        os.chmod(self.path.parent, 0o700)
        for path in (self.path, self.path.with_name(f"{self.path.name}-wal"), self.path.with_name(f"{self.path.name}-shm")):
            if path.exists():
                os.chmod(path, 0o600)

    def open(self) -> OutboxResult:
        if self._schema_identity is not None:
            probe: sqlite3.Connection | None = None
            try:
                stat = self.path.stat()
                identity = (int(stat.st_dev), int(stat.st_ino), int(stat.st_mtime_ns))
                if identity == self._schema_identity:
                    probe = self._connect()
                    version = int(probe.execute("PRAGMA user_version").fetchone()[0] or 0)
                    if version == OUTBOX_SCHEMA_VERSION:
                        return OutboxResult(OutboxResultKind.APPLIED)
            except Exception:
                self._schema_identity = None
            finally:
                if probe is not None:
                    probe.close()
        last: Exception | None = None
        for attempt in range(_INIT_RETRIES):
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect()
                self._initialize(conn)
                self._secure_state_files()
                stat = self.path.stat()
                self._schema_identity = (int(stat.st_dev), int(stat.st_ino), int(stat.st_mtime_ns))
                return OutboxResult(OutboxResultKind.APPLIED)
            except sqlite3.OperationalError as exc:
                last = exc
                if not _busy(exc) or attempt + 1 >= _INIT_RETRIES:
                    return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc))
                self._metric("outbox_busy_retries")
                time.sleep(min(_MAX_INIT_BACKOFF_S, _INIT_BACKOFF_S * (2**attempt)))
            except Exception as exc:
                return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}")
            finally:
                if conn is not None:
                    conn.close()
        return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(last or "outbox open failed"), lock_busy=True)

    @staticmethod
    def _initialize(conn: sqlite3.Connection) -> None:
        for attempt in range(_INIT_RETRIES):
            try:
                version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
                if version > OUTBOX_SCHEMA_VERSION:
                    raise LifecycleOutboxError(
                        f"outbox schema v{version} is newer than supported v{OUTBOX_SCHEMA_VERSION}"
                    )
                if version == OUTBOX_SCHEMA_VERSION:
                    # A validated schema is already adopted. Avoid reopening
                    # WAL negotiation on every short-lived hook process.
                    LifecycleOutboxRepository._validate_schema(conn)
                    return
                if version == OUTBOX_LEGACY_SCHEMA_VERSION:
                    with _transaction(conn):
                        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
                        if "work_kind" not in columns:
                            conn.execute(
                                "ALTER TABLE lifecycle_outbox ADD COLUMN work_kind TEXT NOT NULL DEFAULT 'lifecycle'"
                            )
                        conn.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION}")
                    LifecycleOutboxRepository._validate_schema(conn)
                    return
                if version != 0:
                    raise LifecycleOutboxError(f"unsupported outbox schema v{version}")

                # WAL is negotiated only while creating or upgrading the
                # durable database. Competing first-openers retry the complete
                # sequence rather than using a process-local success flag.
                conn.execute("PRAGMA journal_mode=WAL")
                with _transaction(conn):
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS lifecycle_outbox (
                            intent_id TEXT PRIMARY KEY,
                            work_kind TEXT NOT NULL DEFAULT 'lifecycle',
                            plan_json TEXT NOT NULL,
                            plan_fingerprint TEXT NOT NULL,
                            parent_guard_json TEXT NOT NULL,
                            configuration_fingerprint TEXT NOT NULL,
                            schedule_fingerprint TEXT NOT NULL,
                            lifecycle_stage TEXT NOT NULL,
                            processing_state TEXT NOT NULL,
                            lease_owner TEXT NOT NULL DEFAULT '',
                            lease_expires_at REAL NOT NULL DEFAULT 0,
                            attempts INTEGER NOT NULL DEFAULT 0,
                            failure_json TEXT NOT NULL DEFAULT '',
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            acknowledged_at REAL NOT NULL DEFAULT 0,
                            CHECK (attempts >= 0)
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lifecycle_outbox_claim "
                        "ON lifecycle_outbox (processing_state, lease_expires_at, created_at)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lifecycle_outbox_stage "
                        "ON lifecycle_outbox (lifecycle_stage, processing_state)"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_lifecycle_outbox_ack "
                        "ON lifecycle_outbox (processing_state, acknowledged_at)"
                    )
                    conn.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION}")
                LifecycleOutboxRepository._validate_schema(conn)
                return
            except sqlite3.OperationalError as exc:
                if not _busy(exc) or attempt + 1 >= _INIT_RETRIES:
                    raise
                time.sleep(min(_MAX_INIT_BACKOFF_S, _INIT_BACKOFF_S * (2**attempt)))

    @staticmethod
    def _validate_schema(conn: sqlite3.Connection) -> None:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
        required = {
            "intent_id", "work_kind", "plan_json", "plan_fingerprint", "parent_guard_json",
            "configuration_fingerprint", "schedule_fingerprint", "lifecycle_stage",
            "processing_state", "lease_owner", "lease_expires_at", "attempts",
            "failure_json", "created_at", "updated_at", "acknowledged_at",
        }
        missing = sorted(required - columns)
        if missing:
            raise LifecycleOutboxError(f"outbox schema is incomplete: missing {', '.join(missing)}")

    def _with_connection(self, operation: Callable[[sqlite3.Connection], OutboxResult]) -> OutboxResult:
        conn: sqlite3.Connection | None = None
        started = time.perf_counter()
        self._metric("outbox_operation_scopes")
        try:
            conn = self._connect()
            self._initialize(conn)
            result = operation(conn)
            self._secure_state_files()
            return result
        except sqlite3.OperationalError as exc:
            if _busy(exc):
                self._metric("outbox_busy_failures")
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc))
        except LifecycleOutboxError as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}")
        finally:
            self._metric("outbox_operation_seconds", time.perf_counter() - started)
            if conn is not None:
                conn.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> LifecycleOutboxRecord:
        work_kind = str(row["work_kind"] or "lifecycle")
        if work_kind != "lifecycle":
            raise LifecycleOutboxError(f"unsupported outbox work kind for lifecycle reader: {work_kind}")
        plan = _decode_plan(row["plan_json"])
        intent_id = str(row["intent_id"])
        if intent_id != plan.identity.idempotency_key:
            raise LifecycleOutboxError("outbox row intent id differs from lifecycle plan identity")
        if str(row["plan_fingerprint"] or "") != plan.semantic_key():
            raise LifecycleOutboxError("outbox row plan fingerprint differs from immutable lifecycle plan")
        expected_guard = json.dumps(
            plan.parent_guard.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if _canonical_object_json(row["parent_guard_json"], field="parent guard") != expected_guard:
            raise LifecycleOutboxError("outbox row parent guard differs from immutable lifecycle plan")
        try:
            state = OutboxProcessingState(str(row["processing_state"]))
            stage = ExecutionStage(str(row["lifecycle_stage"]))
        except (TypeError, ValueError) as exc:
            raise LifecycleOutboxError("outbox row has invalid state or stage") from exc
        return LifecycleOutboxRecord(
            intent_id=intent_id,
            plan=plan,
            configuration_fingerprint=str(row["configuration_fingerprint"]),
            schedule_fingerprint=str(row["schedule_fingerprint"]),
            state=state,
            stage=stage,
            lease_owner=str(row["lease_owner"] or ""),
            lease_expires_at=float(row["lease_expires_at"] or 0),
            attempts=int(row["attempts"] or 0),
            failure=OutboxFailure.from_json(str(row["failure_json"] or "")),
            created_at=float(row["created_at"] or 0),
            updated_at=float(row["updated_at"] or 0),
        )

    def enqueue(
        self,
        plan: LifecyclePlan,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
    ) -> OutboxResult:
        if not isinstance(plan, LifecyclePlan):
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox enqueue requires a lifecycle plan")
        config = str(configuration_fingerprint or "").strip()
        schedule = str(schedule_fingerprint or "").strip()
        if not config or not schedule:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox enqueue requires configuration and schedule fingerprints")
        if plan.stage not in {ExecutionStage.PLANNED, ExecutionStage.PERSISTED}:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox enqueue requires a planned lifecycle plan")
        intent_id = plan.identity.idempotency_key
        encoded_plan = _plan_json(plan.with_stage(ExecutionStage.PLANNED))
        plan_fingerprint = plan.semantic_key()
        guard_json = json.dumps(plan.parent_guard.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        now = self._clock()

        def operation(conn: sqlite3.Connection) -> OutboxResult:
            with _transaction(conn):
                row = conn.execute("SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)).fetchone()
                if row is not None:
                    current = self._from_row(row)
                    same_plan = current.plan.compatibility_key() == plan.compatibility_key()
                    if current.state is OutboxProcessingState.ACKNOWLEDGED and same_plan:
                        return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=current)
                    if (
                        same_plan
                        and current.state in {OutboxProcessingState.READY, OutboxProcessingState.RETRY}
                    ):
                        conn.execute(
                            "UPDATE lifecycle_outbox SET plan_json=?, plan_fingerprint=?, parent_guard_json=?, "
                            "configuration_fingerprint=?, schedule_fingerprint=?, failure_json='', updated_at=? "
                            "WHERE intent_id=?",
                            (
                                encoded_plan,
                                plan_fingerprint,
                                guard_json,
                                config,
                                schedule,
                                now,
                                intent_id,
                            ),
                        )
                        refreshed = conn.execute(
                            "SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)
                        ).fetchone()
                        if refreshed is None:
                            return OutboxResult(OutboxResultKind.RETRYABLE, reason="refreshed lifecycle intent disappeared")
                        return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=self._from_row(refreshed))
                    if (
                        same_plan
                        and current.state is OutboxProcessingState.CLAIMED
                        and current.lease_expires_at <= now
                    ):
                        conn.execute(
                            "UPDATE lifecycle_outbox SET plan_json=?, plan_fingerprint=?, parent_guard_json=?, "
                            "configuration_fingerprint=?, schedule_fingerprint=?, processing_state=?, lease_owner='', "
                            "lease_expires_at=0, failure_json='', updated_at=? WHERE intent_id=?",
                            (
                                encoded_plan,
                                plan_fingerprint,
                                guard_json,
                                config,
                                schedule,
                                OutboxProcessingState.RETRY.value,
                                now,
                                intent_id,
                            ),
                        )
                        refreshed = conn.execute(
                            "SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)
                        ).fetchone()
                        if refreshed is None:
                            return OutboxResult(OutboxResultKind.RETRYABLE, reason="expired lifecycle intent disappeared")
                        return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=self._from_row(refreshed))
                    if (
                        same_plan
                        and current.state is OutboxProcessingState.MANUAL_REVIEW
                        and current.failure is not None
                        and current.failure.code in {"mutation_conflict", "mutation_rejected"}
                    ):
                        conn.execute(
                            "UPDATE lifecycle_outbox SET plan_json=?, plan_fingerprint=?, parent_guard_json=?, "
                            "configuration_fingerprint=?, schedule_fingerprint=?, lifecycle_stage=?, processing_state=?, "
                            "failure_json='', updated_at=? WHERE intent_id=?",
                            (
                                encoded_plan,
                                plan_fingerprint,
                                guard_json,
                                config,
                                schedule,
                                ExecutionStage.PLANNED.value,
                                OutboxProcessingState.RETRY.value,
                                now,
                                intent_id,
                            ),
                        )
                        refreshed = conn.execute(
                            "SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)
                        ).fetchone()
                        if refreshed is None:
                            return OutboxResult(OutboxResultKind.RETRYABLE, reason="reopened lifecycle intent disappeared")
                        return OutboxResult(OutboxResultKind.APPLIED, record=self._from_row(refreshed))
                    compatible = (
                        same_plan
                        and current.configuration_fingerprint == config
                        and current.schedule_fingerprint == schedule
                    )
                    if not compatible:
                        return OutboxResult(
                            OutboxResultKind.CONFLICT,
                            record=current,
                            reason=(
                                "deterministic lifecycle intent conflicts with an existing queued transition; "
                                "run `nautical reconcile --apply` to drain it, then inspect the chain's "
                                "nextLink and child before retrying; "
                                f"state={current.state.value}, plan_equal={same_plan}, "
                                f"configuration={current.configuration_fingerprint!r}->{config!r}, "
                                f"schedule={current.schedule_fingerprint!r}->{schedule!r}"
                            ),
                        )
                    return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=current)
                conn.execute(
                    """
                    INSERT INTO lifecycle_outbox (
                        intent_id, plan_json, plan_fingerprint, parent_guard_json,
                        configuration_fingerprint, schedule_fingerprint,
                        lifecycle_stage, processing_state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent_id, encoded_plan, plan_fingerprint, guard_json, config, schedule,
                        ExecutionStage.PLANNED.value, OutboxProcessingState.READY.value, now, now,
                    ),
                )
                record = LifecycleOutboxRecord(
                    intent_id=intent_id,
                    plan=plan.with_stage(ExecutionStage.PLANNED),
                    configuration_fingerprint=config,
                    schedule_fingerprint=schedule,
                    state=OutboxProcessingState.READY,
                    stage=ExecutionStage.PLANNED,
                    created_at=now,
                    updated_at=now,
                )
                return OutboxResult(OutboxResultKind.APPLIED, record=record)

        return self._with_connection(operation)

    def enqueue_integrity(self, envelope: Any) -> OutboxResult:
        """Persist an integrity envelope in the shared outbox table.

        Lifecycle methods deliberately do not decode or claim this work kind;
        the integrity executor owns its later dispatch path.
        """
        try:
            from .integrity_outbox_envelope import IntegrityOutboxEnvelope, OutboxWorkKind

            if not isinstance(envelope, IntegrityOutboxEnvelope):
                return OutboxResult(OutboxResultKind.REJECTED, reason="integrity enqueue requires an integrity envelope")
            encoded = envelope.to_json()
            intent_id = envelope.intent_id
            fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            config = envelope.configuration_fingerprint
            schedule = envelope.schedule_fingerprint
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"invalid integrity envelope: {exc}")
        now = self._clock()

        def operation(conn: sqlite3.Connection) -> OutboxResult:
            with _transaction(conn):
                row = conn.execute("SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)).fetchone()
                if row is not None:
                    if str(row["work_kind"] or "lifecycle") != OutboxWorkKind.INTEGRITY.value:
                        return OutboxResult(OutboxResultKind.CONFLICT, reason="intent ID belongs to another outbox work kind")
                    if str(row["plan_fingerprint"] or "") == fingerprint:
                        return OutboxResult(OutboxResultKind.ALREADY_APPLIED)
                    return OutboxResult(OutboxResultKind.CONFLICT, reason="integrity intent payload differs")
                conn.execute(
                    """
                    INSERT INTO lifecycle_outbox (
                        intent_id, work_kind, plan_json, plan_fingerprint, parent_guard_json,
                        configuration_fingerprint, schedule_fingerprint,
                        lifecycle_stage, processing_state, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent_id, OutboxWorkKind.INTEGRITY.value, encoded, fingerprint, "{}",
                        config, schedule, ExecutionStage.PLANNED.value,
                        OutboxProcessingState.READY.value, now, now,
                    ),
                )
            return OutboxResult(OutboxResultKind.APPLIED)

        return self._with_connection(operation)

    def claim_batch(self, *, owner: str, lease_seconds: float, limit: int) -> tuple[OutboxResult, tuple[LifecycleOutboxRecord, ...]]:
        self._metric("outbox_lease_claims")
        owner = str(owner or "").strip()
        if not owner or lease_seconds <= 0 or limit <= 0:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox claim requires owner, lease, and limit"), ()
        now = self._clock()
        expires = now + float(lease_seconds)
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            self._secure_state_files()
            with _transaction(conn):
                conn.execute(
                    "UPDATE lifecycle_outbox SET processing_state=?, lease_owner='', lease_expires_at=0, updated_at=? "
                    "WHERE processing_state=? AND lease_expires_at <= ?",
                    (OutboxProcessingState.RETRY.value, now, OutboxProcessingState.CLAIMED.value, now),
                )
                candidates = conn.execute(
                    "SELECT intent_id FROM lifecycle_outbox WHERE processing_state IN (?, ?) "
                    "AND work_kind=? ORDER BY created_at, intent_id LIMIT ?",
                    (*_ACTIVE_STATES, "lifecycle", int(limit)),
                ).fetchall()
                records: list[LifecycleOutboxRecord] = []
                for raw in candidates:
                    intent_id = str(raw[0])
                    row = conn.execute("SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)).fetchone()
                    if row is None:
                        continue
                    try:
                        candidate = self._from_row(row)
                    except LifecycleOutboxError as exc:
                        self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                        continue
                    if candidate.attempts >= candidate.plan.max_attempts:
                        self._quarantine_row(
                            conn,
                            intent_id,
                            now,
                            OutboxFailure(
                                "retry_exhausted",
                                "outbox retry budget exhausted before claim",
                                {
                                    "attempts": candidate.attempts,
                                    "max_attempts": candidate.plan.max_attempts,
                                },
                            ),
                        )
                        continue
                    conn.execute(
                        "UPDATE lifecycle_outbox SET processing_state=?, lease_owner=?, lease_expires_at=?, "
                        "attempts=attempts+1, failure_json='', updated_at=? "
                        "WHERE intent_id=? AND processing_state IN (?, ?)",
                        (OutboxProcessingState.CLAIMED.value, owner, expires, now, intent_id, *_ACTIVE_STATES),
                    )
                    row = conn.execute("SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)).fetchone()
                    if row is None:
                        continue
                    try:
                        records.append(self._from_row(row))
                    except LifecycleOutboxError as exc:
                        self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                return OutboxResult(OutboxResultKind.APPLIED), tuple(records)
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc)), ()
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}"), ()
        finally:
            if conn is not None:
                conn.close()

    def claim_integrity_batch(self, *, owner: str, lease_seconds: float, limit: int) -> tuple[OutboxResult, tuple[Any, ...]]:
        self._metric("outbox_lease_claims")
        """Claim integrity work without exposing it to lifecycle executors."""
        from .integrity_outbox_envelope import IntegrityOutboxEnvelope, IntegrityOutboxRecord

        owner = str(owner or "").strip()
        if not owner or lease_seconds <= 0 or limit <= 0:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox claim requires owner, lease, and limit"), ()
        now = self._clock()
        expires = now + float(lease_seconds)
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            self._secure_state_files()
            with _transaction(conn):
                conn.execute(
                    "UPDATE lifecycle_outbox SET processing_state=?, lease_owner='', lease_expires_at=0, updated_at=? "
                    "WHERE work_kind=? AND processing_state=? AND lease_expires_at <= ?",
                    (OutboxProcessingState.RETRY.value, now, "integrity", OutboxProcessingState.CLAIMED.value, now),
                )
                rows = conn.execute(
                    "SELECT * FROM lifecycle_outbox WHERE work_kind=? AND processing_state IN (?, ?) "
                    "ORDER BY created_at, intent_id LIMIT ?",
                    ("integrity", *_ACTIVE_STATES, int(limit)),
                ).fetchall()
                records: list[Any] = []
                for row in rows:
                    intent_id = str(row["intent_id"])
                    try:
                        envelope = IntegrityOutboxEnvelope.from_dict(json.loads(str(row["plan_json"] or "")))
                        if hashlib.sha256(envelope.to_json().encode("utf-8")).hexdigest() != str(row["plan_fingerprint"] or ""):
                            raise LifecycleOutboxError("integrity envelope fingerprint mismatch")
                    except Exception as exc:
                        self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_integrity_row", str(exc)))
                        continue
                    conn.execute(
                        "UPDATE lifecycle_outbox SET processing_state=?, lease_owner=?, lease_expires_at=?, "
                        "attempts=attempts+1, failure_json='', updated_at=? WHERE intent_id=? "
                        "AND work_kind=? AND processing_state IN (?, ?)",
                        (OutboxProcessingState.CLAIMED.value, owner, expires, now, intent_id, "integrity", *_ACTIVE_STATES),
                    )
                    records.append(IntegrityOutboxRecord(
                        envelope, OutboxProcessingState.CLAIMED, ExecutionStage.PLANNED,
                        owner, expires, int(row["attempts"] or 0) + 1,
                    ))
                return OutboxResult(OutboxResultKind.APPLIED), tuple(records)
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc)), ()
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}"), ()
        finally:
            if conn is not None:
                conn.close()

    def _integrity_transition(
        self,
        *,
        intent_id: str,
        owner: str,
        state: OutboxProcessingState,
        failure: OutboxFailure | None = None,
    ) -> OutboxResult:
        intent_id = str(intent_id or "").strip()
        owner = str(owner or "").strip()
        if not intent_id or not owner:
            return OutboxResult(OutboxResultKind.REJECTED, reason="integrity transition requires intent and owner")
        if state not in {OutboxProcessingState.RETRY, OutboxProcessingState.MANUAL_REVIEW, OutboxProcessingState.ACKNOWLEDGED}:
            return OutboxResult(OutboxResultKind.REJECTED, reason="invalid integrity transition state")
        if state is OutboxProcessingState.MANUAL_REVIEW and failure is None:
            return OutboxResult(OutboxResultKind.REJECTED, reason="manual review requires failure evidence")
        now = self._clock()
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            with _transaction(conn):
                row = conn.execute(
                    "SELECT * FROM lifecycle_outbox WHERE intent_id=? AND work_kind=?",
                    (intent_id, "integrity"),
                ).fetchone()
                if row is None:
                    return OutboxResult(OutboxResultKind.CONFLICT, reason="no such integrity intent")
                current = str(row["processing_state"] or "")
                if current == OutboxProcessingState.ACKNOWLEDGED.value and state is OutboxProcessingState.ACKNOWLEDGED:
                    return OutboxResult(OutboxResultKind.ALREADY_APPLIED)
                if current != OutboxProcessingState.CLAIMED.value or str(row["lease_owner"] or "") != owner:
                    return OutboxResult(OutboxResultKind.CONFLICT, reason="integrity intent is not owned by caller")
                encoded_failure = "" if failure is None else failure.to_json()
                acknowledged_at = now if state is OutboxProcessingState.ACKNOWLEDGED else 0
                stage = ExecutionStage.FINALIZED.value if state in {
                    OutboxProcessingState.MANUAL_REVIEW, OutboxProcessingState.ACKNOWLEDGED,
                } else ExecutionStage.PLANNED.value
                conn.execute(
                    "UPDATE lifecycle_outbox SET processing_state=?, lifecycle_stage=?, lease_owner='', "
                    "lease_expires_at=0, failure_json=?, acknowledged_at=?, updated_at=? "
                    "WHERE intent_id=? AND work_kind=? AND processing_state=? AND lease_owner=?",
                    (state.value, stage, encoded_failure, acknowledged_at, now, intent_id, "integrity",
                     OutboxProcessingState.CLAIMED.value, owner),
                )
                return OutboxResult(OutboxResultKind.APPLIED)
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc))
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}")
        finally:
            if conn is not None:
                conn.close()

    def acknowledge_integrity(self, *, intent_id: str, owner: str) -> OutboxResult:
        return self._integrity_transition(
            intent_id=intent_id, owner=owner, state=OutboxProcessingState.ACKNOWLEDGED,
        )

    def release_integrity_retry(self, *, intent_id: str, owner: str, failure: OutboxFailure) -> OutboxResult:
        return self._integrity_transition(
            intent_id=intent_id, owner=owner, state=OutboxProcessingState.RETRY, failure=failure,
        )

    def manual_review_integrity(self, *, intent_id: str, owner: str, failure: OutboxFailure) -> OutboxResult:
        return self._integrity_transition(
            intent_id=intent_id, owner=owner, state=OutboxProcessingState.MANUAL_REVIEW, failure=failure,
        )

    def claim_intent(self, *, owner: str, lease_seconds: float, intent_id: str) -> OutboxResult:
        """Claim one specific intent by id, for a caller that must execute
        exactly the record it just staged (e.g. reconcile, which holds a
        per-parent lock and must not act on an unrelated claimed intent).
        Reuses the same claim/lease/poison-row handling as `claim_batch`,
        scoped to a single row instead of an ordered batch.
        """
        owner = str(owner or "").strip()
        intent_id = str(intent_id or "").strip()
        if not owner or lease_seconds <= 0 or not intent_id:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox claim requires owner, lease, and intent_id")
        now = self._clock()
        expires = now + float(lease_seconds)
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            self._secure_state_files()
            with _transaction(conn):
                conn.execute(
                    "UPDATE lifecycle_outbox SET processing_state=?, lease_owner='', lease_expires_at=0, updated_at=? "
                    "WHERE processing_state=? AND lease_expires_at <= ?",
                    (OutboxProcessingState.RETRY.value, now, OutboxProcessingState.CLAIMED.value, now),
                )
                row = conn.execute(
                    "SELECT * FROM lifecycle_outbox WHERE intent_id=? AND work_kind=?",
                    (intent_id, "lifecycle"),
                ).fetchone()
                if row is None:
                    return OutboxResult(OutboxResultKind.CONFLICT, reason="no such lifecycle intent")
                current_state = str(row["processing_state"])
                if current_state == OutboxProcessingState.ACKNOWLEDGED.value:
                    try:
                        record = self._from_row(row)
                    except LifecycleOutboxError as exc:
                        self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                        return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
                    return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=record)
                if current_state not in _ACTIVE_STATES:
                    try:
                        record = self._from_row(row)
                    except LifecycleOutboxError as exc:
                        self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                        return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
                    return OutboxResult(
                        OutboxResultKind.CONFLICT,
                        record=record,
                        reason=f"lifecycle intent is not claimable (state={current_state!r})",
                    )
                try:
                    candidate = self._from_row(row)
                except LifecycleOutboxError as exc:
                    self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                    return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
                if candidate.attempts >= candidate.plan.max_attempts:
                    failure = OutboxFailure(
                        "retry_exhausted",
                        "outbox retry budget exhausted before claim",
                        {
                            "attempts": candidate.attempts,
                            "max_attempts": candidate.plan.max_attempts,
                        },
                    )
                    self._quarantine_row(conn, intent_id, now, failure)
                    return OutboxResult(OutboxResultKind.REJECTED, reason=failure.message)
                conn.execute(
                    "UPDATE lifecycle_outbox SET processing_state=?, lease_owner=?, lease_expires_at=?, "
                    "attempts=attempts+1, failure_json='', updated_at=? "
                    "WHERE intent_id=? AND processing_state IN (?, ?)",
                    (OutboxProcessingState.CLAIMED.value, owner, expires, now, intent_id, *_ACTIVE_STATES),
                )
                row = conn.execute("SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    return OutboxResult(OutboxResultKind.CONFLICT, reason="no such lifecycle intent")
                if str(row["processing_state"]) == OutboxProcessingState.ACKNOWLEDGED.value:
                    try:
                        record = self._from_row(row)
                    except LifecycleOutboxError as exc:
                        self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                        return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
                    return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=record)
                if str(row["lease_owner"]) != owner or str(row["processing_state"]) != OutboxProcessingState.CLAIMED.value:
                    return OutboxResult(
                        OutboxResultKind.CONFLICT,
                        reason=f"lifecycle intent is not claimable (state={row['processing_state']!r})",
                    )
                try:
                    record = self._from_row(row)
                except LifecycleOutboxError as exc:
                    self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                    return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
                return OutboxResult(OutboxResultKind.APPLIED, record=record)
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc))
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}")
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _quarantine_row(conn: sqlite3.Connection, intent_id: str, now: float, failure: OutboxFailure) -> None:
        conn.execute(
            "UPDATE lifecycle_outbox SET processing_state=?, lease_owner='', lease_expires_at=0, "
            "failure_json=?, updated_at=? WHERE intent_id=?",
            (OutboxProcessingState.QUARANTINED.value, failure.to_json(), now, intent_id),
        )

    def _claimed_update(
        self,
        *,
        intent_id: str,
        owner: str,
        operation: Callable[[sqlite3.Connection, LifecycleOutboxRecord, float], OutboxResult],
    ) -> OutboxResult:
        intent_id = str(intent_id or "").strip()
        owner = str(owner or "").strip()
        if not intent_id or not owner:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox mutation requires intent and lease owner")
        now = self._clock()

        def run(conn: sqlite3.Connection) -> OutboxResult:
            with _transaction(conn):
                row = conn.execute("SELECT * FROM lifecycle_outbox WHERE intent_id=?", (intent_id,)).fetchone()
                if row is None:
                    return OutboxResult(OutboxResultKind.CONFLICT, reason="outbox intent is absent")
                try:
                    record = self._from_row(row)
                except LifecycleOutboxError as exc:
                    self._quarantine_row(conn, intent_id, now, OutboxFailure("poison_row", str(exc)))
                    return OutboxResult(OutboxResultKind.REJECTED, reason=f"outbox intent quarantined: {exc}")
                if record.state in _TERMINAL_STATES:
                    return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=record)
                if (
                    record.state is not OutboxProcessingState.CLAIMED
                    or record.lease_owner != owner
                    or record.lease_expires_at <= now
                ):
                    return OutboxResult(OutboxResultKind.CONFLICT, record=record, reason="outbox claim is not owned")
                return operation(conn, record, now)

        return self._with_connection(run)

    def renew_lease(self, *, intent_id: str, owner: str, lease_seconds: float) -> OutboxResult:
        self._metric("outbox_lease_renewals")
        if lease_seconds <= 0:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox lease must be positive")

        def operation(conn: sqlite3.Connection, record: LifecycleOutboxRecord, now: float) -> OutboxResult:
            expires = now + float(lease_seconds)
            conn.execute(
                "UPDATE lifecycle_outbox SET lease_expires_at=?, updated_at=? WHERE intent_id=?",
                (expires, now, record.intent_id),
            )
            return OutboxResult(
                OutboxResultKind.APPLIED,
                record=replace(record, lease_expires_at=expires, updated_at=now),
            )

        return self._claimed_update(intent_id=intent_id, owner=owner, operation=operation)

    def advance_stage(self, *, intent_id: str, owner: str, stage: ExecutionStage) -> OutboxResult:
        self._metric("outbox_stage_advances")
        try:
            target = ExecutionStage(stage)
        except (TypeError, ValueError):
            return OutboxResult(OutboxResultKind.REJECTED, reason="invalid lifecycle stage")

        def operation(conn: sqlite3.Connection, record: LifecycleOutboxRecord, now: float) -> OutboxResult:
            if target is ExecutionStage.FINALIZED:
                return OutboxResult(OutboxResultKind.REJECTED, record=record, reason="finalization requires acknowledgement")
            if target is record.stage:
                return OutboxResult(OutboxResultKind.ALREADY_APPLIED, record=record)
            if not _transition_allowed(record.stage, target):
                return OutboxResult(
                    OutboxResultKind.CONFLICT,
                    record=record,
                    reason=f"invalid lifecycle stage transition: {record.stage.value} -> {target.value}",
                )
            conn.execute(
                "UPDATE lifecycle_outbox SET lifecycle_stage=?, updated_at=? WHERE intent_id=?",
                (target.value, now, record.intent_id),
            )
            return OutboxResult(OutboxResultKind.APPLIED)

        return self._claimed_update(intent_id=intent_id, owner=owner, operation=operation)

    def acknowledge(self, *, intent_id: str, owner: str) -> OutboxResult:
        self._metric("outbox_acknowledgements")
        def operation(conn: sqlite3.Connection, record: LifecycleOutboxRecord, now: float) -> OutboxResult:
            if record.stage is not ExecutionStage.VERIFIED:
                return OutboxResult(OutboxResultKind.CONFLICT, record=record, reason="outbox acknowledgement requires verified stage")
            conn.execute(
                "UPDATE lifecycle_outbox SET lifecycle_stage=?, processing_state=?, lease_owner='', lease_expires_at=0, "
                "acknowledged_at=?, updated_at=? WHERE intent_id=?",
                (
                    ExecutionStage.FINALIZED.value, OutboxProcessingState.ACKNOWLEDGED.value,
                    now, now, record.intent_id,
                ),
            )
            return OutboxResult(OutboxResultKind.APPLIED)

        return self._claimed_update(intent_id=intent_id, owner=owner, operation=operation)

    def release_retry(self, *, intent_id: str, owner: str, failure: OutboxFailure) -> OutboxResult:
        self._metric("outbox_retry_releases")
        def operation(conn: sqlite3.Connection, record: LifecycleOutboxRecord, now: float) -> OutboxResult:
            if record.attempts >= record.plan.max_attempts:
                self._quarantine_row(
                    conn, record.intent_id, now,
                    OutboxFailure("retry_exhausted", failure.message, failure.evidence),
                )
                return OutboxResult(OutboxResultKind.REJECTED, reason="outbox retry budget exhausted")
            conn.execute(
                "UPDATE lifecycle_outbox SET processing_state=?, lease_owner='', lease_expires_at=0, failure_json=?, updated_at=? "
                "WHERE intent_id=?",
                (OutboxProcessingState.RETRY.value, failure.to_json(), now, record.intent_id),
            )
            return OutboxResult(OutboxResultKind.APPLIED)

        return self._claimed_update(intent_id=intent_id, owner=owner, operation=operation)

    def manual_review(self, *, intent_id: str, owner: str, failure: OutboxFailure) -> OutboxResult:
        self._metric("outbox_manual_reviews")
        def operation(conn: sqlite3.Connection, record: LifecycleOutboxRecord, now: float) -> OutboxResult:
            conn.execute(
                "UPDATE lifecycle_outbox SET lifecycle_stage=?, processing_state=?, lease_owner='', lease_expires_at=0, "
                "failure_json=?, updated_at=? WHERE intent_id=?",
                (
                    ExecutionStage.FINALIZED.value, OutboxProcessingState.MANUAL_REVIEW.value,
                    failure.to_json(), now, record.intent_id,
                ),
            )
            return OutboxResult(OutboxResultKind.APPLIED)

        return self._claimed_update(intent_id=intent_id, owner=owner, operation=operation)

    def status(
        self,
        *,
        limit: int = 20,
        stale_after: float = 300.0,
        retention_seconds: float = OUTBOX_ACK_RETENTION_SECONDS,
    ) -> tuple[OutboxResult, dict[str, Any]]:
        """Return a typed, read-only operational view of the outbox.

        Operator tools use this method instead of decoding the SQLite schema
        themselves.  The repository remains the owner of row validation,
        state names, failure evidence, and lease semantics.
        """
        empty = {
            "schema_version": OUTBOX_SCHEMA_VERSION,
            "integrity": "not_checked",
            "states": {},
            "stale_claims": 0,
            "max_attempts": 0,
            "retention": {
                "retention_seconds": float(retention_seconds),
                "acknowledged": 0,
                "eligible": 0,
                "oldest_age_s": 0,
            },
            "records": [],
        }
        conn: sqlite3.Connection | None = None
        now = self._clock()
        try:
            # Status is an operator read, not an initialization path.  Do not
            # create the state directory, negotiate WAL, or repair a schema
            # merely because a diagnostic command was run.
            if not self.path.exists():
                return OutboxResult(OutboxResultKind.APPLIED), empty
            conn = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro",
                uri=True,
                timeout=self.connect_timeout,
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={int(self.connect_timeout * 2000)}")
            version_row = conn.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0] if version_row else 0)
            empty["schema_version"] = version
            if version != OUTBOX_SCHEMA_VERSION:
                raise LifecycleOutboxError(
                    f"outbox schema v{version} is incompatible with v{OUTBOX_SCHEMA_VERSION}"
                )
            self._validate_schema(conn)
            integrity_row = conn.execute("PRAGMA quick_check").fetchone()
            integrity = str(integrity_row[0] if integrity_row else "unknown")
            empty["integrity"] = integrity
            states = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT processing_state, COUNT(*) FROM lifecycle_outbox GROUP BY processing_state"
                )
            }
            stale_after = max(0.0, float(stale_after))
            empty["stale_claims"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM lifecycle_outbox "
                    "WHERE processing_state=? AND lease_expires_at <= ?",
                    (OutboxProcessingState.CLAIMED.value, now - stale_after),
                ).fetchone()[0]
                or 0
            )
            empty["max_attempts"] = int(
                conn.execute("SELECT COALESCE(MAX(attempts), 0) FROM lifecycle_outbox").fetchone()[0] or 0
            )
            retention = float(retention_seconds)
            if retention < 0 or retention != retention or retention in {float("inf"), float("-inf")}:
                raise LifecycleOutboxError("retention_seconds must be finite and non-negative")
            retention_cutoff = now - retention
            retention_row = conn.execute(
                "SELECT COUNT(*), COALESCE(MIN(acknowledged_at), 0), "
                "SUM(CASE WHEN acknowledged_at > 0 AND acknowledged_at <= ? THEN 1 ELSE 0 END) "
                "FROM lifecycle_outbox WHERE processing_state=?",
                (retention_cutoff, OutboxProcessingState.ACKNOWLEDGED.value),
            ).fetchone()
            oldest_ack = float(retention_row[1] or 0)
            empty["retention"] = {
                "retention_seconds": retention,
                "acknowledged": int(retention_row[0] or 0),
                "eligible": int(retention_row[2] or 0),
                "oldest_age_s": max(0, int(now - oldest_ack)) if oldest_ack else 0,
            }
            records = []
            for row in conn.execute(
                "SELECT * FROM lifecycle_outbox ORDER BY updated_at ASC, intent_id ASC LIMIT ?",
                (max(0, int(limit)),),
            ):
                try:
                    record = self._from_row(row)
                    records.append(
                        {
                            "intent_id": record.intent_id,
                            "state": record.state.value,
                            "stage": record.stage.value,
                            "attempts": record.attempts,
                            "lease_expires_at": record.lease_expires_at,
                            "lease_age_s": (
                                max(0, int(now - record.lease_expires_at))
                                if record.lease_expires_at
                                else 0
                            ),
                            "failure": None if record.failure is None else {
                                "code": record.failure.code,
                                "message": record.failure.message,
                            },
                        }
                    )
                except LifecycleOutboxError as exc:
                    records.append(
                        {"intent_id": str(row["intent_id"]), "state": "poison", "reason": str(exc)}
                    )
            empty["states"] = states
            empty["records"] = records
            return OutboxResult(OutboxResultKind.APPLIED), empty
        except LifecycleOutboxError as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc)), empty
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc)), empty
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}"), empty
        finally:
            if conn is not None:
                conn.close()

    def snapshot_records(self) -> tuple[OutboxResult, tuple[Any, ...]]:
        """Read every validated intent for one immutable integrity snapshot.

        This is deliberately a repository operation: callers do not inspect
        SQLite rows or reconstruct lifecycle plans themselves.  A poison row
        makes the complete snapshot rejected rather than silently omitted.
        """
        if not self.path.exists():
            return OutboxResult(OutboxResultKind.APPLIED), ()
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro",
                uri=True,
                timeout=self.connect_timeout,
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={int(self.connect_timeout * 2000)}")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            if version != OUTBOX_SCHEMA_VERSION:
                raise LifecycleOutboxError(
                    f"outbox schema v{version} is incompatible with v{OUTBOX_SCHEMA_VERSION}"
                )
            self._validate_schema(conn)
            from .integrity_outbox_envelope import IntegrityOutboxEnvelope, IntegrityOutboxRecord

            records: list[Any] = []
            for row in conn.execute("SELECT * FROM lifecycle_outbox ORDER BY intent_id ASC"):
                try:
                    if str(row["work_kind"] or "lifecycle") == "integrity":
                        encoded = str(row["plan_json"] or "")
                        envelope = IntegrityOutboxEnvelope.from_dict(json.loads(encoded))
                        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != str(row["plan_fingerprint"] or ""):
                            raise LifecycleOutboxError("integrity envelope fingerprint mismatch")
                        records.append(IntegrityOutboxRecord(
                            envelope,
                            OutboxProcessingState(str(row["processing_state"] or "")),
                            ExecutionStage(str(row["lifecycle_stage"] or "")),
                            str(row["lease_owner"] or ""),
                            float(row["lease_expires_at"] or 0.0),
                            int(row["attempts"] or 0),
                        ))
                    else:
                        records.append(self._from_row(row))
                except LifecycleOutboxError as exc:
                    return OutboxResult(OutboxResultKind.REJECTED, reason=f"poison outbox row: {exc}"), ()
                except Exception as exc:
                    return OutboxResult(OutboxResultKind.REJECTED, reason=f"poison outbox row: {type(exc).__name__}: {exc}"), ()
            return OutboxResult(OutboxResultKind.APPLIED), tuple(records)
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc)), ()
        except LifecycleOutboxError as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc)), ()
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}"), ()
        finally:
            if conn is not None:
                conn.close()

    def prune_acknowledged(
        self,
        *,
        retention_seconds: float = OUTBOX_ACK_RETENTION_SECONDS,
        limit: int = 1000,
        checkpoint: bool = False,
    ) -> OutboxMaintenanceResult:
        """Remove only acknowledged records older than a conservative boundary.

        Cleanup is explicit and bounded.  No retry, claimed, quarantined, or
        manual-review row can match the delete predicate, so their evidence is
        retained even when an operator runs maintenance repeatedly.
        """
        try:
            retention = float(retention_seconds)
        except (TypeError, ValueError):
            return OutboxMaintenanceResult(OutboxResultKind.REJECTED, reason="retention_seconds must be finite")
        if retention < 0 or not retention == retention or retention in {float("inf"), float("-inf")}:
            return OutboxMaintenanceResult(OutboxResultKind.REJECTED, reason="retention_seconds must be finite and non-negative")
        if isinstance(limit, bool) or int(limit) <= 0:
            return OutboxMaintenanceResult(OutboxResultKind.REJECTED, reason="maintenance limit must be positive")
        now = self._clock()
        cutoff = now - retention
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            self._secure_state_files()
            with _transaction(conn):
                rows = conn.execute(
                    "SELECT intent_id FROM lifecycle_outbox "
                    "WHERE processing_state=? AND acknowledged_at > 0 AND acknowledged_at <= ? "
                    "ORDER BY acknowledged_at ASC, intent_id ASC LIMIT ?",
                    (OutboxProcessingState.ACKNOWLEDGED.value, cutoff, int(limit)),
                ).fetchall()
                removed = 0
                for row in rows:
                    result = conn.execute(
                        "DELETE FROM lifecycle_outbox WHERE intent_id=? AND processing_state=? "
                        "AND acknowledged_at > 0 AND acknowledged_at <= ?",
                        (str(row[0]), OutboxProcessingState.ACKNOWLEDGED.value, cutoff),
                    )
                    removed += int(result.rowcount or 0)
            checkpoint_state = "not_requested"
            if checkpoint:
                checkpoint_row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                checkpoint_state = "completed" if checkpoint_row is not None else "unavailable"
            return OutboxMaintenanceResult(
                OutboxResultKind.APPLIED,
                removed=removed,
                cutoff=cutoff,
                retention_seconds=retention,
                checkpoint=checkpoint_state,
            )
        except sqlite3.OperationalError as exc:
            return OutboxMaintenanceResult(
                OutboxResultKind.RETRYABLE,
                cutoff=cutoff,
                retention_seconds=retention,
                reason=str(exc),
                lock_busy=_busy(exc),
            )
        except Exception as exc:
            return OutboxMaintenanceResult(
                OutboxResultKind.REJECTED,
                cutoff=cutoff,
                retention_seconds=retention,
                reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if conn is not None:
                conn.close()

    def opportunistic_housekeeping(
        self,
        *,
        retention_seconds: float = OUTBOX_ACK_RETENTION_SECONDS,
        interval_seconds: float = OUTBOX_HOUSEKEEPING_INTERVAL_SECONDS,
        size_threshold_bytes: int = OUTBOX_HOUSEKEEPING_SIZE_THRESHOLD_BYTES,
        limit: int = OUTBOX_HOUSEKEEPING_ROW_LIMIT,
        checkpoint: bool = True,
    ) -> OutboxMaintenanceResult:
        """Run bounded maintenance when the persisted cooldown allows it.

        This is intentionally separate from hook paths.  Reconcile can call it
        after a successful apply; a cooldown and size/eligibility gates make
        routine runs cheap while keeping cleanup automatic for operators.
        """
        try:
            retention = float(retention_seconds)
            interval = float(interval_seconds)
            size_threshold = int(size_threshold_bytes)
            row_limit = int(limit)
        except (TypeError, ValueError):
            return OutboxMaintenanceResult(OutboxResultKind.REJECTED, reason="invalid housekeeping limits")
        if (
            retention < 0
            or retention != retention
            or retention in {float("inf"), float("-inf")}
            or interval < 0
            or interval != interval
            or interval in {float("inf"), float("-inf")}
            or size_threshold < 0
            or row_limit <= 0
        ):
            return OutboxMaintenanceResult(OutboxResultKind.REJECTED, reason="invalid housekeeping limits")
        if not self.path.exists():
            return OutboxMaintenanceResult(OutboxResultKind.APPLIED, skipped=True, reason="outbox_absent")

        now = self._clock()
        cutoff = now - retention
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            self._secure_state_files()
            with _transaction(conn):
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS lifecycle_maintenance ("
                    "key TEXT PRIMARY KEY, value REAL NOT NULL)"
                )
                previous = conn.execute(
                    "SELECT value FROM lifecycle_maintenance WHERE key='housekeeping_last_attempt'"
                ).fetchone()
                last_attempt = float(previous[0]) if previous is not None else 0.0
                eligible = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM lifecycle_outbox "
                        "WHERE processing_state=? AND acknowledged_at > 0 AND acknowledged_at <= ?",
                        (OutboxProcessingState.ACKNOWLEDGED.value, cutoff),
                    ).fetchone()[0]
                    or 0
                )
                db_size = int(self.path.stat().st_size) if self.path.exists() else 0
                if last_attempt > 0 and now - last_attempt < interval:
                    return OutboxMaintenanceResult(
                        OutboxResultKind.APPLIED,
                        cutoff=cutoff,
                        retention_seconds=retention,
                        skipped=True,
                        reason="cooldown",
                    )
                if eligible == 0 and db_size < size_threshold:
                    return OutboxMaintenanceResult(
                        OutboxResultKind.APPLIED,
                        cutoff=cutoff,
                        retention_seconds=retention,
                        skipped=True,
                        reason="no_work",
                    )
                conn.execute(
                    "INSERT INTO lifecycle_maintenance(key, value) VALUES('housekeeping_last_attempt', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (now,),
                )
                rows = conn.execute(
                    "SELECT intent_id FROM lifecycle_outbox "
                    "WHERE processing_state=? AND acknowledged_at > 0 AND acknowledged_at <= ? "
                    "ORDER BY acknowledged_at ASC, intent_id ASC LIMIT ?",
                    (OutboxProcessingState.ACKNOWLEDGED.value, cutoff, row_limit),
                ).fetchall()
                removed = 0
                for row in rows:
                    result = conn.execute(
                        "DELETE FROM lifecycle_outbox WHERE intent_id=? AND processing_state=? "
                        "AND acknowledged_at > 0 AND acknowledged_at <= ?",
                        (str(row[0]), OutboxProcessingState.ACKNOWLEDGED.value, cutoff),
                    )
                    removed += int(result.rowcount or 0)
            checkpoint_state = "not_requested"
            if checkpoint and removed:
                checkpoint_row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                checkpoint_state = "completed" if checkpoint_row is not None else "unavailable"
            return OutboxMaintenanceResult(
                OutboxResultKind.APPLIED,
                removed=removed,
                cutoff=cutoff,
                retention_seconds=retention,
                checkpoint=checkpoint_state,
            )
        except sqlite3.OperationalError as exc:
            return OutboxMaintenanceResult(
                OutboxResultKind.RETRYABLE,
                cutoff=cutoff,
                retention_seconds=retention,
                reason=str(exc),
                lock_busy=_busy(exc),
            )
        except Exception as exc:
            return OutboxMaintenanceResult(
                OutboxResultKind.REJECTED,
                cutoff=cutoff,
                retention_seconds=retention,
                reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            if conn is not None:
                conn.close()


__all__ = (
    "LifecycleOutboxError",
    "LifecycleOutboxRecord",
    "LifecycleOutboxRepository",
    "OUTBOX_SCHEMA_VERSION",
    "OUTBOX_ACK_RETENTION_SECONDS",
    "OUTBOX_HOUSEKEEPING_INTERVAL_SECONDS",
    "OUTBOX_HOUSEKEEPING_SIZE_THRESHOLD_BYTES",
    "OUTBOX_HOUSEKEEPING_ROW_LIMIT",
    "OutboxFailure",
    "OutboxMaintenanceResult",
    "OutboxProcessingState",
    "OutboxResult",
    "OutboxResultKind",
    "lifecycle_outbox_path",
)
