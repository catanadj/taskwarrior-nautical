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
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Callable, Iterator, Sequence
import uuid

from nautical_core.lifecycle_models import ExecutionStage, LifecycleContractError, LifecyclePlan


OUTBOX_SCHEMA_VERSION = 1
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
        durable: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.taskdata = Path(taskdata).resolve()
        self.path = lifecycle_outbox_path(self.taskdata)
        self.connect_timeout = max(0.1, float(connect_timeout))
        self.durable = bool(durable)
        self._clock = clock

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        conn = sqlite3.connect(str(self.path), timeout=self.connect_timeout)
        os.chmod(self.path, 0o600)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA synchronous={'FULL' if self.durable else 'NORMAL'}")
        conn.execute(f"PRAGMA busy_timeout={int(self.connect_timeout * 2000)}")
        return conn

    def _secure_state_files(self) -> None:
        """Keep the outbox and SQLite sidecars private to the Taskwarrior user."""
        os.chmod(self.path.parent, 0o700)
        for path in (self.path, self.path.with_name(f"{self.path.name}-wal"), self.path.with_name(f"{self.path.name}-shm")):
            if path.exists():
                os.chmod(path, 0o600)

    def open(self) -> OutboxResult:
        last: Exception | None = None
        for attempt in range(_INIT_RETRIES):
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect()
                self._initialize(conn)
                self._secure_state_files()
                return OutboxResult(OutboxResultKind.APPLIED)
            except sqlite3.OperationalError as exc:
                last = exc
                if not _busy(exc) or attempt + 1 >= _INIT_RETRIES:
                    return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc))
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
                # WAL is negotiated once per durable database. Competing
                # first-openers retry the complete sequence rather than using
                # a process-local success flag.
                conn.execute("PRAGMA journal_mode=WAL")
                with _transaction(conn):
                    version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
                    if version > OUTBOX_SCHEMA_VERSION:
                        raise LifecycleOutboxError(
                            f"outbox schema v{version} is newer than supported v{OUTBOX_SCHEMA_VERSION}"
                        )
                    if version not in {0, OUTBOX_SCHEMA_VERSION}:
                        raise LifecycleOutboxError(f"unsupported outbox schema v{version}")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS lifecycle_outbox (
                            intent_id TEXT PRIMARY KEY,
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
            "intent_id", "plan_json", "plan_fingerprint", "parent_guard_json",
            "configuration_fingerprint", "schedule_fingerprint", "lifecycle_stage",
            "processing_state", "lease_owner", "lease_expires_at", "attempts",
            "failure_json", "created_at", "updated_at", "acknowledged_at",
        }
        missing = sorted(required - columns)
        if missing:
            raise LifecycleOutboxError(f"outbox schema is incomplete: missing {', '.join(missing)}")

    def _with_connection(self, operation: Callable[[sqlite3.Connection], OutboxResult]) -> OutboxResult:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            result = operation(conn)
            self._secure_state_files()
            return result
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc))
        except LifecycleOutboxError as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=str(exc))
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}")
        finally:
            if conn is not None:
                conn.close()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> LifecycleOutboxRecord:
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
                    compatible = (
                        str(row["plan_fingerprint"]) == plan_fingerprint
                        and current.configuration_fingerprint == config
                        and current.schedule_fingerprint == schedule
                    )
                    if not compatible:
                        return OutboxResult(
                            OutboxResultKind.CONFLICT,
                            record=current,
                            reason="deterministic lifecycle intent already exists with different immutable inputs",
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

    def claim_batch(self, *, owner: str, lease_seconds: float, limit: int) -> tuple[OutboxResult, tuple[LifecycleOutboxRecord, ...]]:
        owner = str(owner or "").strip()
        if not owner or lease_seconds <= 0 or limit <= 0:
            return OutboxResult(OutboxResultKind.REJECTED, reason="outbox claim requires owner, lease, and limit"), ()
        now = self._clock()
        expires = now + float(lease_seconds)
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            with _transaction(conn):
                conn.execute(
                    "UPDATE lifecycle_outbox SET processing_state=?, lease_owner='', lease_expires_at=0, updated_at=? "
                    "WHERE processing_state=? AND lease_expires_at <= ?",
                    (OutboxProcessingState.RETRY.value, now, OutboxProcessingState.CLAIMED.value, now),
                )
                candidates = conn.execute(
                    "SELECT intent_id FROM lifecycle_outbox WHERE processing_state IN (?, ?) "
                    "ORDER BY created_at, intent_id LIMIT ?",
                    (*_ACTIVE_STATES, int(limit)),
                ).fetchall()
                records: list[LifecycleOutboxRecord] = []
                for raw in candidates:
                    intent_id = str(raw[0])
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

    def status(self, *, limit: int = 20) -> tuple[OutboxResult, dict[str, Any]]:
        empty = {"schema_version": OUTBOX_SCHEMA_VERSION, "states": {}, "records": []}
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialize(conn)
            states = {
                str(row[0]): int(row[1])
                for row in conn.execute(
                    "SELECT processing_state, COUNT(*) FROM lifecycle_outbox GROUP BY processing_state"
                )
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
                            "failure": None if record.failure is None else {
                                "code": record.failure.code,
                                "message": record.failure.message,
                            },
                        }
                    )
                except LifecycleOutboxError as exc:
                    records.append({"intent_id": str(row["intent_id"]), "state": "poison", "reason": str(exc)})
            return OutboxResult(OutboxResultKind.APPLIED), {
                "schema_version": OUTBOX_SCHEMA_VERSION,
                "states": states,
                "records": records,
            }
        except sqlite3.OperationalError as exc:
            return OutboxResult(OutboxResultKind.RETRYABLE, reason=str(exc), lock_busy=_busy(exc)), empty
        except Exception as exc:
            return OutboxResult(OutboxResultKind.REJECTED, reason=f"{type(exc).__name__}: {exc}"), empty
        finally:
            if conn is not None:
                conn.close()


__all__ = (
    "LifecycleOutboxError",
    "LifecycleOutboxRecord",
    "LifecycleOutboxRepository",
    "OUTBOX_SCHEMA_VERSION",
    "OutboxFailure",
    "OutboxProcessingState",
    "OutboxResult",
    "OutboxResultKind",
    "lifecycle_outbox_path",
)
