"""Read-only SQL and projections for lifecycle outbox operator views."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Callable, Mapping, Protocol, TypeVar

from .lifecycle_outbox_schema import OUTBOX_SCHEMA_VERSION


class StatusRowPoison(Exception):
    """A lifecycle row failed its repository decoder but remains reportable."""


class _EnumLike(Protocol):
    value: str


class _Identity(Protocol):
    event: _EnumLike
    chain_id: str
    parent_uuid: str
    source_link: int
    target_link: int


class _Guard(Protocol):
    def to_dict(self) -> dict[str, object]: ...


class _Plan(Protocol):
    action: _EnumLike
    identity: _Identity
    parent_guard: _Guard

    def child_dict(self) -> dict[str, object]: ...


class _Failure(Protocol):
    code: str
    message: str


class StatusLifecycleRecord(Protocol):
    intent_id: str
    state: _EnumLike
    stage: _EnumLike
    attempts: int
    lease_expires_at: float
    failure: _Failure | None
    plan: _Plan


@dataclass(frozen=True, slots=True)
class OutboxStatusFailureSummary:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class OutboxStatusPlanSummary:
    schema_version: int
    action: str
    event: str
    chainID: str
    parent_uuid: str
    source_link: int
    target_link: int
    parent_guard: Mapping[str, object]
    child_uuid: object


@dataclass(frozen=True, slots=True)
class OutboxStatusRecordSummary:
    intent_id: str
    state: str
    stage: str | None = None
    attempts: int = 0
    lease_expires_at: float = 0.0
    lease_age_s: int = 0
    failure: OutboxStatusFailureSummary | None = None
    plan: OutboxStatusPlanSummary | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class OutboxStatusSummary:
    schema_version: int
    integrity: str
    states: Mapping[str, int]
    stale_claims: int
    max_attempts: int
    retention_seconds: float
    acknowledged: int
    eligible: int
    oldest_age_s: int
    records: tuple[OutboxStatusRecordSummary, ...]


def status_summary(
    connection: sqlite3.Connection,
    *,
    now: float,
    limit: int,
    stale_after: float,
    retention_seconds: float,
    intent_id: str | None,
    decode_row: Callable[[sqlite3.Row], StatusLifecycleRecord],
) -> OutboxStatusSummary:
    """Query and project status using a caller-owned, already-validated connection."""
    version_row = connection.execute("PRAGMA user_version").fetchone()
    schema_version = int(version_row[0] if version_row else 0)
    integrity_row = connection.execute("PRAGMA quick_check").fetchone()
    integrity = str(integrity_row[0] if integrity_row else "unknown")
    states = {
        str(row[0]): int(row[1])
        for row in connection.execute(
            "SELECT processing_state, COUNT(*) FROM lifecycle_outbox GROUP BY processing_state"
        )
    }
    stale_after = max(0.0, float(stale_after))
    stale_claims = int(connection.execute(
        "SELECT COUNT(*) FROM lifecycle_outbox "
        "WHERE processing_state='claimed' AND lease_expires_at <= ?",
        (float(now) - stale_after,),
    ).fetchone()[0] or 0)
    max_attempts = int(connection.execute(
        "SELECT COALESCE(MAX(attempts), 0) FROM lifecycle_outbox"
    ).fetchone()[0] or 0)
    retention = float(retention_seconds)
    if retention < 0 or retention != retention or retention in {float("inf"), float("-inf")}:
        raise ValueError("retention_seconds must be finite and non-negative")
    cutoff = float(now) - retention
    retention_row = connection.execute(
        "SELECT COUNT(*), COALESCE(MIN(acknowledged_at), 0), "
        "SUM(CASE WHEN acknowledged_at > 0 AND acknowledged_at <= ? THEN 1 ELSE 0 END) "
        "FROM lifecycle_outbox WHERE processing_state='acknowledged'",
        (cutoff,),
    ).fetchone()
    oldest_ack = float(retention_row[1] or 0)
    if intent_id:
        rows = connection.execute(
            "SELECT * FROM lifecycle_outbox WHERE intent_id=?", (str(intent_id).strip(),)
        )
    else:
        rows = connection.execute(
            "SELECT * FROM lifecycle_outbox "
            "ORDER BY CASE processing_state "
            "WHEN 'manual_review' THEN 0 WHEN 'quarantined' THEN 1 "
            "WHEN 'retry' THEN 2 WHEN 'claimed' THEN 3 WHEN 'ready' THEN 4 "
            "ELSE 5 END, updated_at ASC, intent_id ASC LIMIT ?",
            (max(0, int(limit)),),
        )
    records: list[OutboxStatusRecordSummary] = []
    for row in rows:
        try:
            record = decode_row(row)
        except StatusRowPoison as exc:
            records.append(OutboxStatusRecordSummary(
                intent_id=str(row["intent_id"]), state="poison", reason=str(exc)
            ))
            continue
        plan = record.plan
        failure = None if record.failure is None else OutboxStatusFailureSummary(
            code=record.failure.code, message=record.failure.message
        )
        records.append(OutboxStatusRecordSummary(
            intent_id=record.intent_id,
            state=record.state.value,
            stage=record.stage.value,
            attempts=record.attempts,
            lease_expires_at=record.lease_expires_at,
            lease_age_s=max(0, int(float(now) - record.lease_expires_at)) if record.lease_expires_at else 0,
            failure=failure,
            plan=OutboxStatusPlanSummary(
                schema_version=OUTBOX_SCHEMA_VERSION,
                action=plan.action.value,
                event=plan.identity.event.value,
                chainID=plan.identity.chain_id,
                parent_uuid=plan.identity.parent_uuid,
                source_link=plan.identity.source_link,
                target_link=plan.identity.target_link,
                parent_guard=plan.parent_guard.to_dict(),
                child_uuid=plan.child_dict().get("uuid"),
            ),
        ))
    return OutboxStatusSummary(
        schema_version=schema_version,
        integrity=integrity,
        states=states,
        stale_claims=stale_claims,
        max_attempts=max_attempts,
        retention_seconds=retention,
        acknowledged=int(retention_row[0] or 0),
        eligible=int(retention_row[2] or 0),
        oldest_age_s=max(0, int(float(now) - oldest_ack)) if oldest_ack else 0,
        records=tuple(records),
    )


T = TypeVar("T")
U = TypeVar("U")


def snapshot_rows(
    connection: sqlite3.Connection,
    *,
    decode_lifecycle_row: Callable[[sqlite3.Row], T],
    decode_integrity_row: Callable[[sqlite3.Row], U],
) -> tuple[T | U, ...]:
    """Decode a complete deterministic snapshot without managing connection lifetime."""
    records: list[T | U] = []
    for row in connection.execute("SELECT * FROM lifecycle_outbox ORDER BY intent_id ASC"):
        if str(row["work_kind"] or "lifecycle") == "integrity":
            records.append(decode_integrity_row(row))
        else:
            records.append(decode_lifecycle_row(row))
    return tuple(records)


__all__ = (
    "OutboxStatusFailureSummary",
    "OutboxStatusPlanSummary",
    "OutboxStatusRecordSummary",
    "OutboxStatusSummary",
    "StatusLifecycleRecord",
    "StatusRowPoison",
    "snapshot_rows",
    "status_summary",
)
