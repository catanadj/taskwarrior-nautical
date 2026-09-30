"""Connection-scoped SQL operations for bounded lifecycle outbox maintenance."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import sqlite3
from typing import Callable


TransactionFactory = Callable[[sqlite3.Connection], AbstractContextManager[None]]


@dataclass(frozen=True, slots=True)
class HousekeepingOutcome:
    removed: int
    skipped: bool
    reason: str
    checkpoint: str


def _delete_acknowledged_rows(
    connection: sqlite3.Connection,
    *,
    cutoff: float,
    limit: int,
) -> int:
    rows = connection.execute(
        "SELECT intent_id FROM lifecycle_outbox "
        "WHERE processing_state='acknowledged' AND acknowledged_at > 0 AND acknowledged_at <= ? "
        "ORDER BY acknowledged_at ASC, intent_id ASC LIMIT ?",
        (cutoff, int(limit)),
    ).fetchall()
    removed = 0
    for row in rows:
        result = connection.execute(
            "DELETE FROM lifecycle_outbox WHERE intent_id=? AND processing_state='acknowledged' "
            "AND acknowledged_at > 0 AND acknowledged_at <= ?",
            (str(row[0]), cutoff),
        )
        removed += int(result.rowcount or 0)
    return removed


def prune_acknowledged_rows(
    connection: sqlite3.Connection,
    *,
    cutoff: float,
    limit: int,
    transaction: TransactionFactory,
) -> int:
    """Delete only the oldest acknowledged rows at or before the cutoff."""
    with transaction(connection):
        return _delete_acknowledged_rows(connection, cutoff=cutoff, limit=limit)


def checkpoint_wal(
    connection: sqlite3.Connection,
    *,
    requested: bool,
    removed: int | None = None,
) -> str:
    """Run the passive checkpoint when the owning operation's policy allows."""
    if not requested or (removed is not None and removed <= 0):
        return "not_requested"
    row = connection.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
    return "completed" if row is not None else "unavailable"


def housekeeping_rows(
    connection: sqlite3.Connection,
    *,
    now: float,
    cutoff: float,
    interval_seconds: float,
    size_threshold_bytes: int,
    limit: int,
    checkpoint: bool,
    database_size: Callable[[], int],
    transaction: TransactionFactory,
) -> HousekeepingOutcome:
    """Apply persisted cooldown/size gates, prune a bounded wave, then checkpoint."""
    with transaction(connection):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS lifecycle_maintenance ("
            "key TEXT PRIMARY KEY, value REAL NOT NULL)"
        )
        previous = connection.execute(
            "SELECT value FROM lifecycle_maintenance WHERE key='housekeeping_last_attempt'"
        ).fetchone()
        last_attempt = float(previous[0]) if previous is not None else 0.0
        eligible = int(connection.execute(
            "SELECT COUNT(*) FROM lifecycle_outbox "
            "WHERE processing_state='acknowledged' AND acknowledged_at > 0 AND acknowledged_at <= ?",
            (cutoff,),
        ).fetchone()[0] or 0)
        db_size = int(database_size())
        if last_attempt > 0 and now - last_attempt < interval_seconds:
            return HousekeepingOutcome(0, True, "cooldown", "not_requested")
        if eligible == 0 and db_size < size_threshold_bytes:
            return HousekeepingOutcome(0, True, "no_work", "not_requested")
        connection.execute(
            "INSERT INTO lifecycle_maintenance(key, value) VALUES('housekeeping_last_attempt', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (now,),
        )
        removed = _delete_acknowledged_rows(connection, cutoff=cutoff, limit=limit)
    checkpoint_state = checkpoint_wal(connection, requested=checkpoint, removed=removed)
    return HousekeepingOutcome(removed, False, "", checkpoint_state)


__all__ = (
    "HousekeepingOutcome",
    "TransactionFactory",
    "checkpoint_wal",
    "housekeeping_rows",
    "prune_acknowledged_rows",
)
