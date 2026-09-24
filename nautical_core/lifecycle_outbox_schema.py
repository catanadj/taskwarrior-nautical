"""SQLite schema ownership for the lifecycle outbox."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import AbstractContextManager


OUTBOX_SCHEMA_VERSION = 2
OUTBOX_LEGACY_SCHEMA_VERSION = 1
_INIT_RETRIES = 8
_INIT_BACKOFF_S = 0.025
_MAX_INIT_BACKOFF_S = 0.25


def _busy(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def validate_schema(conn: sqlite3.Connection, *, error_type: type[Exception] = ValueError) -> None:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
    required = {
        "intent_id", "work_kind", "plan_json", "plan_fingerprint", "parent_guard_json",
        "configuration_fingerprint", "schedule_fingerprint", "lifecycle_stage",
        "processing_state", "lease_owner", "lease_expires_at", "attempts",
        "failure_json", "created_at", "updated_at", "acknowledged_at",
    }
    missing = sorted(required - columns)
    if missing:
        raise error_type(f"outbox schema is incomplete: missing {', '.join(missing)}")


def initialize(
    conn: sqlite3.Connection,
    *,
    transaction: Callable[[sqlite3.Connection], AbstractContextManager[None]],
    error_type: type[Exception] = ValueError,
) -> None:
    for attempt in range(_INIT_RETRIES):
        try:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
            if version > OUTBOX_SCHEMA_VERSION:
                raise error_type(
                    f"outbox schema v{version} is newer than supported v{OUTBOX_SCHEMA_VERSION}"
                )
            if version == OUTBOX_SCHEMA_VERSION:
                validate_schema(conn, error_type=error_type)
                return
            if version == OUTBOX_LEGACY_SCHEMA_VERSION:
                with transaction(conn):
                    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
                    if "work_kind" not in columns:
                        conn.execute(
                            "ALTER TABLE lifecycle_outbox ADD COLUMN work_kind TEXT NOT NULL DEFAULT 'lifecycle'"
                        )
                    conn.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION}")
                validate_schema(conn, error_type=error_type)
                return
            if version != 0:
                raise error_type(f"unsupported outbox schema v{version}")

            conn.execute("PRAGMA journal_mode=WAL")
            with transaction(conn):
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
            validate_schema(conn, error_type=error_type)
            return
        except sqlite3.OperationalError as exc:
            if not _busy(exc) or attempt + 1 >= _INIT_RETRIES:
                raise
            time.sleep(min(_MAX_INIT_BACKOFF_S, _INIT_BACKOFF_S * (2**attempt)))
