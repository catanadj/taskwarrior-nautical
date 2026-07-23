from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

from nautical_core.queue_models import (
    QueueEntriesBatch,
    QueueOpenResult,
    QueueRowClaimResult,
    QueueStoredRow,
    QueueWriteResult,
)


QUEUE_DB_SCHEMA_VERSION = 1
_QUEUE_COLUMNS = (
    ("id", "INTEGER", 0, None, 1),
    ("spawn_intent_id", "TEXT", 0, None, 0),
    ("payload", "TEXT", 1, None, 0),
    ("attempts", "INTEGER", 1, "0", 0),
    ("state", "TEXT", 1, "'queued'", 0),
    ("claim_token", "TEXT", 0, None, 0),
    ("claimed_at", "REAL", 0, None, 0),
    ("created_at", "REAL", 1, None, 0),
    ("updated_at", "REAL", 1, None, 0),
)
_QUEUE_INDEXES = {
    "idx_queue_entries_spawn_intent": (("spawn_intent_id",), 1, 1),
    "idx_queue_entries_state_id": (("state", "id"), 0, 0),
    "idx_queue_entries_claimed_at": (("claimed_at",), 0, 0),
}


class QueueSchemaError(RuntimeError):
    pass


def nautical_state_dir_path(tw_data_dir: Path) -> Path:
    return tw_data_dir / ".nautical-state"


def nautical_lock_dir_path(tw_data_dir: Path) -> Path:
    return tw_data_dir / ".nautical-locks"


def parent_nextlink_lock_path(tw_data_dir: Path, parent_uuid: str) -> Path:
    raw = str(parent_uuid or "").strip().lower()
    safe = "".join(ch for ch in raw if ch.isalnum())[:64] or "unknown"
    return nautical_lock_dir_path(tw_data_dir) / f".nautical_parent_nextlink.{safe}.lock"


def legacy_state_path(tw_data_dir: Path, name: str) -> Path:
    return tw_data_dir / name


def queue_db_path(tw_data_dir: Path) -> Path:
    return nautical_state_dir_path(tw_data_dir) / ".nautical_queue.db"


def dead_letter_path(tw_data_dir: Path) -> Path:
    return nautical_state_dir_path(tw_data_dir) / ".nautical_dead_letter.jsonl"


def dead_letter_lock_path(tw_data_dir: Path) -> Path:
    return nautical_lock_dir_path(tw_data_dir) / ".nautical_dead_letter.lock"


def maybe_migrate_state_file(current: Path, legacy: Path) -> None:
    try:
        if current.exists() or not legacy.exists():
            return
        current.parent.mkdir(parents=True, exist_ok=True)
        os.replace(legacy, current)
    except Exception:
        pass


def maybe_migrate_state_sidecars(current: Path, legacy: Path) -> None:
    for suffix in ("-wal", "-shm"):
        maybe_migrate_state_file(Path(str(current) + suffix), Path(str(legacy) + suffix))


def migrate_legacy_state(
    *,
    file_pairs: Sequence[tuple[Path, Path]],
    db_sidecars: Sequence[tuple[Path, Path]] = (),
) -> None:
    for current, legacy in file_pairs:
        maybe_migrate_state_file(current, legacy)
    for current, legacy in db_sidecars:
        maybe_migrate_state_sidecars(current, legacy)


def migrate_nautical_state(
    *,
    tw_data_dir: Path,
    extra_file_pairs: Sequence[tuple[Path, Path]] = (),
) -> None:
    file_pairs = [
        (queue_db_path(tw_data_dir), legacy_state_path(tw_data_dir, ".nautical_queue.db")),
        (dead_letter_path(tw_data_dir), legacy_state_path(tw_data_dir, ".nautical_dead_letter.jsonl")),
    ]
    file_pairs.extend(extra_file_pairs)
    migrate_legacy_state(
        file_pairs=file_pairs,
        db_sidecars=((queue_db_path(tw_data_dir), legacy_state_path(tw_data_dir, ".nautical_queue.db")),),
    )


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass


def build_dead_letter_payload(
    *,
    hook: str,
    hook_version: str,
    entry: dict[str, Any],
    reason: str,
    now_fn: Callable[[], str] | None = None,
) -> dict[str, Any]:
    ts = now_fn() if callable(now_fn) else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "ts": ts,
        "hook": hook,
        "hook_version": hook_version,
        "reason": reason,
        "spawn_intent_id": (entry.get("spawn_intent_id") or "").strip(),
        "parent_uuid": (entry.get("parent_uuid") or "").strip(),
        "child_short": (entry.get("child_short") or "").strip(),
        "child_uuid": ((entry.get("child") or {}).get("uuid") or "").strip(),
        "entry": entry,
    }


def append_dead_letter_jsonl(
    *,
    path: Path,
    payload: dict[str, Any],
    durable: bool,
    acquire_lock: Callable[[], Any] | None = None,
    diag: Callable[[str], None] | None = None,
    max_bytes: int = 0,
    retention_days: int = 0,
) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    lock_ctx = acquire_lock() if callable(acquire_lock) else nullcontext(True)
    try:
        with lock_ctx as locked:
            if locked is False:
                if callable(diag):
                    diag("dead-letter lock busy; entry not recorded")
                return False
            try:
                if max_bytes > 0 and path.exists():
                    try:
                        st = path.stat()
                        if st.st_size > max_bytes:
                            ts = int(time.time())
                            overflow = path.with_suffix(f".overflow.{ts}.jsonl")
                            os.replace(path, overflow)
                            if callable(diag):
                                diag(f"dead-letter rotated: {overflow}")
                            if retention_days > 0:
                                cutoff = time.time() - (retention_days * 86400)
                                candidates = sorted(path.parent.glob(f"{path.stem}.overflow.*.jsonl"))
                                for old in candidates:
                                    try:
                                        if old.stat().st_mtime < cutoff:
                                            old.unlink()
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                fd = os.open(str(path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
                try:
                    os.fchmod(fd, 0o600)
                except Exception:
                    pass
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if durable:
                        try:
                            f.flush()
                            os.fsync(f.fileno())
                            fsync_dir(path.parent)
                        except Exception:
                            pass
                return True
            except Exception:
                return False
    except Exception:
        return False


def sqlite_error_looks_corrupt(exc: Exception) -> bool:
    msg = str(exc or "").lower()
    return (
        "database disk image is malformed" in msg
        or "file is not a database" in msg
        or "malformed database schema" in msg
        or "database corrupted" in msg
    )


def quarantine_sqlite_db(
    db_path: Path,
    reason: Exception | str,
    *,
    diag: Callable[[str], None] | None = None,
) -> bool:
    moved = False
    ts = int(time.time())
    for candidate in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        try:
            if not candidate.exists():
                continue
            bad = candidate.with_name(f"{candidate.name}.corrupt.{ts}")
            os.replace(candidate, bad)
            moved = True
        except Exception:
            continue
    if moved and callable(diag):
        try:
            diag(f"queue db quarantined after corruption: {reason}")
        except Exception:
            pass
    return moved



def _queue_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='queue_entries'"
    ).fetchone()
    return row is not None


def _validate_queue_table(conn: sqlite3.Connection, *, require_indexes: bool) -> None:
    rows = list(conn.execute("PRAGMA table_info(queue_entries)"))
    actual = tuple(
        (
            str(row[1]),
            str(row[2]).upper(),
            int(row[3] or 0),
            None if row[4] is None else str(row[4]),
            int(row[5] or 0),
        )
        for row in rows
    )
    if actual != _QUEUE_COLUMNS:
        actual_names = [item[0] for item in actual]
        expected_names = [item[0] for item in _QUEUE_COLUMNS]
        raise QueueSchemaError(
            "queue database schema is incompatible: "
            f"queue_entries columns are {actual_names}, expected {expected_names}"
        )

    if not require_indexes:
        return

    index_rows = {str(row[1]): row for row in conn.execute("PRAGMA index_list(queue_entries)")}
    for name, (expected_columns, expected_unique, expected_partial) in _QUEUE_INDEXES.items():
        row = index_rows.get(name)
        if row is None:
            raise QueueSchemaError(f"queue database schema is incomplete: missing index {name}")
        actual_columns = tuple(
            str(index_row[2])
            for index_row in conn.execute(f"PRAGMA index_info({name})")
        )
        unique = int(row[2] or 0)
        partial = int(row[4] or 0) if len(row) > 4 else 0
        if (
            actual_columns != expected_columns
            or unique != expected_unique
            or partial != expected_partial
        ):
            raise QueueSchemaError(f"queue database schema is incompatible: index {name} differs")


def _validate_no_duplicate_intents(conn: sqlite3.Connection) -> None:
    duplicate = conn.execute(
        """
        SELECT spawn_intent_id
        FROM queue_entries
        WHERE spawn_intent_id IS NOT NULL AND spawn_intent_id <> ''
        GROUP BY spawn_intent_id
        HAVING COUNT(*) > 1
        LIMIT 1
        """
    ).fetchone()
    if duplicate is not None:
        raise QueueSchemaError(
            "queue database schema is incompatible: duplicate spawn_intent_id values"
        )


def queue_schema_status(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
        version = int(row[0] if row else 0)
        table_present = _queue_table_exists(conn)
        if version > QUEUE_DB_SCHEMA_VERSION:
            raise QueueSchemaError(
                f"queue database schema v{version} is newer than supported v{QUEUE_DB_SCHEMA_VERSION}"
            )
        if version < 0:
            raise QueueSchemaError(f"queue database schema version is invalid: {version}")
        if version == 0:
            if table_present:
                _validate_queue_table(conn, require_indexes=False)
                _validate_no_duplicate_intents(conn)
            return {
                "status": "legacy",
                "version": 0,
                "expected_version": QUEUE_DB_SCHEMA_VERSION,
                "compatible": True,
                "table_present": table_present,
                "error": "",
            }
        if not table_present:
            raise QueueSchemaError("queue database schema is incomplete: queue_entries table is missing")
        _validate_queue_table(conn, require_indexes=True)
        return {
            "status": "ok",
            "version": version,
            "expected_version": QUEUE_DB_SCHEMA_VERSION,
            "compatible": True,
            "table_present": True,
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "version": locals().get("version", -1),
            "expected_version": QUEUE_DB_SCHEMA_VERSION,
            "compatible": False,
            "table_present": locals().get("table_present", False),
            "error": str(exc),
        }


def _create_queue_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spawn_intent_id TEXT,
            payload TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'queued',
            claim_token TEXT,
            claimed_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_entries_spawn_intent
        ON queue_entries (spawn_intent_id)
        WHERE spawn_intent_id IS NOT NULL AND spawn_intent_id <> ''
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_entries_state_id ON queue_entries (state, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_entries_claimed_at ON queue_entries (claimed_at)")


def init_queue_db(conn: sqlite3.Connection) -> None:
    status = queue_schema_status(conn)
    if status["status"] == "ok":
        return
    if status["status"] == "error":
        raise QueueSchemaError(str(status["error"]))

    try:
        conn.execute("BEGIN IMMEDIATE")
        status = queue_schema_status(conn)
        if status["status"] == "error":
            raise QueueSchemaError(str(status["error"]))
        if status["status"] == "legacy":
            _create_queue_schema(conn)
            _validate_queue_table(conn, require_indexes=True)
            conn.execute(f"PRAGMA user_version = {QUEUE_DB_SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def connect_queue_db_result(
    db_path: Path,
    *,
    attempts: int,
    timeout_base: float,
    timeout_max: float,
    backoff_base: float,
    row_factory: Any = None,
    diag: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> QueueOpenResult:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    max_attempts = max(1, int(attempts or 1))
    timeout_floor = max(1.0, float(timeout_base))
    timeout_cap = max(timeout_floor, float(timeout_max or timeout_floor))
    last_err: Exception | None = None
    recovered_corrupt = False
    for attempt in range(1, max_attempts + 1):
        connect_timeout = min(timeout_cap, timeout_floor * (2 ** (attempt - 1)))
        busy_timeout_ms = int(min(60_000, max(1_500, connect_timeout * 1000.0 * 2.0)))
        try:
            conn = sqlite3.connect(str(db_path), timeout=connect_timeout)
            if row_factory is not None:
                conn.row_factory = row_factory
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
            try:
                if db_path.exists():
                    os.chmod(db_path, 0o600)
            except Exception:
                pass
            return QueueOpenResult(conn=conn, recovered_corrupt=recovered_corrupt)
        except sqlite3.OperationalError as exc:
            if sqlite_error_looks_corrupt(exc) and not recovered_corrupt and quarantine_sqlite_db(db_path, exc, diag=diag):
                recovered_corrupt = True
                last_err = exc
                continue
            last_err = exc
            if attempt >= max_attempts:
                break
            delay = max(0.0, float(backoff_base or 0.0)) * (2 ** (attempt - 1))
            jitter = ((time.monotonic_ns() % 1_000_000) / 1_000_000.0) * max(0.001, delay) if delay > 0 else 0.0
            if callable(sleep_fn):
                sleep_fn(delay + jitter)
        except Exception as exc:
            if sqlite_error_looks_corrupt(exc) and not recovered_corrupt and quarantine_sqlite_db(db_path, exc, diag=diag):
                recovered_corrupt = True
                last_err = exc
                continue
            last_err = exc
            break
    if last_err is not None and callable(diag):
        try:
            diag(f"queue db connect failed: {last_err}")
        except Exception:
            pass
    return QueueOpenResult(conn=None, recovered_corrupt=recovered_corrupt, err=str(last_err or ""))


def open_ready_queue_db_result(
    db_path: Path,
    *,
    connect_fn: Callable[[], QueueOpenResult],
    init_fn: Callable[[sqlite3.Connection], None],
    close_fn: Callable[[sqlite3.Connection], None],
    diag: Callable[[str], None] | None = None,
) -> QueueOpenResult:
    for _ in range(2):
        open_result = connect_fn()
        conn = open_result.conn
        if conn is None:
            return open_result
        try:
            init_fn(conn)
            return QueueOpenResult(conn=conn, recovered_corrupt=open_result.recovered_corrupt, err=open_result.err)
        except Exception as exc:
            try:
                close_fn(conn)
            except Exception:
                pass
            if sqlite_error_looks_corrupt(exc) and quarantine_sqlite_db(db_path, exc, diag=diag):
                continue
            if callable(diag):
                try:
                    diag(f"queue db init failed: {exc}")
                except Exception:
                    pass
            return QueueOpenResult(conn=None, recovered_corrupt=open_result.recovered_corrupt, err=str(exc))
    return QueueOpenResult(conn=None)


def close_silent(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except Exception:
        pass


def repair_sqlite_permissions(db_path: Path) -> None:
    try:
        if db_path.exists():
            os.chmod(db_path, 0o600)
        wal = Path(str(db_path) + "-wal")
        if wal.exists():
            os.chmod(wal, 0o600)
        shm = Path(str(db_path) + "-shm")
        if shm.exists():
            os.chmod(shm, 0o600)
    except Exception:
        pass


def select_queued_rows(conn: sqlite3.Connection, *, max_lines: int) -> list[sqlite3.Row]:
    query = "SELECT id, spawn_intent_id, payload, attempts FROM queue_entries WHERE state='queued' ORDER BY id"
    params: tuple[Any, ...] = ()
    if max_lines > 0:
        query += " LIMIT ?"
        params = (int(max_lines),)
    return list(conn.execute(query, params))


def queue_rows_from_sqlite(rows: list[sqlite3.Row]) -> list[QueueStoredRow]:
    stored_rows: list[QueueStoredRow] = []
    for row in rows:
        try:
            stored_rows.append(QueueStoredRow.from_mapping(row))
        except Exception:
            continue
    return stored_rows


def row_ids(rows: list[QueueStoredRow | sqlite3.Row]) -> list[int]:
    ids: list[int] = []
    for row in rows:
        try:
            rid = int(row.id) if isinstance(row, QueueStoredRow) else int(row["id"])
        except Exception:
            continue
        if rid > 0:
            ids.append(rid)
    return ids


def claim_rows_sqlite_result(
    conn: sqlite3.Connection,
    *,
    token: str,
    now: float,
    processing_stale_after: float,
    max_lines: int,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> QueueRowClaimResult:
    rows: list[QueueStoredRow] = []
    try:
        if processing_stale_after > 0:
            cutoff = now - processing_stale_after
            conn.execute(
                "UPDATE queue_entries SET state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                "WHERE state='processing' AND claimed_at IS NOT NULL AND claimed_at < ?",
                (now, cutoff),
            )
            conn.commit()

        conn.execute("BEGIN IMMEDIATE")
        rows = queue_rows_from_sqlite(select_queued_rows(conn, max_lines=max_lines))
        ids = row_ids(rows)
        if ids:
            conn.executemany(
                "UPDATE queue_entries SET state='processing', claim_token=?, claimed_at=?, updated_at=? "
                "WHERE id=? AND state='queued'",
                [(token, now, now, rid) for rid in ids],
            )
            rows = [
                QueueStoredRow(
                    id=row.id,
                    spawn_intent_id=row.spawn_intent_id,
                    payload=row.payload,
                    attempts=row.attempts,
                    claim_token=token,
                )
                for row in rows
            ]
        conn.commit()
        return QueueRowClaimResult(rows=rows)
    except sqlite3.OperationalError as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        msg = str(exc).lower()
        lock_busy = "locked" in msg or "busy" in msg
        if lock_busy:
            if callable(on_lock_busy):
                on_lock_busy()
        elif callable(diag):
            diag(f"queue db claim failed: {exc}")
        return QueueRowClaimResult(rows=[], lock_busy=lock_busy, err=str(exc))
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        if callable(diag):
            diag(f"queue db claim failed: {exc}")
        return QueueRowClaimResult(rows=[], err=str(exc))


def rows_to_entries_result(rows: list[QueueStoredRow | sqlite3.Row]) -> QueueEntriesBatch:
    entries: list[dict[str, Any]] = []
    for raw_row in rows:
        row = raw_row if isinstance(raw_row, QueueStoredRow) else QueueStoredRow.from_mapping(raw_row)
        rid = row.id
        spawn_intent_id = row.spawn_intent_id
        payload = row.payload
        attempts_db = row.attempts
        try:
            obj = json.loads(payload) if payload else {}
        except Exception:
            obj = {"raw": payload}
        if not isinstance(obj, dict):
            obj = {"raw": payload}
        if spawn_intent_id and not str(obj.get("spawn_intent_id") or "").strip():
            obj["spawn_intent_id"] = spawn_intent_id
        if "attempts" not in obj:
            obj["attempts"] = attempts_db
        obj["__queue_backend"] = "sqlite"
        obj["__queue_id"] = rid
        obj["__queue_claim_token"] = row.claim_token
        entries.append(obj)
    return QueueEntriesBatch(entries=entries)


def ack_entry_claims_sqlite_result(
    conn: sqlite3.Connection,
    entry_claims: Sequence[tuple[Any, Any]],
    *,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> QueueWriteResult:
    claims: list[tuple[int, str]] = []
    for raw_id, raw_token in entry_claims or ():
        try:
            rid = int(raw_id)
        except Exception:
            continue
        token = str(raw_token or "").strip()
        if rid > 0 and token:
            claims.append((rid, token))
    if not claims:
        return QueueWriteResult(ok=True, count=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        deleted = 0
        for rid, token in claims:
            cur = conn.execute(
                "DELETE FROM queue_entries WHERE id=? AND state='processing' AND claim_token=?",
                (rid, token),
            )
            deleted += max(0, int(getattr(cur, "rowcount", 0) or 0))
        conn.commit()
        lost = len(claims) - deleted
        if lost:
            if callable(diag):
                diag(f"queue db ack lost claim ownership for {lost} entr{'y' if lost == 1 else 'ies'}")
            return QueueWriteResult(ok=False, count=lost, err="queue claim ownership lost")
        return QueueWriteResult(ok=True, count=deleted)
    except sqlite3.OperationalError as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        msg = str(exc).lower()
        lock_busy = "locked" in msg or "busy" in msg
        if lock_busy:
            if callable(on_lock_busy):
                on_lock_busy()
        elif callable(diag):
            diag(f"queue db ack failed: {exc}")
        return QueueWriteResult(ok=False, count=len(claims), lock_busy=lock_busy, err=str(exc))
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        if callable(diag):
            diag(f"queue db ack failed: {exc}")
        return QueueWriteResult(ok=False, count=len(claims), err=str(exc))


def requeue_entries_sqlite_result(
    conn: sqlite3.Connection,
    entries: Sequence[dict[str, Any]],
    *,
    now: float,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> QueueWriteResult:
    items = [
        entry
        for entry in (entries or [])
        if isinstance(entry, dict) and entry.get("__queue_backend") == "sqlite" and entry.get("__queue_id")
    ]
    if not items:
        return QueueWriteResult(ok=True, count=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        updated = 0
        for entry in items:
            rid = int(entry.get("__queue_id") or 0)
            token = str(entry.get("__queue_claim_token") or "").strip()
            if rid <= 0 or not token:
                continue
            out = dict(entry)
            out.pop("__queue_backend", None)
            out.pop("__queue_id", None)
            out.pop("__queue_claim_token", None)
            try:
                attempts = int(out.get("attempts") or 0)
            except Exception:
                attempts = 0
            payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
            cur = conn.execute(
                "UPDATE queue_entries SET state='queued', claim_token=NULL, claimed_at=NULL, attempts=?, payload=?, updated_at=? "
                "WHERE id=? AND state='processing' AND claim_token=?",
                (attempts, payload, now, rid, token),
            )
            updated += max(0, int(getattr(cur, "rowcount", 0) or 0))
        conn.commit()
        lost = len(items) - updated
        if lost:
            if callable(diag):
                diag(f"queue db requeue lost claim ownership for {lost} entr{'y' if lost == 1 else 'ies'}")
            return QueueWriteResult(ok=False, count=lost, err="queue claim ownership lost")
        return QueueWriteResult(ok=True, count=updated)
    except sqlite3.OperationalError as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        msg = str(exc).lower()
        lock_busy = "locked" in msg or "busy" in msg
        if lock_busy:
            if callable(on_lock_busy):
                on_lock_busy()
        elif callable(diag):
            diag(f"queue db requeue failed: {exc}")
        return QueueWriteResult(ok=False, count=len(items), lock_busy=lock_busy, err=str(exc))
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        if callable(diag):
            diag(f"queue db requeue failed: {exc}")
        return QueueWriteResult(ok=False, count=len(items), err=str(exc))


def enqueue_entries_sqlite_result(
    conn: sqlite3.Connection,
    entries: Sequence[dict[str, Any]],
    *,
    now: float,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> QueueWriteResult:
    items = [entry for entry in (entries or []) if isinstance(entry, dict)]
    if not items:
        return QueueWriteResult(ok=True, count=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        for entry in items:
            out = dict(entry)
            out.pop("__queue_backend", None)
            out.pop("__queue_id", None)
            out.pop("__queue_claim_token", None)
            try:
                attempts = int(out.get("attempts") or 0)
            except Exception:
                attempts = 0
            payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
            sid = (out.get("spawn_intent_id") or "").strip()
            if sid:
                cur = conn.execute(
                    "UPDATE queue_entries SET payload=?, attempts=?, state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                    "WHERE spawn_intent_id=?",
                    (payload, attempts, now, sid),
                )
                if int(getattr(cur, "rowcount", 0) or 0) <= 0:
                    try:
                        conn.execute(
                            "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) "
                            "VALUES (?, ?, ?, 'queued', NULL, NULL, ?, ?)",
                            (sid, payload, attempts, now, now),
                        )
                    except sqlite3.IntegrityError:
                        conn.execute(
                            "UPDATE queue_entries SET payload=?, attempts=?, state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                            "WHERE spawn_intent_id=?",
                            (payload, attempts, now, sid),
                        )
            else:
                conn.execute(
                    "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) "
                    "VALUES (NULL, ?, ?, 'queued', NULL, NULL, ?, ?)",
                    (payload, attempts, now, now),
                )
        conn.commit()
        return QueueWriteResult(ok=True, count=len(items))
    except sqlite3.OperationalError as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        msg = str(exc).lower()
        lock_busy = "locked" in msg or "busy" in msg
        if lock_busy:
            if callable(on_lock_busy):
                on_lock_busy()
        elif callable(diag):
            diag(f"queue db enqueue failed: {exc}")
        return QueueWriteResult(ok=False, count=len(items), lock_busy=lock_busy, err=str(exc))
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        if callable(diag):
            diag(f"queue db enqueue failed: {exc}")
        return QueueWriteResult(ok=False, count=len(items), err=str(exc))
