from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from nautical_core.queue_models import QueueEntriesBatch, QueueRowClaimResult, QueueWriteResult


def nautical_state_dir_path(tw_data_dir: Path) -> Path:
    return tw_data_dir / ".nautical-state"


def nautical_lock_dir_path(tw_data_dir: Path) -> Path:
    return tw_data_dir / ".nautical-locks"


def legacy_state_path(tw_data_dir: Path, name: str) -> Path:
    return tw_data_dir / name


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


def resolve_queue_db_path(queue_path: Path, db_path: Path) -> Path:
    try:
        if not db_path.exists() and db_path.parent != queue_path.parent:
            return queue_path.parent / db_path.name
    except Exception:
        pass
    return db_path


def init_queue_db(conn: sqlite3.Connection) -> None:
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
    conn.commit()


def connect_queue_db(
    db_path: Path,
    *,
    attempts: int,
    timeout_base: float,
    timeout_max: float,
    backoff_base: float,
    row_factory: Any = None,
    diag: Callable[[str], None] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> sqlite3.Connection | None:
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
            return conn
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
    return None


def open_ready_queue_db(
    db_path: Path,
    *,
    connect_fn: Callable[[], sqlite3.Connection | None],
    init_fn: Callable[[sqlite3.Connection], None],
    close_fn: Callable[[sqlite3.Connection], None],
    diag: Callable[[str], None] | None = None,
) -> sqlite3.Connection | None:
    for _ in range(2):
        conn = connect_fn()
        if conn is None:
            return None
        try:
            init_fn(conn)
            return conn
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
            return None
    return None


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


def row_ids(rows: list[sqlite3.Row]) -> list[int]:
    ids: list[int] = []
    for row in rows:
        try:
            rid = int(row["id"])
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
    rows: list[sqlite3.Row] = []
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
        rows = select_queued_rows(conn, max_lines=max_lines)
        ids = row_ids(rows)
        if ids:
            conn.executemany(
                "UPDATE queue_entries SET state='processing', claim_token=?, claimed_at=?, updated_at=? "
                "WHERE id=?",
                [(token, now, now, rid) for rid in ids],
            )
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


def claim_rows_sqlite(
    conn: sqlite3.Connection,
    *,
    token: str,
    now: float,
    processing_stale_after: float,
    max_lines: int,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> list[sqlite3.Row]:
    return claim_rows_sqlite_result(
        conn,
        token=token,
        now=now,
        processing_stale_after=processing_stale_after,
        max_lines=max_lines,
        diag=diag,
        on_lock_busy=on_lock_busy,
    ).rows


def rows_to_entries_result(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        rid = int(row["id"])
        spawn_intent_id = str(row["spawn_intent_id"] or "").strip()
        payload = (row["payload"] or "").strip()
        attempts_db = int(row["attempts"] or 0)
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
        entries.append(obj)
    return QueueEntriesBatch(entries=entries)


def rows_to_entries(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return rows_to_entries_result(rows).entries


def ack_entry_ids_sqlite_result(
    conn: sqlite3.Connection,
    entry_ids: Sequence[Any],
    *,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> QueueWriteResult:
    ids: list[int] = []
    for raw in entry_ids or ():
        try:
            rid = int(raw)
        except Exception:
            continue
        if rid > 0:
            ids.append(rid)
    if not ids:
        return QueueWriteResult(ok=True, count=0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany("DELETE FROM queue_entries WHERE id=?", [(rid,) for rid in ids])
        conn.commit()
        return QueueWriteResult(ok=True, count=len(ids))
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
        return QueueWriteResult(ok=False, count=len(ids), lock_busy=lock_busy, err=str(exc))
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        if callable(diag):
            diag(f"queue db ack failed: {exc}")
        return QueueWriteResult(ok=False, count=len(ids), err=str(exc))


def ack_entry_ids_sqlite(
    conn: sqlite3.Connection,
    entry_ids: Sequence[Any],
    *,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> bool:
    return ack_entry_ids_sqlite_result(
        conn,
        entry_ids,
        diag=diag,
        on_lock_busy=on_lock_busy,
    ).ok


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
        for entry in items:
            rid = int(entry.get("__queue_id") or 0)
            if rid <= 0:
                continue
            out = dict(entry)
            out.pop("__queue_backend", None)
            out.pop("__queue_id", None)
            try:
                attempts = int(out.get("attempts") or 0)
            except Exception:
                attempts = 0
            payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
            conn.execute(
                "UPDATE queue_entries SET state='queued', claim_token=NULL, claimed_at=NULL, attempts=?, payload=?, updated_at=? "
                "WHERE id=?",
                (attempts, payload, now, rid),
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


def requeue_entries_sqlite(
    conn: sqlite3.Connection,
    entries: Sequence[dict[str, Any]],
    *,
    now: float,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> bool:
    return requeue_entries_sqlite_result(
        conn,
        entries,
        now=now,
        diag=diag,
        on_lock_busy=on_lock_busy,
    ).ok


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


def enqueue_entries_sqlite(
    conn: sqlite3.Connection,
    entries: Sequence[dict[str, Any]],
    *,
    now: float,
    diag: Callable[[str], None] | None = None,
    on_lock_busy: Callable[[], None] | None = None,
) -> bool:
    return enqueue_entries_sqlite_result(
        conn,
        entries,
        now=now,
        diag=diag,
        on_lock_busy=on_lock_busy,
    ).ok


def recover_processing_file(
    *,
    queue_processing_path: Path,
    queue_path: Path,
    durable_queue: bool,
    fsync_dir_fn: Callable[[Path], None],
) -> bool:
    try:
        if not queue_processing_path.exists():
            return True
        if not queue_path.exists():
            os.replace(queue_processing_path, queue_path)
            return True
        with open(queue_processing_path, "r", encoding="utf-8") as f_in:
            fd = os.open(str(queue_path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            with os.fdopen(fd, "a", encoding="utf-8") as f_out:
                for line in f_in:
                    f_out.write(line)
                if durable_queue:
                    try:
                        f_out.flush()
                        os.fsync(f_out.fileno())
                    except Exception:
                        pass
        os.unlink(queue_processing_path)
        if durable_queue:
            fsync_dir_fn(queue_path.parent)
        return True
    except Exception:
        return False


def source_path_with_overflow(
    *,
    queue_path: Path,
    queue_max_bytes: int,
    diag: Callable[[str], None] | None = None,
) -> tuple[Path, Path | None]:
    overflow_path = None
    try:
        st = queue_path.stat()
        if queue_max_bytes > 0 and st.st_size > queue_max_bytes:
            try:
                ts = int(time.time())
                overflow_path = queue_path.with_suffix(f".overflow.{ts}.jsonl")
                os.replace(queue_path, overflow_path)
                if callable(diag):
                    diag(f"queue rotated: {overflow_path}")
            except Exception:
                pass
    except Exception:
        pass
    return (overflow_path or queue_path), overflow_path


def split_source_to_staging(
    *,
    src_path: Path,
    tmp_path: Path,
    tmp_processing: Path,
    queue_max_lines: int,
    durable_queue: bool,
    quarantine_line: Callable[[str, str], None],
    write_dead_letter: Callable[[dict[str, Any], str], None],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    fd_out = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    fd_proc = os.open(str(tmp_processing), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with open(src_path, "r", encoding="utf-8") as f_in, os.fdopen(fd_out, "w", encoding="utf-8") as f_out, os.fdopen(fd_proc, "w", encoding="utf-8") as f_proc:
        for line in f_in:
            ln = line.strip()
            if not ln:
                continue
            if queue_max_lines > 0 and len(entries) >= queue_max_lines:
                f_out.write(line)
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                quarantine_line(line, "queue json parse")
                write_dead_letter({"raw": line}, "queue json parse")
                continue
            if isinstance(obj, dict):
                entries.append(obj)
                f_proc.write(line)
                continue
            quarantine_line(line, "queue json not object")
            write_dead_letter({"raw": line}, "queue json not object")
        if durable_queue:
            try:
                f_out.flush()
                os.fsync(f_out.fileno())
                f_proc.flush()
                os.fsync(f_proc.fileno())
            except Exception:
                pass
    return entries


def commit_staging(
    *,
    queue_path: Path,
    queue_processing_path: Path,
    tmp_path: Path,
    tmp_processing: Path,
    entries: Sequence[dict[str, Any]],
    overflow_path: Path | None,
    durable_queue: bool,
    fsync_dir_fn: Callable[[Path], None],
) -> None:
    os.replace(tmp_path, queue_path)
    if entries:
        os.replace(tmp_processing, queue_processing_path)
    else:
        try:
            tmp_processing.unlink()
        except Exception:
            pass
    if durable_queue:
        fsync_dir_fn(queue_path.parent)
    if overflow_path:
        try:
            overflow_path.unlink()
        except Exception:
            pass


def cleanup_staging(tmp_path: Path, tmp_processing: Path) -> None:
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass
    try:
        if tmp_processing.exists():
            tmp_processing.unlink()
    except Exception:
        pass


def take_queue_entries_jsonl_result(
    *,
    queue_path: Path,
    queue_processing_path: Path,
    queue_max_bytes: int,
    queue_max_lines: int,
    durable_queue: bool,
    lock_queue: Callable[[], Any],
    record_lock_failure: Callable[[], None],
    quarantine_line: Callable[[str, str], None],
    write_dead_letter: Callable[[dict[str, Any], str], None],
    fsync_dir_fn: Callable[[Path], None],
    diag: Callable[[str], None] | None = None,
) -> QueueEntriesBatch:
    entries: list[dict[str, Any]] = []
    with lock_queue() as locked:
        if not locked:
            record_lock_failure()
            return QueueEntriesBatch(entries=entries)
        if not recover_processing_file(
            queue_processing_path=queue_processing_path,
            queue_path=queue_path,
            durable_queue=durable_queue,
            fsync_dir_fn=fsync_dir_fn,
        ):
            return QueueEntriesBatch(entries=entries)
        try:
            if not queue_path.exists():
                return QueueEntriesBatch(entries=entries)
        except Exception:
            return QueueEntriesBatch(entries=entries)
        src_path, overflow_path = source_path_with_overflow(
            queue_path=queue_path,
            queue_max_bytes=queue_max_bytes,
            diag=diag,
        )
        tmp_path = queue_path.with_suffix('.staging')
        tmp_processing = queue_processing_path.with_suffix('.staging')
        try:
            entries = split_source_to_staging(
                src_path=src_path,
                tmp_path=tmp_path,
                tmp_processing=tmp_processing,
                queue_max_lines=queue_max_lines,
                durable_queue=durable_queue,
                quarantine_line=quarantine_line,
                write_dead_letter=write_dead_letter,
            )
            commit_staging(
                queue_path=queue_path,
                queue_processing_path=queue_processing_path,
                tmp_path=tmp_path,
                tmp_processing=tmp_processing,
                entries=entries,
                overflow_path=overflow_path,
                durable_queue=durable_queue,
                fsync_dir_fn=fsync_dir_fn,
            )
        except Exception:
            cleanup_staging(tmp_path, tmp_processing)
    return QueueEntriesBatch(entries=entries)


def take_queue_entries_jsonl(
    *,
    queue_path: Path,
    queue_processing_path: Path,
    queue_max_bytes: int,
    queue_max_lines: int,
    durable_queue: bool,
    lock_queue: Callable[[], Any],
    record_lock_failure: Callable[[], None],
    quarantine_line: Callable[[str, str], None],
    write_dead_letter: Callable[[dict[str, Any], str], None],
    fsync_dir_fn: Callable[[Path], None],
    diag: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    return take_queue_entries_jsonl_result(
        queue_path=queue_path,
        queue_processing_path=queue_processing_path,
        queue_max_bytes=queue_max_bytes,
        queue_max_lines=queue_max_lines,
        durable_queue=durable_queue,
        lock_queue=lock_queue,
        record_lock_failure=record_lock_failure,
        quarantine_line=quarantine_line,
        write_dead_letter=write_dead_letter,
        fsync_dir_fn=fsync_dir_fn,
        diag=diag,
    ).entries


def requeue_entries_jsonl_result(
    *,
    queue_path: Path,
    entries: Sequence[dict[str, Any]],
    durable_queue: bool,
    lock_queue: Callable[[], Any],
    record_lock_failure: Callable[[], None],
    fsync_dir_fn: Callable[[Path], None],
    diag: Callable[[str], None] | None = None,
) -> QueueWriteResult:
    items = [entry for entry in (entries or []) if isinstance(entry, dict)]
    if not items:
        return QueueWriteResult(ok=True, count=0)
    with lock_queue() as locked:
        if not locked:
            record_lock_failure()
            return QueueWriteResult(ok=False, count=len(items), lock_busy=True, err='queue lock busy')
        try:
            fd = os.open(str(queue_path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            with os.fdopen(fd, 'a', encoding='utf-8') as f:
                for obj in items:
                    f.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + "\n")
                if durable_queue:
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                        fsync_dir_fn(queue_path.parent)
                    except Exception:
                        pass
            return QueueWriteResult(ok=True, count=len(items))
        except Exception as exc:
            if callable(diag):
                diag(f'requeue write failed: {exc}')
            return QueueWriteResult(ok=False, count=len(items), err=str(exc))


def requeue_entries_jsonl(
    *,
    queue_path: Path,
    entries: Sequence[dict[str, Any]],
    durable_queue: bool,
    lock_queue: Callable[[], Any],
    record_lock_failure: Callable[[], None],
    fsync_dir_fn: Callable[[Path], None],
    diag: Callable[[str], None] | None = None,
) -> bool:
    return requeue_entries_jsonl_result(
        queue_path=queue_path,
        entries=entries,
        durable_queue=durable_queue,
        lock_queue=lock_queue,
        record_lock_failure=record_lock_failure,
        fsync_dir_fn=fsync_dir_fn,
        diag=diag,
    ).ok


def queue_jsonl_has_data(*paths: Path) -> bool:
    for path in paths:
        try:
            if not path.exists():
                continue
            try:
                if path.stat().st_size > 0:
                    return True
            except Exception:
                return True
        except Exception:
            continue
    return False
