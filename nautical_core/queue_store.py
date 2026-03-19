from __future__ import annotations

import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Sequence


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
            jitter = random.uniform(0.0, max(0.001, delay)) if delay > 0 else 0.0
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
