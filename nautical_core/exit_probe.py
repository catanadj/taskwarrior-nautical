from __future__ import annotations

import os
from pathlib import Path


_LEGACY_QUEUE_NAMES = (
    ".nautical_spawn_queue.jsonl",
    ".nautical_spawn_queue.processing.jsonl",
)
_MIGRATABLE_STATE_NAMES = (
    ".nautical_queue.db",
    ".nautical_dead_letter.jsonl",
    ".nautical_spawn_intents.jsonl",
)


class ExitWorkProbe:
    __slots__ = ("may_have_work", "reason")

    def __init__(self, may_have_work: bool, reason: str) -> None:
        self.may_have_work = bool(may_have_work)
        self.reason = str(reason or "")

    @property
    def definitely_empty(self) -> bool:
        return not self.may_have_work


def _file_nonempty_or_uncertain(path: Path) -> bool | None:
    try:
        if not path.exists():
            return False
        if not path.is_file():
            return None
        return path.stat().st_size > 0
    except Exception:
        return None


def _sqlite_may_have_work(path: Path) -> bool | None:
    try:
        import sqlite3

        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=0.0)
        try:
            row = conn.execute(
                "SELECT 1 FROM queue_entries WHERE state IN ('queued', 'processing') LIMIT 1"
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return None


def probe_exit_work(taskdata: str | os.PathLike[str]) -> ExitWorkProbe:
    try:
        root = Path(taskdata).expanduser().resolve()
    except Exception:
        return ExitWorkProbe(True, "taskdata path could not be resolved")

    state_dir = root / ".nautical-state"
    for name in _MIGRATABLE_STATE_NAMES:
        current = state_dir / name
        legacy = root / name
        try:
            if legacy.exists() and not current.exists():
                return ExitWorkProbe(True, f"legacy state migration is pending: {legacy}")
        except Exception:
            return ExitWorkProbe(True, f"legacy state migration status is uncertain: {legacy}")

    for directory in (state_dir, root):
        for name in _LEGACY_QUEUE_NAMES:
            path = directory / name
            nonempty = _file_nonempty_or_uncertain(path)
            if nonempty is None:
                return ExitWorkProbe(True, f"queue file state is uncertain: {path}")
            if nonempty:
                return ExitWorkProbe(True, f"legacy queue file is non-empty: {path}")

    database_paths = (
        state_dir / ".nautical_queue.db",
        root / ".nautical_queue.db",
    )
    for path in database_paths:
        try:
            database_exists = path.exists()
            database_is_file = path.is_file() if database_exists else False
        except Exception:
            return ExitWorkProbe(True, f"queue database state is uncertain: {path}")
        if not database_exists:
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(path) + suffix)
                try:
                    if sidecar.exists():
                        return ExitWorkProbe(True, f"queue database sidecar exists without database: {sidecar}")
                except Exception:
                    return ExitWorkProbe(True, f"queue database sidecar state is uncertain: {sidecar}")
            continue
        if not database_is_file:
            return ExitWorkProbe(True, f"queue database is not a regular file: {path}")
        may_have_work = _sqlite_may_have_work(path)
        if may_have_work is None:
            return ExitWorkProbe(True, f"queue database could not be inspected: {path}")
        if may_have_work:
            return ExitWorkProbe(True, f"queue database has active rows: {path}")

    return ExitWorkProbe(False, "no active queue state")


__all__ = ("ExitWorkProbe", "probe_exit_work")
