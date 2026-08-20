from __future__ import annotations

import os
from pathlib import Path


_OUTBOX_DB_SCHEMA_VERSION = 1


class ExitWorkProbe:
    __slots__ = ("may_have_work", "reason")

    def __init__(self, may_have_work: bool, reason: str) -> None:
        self.may_have_work = bool(may_have_work)
        self.reason = str(reason or "")

    @property
    def definitely_empty(self) -> bool:
        return not self.may_have_work


def _outbox_may_have_work(path: Path) -> bool | None:
    try:
        import sqlite3

        uri = path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=0.0)
        try:
            row = conn.execute("PRAGMA user_version").fetchone()
            version = int(row[0] if row else 0)
            if version > _OUTBOX_DB_SCHEMA_VERSION:
                return None
            row = conn.execute(
                "SELECT 1 FROM lifecycle_outbox "
                "WHERE processing_state IN ('ready', 'claimed', 'retry') LIMIT 1"
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

    path = root / ".nautical-state" / ".nautical_lifecycle_outbox.db"
    try:
        if not path.exists():
            return ExitWorkProbe(False, "no lifecycle outbox")
        if not path.is_file():
            return ExitWorkProbe(True, f"lifecycle outbox is not a regular file: {path}")
    except Exception:
        return ExitWorkProbe(True, f"lifecycle outbox state is uncertain: {path}")
    may_have_work = _outbox_may_have_work(path)
    if may_have_work is None:
        return ExitWorkProbe(True, f"lifecycle outbox could not be inspected: {path}")
    if may_have_work:
        return ExitWorkProbe(True, "lifecycle outbox has active intents")
    return ExitWorkProbe(False, "no active lifecycle intents")


__all__ = ("ExitWorkProbe", "probe_exit_work")
