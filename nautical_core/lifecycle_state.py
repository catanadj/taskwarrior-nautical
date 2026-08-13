"""Small filesystem paths shared by lifecycle hooks and operator tools."""

from __future__ import annotations

import os
from pathlib import Path


def nautical_state_dir_path(taskdata: Path) -> Path:
    return Path(taskdata) / ".nautical-state"


def nautical_lock_dir_path(taskdata: Path) -> Path:
    return Path(taskdata) / ".nautical-locks"


def parent_nextlink_lock_path(taskdata: Path, parent_uuid: str) -> Path:
    raw = str(parent_uuid or "").strip().lower()
    safe = "".join(char for char in raw if char.isalnum())[:64] or "unknown"
    return nautical_lock_dir_path(taskdata) / f".nautical_parent_nextlink.{safe}.lock"


def reconcile_lock_path(taskdata: Path) -> Path:
    return nautical_lock_dir_path(taskdata) / ".nautical_reconcile.lock"


def fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


__all__ = (
    "fsync_dir",
    "nautical_lock_dir_path",
    "nautical_state_dir_path",
    "parent_nextlink_lock_path",
    "reconcile_lock_path",
)
