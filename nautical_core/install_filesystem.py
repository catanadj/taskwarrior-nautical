"""Filesystem transactions used by the Nautical runtime installer."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

fcntl: Any
try:
    import fcntl
except Exception:
    fcntl = None


class InstallError(RuntimeError):
    """Raised when an installation transaction cannot be completed safely."""


class InstallLock:
    """Serialize install and cleanup transactions for one runtime root."""

    def __init__(self, path: Path):
        self.path = path
        self.handle: Any = None
        self.fallback_fd: int | None = None

    def __enter__(self) -> "InstallLock":
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if fcntl is not None:
            self.handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.handle.close()
                self.handle = None
                raise InstallError("another Nautical installation is already running") from exc
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(f"{os.getpid()}\n")
            self.handle.flush()
            return self
        try:
            self.fallback_fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(self.fallback_fd, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError as exc:
            raise InstallError("another Nautical installation is already running") from exc
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
        if self.fallback_fd is not None:
            os.close(self.fallback_fd)
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def atomic_symlink(target: str, path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        os.symlink(target, temp)
        os.replace(temp, path)
    finally:
        if lexists(temp):
            temp.unlink()


def atomic_copy(source: Path, target: Path, *, executable: bool = False) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copy2(source, temp)
        if executable:
            temp.chmod(0o755)
        os.replace(temp, target)
    finally:
        if lexists(temp):
            temp.unlink()


def atomic_write_text(text: str, target: Path) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        temp.write_text(text, encoding="utf-8")
        temp.chmod(0o600)
        os.replace(temp, target)
    finally:
        if lexists(temp):
            temp.unlink()


def snapshot_file(path: Path, backup_dir: Path) -> dict[str, Any]:
    if not lexists(path):
        return {"kind": "missing"}
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    if not path.is_file():
        raise InstallError(f"managed install path is not a file or symlink: {path}")
    backup = backup_dir / f"{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:12]}-{path.name}"
    shutil.copy2(path, backup)
    return {"kind": "file", "backup": str(backup)}


def restore_file(path: Path, snapshot: dict[str, Any]) -> None:
    kind = snapshot["kind"]
    if kind == "missing":
        if lexists(path):
            path.unlink()
    elif kind == "symlink":
        atomic_symlink(str(snapshot["target"]), path)
    else:
        backup = Path(str(snapshot["backup"]))
        atomic_copy(backup, path, executable=os.access(str(backup), os.X_OK))


def pointer_snapshot(path: Path) -> dict[str, Any]:
    if not lexists(path):
        return {"kind": "missing"}
    if not path.is_symlink():
        raise InstallError(f"managed runtime pointer is not a symlink: {path}")
    return {"kind": "symlink", "target": os.readlink(path)}


def restore_pointer(path: Path, snapshot: dict[str, Any]) -> None:
    if snapshot["kind"] == "missing":
        if lexists(path):
            path.unlink()
    else:
        atomic_symlink(str(snapshot["target"]), path)

