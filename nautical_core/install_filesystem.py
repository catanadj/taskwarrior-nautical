"""Filesystem transactions used by the Nautical runtime installer."""

from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

fcntl: Any
try:
    import fcntl
except Exception:
    fcntl = None


class InstallError(RuntimeError):
    """Raised when an installation transaction cannot be completed safely."""


class MissingSnapshot(TypedDict):
    kind: Literal["missing"]


class SymlinkSnapshot(TypedDict):
    kind: Literal["symlink"]
    target: str


class FileSnapshot(TypedDict):
    kind: Literal["file"]
    backup: str


FileSystemSnapshot: TypeAlias = MissingSnapshot | SymlinkSnapshot | FileSnapshot


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


def snapshot_file(path: Path, backup_dir: Path) -> FileSystemSnapshot:
    if not lexists(path):
        return MissingSnapshot(kind="missing")
    if path.is_symlink():
        return SymlinkSnapshot(kind="symlink", target=os.readlink(path))
    if not path.is_file():
        raise InstallError(f"managed install path is not a file or symlink: {path}")
    if backup_dir.is_symlink() or (backup_dir.exists() and not backup_dir.is_dir()):
        raise InstallError(f"install backup path is not a directory: {backup_dir}")
    backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    backup_dir.chmod(0o700)
    backup = backup_dir / f"{hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:12]}-{path.name}"
    shutil.copy2(path, backup)
    backup.chmod(0o600)
    return FileSnapshot(kind="file", backup=str(backup))


def restore_file(path: Path, snapshot: FileSystemSnapshot) -> None:
    if snapshot["kind"] == "missing":
        if lexists(path):
            path.unlink()
    elif snapshot["kind"] == "symlink":
        atomic_symlink(snapshot["target"], path)
    else:
        backup = Path(snapshot["backup"])
        atomic_copy(backup, path, executable=os.access(str(backup), os.X_OK))


def pointer_snapshot(path: Path) -> MissingSnapshot | SymlinkSnapshot:
    if not lexists(path):
        return MissingSnapshot(kind="missing")
    if not path.is_symlink():
        raise InstallError(f"managed runtime pointer is not a symlink: {path}")
    return SymlinkSnapshot(kind="symlink", target=os.readlink(path))


def restore_pointer(path: Path, snapshot: MissingSnapshot | SymlinkSnapshot) -> None:
    if snapshot["kind"] == "missing":
        if lexists(path):
            path.unlink()
    else:
        atomic_symlink(str(snapshot["target"]), path)
