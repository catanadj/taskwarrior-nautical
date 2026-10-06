"""Shared capability contracts for cache encoding and file persistence."""

from __future__ import annotations

from os import PathLike, stat_result
from typing import IO, Protocol


class ClockPort(Protocol):
    def time(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class CacheClockPort(ClockPort, Protocol):
    def time_ns(self) -> int: ...


class RandomPort(Protocol):
    def uniform(self, start: float, end: float) -> float: ...


class PathPort(Protocol):
    def dirname(self, path: str) -> str: ...

    def abspath(self, path: str) -> str: ...

    def join(self, *parts: str) -> str: ...

    def exists(self, path: str) -> bool: ...

    def getsize(self, path: str) -> int: ...

    def isdir(self, path: str) -> bool: ...

    def isfile(self, path: str) -> bool: ...


class StatResultPort(Protocol):
    st_dev: int
    st_ino: int
    st_mtime: float
    st_mtime_ns: int
    st_size: int


class EnvironmentPort(Protocol):
    def get(self, key: str, default: str | None = None) -> str | None: ...


class FilesystemPort(Protocol):
    O_CREAT: int
    O_EXCL: int
    O_RDWR: int
    O_WRONLY: int
    name: str
    path: PathPort
    environ: EnvironmentPort

    def stat(self, path: str) -> StatResultPort: ...

    def getpid(self) -> int: ...

    def replace(self, src: str, dst: str) -> None: ...

    def listdir(self, path: str) -> list[str]: ...

    def unlink(self, path: str) -> None: ...

    def close(self, file_descriptor: int) -> None: ...

    def fchmod(self, file_descriptor: int, mode: int) -> None: ...

    def write(self, file_descriptor: int, data: bytes) -> int: ...

    def open(self, path: str, flags: int, mode: int = 0o777) -> int: ...

    def fdopen(self, file_descriptor: int, mode: str, *, encoding: str) -> IO[str]: ...

    def makedirs(self, path: str, *, exist_ok: bool = False) -> None: ...

    def kill(self, pid: int, signal: int) -> None: ...


class LockFilesystemPort(Protocol):
    """Filesystem capabilities used by lock operations (without environment access)."""

    O_CREAT: int
    O_EXCL: int
    O_RDWR: int
    O_WRONLY: int
    path: PathPort

    def stat(
        self,
        path: int | str | bytes | PathLike[str] | PathLike[bytes],
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> stat_result: ...

    def getpid(self) -> int: ...

    def unlink(self, path: str) -> None: ...

    def close(self, file_descriptor: int) -> None: ...

    def fchmod(self, file_descriptor: int, mode: int) -> None: ...

    def write(self, file_descriptor: int, data: bytes) -> int: ...

    def open(self, path: str, flags: int, mode: int = 0o777) -> int: ...

    def fdopen(self, file_descriptor: int, mode: str, *, encoding: str) -> IO[str]: ...

    def makedirs(self, path: str, *, exist_ok: bool = False) -> None: ...

    def kill(self, pid: int, signal: int) -> None: ...


class FcntlPort(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


class JsonPort(Protocol):
    JSONDecodeError: type[ValueError]

    def loads(self, s: str) -> object: ...

    def dumps(
        self,
        obj: object,
        *,
        ensure_ascii: bool = True,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
    ) -> str: ...


class DecompressorPort(Protocol):
    unconsumed_tail: bytes
    eof: bool
    unused_data: bytes

    def decompress(self, data: bytes, max_length: int = 0) -> bytes: ...

    def flush(self, length: int = 16384) -> bytes: ...


class CompressionPort(Protocol):
    error: type[Exception]

    def compress(self, data: bytes, level: int) -> bytes: ...

    def decompressobj(self) -> DecompressorPort: ...


class Base64Port(Protocol):
    def b85decode(self, value: bytes | str) -> bytes: ...

    def b85encode(self, value: bytes) -> bytes: ...


class TemporaryFilePort(Protocol):
    def mkstemp(
        self,
        *,
        dir: str,
        prefix: str,
        suffix: str,
    ) -> tuple[int, str]: ...


class AtomicReplacePort(Protocol):
    name: str

    def replace(self, src: str, dst: str) -> None: ...
