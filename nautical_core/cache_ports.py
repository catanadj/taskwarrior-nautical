"""Shared capability contracts for cache encoding and file persistence."""

from __future__ import annotations

from typing import Protocol


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
