"""Compatibility helpers for facade-wide cache control."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from functools import partial
from typing import Protocol


class CacheInfoPort(Protocol):
    def cache_info(self) -> object: ...


class CacheClearPort(Protocol):
    def cache_clear(self) -> None: ...


class MemoryCacheClearPort(Protocol):
    def clear(self) -> None: ...


class PositionSelectionClearPort(Protocol):
    def clear_candidate_cache(self) -> None: ...


def emit_metrics(
    caches: Iterable[tuple[str, CacheInfoPort]],
    warn_once: Callable[[str, str], None],
) -> None:
    if os.environ.get("NAUTICAL_DIAG_METRICS") != "1":
        return
    lines: list[str] = []
    for name, cache in caches:
        try:
            lines.append(f"{name}: {cache.cache_info()}")
        except Exception:
            # Metrics are optional; a broken cache must not hide other metrics.
            continue
    if lines:
        warn_once("cache_metrics", "[nautical-metrics] " + " | ".join(lines))


def clear_all(
    memory_cache: MemoryCacheClearPort,
    caches: Iterable[CacheClearPort],
    *,
    position_selection: PositionSelectionClearPort,
    selection_matcher: CacheClearPort,
) -> None:
    operations: list[Callable[[], None]] = [lambda: memory_cache.clear()]
    def clear_cache(cache: CacheClearPort) -> None:
        cache.cache_clear()
    for cache in caches:
        operations.append(partial(clear_cache, cache))
    operations.extend(
        [
            lambda: position_selection.clear_candidate_cache(),
            lambda: selection_matcher.cache_clear(),
        ]
    )
    for operation in operations:
        try:
            operation()
        except Exception:
            # Caches are disposable; keep clearing independent caches after one fails.
            continue


__all__ = ("emit_metrics", "clear_all")
