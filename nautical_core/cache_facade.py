"""Compatibility helpers for facade-wide cache control."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any


def emit_metrics(
    caches: Iterable[tuple[str, Any]],
    warn_once: Callable[[str, str], Any],
) -> None:
    if os.environ.get("NAUTICAL_DIAG_METRICS") != "1":
        return
    lines: list[str] = []
    for name, cache in caches:
        try:
            lines.append(f"{name}: {cache.cache_info()}")
        except Exception:
            continue
    if lines:
        warn_once("cache_metrics", "[nautical-metrics] " + " | ".join(lines))


def clear_all(
    memory_cache: Any,
    caches: Iterable[Any],
    *,
    position_selection: Any,
    selection_matcher: Any,
) -> None:
    operations: list[Callable[[], Any]] = [lambda: memory_cache.clear()]
    def clear_cache(cache: Any) -> None:
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
            continue


__all__ = ("emit_metrics", "clear_all")
