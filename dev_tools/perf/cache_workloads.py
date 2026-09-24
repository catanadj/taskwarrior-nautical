"""Cache benchmark workloads with explicit runtime dependencies."""

from __future__ import annotations

import tempfile
import time
from contextlib import contextmanager
from typing import Any, Iterator


def cache_key_hot(core: Any, clear_caches: Any, exprs: list[str], rounds: int) -> float:
    clear_caches()
    started = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.cache_key_for_task(expr, "skip")
    return time.perf_counter() - started


@contextmanager
def cache_context(core: Any, clear_caches: Any) -> Iterator[str]:
    """Isolate cache I/O benchmarks from user cache directories."""
    saved_enable = bool(getattr(core, "ENABLE_ANCHOR_CACHE", False))
    saved_override = str(getattr(core, "ANCHOR_CACHE_DIR_OVERRIDE", "") or "")
    saved_cache_dir = getattr(core, "_CACHE_DIR", None)
    saved_ttl = int(getattr(core, "ANCHOR_CACHE_TTL", 0) or 0)
    cache_bundle = getattr(core, "_cache_api", None)
    hint_bundle = getattr(core, "_hint_builder_api", None)
    saved_cache_binding = getattr(cache_bundle, "_bindings", None)
    saved_hint_binding = getattr(hint_bundle, "_bindings", None)
    with tempfile.TemporaryDirectory(prefix="nautical-perf-cache-") as td:
        try:
            core.ENABLE_ANCHOR_CACHE = True
            core.ANCHOR_CACHE_DIR_OVERRIDE = td
            core.ANCHOR_CACHE_TTL = 0
            core._CACHE_DIR = None
            if cache_bundle is not None:
                cache_bundle._bindings = None
            if hint_bundle is not None:
                hint_bundle._bindings = None
            try:
                core._CACHE_LOAD_MEM.clear()
            except Exception:
                pass
            yield td
        finally:
            clear_caches()
            core.ENABLE_ANCHOR_CACHE = saved_enable
            core.ANCHOR_CACHE_DIR_OVERRIDE = saved_override
            core.ANCHOR_CACHE_TTL = saved_ttl
            core._CACHE_DIR = saved_cache_dir
            if cache_bundle is not None:
                cache_bundle._bindings = saved_cache_binding
            if hint_bundle is not None:
                hint_bundle._bindings = saved_hint_binding
            clear_caches()


def cache_payload(expr: str, idx: int) -> dict[str, Any]:
    return {
        "natural": expr,
        "next_dates": ["2026-01-01", "2026-01-08", "2026-01-15"],
        "meta": {"i": idx},
        "dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]],
    }


def cache_save(core: Any, clear_caches: Any, exprs: list[str], rounds: int) -> float:
    with cache_context(core, clear_caches):
        keys = [f"perf-save-{i}" for i in range(max(1, len(exprs)))]
        started = time.perf_counter()
        idx = 0
        for _ in range(rounds):
            for i, expr in enumerate(exprs):
                if not core.cache_save(keys[i], cache_payload(expr, idx)):
                    raise RuntimeError("cache_save benchmark write failed")
                idx += 1
        return time.perf_counter() - started


def cache_load_hot(core: Any, clear_caches: Any, exprs: list[str], rounds: int) -> float:
    with cache_context(core, clear_caches):
        keys = [f"perf-load-{i}" for i in range(max(1, len(exprs)))]
        for i, expr in enumerate(exprs):
            if not core.cache_save(keys[i], cache_payload(expr, i)):
                raise RuntimeError("cache_load benchmark setup write failed")
        clear_caches()
        started = time.perf_counter()
        for _ in range(rounds):
            for key in keys:
                if not isinstance(core.cache_load(key), dict):
                    raise RuntimeError("cache_load benchmark read failed")
        return time.perf_counter() - started
