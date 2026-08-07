"""Public cache entry points bound to one core facade instance."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any):
    """Create cache APIs without sharing cache state across core loaders."""
    return SimpleNamespace(
        cache_load=module._cache_load_impl,
        cache_save=module._cache_save_impl,
        cache_gc=module._cache_gc_impl,
        cache_key_for_task=module._cache_key_for_task_impl,
    )


__all__ = ("for_core",)
