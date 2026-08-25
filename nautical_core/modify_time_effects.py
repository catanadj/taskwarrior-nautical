"""Time-slot normalization for typed on-modify validation."""

from __future__ import annotations

from typing import Any


def normalize_hhmm_list(host: Any, value: Any, target_date=None) -> list[tuple[int, int]]:
    if value is None:
        return []
    time_slots = host.core._import_sibling("time_slots")
    return time_slots.resolve_time_slots(
        value,
        target_date,
        config=getattr(host.core, "ASTRONOMY_CONFIG", {}),
        to_local=host.core.to_local,
    )


__all__ = ("normalize_hhmm_list",)
