"""Time-slot normalization for typed on-modify validation."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from .callback_ports import CallbackPort


@dataclass(frozen=True, slots=True)
class TimeSlotPorts:
    resolve_time_slots: CallbackPort


def time_slot_ports_for(host: Any) -> TimeSlotPorts:
    time_slots = host.core._import_sibling("time_slots")
    return TimeSlotPorts(
        resolve_time_slots=lambda value, target_date, config=getattr(host.core, "ASTRONOMY_CONFIG", {}): time_slots.resolve_time_slots(
            value, target_date, config=config, to_local=host.core.to_local
        )
    )


def normalize_hhmm_list(ports: TimeSlotPorts, value: Any, target_date: Any = None) -> list[tuple[int, int]]:
    if value is None:
        return []
    return ports.resolve_time_slots(value, target_date)


__all__ = ("TimeSlotPorts", "time_slot_ports_for", "normalize_hhmm_list")
