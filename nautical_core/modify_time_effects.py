"""Time-slot normalization for typed on-modify validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Protocol


class _TimeSlotsOwner(Protocol):
    def resolve_time_slots(
        self,
        value: object,
        target_date: date | None,
        *,
        config: dict[str, Any] | None = None,
        to_local: Callable[[Any], Any] | None = None,
    ) -> list[tuple[int, int]]: ...


class _TimeSlotCore(Protocol):
    ASTRONOMY_CONFIG: dict[str, Any]

    def _import_sibling(self, name: str) -> _TimeSlotsOwner: ...

    def to_local(self, value: Any) -> Any: ...


class TimeSlotHost(Protocol):
    core: _TimeSlotCore


@dataclass(frozen=True, slots=True)
class TimeSlotPorts:
    resolve_time_slots: Callable[[object, date | None], list[tuple[int, int]]]


def time_slot_ports_for(host: TimeSlotHost) -> TimeSlotPorts:
    time_slots = host.core._import_sibling("time_slots")
    config = host.core.ASTRONOMY_CONFIG
    to_local = host.core.to_local

    def resolve(value: object, target_date: date | None) -> list[tuple[int, int]]:
        return time_slots.resolve_time_slots(value, target_date, config=config, to_local=to_local)

    return TimeSlotPorts(
        resolve_time_slots=resolve,
    )


def normalize_hhmm_list(
    ports: TimeSlotPorts,
    value: object,
    target_date: date | None = None,
) -> list[tuple[int, int]]:
    if value is None:
        return []
    return ports.resolve_time_slots(value, target_date)


__all__ = ("TimeSlotHost", "TimeSlotPorts", "time_slot_ports_for", "normalize_hhmm_list")
