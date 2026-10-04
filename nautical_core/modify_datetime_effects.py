"""Datetime parsing effects shared by the typed on-modify workflow."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Protocol

from .task_datetime import datetime_value, parser_for_host


@dataclass(frozen=True, slots=True)
class DatetimeEffectPorts:
    parse_datetime: Callable[[object], datetime | None]
    utc_to_local: Callable[[datetime], datetime]
    local_to_utc: Callable[[datetime], datetime]


class _DatetimeEffectsCore(Protocol):
    utc_to_local_naive: Callable[[datetime], datetime]
    local_naive_to_utc: Callable[[datetime], datetime]


class DatetimeEffectsHost(Protocol):
    core: _DatetimeEffectsCore


def safe_dt(ports: DatetimeEffectPorts, value: object) -> datetime | None:
    try:
        return value if isinstance(value, datetime) else ports.parse_datetime(value)
    except (TypeError, ValueError):
        return None


def utc_to_local_naive(ports: DatetimeEffectPorts, value: datetime) -> datetime:
    if not hasattr(value, "tzinfo"):
        raise TypeError("dt_utc must be datetime")
    return ports.utc_to_local(value)


def local_naive_to_utc(ports: DatetimeEffectPorts, value: datetime) -> datetime:
    if not hasattr(value, "tzinfo"):
        raise TypeError("dt_local_naive must be datetime")
    return ports.local_to_utc(value.replace(microsecond=0))


def datetime_effect_ports_for(host: DatetimeEffectsHost) -> DatetimeEffectPorts:
    return DatetimeEffectPorts(
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        utc_to_local=host.core.utc_to_local_naive,
        local_to_utc=host.core.local_naive_to_utc,
    )


__all__ = ("DatetimeEffectPorts", "datetime_effect_ports_for", "safe_dt", "utc_to_local_naive", "local_naive_to_utc")
