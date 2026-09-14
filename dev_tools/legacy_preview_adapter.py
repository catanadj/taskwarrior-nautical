"""Test-only compatibility adapter for evaluator-less preview collection.

Production preview collection requires the shared scheduler/evaluator boundary.
This module preserves the old direct-helper path for characterization fixtures
that intentionally exercise the legacy anchor-inclusion implementation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nautical_core import anchor_inclusion
from nautical_core.occurrence_provider import (
    AnchorEventOccurrenceProvider,
    Occurrence,
    OccurrenceBatch,
    collect_after,
)
from nautical_core.recurrence_protocols import NextOccurrenceCallback, PickOccurrenceCallback


def _build_slot_datetime(day: Any, hhmm: tuple[int, int]) -> datetime:
    return datetime.combine(day, datetime.min.time().replace(hour=int(hhmm[0]), minute=int(hhmm[1])))


def collect_events_legacy(
    *,
    dnf: Any,
    anchor_file_str: str,
    after_local_dt: datetime,
    inclusive: bool,
    limit_included: int,
    fallback_hhmm: tuple[int, int],
    default_seed_date: Any,
    seed_base: str,
    omit_dnf: Any,
    core: Any,
    next_occurrence_after_local_dt: NextOccurrenceCallback,
    pick_occurrence_local: PickOccurrenceCallback | None = None,
    anchor_file_dir: str = "",
    max_iterations: int = 512,
    return_occurrences: bool = False,
    anchor_file_provider: Any | None = None,
    max_file_skips: int | None = None,
) -> list[Any]:
    """Collect using the pre-service path retained solely for test fixtures."""
    from nautical_core.anchor_inclusion import next_included_occurrence, next_occurrence_event_local

    if (
        anchor_file_provider is not None
        and (
            getattr(anchor_file_provider, "name", None) != anchor_file_str
            or getattr(anchor_file_provider, "anchor_file_dir", None) != anchor_file_dir
            or getattr(anchor_file_provider, "fallback_hhmm", None) != fallback_hhmm
        )
    ):
        anchor_file_provider = None
    if anchor_file_provider is None:
        anchor_file_provider = (
            anchor_inclusion._build_anchor_file_provider(
                anchor_file_str,
                anchor_file_dir=anchor_file_dir,
                fallback_hhmm=fallback_hhmm,
                seed_base=seed_base,
                core=core,
            )
            if anchor_file_str
            else None
        )
    def next_event(value: datetime) -> Any:
        callback = next_included_occurrence if max_file_skips is not None else next_occurrence_event_local
        kwargs = dict(
            dnf=dnf,
            anchor_file_str=anchor_file_str,
            after_local_dt=value,
            inclusive=False,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=next_occurrence_after_local_dt,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_dir=anchor_file_dir,
            anchor_file_provider=anchor_file_provider,
        )
        if max_file_skips is not None:
            kwargs["max_file_skips"] = max_file_skips
        return callback(**kwargs)

    provider = AnchorEventOccurrenceProvider(
        next_event,
        source="anchor+anchor_file" if anchor_file_str and dnf else ("anchor_file" if anchor_file_str else "anchor"),
    )
    collected = collect_after(
        provider,
        after_local_dt,
        limit=limit_included,
        inclusive=inclusive,
        max_iterations=max_iterations,
        build_local_datetime=getattr(core, "build_local_datetime", _build_slot_datetime),
        to_local=lambda value: value,
    )
    if return_occurrences:
        return list(collected)
    return OccurrenceBatch(
        [
            (occurrence.local_datetime, occurrence.omitted)
            for occurrence in collected
            if occurrence.local_datetime is not None
        ],
        terminal=getattr(collected, "terminal", None),
    )


def collect_included_legacy(**kwargs: Any) -> list[datetime]:
    """Return only included datetimes for legacy omission-scan fixtures."""
    kwargs["max_file_skips"] = kwargs.get("max_file_skips", kwargs.get("max_iterations", 512))
    kwargs["return_occurrences"] = True
    events = collect_events_legacy(**kwargs)
    return [event.local_datetime for event in events if isinstance(event, Occurrence) and event.local_datetime is not None]
