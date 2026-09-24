"""Anchor-expression and scheduler benchmark workloads."""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any


def describe(core: Any, exprs: list[str], rounds: int, clear_caches: Any) -> float:
    clear_caches()
    started = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.describe_anchor_expr(expr)
    return time.perf_counter() - started


def next_after(core: Any, exprs: list[str], rounds: int, clear_caches: Any) -> float:
    clear_caches()
    dnfs = [core.validate_anchor_expr_strict(expr) for expr in exprs]
    reference = date(2026, 1, 1)
    started = time.perf_counter()
    for _ in range(rounds):
        for dnf in dnfs:
            core.next_after_expr(dnf, reference)
    return time.perf_counter() - started


def decisions(core: Any, exprs: list[str], task_codec: Any, resource_details: dict[str, object]) -> float:
    from nautical_core.scheduler_cursor import OccurrenceCursor
    from nautical_core.scheduler_service import SchedulerService
    from nautical_core.scheduler_trace import SchedulerTrace

    row = {
        "uuid": "00000000-0000-4000-8000-000000000099", "description": "scheduler decision benchmark",
        "status": "pending", "chain": "on", "chainID": "scheduler-perf", "link": 1,
        "due": "20260824T090000Z",
    }
    started = time.perf_counter()
    measured: dict[str, int] = {}
    for expr in exprs:
        observation = task_codec.DEFAULT_TASK_CODEC.decode_row({**row, "anchor": expr}, source_query="perf:scheduler-decisions")
        trace = SchedulerTrace(enabled=True, max_events=1)
        service = SchedulerService.from_observation(observation, trace=trace)
        cursor = OccurrenceCursor.strict_after(datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc)
        service.next(cursor)
        measured[expr] = trace.last_decision_count
    resource_details["scheduler_decisions"] = measured
    return time.perf_counter() - started
