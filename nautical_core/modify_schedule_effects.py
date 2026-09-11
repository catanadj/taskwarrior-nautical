"""Task-scoped schedule projections used by the typed on-modify flow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any

from .task_models import TaskPayload
from .task_datetime import datetime_value, parser_for_host


@dataclass(frozen=True, slots=True)
class SchedulePorts:
    """Minimal clock/calendar ports needed by schedule projection helpers."""

    to_local: Any
    build_local_datetime: Any


@dataclass(frozen=True, slots=True)
class SequencePorts:
    """Minimal port for computing a CP sequence period."""

    sequence_interval: Any


@dataclass(frozen=True, slots=True)
class OccurrencePorts:
    next_occurrence: Any


def scheduler_callbacks(host: Any) -> tuple[Any, Any]:
    """Return the stable one-argument callbacks used by projection services."""
    def service_for_task(task: TaskPayload) -> Any:
        return host._module("modify_runtime").scheduler_service_for_task(
            task,
            state=host._modify_runtime_state(),
            core=host.core,
            recurrence_seed_base=recurrence_seed_base,
        )

    return lambda task: service_for_task(task).session.evaluator, service_for_task


def recurrence_seed_base(task: TaskPayload) -> str:
    return str(task.get("chainID") or task.get("uuid") or "preview").strip()


def cp_add_period(ports: SchedulePorts, dt: datetime, td: timedelta) -> datetime:
    secs = int(td.total_seconds())
    if secs % 86400 == 0:
        local = ports.to_local(dt)
        return ports.build_local_datetime(
            (local + timedelta(days=int(secs // 86400))).date(),
            (local.hour, local.minute),
        ).astimezone(timezone.utc)
    return (dt + td).replace(microsecond=0)


def sequence_period_for_link(ports: SequencePorts, tokens: list[dict], cp_str: str, link_no: int, chain_id: str | None = None) -> timedelta:
    index = (max(1, int(link_no)) - 1) % len(tokens)
    return ports.sequence_interval(
        tokens[index], cp=cp_str, link_no=link_no, token_index=index, chain_id=chain_id
    ) or timedelta()


def next_occurrence_after_local_dt(
    ports: OccurrencePorts,
    dnf: Any,
    after_local_dt: datetime,
    default_seed_date: Any,
    seed_base: str,
    omit_dnf: Any = None,
    fallback_hhmm: tuple[int, int] | None = None,
) -> Any:
    if not dnf:
        return None
    return ports.next_occurrence(
        dnf, after_local_dt, fallback_hhmm=fallback_hhmm or (0, 0),
        interval_seed=default_seed_date, seed_base=seed_base,
        omit_dnf=omit_dnf, default_seed_date=default_seed_date,
    )


def anchor_included_occurrences(
    host: Any,
    parent: TaskPayload,
    *,
    after_local_dt: datetime,
    inclusive: bool,
    limit: int,
    **_kwargs: Any,
) -> Any:
    service = scheduler_callbacks(host)[1](parent)
    return service.included_occurrences_after(after_local_dt, inclusive=inclusive, limit=limit)


def estimate_cp_final_by_max(host: Any, task: TaskPayload, next_due_utc: Any) -> Any:
    ports = SequencePorts(host.core.cp_sequence_interval_for_token)
    schedule_ports = SchedulePorts(host._tolocal, host.core.build_local_datetime)
    return host._module("modify_completion_compute").estimate_cp_final_by_max(
        task,
        next_due_utc,
        coerce_int=host.core.coerce_int,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        sequence_period_for_link=lambda tokens, cp, link, chain=None: sequence_period_for_link(ports, tokens, cp, link, chain),
        add_period=lambda dt, td: cp_add_period(schedule_ports, dt, td),
        max_iterations=host._MAX_ITERATIONS,
        diagnostic=host._diag,
    )


def estimate_anchor_final_by_max(host: Any, task: TaskPayload, next_due_utc: Any, dnf: Any) -> Any:
    evaluator_callback, _service_callback = scheduler_callbacks(host)
    return host._module("modify_completion_compute").estimate_anchor_final_by_max(
        task,
        next_due_utc,
        dnf,
        coerce_int=host.core.coerce_int,
        recurrence_seed_base=recurrence_seed_base,
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=host._TASK_DATETIME_PARSER.parse,
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=lambda task: host._module("modify_anchor_effects").omit_dnf_from_parent(
            host._module("modify_anchor_effects").omit_ports_for(host), task
        ),
        recurrence_evaluator_for_task=evaluator_callback,
        anchor_file_provider_for=host._anchor_file_provider_for,
        anchor_included_occurrences=lambda *args, **kwargs: anchor_included_occurrences(host, *args, **kwargs),
        diagnostic=host._diag,
        max_iterations=host._MAX_ITERATIONS,
    )


def cap_from_until_cp(host: Any, task: TaskPayload, next_due_utc: Any) -> Any:
    ports = SequencePorts(host.core.cp_sequence_interval_for_token)
    schedule_ports = SchedulePorts(host._tolocal, host.core.build_local_datetime)
    return host._module("modify_completion_compute").cap_from_until_cp(
        task,
        next_due_utc,
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        coerce_int=host.core.coerce_int,
        sequence_period_for_link=lambda tokens, cp, link, chain=None: sequence_period_for_link(ports, tokens, cp, link, chain),
        add_period=lambda dt, td: cp_add_period(schedule_ports, dt, td),
        max_iterations=host._MAX_ITERATIONS,
    )


def cap_from_until_anchor(host: Any, task: TaskPayload, next_due_utc: Any, dnf: Any) -> Any:
    evaluator_callback, _service_callback = scheduler_callbacks(host)
    return host._module("modify_completion_compute").cap_from_until_anchor(
        task,
        next_due_utc,
        dnf,
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        coerce_int=host.core.coerce_int,
        recurrence_seed_base=recurrence_seed_base,
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=host._TASK_DATETIME_PARSER.parse,
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=lambda task: host._module("modify_anchor_effects").omit_dnf_from_parent(
            host._module("modify_anchor_effects").omit_ports_for(host), task
        ),
        recurrence_evaluator_for_task=evaluator_callback,
        anchor_file_provider_for=host._anchor_file_provider_for,
        anchor_included_occurrences=lambda *args, **kwargs: anchor_included_occurrences(host, *args, **kwargs),
        compare_datetimes=lambda left, right: host._module("modify_value_effects").compare_datetimes(
            host._module("modify_value_effects").DatetimePorts(host._module("timeutil").compare_datetimes), left, right
        ),
        max_iterations=host._MAX_ITERATIONS,
    )


__all__ = (
    "estimate_cp_final_by_max",
    "estimate_anchor_final_by_max",
    "cap_from_until_cp",
    "cap_from_until_anchor",
)
