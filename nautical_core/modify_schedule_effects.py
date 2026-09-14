"""Task-scoped schedule projections used by the typed on-modify flow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any

from .task_models import TaskPayload
from .task_datetime import datetime_value, parser_for_host
from .timeutil import compare_datetimes


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


@dataclass(frozen=True, slots=True)
class SchedulerPorts:
    runtime_module: Any
    state: Any
    core: Any


@dataclass(frozen=True, slots=True)
class AnchorOccurrencePorts:
    scheduler: SchedulerPorts


@dataclass(frozen=True, slots=True)
class CPCompletionPorts:
    compute: Any
    parse_datetime: Any
    coerce_int: Any
    parse_cp_sequence_tokens: Any
    sequence: SequencePorts
    schedule: SchedulePorts
    max_iterations: int
    diagnostic: Any


@dataclass(frozen=True, slots=True)
class AnchorCompletionPorts:
    compute: Any
    parse_datetime: Any
    coerce_int: Any
    scheduler: SchedulerPorts
    to_local_cached: Any
    safe_parse_datetime: Any
    anchor_file_fallback_hhmm: Any
    omit_dnf_from_parent: Any
    anchor_file_provider_for: Any
    compare_datetimes: Any
    max_iterations: int
    diagnostic: Any


def scheduler_ports_for(host: Any) -> SchedulerPorts:
    return SchedulerPorts(
        runtime_module=host._module("modify_runtime"),
        state=host._modify_runtime_state(),
        core=host.core,
    )


def cp_completion_ports_for(host: Any) -> CPCompletionPorts:
    return CPCompletionPorts(
        compute=host._module("modify_completion_compute"),
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        coerce_int=host.core.coerce_int,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        sequence=SequencePorts(host.core.cp_sequence_interval_for_token),
        schedule=SchedulePorts(host._tolocal, host.core.build_local_datetime),
        max_iterations=host._MAX_ITERATIONS,
        diagnostic=host._diag,
    )


def anchor_completion_ports_for(host: Any) -> AnchorCompletionPorts:
    scheduler = scheduler_ports_for(host)
    anchor_effects = host._module("modify_anchor_effects")
    value_effects = host._module("modify_value_effects")
    return AnchorCompletionPorts(
        compute=host._module("modify_completion_compute"),
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        coerce_int=host.core.coerce_int,
        scheduler=scheduler,
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=host._TASK_DATETIME_PARSER.parse,
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=lambda task: anchor_effects.omit_dnf_from_parent(
            anchor_effects.omit_ports_for(host), task
        ),
        anchor_file_provider_for=host._anchor_file_provider_for,
        compare_datetimes=lambda left, right: value_effects.compare_datetimes(
            value_effects.DatetimePorts(compare_datetimes), left, right
        ),
        max_iterations=host._MAX_ITERATIONS,
        diagnostic=host._diag,
    )


def scheduler_callbacks(ports: SchedulerPorts) -> tuple[Any, Any]:
    """Return the stable one-argument callbacks used by projection services."""
    def service_for_task(task: TaskPayload) -> Any:
        return ports.runtime_module.scheduler_service_for_task(
            task,
            state=ports.state,
            core=ports.core,
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
    ports: AnchorOccurrencePorts,
    parent: TaskPayload,
    *,
    after_local_dt: datetime,
    inclusive: bool,
    limit: int,
    **_kwargs: Any,
) -> Any:
    service = scheduler_callbacks(ports.scheduler)[1](parent)
    return service.included_occurrences_after(after_local_dt, inclusive=inclusive, limit=limit)


def estimate_cp_final_by_max(ports: CPCompletionPorts, task: TaskPayload, next_due_utc: Any) -> Any:
    return ports.compute.estimate_cp_final_by_max(
        task,
        next_due_utc,
        coerce_int=ports.coerce_int,
        parse_cp_sequence_tokens=ports.parse_cp_sequence_tokens,
        sequence_period_for_link=lambda tokens, cp, link, chain=None: sequence_period_for_link(ports.sequence, tokens, cp, link, chain),
        add_period=lambda dt, td: cp_add_period(ports.schedule, dt, td),
        max_iterations=ports.max_iterations,
        diagnostic=ports.diagnostic,
    )


def estimate_anchor_final_by_max(ports: AnchorCompletionPorts, task: TaskPayload, next_due_utc: Any, dnf: Any) -> Any:
    evaluator_callback, _service_callback = scheduler_callbacks(ports.scheduler)
    return ports.compute.estimate_anchor_final_by_max(
        task,
        next_due_utc,
        dnf,
        coerce_int=ports.coerce_int,
        recurrence_seed_base=recurrence_seed_base,
        to_local_cached=ports.to_local_cached,
        safe_parse_datetime=ports.safe_parse_datetime,
        anchor_file_fallback_hhmm=ports.anchor_file_fallback_hhmm,
        omit_dnf_from_parent=ports.omit_dnf_from_parent,
        recurrence_evaluator_for_task=evaluator_callback,
        anchor_file_provider_for=ports.anchor_file_provider_for,
        anchor_included_occurrences=lambda *args, **kwargs: anchor_included_occurrences(
            AnchorOccurrencePorts(ports.scheduler), *args, **kwargs
        ),
        diagnostic=ports.diagnostic,
        max_iterations=ports.max_iterations,
    )


def cap_from_until_cp(ports: CPCompletionPorts, task: TaskPayload, next_due_utc: Any) -> Any:
    return ports.compute.cap_from_until_cp(
        task,
        next_due_utc,
        parse_datetime=ports.parse_datetime,
        parse_cp_sequence_tokens=ports.parse_cp_sequence_tokens,
        coerce_int=ports.coerce_int,
        sequence_period_for_link=lambda tokens, cp, link, chain=None: sequence_period_for_link(ports.sequence, tokens, cp, link, chain),
        add_period=lambda dt, td: cp_add_period(ports.schedule, dt, td),
        max_iterations=ports.max_iterations,
    )


def cap_from_until_anchor(ports: AnchorCompletionPorts, task: TaskPayload, next_due_utc: Any, dnf: Any) -> Any:
    evaluator_callback, _service_callback = scheduler_callbacks(ports.scheduler)
    return ports.compute.cap_from_until_anchor(
        task,
        next_due_utc,
        dnf,
        parse_datetime=ports.parse_datetime,
        coerce_int=ports.coerce_int,
        recurrence_seed_base=recurrence_seed_base,
        to_local_cached=ports.to_local_cached,
        safe_parse_datetime=ports.safe_parse_datetime,
        anchor_file_fallback_hhmm=ports.anchor_file_fallback_hhmm,
        omit_dnf_from_parent=ports.omit_dnf_from_parent,
        recurrence_evaluator_for_task=evaluator_callback,
        anchor_file_provider_for=ports.anchor_file_provider_for,
        anchor_included_occurrences=lambda *args, **kwargs: anchor_included_occurrences(
            AnchorOccurrencePorts(ports.scheduler), *args, **kwargs
        ),
        compare_datetimes=ports.compare_datetimes,
        max_iterations=ports.max_iterations,
    )


__all__ = (
    "estimate_cp_final_by_max",
    "estimate_anchor_final_by_max",
    "cap_from_until_cp",
    "cap_from_until_anchor",
    "CPCompletionPorts",
    "AnchorCompletionPorts",
    "cp_completion_ports_for",
    "anchor_completion_ports_for",
)
