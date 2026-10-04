"""Task-scoped schedule projections used by the typed on-modify flow."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .task_models import TaskPayload
from .task_datetime import datetime_value, parser_for_host
from .modify_models import (
    CoerceIntCallback,
    DatetimeParserCallback,
    DiagnosticCallback,
    SafeParseDatetimeCallback,
    AnchorFileProviderFactory,
    AnchorIncludedOccurrencesCallback,
    AnchorDNF,
    OmitState,
    OccurrenceProvider,
)
from .scheduler_service import SchedulerService
from .recurrence_evaluator import RecurrenceEvaluator
from .timeutil import compare_datetimes


@dataclass(frozen=True, slots=True)
class SchedulePorts:
    """Minimal clock/calendar ports needed by schedule projection helpers."""

    to_local: Callable[[datetime], datetime]
    build_local_datetime: Callable[[date, tuple[int, int]], datetime]


class SequenceIntervalForToken(Protocol):
    """Resolve one parsed CP token to its recurrence interval."""

    def __call__(
        self,
        token: dict[str, Any],
        *,
        cp: str,
        link_no: int,
        token_index: int,
        chain_id: str | None = None,
    ) -> timedelta | None: ...


@dataclass(frozen=True, slots=True)
class SequencePorts:
    """Minimal port for computing a CP sequence period."""

    sequence_interval: SequenceIntervalForToken


class CPCompletionCompute(Protocol):
    """CP final-date calculations exposed by the completion compute owner."""

    estimate_cp_final_by_max: Callable[..., datetime | None]
    cap_from_until_cp: Callable[..., tuple[int | None, datetime | None]]


class AnchorCompletionCompute(Protocol):
    """Anchor final-date calculations exposed by the completion compute owner."""

    estimate_anchor_final_by_max: Callable[..., datetime | None]
    cap_from_until_anchor: Callable[..., tuple[int | None, datetime | None]]


class NextOccurrenceAfterLocalDateTime(Protocol):
    """Resolve the next anchor occurrence using an explicit date context."""

    def __call__(
        self,
        dnf: Any,
        after_local_dt: datetime,
        *,
        fallback_hhmm: tuple[int, int],
        interval_seed: date | None,
        seed_base: str,
        omit_dnf: OmitState | None,
        default_seed_date: date | None,
    ) -> datetime | None: ...


@dataclass(frozen=True, slots=True)
class OccurrencePorts:
    next_occurrence: NextOccurrenceAfterLocalDateTime


@dataclass(frozen=True, slots=True)
class SchedulerPorts:
    service_for_task: SchedulerServiceForTask


class SchedulerServiceForTask(Protocol):
    """Bind one Taskwarrior task to its task-scoped scheduler service."""

    def __call__(self, task: TaskPayload) -> SchedulerService: ...


@dataclass(frozen=True, slots=True)
class AnchorOccurrencePorts:
    scheduler: SchedulerPorts


@dataclass(frozen=True, slots=True)
class CPCompletionPorts:
    compute: CPCompletionCompute
    parse_datetime: DatetimeParserCallback
    coerce_int: CoerceIntCallback
    parse_cp_sequence_tokens: Callable[[str], list[dict[str, Any]] | None]
    sequence: SequencePorts
    schedule: SchedulePorts
    max_iterations: int
    diagnostic: DiagnosticCallback


@dataclass(frozen=True, slots=True)
class AnchorCompletionPorts:
    compute: AnchorCompletionCompute
    parse_datetime: DatetimeParserCallback
    coerce_int: CoerceIntCallback
    scheduler: SchedulerPorts
    to_local_cached: Callable[[datetime], datetime]
    safe_parse_datetime: SafeParseDatetimeCallback
    anchor_file_fallback_hhmm: Callable[[dict[str, Any], datetime], tuple[int, int]]
    omit_dnf_from_parent: Callable[[dict[str, Any]], tuple[str, Any]]
    anchor_file_provider_for: AnchorFileProviderFactory
    compare_datetimes: Callable[[datetime, datetime], int]
    max_iterations: int
    diagnostic: DiagnosticCallback


def scheduler_ports_for(host: Any) -> SchedulerPorts:
    runtime_module = host._module("modify_runtime")
    state = host._modify_runtime_state()
    core = host.core

    def service_for_task(task: TaskPayload) -> SchedulerService:
        return runtime_module.scheduler_service_for_task(
            task,
            state=state,
            core=core,
            recurrence_seed_base=recurrence_seed_base,
        )

    return SchedulerPorts(service_for_task=service_for_task)


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


def scheduler_callbacks(
    ports: SchedulerPorts,
) -> tuple[Callable[[TaskPayload], RecurrenceEvaluator], SchedulerServiceForTask]:
    """Return the stable one-argument callbacks used by projection services."""
    service_for_task = ports.service_for_task

    def evaluator_for_task(task: TaskPayload) -> RecurrenceEvaluator:
        return service_for_task(task).session.evaluator

    return evaluator_for_task, service_for_task


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
    default_seed_date: date | None,
    seed_base: str,
    omit_dnf: Any = None,
    fallback_hhmm: tuple[int, int] | None = None,
) -> datetime | None:
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
) -> list[datetime]:
    service = scheduler_callbacks(ports.scheduler)[1](parent)
    return service.included_occurrences_after(after_local_dt, inclusive=inclusive, limit=limit)


def _anchor_included_occurrences_callback(
    scheduler: SchedulerPorts,
) -> AnchorIncludedOccurrencesCallback:
    ports = AnchorOccurrencePorts(scheduler)

    def include(
        task: TaskPayload,
        *,
        after_local_dt: datetime,
        inclusive: bool,
        limit: int,
        fallback_hhmm: tuple[int, int],
        omit_dnf: OmitState | None,
        seed_base: str,
        default_seed_date: date | None,
        dnf: AnchorDNF | None,
        anchor_file_provider: OccurrenceProvider | None,
    ) -> list[datetime]:
        return anchor_included_occurrences(
            ports,
            task,
            after_local_dt=after_local_dt,
            inclusive=inclusive,
            limit=limit,
            fallback_hhmm=fallback_hhmm,
            omit_dnf=omit_dnf,
            seed_base=seed_base,
            default_seed_date=default_seed_date,
            dnf=dnf,
            anchor_file_provider=anchor_file_provider,
        )

    return include


def estimate_cp_final_by_max(
    ports: CPCompletionPorts,
    task: TaskPayload,
    next_due_utc: Any,
) -> datetime | None:
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


def estimate_anchor_final_by_max(
    ports: AnchorCompletionPorts,
    task: TaskPayload,
    next_due_utc: Any,
    dnf: Any,
) -> datetime | None:
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
        anchor_included_occurrences=_anchor_included_occurrences_callback(
            ports.scheduler
        ),
        diagnostic=ports.diagnostic,
        max_iterations=ports.max_iterations,
    )


def cap_from_until_cp(
    ports: CPCompletionPorts,
    task: TaskPayload,
    next_due_utc: Any,
) -> tuple[int | None, datetime | None]:
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


def cap_from_until_anchor(
    ports: AnchorCompletionPorts,
    task: TaskPayload,
    next_due_utc: Any,
    dnf: Any,
) -> tuple[int | None, datetime | None]:
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
        anchor_included_occurrences=_anchor_included_occurrences_callback(
            ports.scheduler
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
