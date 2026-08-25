"""Task-scoped schedule projections used by the typed on-modify flow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from .task_models import TaskPayload


def scheduler_callbacks(host: Any) -> tuple[Any, Any]:
    """Return the stable one-argument callbacks used by projection services."""
    def service_for_task(task: TaskPayload):
        return host._module("modify_runtime").scheduler_service_for_task(
            task,
            state=host._modify_runtime_state(),
            core=host.core,
            recurrence_seed_base=lambda value: recurrence_seed_base(host, value),
        )

    return lambda task: service_for_task(task).session.evaluator, service_for_task


def recurrence_seed_base(_host: Any, task: TaskPayload) -> str:
    return str(task.get("chainID") or task.get("uuid") or "preview").strip()


def cp_add_period(host: Any, dt: datetime, td: timedelta) -> datetime:
    secs = int(td.total_seconds())
    if secs % 86400 == 0:
        local = host._tolocal(dt)
        return host.core.build_local_datetime(
            (local + timedelta(days=int(secs // 86400))).date(),
            (local.hour, local.minute),
        ).astimezone(timezone.utc)
    return (dt + td).replace(microsecond=0)


def sequence_period_for_link(host: Any, tokens: list[dict], cp_str: str, link_no: int, chain_id: str | None = None) -> timedelta:
    index = (max(1, int(link_no)) - 1) % len(tokens)
    return host.core.cp_sequence_interval_for_token(
        tokens[index], cp=cp_str, link_no=link_no, token_index=index, chain_id=chain_id
    ) or timedelta()


def next_occurrence_after_local_dt(host: Any, dnf, after_local_dt: datetime, default_seed_date, seed_base: str, omit_dnf=None, fallback_hhmm: tuple[int, int] | None = None):
    if not dnf:
        return None
    return host._module("add_anchor_compute").anchor_next_occurrence_after_local_dt(
        dnf, after_local_dt, fallback_hhmm=fallback_hhmm or (0, 0),
        interval_seed=default_seed_date, seed_base=seed_base,
        omit_dnf=omit_dnf, default_seed_date=default_seed_date, core=host.core,
    )


def anchor_included_occurrences(host: Any, parent: TaskPayload, *, after_local_dt: datetime, inclusive: bool, limit: int, **_kwargs):
    service = scheduler_callbacks(host)[1](parent)
    return service.included_occurrences_after(after_local_dt, inclusive=inclusive, limit=limit)


def estimate_cp_final_by_max(host: Any, task: TaskPayload, next_due_utc: Any):
    return host._module("modify_completion_compute").estimate_cp_final_by_max(
        task,
        next_due_utc,
        coerce_int=host.core.coerce_int,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        sequence_period_for_link=lambda tokens, cp, link, chain=None: sequence_period_for_link(host, tokens, cp, link, chain),
        add_period=lambda dt, td: cp_add_period(host, dt, td),
        max_iterations=host._MAX_ITERATIONS,
        diagnostic=host._diag,
    )


def estimate_anchor_final_by_max(host: Any, task: TaskPayload, next_due_utc: Any, dnf: Any):
    evaluator_callback, _service_callback = scheduler_callbacks(host)
    return host._module("modify_completion_compute").estimate_anchor_final_by_max(
        task,
        next_due_utc,
        dnf,
        coerce_int=host.core.coerce_int,
        recurrence_seed_base=lambda task: recurrence_seed_base(host, task),
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=lambda value: host._module("modify_datetime_effects").safe_parse_datetime(host, value),
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=lambda task: host._module("modify_anchor_effects").omit_dnf_from_parent(host, task),
        recurrence_evaluator_for_task=evaluator_callback,
        anchor_file_provider_for=host._anchor_file_provider_for,
        anchor_included_occurrences=lambda *args, **kwargs: anchor_included_occurrences(host, *args, **kwargs),
        diagnostic=host._diag,
        max_iterations=host._MAX_ITERATIONS,
    )


def cap_from_until_cp(host: Any, task: TaskPayload, next_due_utc: Any):
    return host._module("modify_completion_compute").cap_from_until_cp(
        task,
        next_due_utc,
        parse_datetime=host._dtparse,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        coerce_int=host.core.coerce_int,
        sequence_period_for_link=lambda tokens, cp, link, chain=None: sequence_period_for_link(host, tokens, cp, link, chain),
        add_period=lambda dt, td: cp_add_period(host, dt, td),
        max_iterations=host._MAX_ITERATIONS,
    )


def cap_from_until_anchor(host: Any, task: TaskPayload, next_due_utc: Any, dnf: Any):
    evaluator_callback, _service_callback = scheduler_callbacks(host)
    return host._module("modify_completion_compute").cap_from_until_anchor(
        task,
        next_due_utc,
        dnf,
        parse_datetime=host._dtparse,
        coerce_int=host.core.coerce_int,
        recurrence_seed_base=lambda task: recurrence_seed_base(host, task),
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=lambda value: host._module("modify_datetime_effects").safe_parse_datetime(host, value),
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=lambda task: host._module("modify_anchor_effects").omit_dnf_from_parent(host, task),
        recurrence_evaluator_for_task=evaluator_callback,
        anchor_file_provider_for=host._anchor_file_provider_for,
        anchor_included_occurrences=lambda *args, **kwargs: anchor_included_occurrences(host, *args, **kwargs),
        compare_datetimes=host._compare_datetimes,
        max_iterations=host._MAX_ITERATIONS,
    )


__all__ = (
    "estimate_cp_final_by_max",
    "estimate_anchor_final_by_max",
    "cap_from_until_cp",
    "cap_from_until_anchor",
)
