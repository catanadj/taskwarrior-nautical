"""Task-scoped schedule projections used by the typed on-modify flow."""

from __future__ import annotations

from typing import Any

from .task_models import TaskPayload


def estimate_cp_final_by_max(host: Any, task: TaskPayload, next_due_utc: Any):
    return host._module("modify_completion_compute").estimate_cp_final_by_max(
        task,
        next_due_utc,
        coerce_int=host.core.coerce_int,
        parse_cp_sequence_tokens=host.core.parse_cp_sequence_tokens,
        sequence_period_for_link=host._cp_sequence_period_for_link,
        add_period=host._cp_add_td,
        max_iterations=host._MAX_ITERATIONS,
        diagnostic=host._diag,
    )


def estimate_anchor_final_by_max(host: Any, task: TaskPayload, next_due_utc: Any, dnf: Any):
    return host._module("modify_completion_compute").estimate_anchor_final_by_max(
        task,
        next_due_utc,
        dnf,
        coerce_int=host.core.coerce_int,
        recurrence_seed_base=host._recurrence_seed_base,
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=host._safe_parse_datetime,
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=host._omit_dnf_from_parent,
        recurrence_evaluator_for_task=host._recurrence_evaluator_for_task,
        anchor_file_provider_for=host._anchor_file_provider_for,
        anchor_included_occurrences=host._anchor_included_occurrences,
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
        sequence_period_for_link=host._cp_sequence_period_for_link,
        add_period=host._cp_add_td,
        max_iterations=host._MAX_ITERATIONS,
    )


def cap_from_until_anchor(host: Any, task: TaskPayload, next_due_utc: Any, dnf: Any):
    return host._module("modify_completion_compute").cap_from_until_anchor(
        task,
        next_due_utc,
        dnf,
        parse_datetime=host._dtparse,
        coerce_int=host.core.coerce_int,
        recurrence_seed_base=host._recurrence_seed_base,
        to_local_cached=host._to_local_cached,
        safe_parse_datetime=host._safe_parse_datetime,
        anchor_file_fallback_hhmm=host._anchor_file_fallback_hhmm,
        omit_dnf_from_parent=host._omit_dnf_from_parent,
        recurrence_evaluator_for_task=host._recurrence_evaluator_for_task,
        anchor_file_provider_for=host._anchor_file_provider_for,
        anchor_included_occurrences=host._anchor_included_occurrences,
        compare_datetimes=host._compare_datetimes,
        max_iterations=host._MAX_ITERATIONS,
    )


__all__ = (
    "estimate_cp_final_by_max",
    "estimate_anchor_final_by_max",
    "cap_from_until_cp",
    "cap_from_until_anchor",
)
