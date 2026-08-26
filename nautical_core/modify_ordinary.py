"""Ordinary (non-completion) on-modify orchestration.

The hook facade owns Taskwarrior process state and compatibility names.  This
module owns the ordinary lifecycle decision flow and receives its policy and
UI callbacks explicitly so it can be loaded only when that lifecycle runs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Callable

from .task_changes import TaskTransition
from .task_models import TaskPayload, TaskTimestamp
from .modify_carry_workflow import NativeUntilDecision, TemporalCarryDecision
from .modify_workflow import ChainCompletionDecision, RecurrenceTransitionDecision


class RecurrenceActivationError(RuntimeError):
    """Raised when a recurrence transition cannot be applied safely."""


class _LifecyclePolicy:
    def recurrence_setting_changes(self, old: TaskPayload, new: TaskPayload, *, transition: TaskTransition | None = None) -> list[tuple[str, str, str]]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class OrdinaryModifyServices:
    field_changed: Callable[[TaskPayload, TaskPayload, str], bool]
    strip_quotes: Callable[[str], str]
    validate_anchor: Callable[[TaskPayload, TaskPayload, str], None]
    validate_omit: Callable[[str, str, str, str], None]
    reject_conflicting_types: Callable[[str, str, str], None]
    validate_chain_limits: Callable[[TaskPayload], None]
    preserve_cp_offsets: Callable[[TaskPayload, TaskPayload, str], TemporalCarryDecision]
    task_has_recurrence: Callable[[TaskPayload], bool]
    preserve_native_until: Callable[[TaskPayload, TaskPayload, str], NativeUntilDecision]
    validate_native_until: Callable[[TaskPayload], None]
    validate_native_until_slots: Callable[[TaskPayload], None]
    render_cp_adjustment: Callable[[TemporalCarryDecision], None]
    render_timing_warning: Callable[[TaskPayload, tuple[str, ...]], None]
    apply_transition: Callable[[TaskPayload, TaskPayload], RecurrenceTransitionDecision]
    short_uuid: Callable[[str], str]
    recurrence_enabled_rows: Callable[[TaskPayload, str], list[tuple[str, str]]]
    panel: Callable[..., None]
    render_disabled_summary: Callable[[TaskPayload, TaskPayload, ChainCompletionDecision], None]
    semantic_diff_value: Callable[[str, str], str]
    first_recurrence_target: Callable[[TaskPayload, str], datetime | None]
    fmtlocal: Callable[[datetime], str]
    render_recurrence_updated: Callable[[list[tuple[str, str, str]], TaskPayload], None]
    print_task: Callable[[TaskPayload], None]


def handle_non_completion_modify(
    old: TaskPayload,
    new: TaskPayload,
    *,
    services: OrdinaryModifyServices,
    lifecycle: _LifecyclePolicy,
    transition: TaskTransition | None = None,
) -> None:
    """Apply ordinary edit validation, carry-forward, and feedback policy."""
    input_transition = transition
    explicit_timing_changes = tuple(
        field
        for field in ("due", "scheduled", "wait")
        if services.field_changed(old, new, field)
    )
    new_anchor = services.strip_quotes(str(new.get("anchor") or "").strip())
    new_anchor_file = services.strip_quotes(str(new.get("anchor_file") or "").strip())
    if new_anchor_file:
        new["anchor_file"] = new_anchor_file
    new_omit = services.strip_quotes(str(new.get("omit") or "").strip())
    if new_omit:
        new["omit"] = new_omit
    new_omit_file = services.strip_quotes(str(new.get("omit_file") or "").strip())
    if new_omit_file:
        new["omit_file"] = new_omit_file

    if new_anchor:
        services.validate_anchor(old, new, new_anchor)
    services.validate_omit(new_anchor, new_anchor_file, new_omit, new_omit_file)

    new_cp = services.strip_quotes(str(new.get("cp") or "").strip())
    services.reject_conflicting_types(new_anchor, new_anchor_file, new_cp)
    recurrence_or_cap_changed = any(
        services.field_changed(old, new, field)
        for field in ("cp", "anchor", "anchor_file", "chainMax", "chainUntil")
    )
    if recurrence_or_cap_changed and (new_cp or new_anchor or new_anchor_file):
        services.validate_chain_limits(new)

    schedule_adjustment = services.preserve_cp_offsets(old, new, new_cp)
    if transition is not None:
        def observation_has_recurrence(observation: Any) -> bool:
            return any(
                bool(str(observation.field(field).raw_value() or "").strip())
                for field in ("cp", "anchor", "anchor_file")
            )

        old_has_recurrence = observation_has_recurrence(transition.old)
        new_has_recurrence = observation_has_recurrence(transition.new)
    else:
        old_has_recurrence = services.task_has_recurrence(old)
        new_has_recurrence = services.task_has_recurrence(new)
    recurrence_enabled = (
        new_has_recurrence and not old_has_recurrence
    )
    if new_has_recurrence and not recurrence_enabled:
        recurrence_kind = "cp" if new_cp else "anchor_file" if new_anchor_file else "anchor"
        native_until_decision = services.preserve_native_until(old, new, recurrence_kind)
    else:
        native_until_decision = NativeUntilDecision("unchanged")
    if native_until_decision.status == "rejected":
        raise RecurrenceActivationError(native_until_decision.reason)
    native_window_changed = any(
        services.field_changed(old, new, field)
        for field in ("due", "scheduled", "until", "anchor", "anchor_file", "anchor_mode")
    )
    if new_has_recurrence and (native_window_changed or recurrence_enabled):
        services.validate_native_until(new)
        services.validate_native_until_slots(new)
    if schedule_adjustment:
        services.render_cp_adjustment(schedule_adjustment)
    services.render_timing_warning(new, explicit_timing_changes)

    try:
        lifecycle_transition = services.apply_transition(old, new)
    except Exception as exc:
        raise RecurrenceActivationError(
            f"Nautical recurrence transition failed: {type(exc).__name__}: {exc}"
        ) from exc
    if lifecycle_transition and lifecycle_transition.state in {"enabled", "resumed"}:
        source = lifecycle_transition.source or (
            "anchor_file" if new_anchor_file else "anchor" if new_anchor else "cp"
        )
        first = services.first_recurrence_target(new, source)
        if first is not None and getattr(lifecycle_transition, "next_occurrence", None) is None:
            lifecycle_transition = replace(
                lifecycle_transition,
                next_occurrence=TaskTimestamp(first),
            )
    recurrence_removed = (
        old_has_recurrence
        and not new_has_recurrence
    )
    if input_transition is not None:
        recurrence_removed = old_has_recurrence and not new_has_recurrence
    chain_was_disabled = (
        (
            str(input_transition.old.field("chain").raw_value() or "").strip().lower() == "on"
            and str(input_transition.new.field("chain").raw_value() or "").strip().lower() == "off"
        )
        if input_transition is not None
        else (
            str(old.get("chain") or "").strip().lower() == "on"
            and str(new.get("chain") or "").strip().lower() == "off"
        )
    )
    if lifecycle_transition and lifecycle_transition.state == "enabled":
        rows = [
            (
                "Reason",
                lifecycle_transition.reason
                or "This task just gained Nautical recurrence and was promoted to chain:on.",
            ),
            ("Source", lifecycle_transition.source),
        ]
        rows.extend(services.recurrence_enabled_rows(new, lifecycle_transition.source))
        services.panel("⚓ Nautical enabled", rows, kind="note")
    elif lifecycle_transition and lifecycle_transition.state == "disabled":
        rows = [
            (
                "Reason",
                lifecycle_transition.reason or "This task's Nautical recurrence is disabled.",
            )
        ]
        if lifecycle_transition.source:
            rows.append(("Source", lifecycle_transition.source))
        rows.append(("Chain", "off"))
        services.panel("⚓ Nautical disabled", rows, kind="disabled")
        if recurrence_removed or chain_was_disabled:
            reason = (
                "Nautical recurrence removed."
                if recurrence_removed
                else "Chain manually disabled."
            )
            completion = ChainCompletionDecision(
                reason=reason,
                source="recurrence_removal" if recurrence_removed else "manual_chain_off",
            )
            services.render_disabled_summary(old, new, completion)
    elif lifecycle_transition and lifecycle_transition.state == "resumed":
        rows = [
            ("Reason", lifecycle_transition.reason or "This task's Nautical recurrence was resumed.")
        ]
        if lifecycle_transition.source:
            rows.append(("Source", lifecycle_transition.source))
        rows.append(("Chain", services.semantic_diff_value("off", "on")))
        source = "anchor" if new.get("anchor") else "anchor_file" if new.get("anchor_file") else "cp"
        first = lifecycle_transition.next_occurrence.value if lifecycle_transition.next_occurrence else None
        if first:
            rows.append(("Next", services.fmtlocal(first)))
        services.panel("⚓ Nautical resumed", rows, kind="note")
    else:
        try:
            changes = lifecycle.recurrence_setting_changes(old, new, transition=input_transition)
        except Exception:
            changes = []
        services.render_recurrence_updated(changes, new)
    services.print_task(new)


__all__ = (
    "OrdinaryModifyServices",
    "RecurrenceActivationError",
    "handle_non_completion_modify",
)
