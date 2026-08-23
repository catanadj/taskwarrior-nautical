from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .task_models import TaskPayload

from nautical_core.timeutil import compare_datetimes
from nautical_core.lifecycle_models import DeletionEvidence, LifecycleEvent
from nautical_core.modify_lifecycle import apply_terminal_transition
from nautical_core.task_codec import DEFAULT_TASK_CODEC


@dataclass(slots=True)
class ExpirationServices:
    core: Any
    reconcile: Any
    safe_parse_datetime: Any
    compute_anchor_child_due: Any
    compute_cp_child_due: Any
    build_child_draft: Any
    spawn_child_atomic: Any
    panel: Any
    short: Any
    diag: Any


@dataclass(slots=True)
class DeletedModifyServices:
    expiration: ExpirationServices
    terminal_chain_off: Any
    now_utc: Any
    end_chain_summary: Any
    format_root_and_age: Any
    short: Any
    panel: Any
    diag: Any
    recovery_warning: Any


def has_expiration_evidence(task: TaskPayload, *, safe_parse_datetime) -> bool:
    try:
        until_dt, until_err = safe_parse_datetime(task.get("until"))
        end_dt, end_err = safe_parse_datetime(task.get("end"))
        return bool(
            not until_err
            and not end_err
            and until_dt is not None
            and end_dt is not None
            and compare_datetimes(until_dt, end_dt) <= 0
        )
    except Exception:
        return False


def classify_deleted_task(
    task: TaskPayload,
    *,
    services: ExpirationServices,
    observation: Any = None,
) -> DeletionEvidence:
    """Return the deletion disposition without turning unavailable evidence into manual stop."""
    if observation is None:
        observation = DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify deletion classification")
    return services.reconcile.deleted_chain_disposition(
        observation,
        safe_parse_datetime=services.safe_parse_datetime,
    )


def render_recovery_warning(task: TaskPayload, reason: str, *, services: ExpirationServices) -> None:
    services.panel(
        "⚠ Nautical expiration recovery deferred",
        [
            ("Task", services.short(task.get("uuid")) or "–"),
            ("Reason", reason or "The next occurrence could not be prepared."),
            ("Action", "Run nautical reconcile --apply."),
        ],
        kind="warning",
    )


def _render_recovery_panel(
    task: TaskPayload,
    plan,
    *,
    services: ExpirationServices,
    result: str = "",
    child_short: str = "",
) -> None:
    current_link = services.core.coerce_int(task.get("link"), 1)
    description = str(task.get("description") or "").strip()
    task_label = f"#{current_link}" + (f" · {description}" if description else "")
    rows = [("Expired", task_label)]
    if result:
        rows.append(("Result", result))
    if plan.child_due is not None:
        next_label = "Blocked next" if plan.action == "legitimate_final" else "Next"
        rows.append((next_label, services.core.fmt_dt_local(plan.child_due)))
    child_until = plan.child.get("until") if isinstance(plan.child, dict) else None
    child_until_dt, child_until_err = services.safe_parse_datetime(child_until)
    if child_until_dt is not None and not child_until_err:
        if plan.child_due is not None:
            try:
                add_validation = services.core._import_sibling("add_validation")
                carry = add_validation.describe_native_until_carry(
                    child_until_dt,
                    plan.child_due,
                    to_local=services.core.to_local,
                )
            except Exception:
                carry = None
            if carry:
                rows.append(("Expiration", carry))
        rows.append(("Next expires", services.core.fmt_dt_local(child_until_dt)))
    rows.append(("Link", f"#{plan.next_link}"))
    if child_short:
        rows.append(("Child", child_short))
    if plan.action == "legitimate_final":
        rows.append(("Boundary", plan.reason))
    panel_kind = "summary" if plan.action == "legitimate_final" else "note"
    services.panel("⌛ Nautical occurrence expired", rows, kind=panel_kind)


def handle_expired_deleted_modify(task: TaskPayload, *, services: ExpirationServices) -> bool:
    reconcile = services.reconcile
    try:
        observation = DEFAULT_TASK_CODEC.decode_row(
            task,
            source_query="on-modify expiration recovery",
        )
    except Exception as exc:
        services.diag(f"expiration recovery task decode failed: {exc}")
        render_recovery_warning(task, "The expired task could not be validated for recovery.", services=services)
        return True
    if not reconcile.is_orphan_expiration_candidate(
        observation,
        safe_parse_datetime=services.safe_parse_datetime,
    ):
        return False

    plan_hook = SimpleNamespace(
        core=services.core,
        _safe_parse_datetime=services.safe_parse_datetime,
        _compute_anchor_child_due=services.compute_anchor_child_due,
        _compute_cp_child_due=services.compute_cp_child_due,
        _build_child_draft=services.build_child_draft,
    )
    plan = reconcile.plan_recovery_decision(observation, existing_children=[], hook=plan_hook)

    if plan.action == "legitimate_final":
        apply_terminal_transition(task, LifecycleEvent.EXPIRE)
        _render_recovery_panel(
            task,
            plan,
            services=services,
            result="[yellow]Chain finished at configured limit[/]",
        )
        return True
    if plan.action != "spawn" or not plan.child:
        render_recovery_warning(task, plan.reason, services=services)
        return True

    try:
        child_short, _stripped, verified, deferred, reason, _intent_id = services.spawn_child_atomic(
            plan.child,
            task,
        )
    except Exception as exc:
        services.diag(f"expiration child queue failed: {exc}")
        render_recovery_warning(task, "The next occurrence could not be queued.", services=services)
        return True
    if verified or deferred:
        task["nextLink"] = child_short
    if verified:
        _render_recovery_panel(
            task,
            plan,
            services=services,
            result="[green]Next occurrence created[/]",
            child_short=child_short,
        )
    elif not deferred:
        render_recovery_warning(
            task,
            reason or "The next occurrence could not be queued.",
            services=services,
        )
    return True


def handle_deleted_modify(
    old: dict[str, Any],
    new: dict[str, Any],
    *,
    services: DeletedModifyServices,
    transition: Any = None,
) -> None:
    """Classify one deleted pending task and converge its chain state."""
    old_status = (
        transition.old.field("status").raw_value()
        if transition is not None
        else old.get("status")
    )
    if str(old_status or "").strip().lower() != "pending":
        return
    old_chain_id = (
        transition.old.field("chainID").raw_value()
        if transition is not None
        else old.get("chainID")
    )
    new_chain_id = (
        transition.new.field("chainID").raw_value()
        if transition is not None
        else new.get("chainID")
    )
    if not ((old_chain_id or new_chain_id or "").strip()):
        return
    expiration = services.expiration
    try:
        evidence = classify_deleted_task(
            new,
            services=expiration,
            observation=(transition.new if transition is not None else None),
        )
        disposition = evidence.disposition.value
        disposition_reason = evidence.reason
    except Exception as exc:
        services.diag(f"deleted-task disposition failed: {exc}")
        services.recovery_warning(new, "Deletion evidence could not be classified safely.")
        return
    if disposition == "ambiguous":
        services.recovery_warning(
            new,
            disposition_reason or "Deletion evidence is unavailable or malformed.",
        )
        return
    if disposition == "expiration":
        try:
            if handle_expired_deleted_modify(new, services=expiration):
                return
        except Exception as exc:
            services.diag(f"expiration recovery failed: {exc}")
        services.recovery_warning(
            new,
            "Expiration recovery could not be initialized; the chain remains active.",
        )
        return
    if disposition == "manual":
        services.diag("deleted Nautical task classified as manual stop")

    services.terminal_chain_off(new, "manual_delete")
    now_utc = services.now_utc()
    try:
        services.end_chain_summary(new, "Pending task deleted.", now_utc, current_task=old)
    except Exception as exc:
        services.diag(f"delete chain summary failed: {exc}")
        services.panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", "Pending Nautical task was deleted."),
                ("Root", services.format_root_and_age(old, now_utc)),
                ("Task", services.short(old.get("uuid")) or "–"),
            ],
            kind="summary",
        )


__all__ = (
    "ExpirationServices",
    "DeletedModifyServices",
    "classify_deleted_task",
    "handle_deleted_modify",
    "handle_expired_deleted_modify",
    "has_expiration_evidence",
    "render_recovery_warning",
)
