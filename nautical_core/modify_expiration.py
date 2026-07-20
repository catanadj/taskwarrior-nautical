from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(slots=True)
class ExpirationServices:
    core: Any
    reconcile: Any
    safe_parse_datetime: Any
    compute_anchor_child_due: Any
    compute_cp_child_due: Any
    build_child_from_parent: Any
    spawn_child_atomic: Any
    panel: Any
    short: Any
    diag: Any


def has_expiration_evidence(task: dict, *, safe_parse_datetime) -> bool:
    try:
        until_dt, until_err = safe_parse_datetime(task.get("until"))
        end_dt, end_err = safe_parse_datetime(task.get("end"))
        return bool(
            not until_err
            and not end_err
            and until_dt is not None
            and end_dt is not None
            and until_dt <= end_dt
        )
    except Exception:
        return False


def render_recovery_warning(task: dict, reason: str, *, services: ExpirationServices) -> None:
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
    task: dict,
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
        rows.append(("Next expires", services.core.fmt_dt_local(child_until_dt)))
    rows.append(("Link", f"#{plan.next_link}"))
    if child_short:
        rows.append(("Child", child_short))
    if plan.action == "legitimate_final":
        rows.append(("Boundary", plan.reason))
    panel_kind = "summary" if plan.action == "legitimate_final" else "note"
    services.panel("⌛ Nautical occurrence expired", rows, kind=panel_kind)


def handle_expired_deleted_modify(task: dict, *, services: ExpirationServices) -> bool:
    reconcile = services.reconcile
    if not reconcile.is_orphan_expiration_candidate(
        task,
        safe_parse_datetime=services.safe_parse_datetime,
    ):
        return False

    plan_hook = SimpleNamespace(
        core=services.core,
        _safe_parse_datetime=services.safe_parse_datetime,
        _compute_anchor_child_due=services.compute_anchor_child_due,
        _compute_cp_child_due=services.compute_cp_child_due,
        _build_child_from_parent=services.build_child_from_parent,
    )
    plan = reconcile.build_reconcile_plan(task, existing_children=[], hook=plan_hook)

    if plan.action == "legitimate_final":
        task["chain"] = "off"
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
    if verified:
        task["nextLink"] = child_short
    if verified or deferred:
        _render_recovery_panel(
            task,
            plan,
            services=services,
            result="[cyan]Next occurrence queued[/]" if deferred else "[green]Next occurrence created[/]",
            child_short=child_short,
        )
    else:
        render_recovery_warning(
            task,
            reason or "The next occurrence could not be queued.",
            services=services,
        )
    return True


__all__ = (
    "ExpirationServices",
    "handle_expired_deleted_modify",
    "has_expiration_evidence",
    "render_recovery_warning",
)
