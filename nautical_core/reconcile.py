from __future__ import annotations

from datetime import timedelta
from dataclasses import dataclass
from typing import Any

from nautical_core import native_until


RECURRENCE_FIELDS = ("anchor", "anchor_file", "cp")


@dataclass(frozen=True)
class ReconcilePlan:
    action: str
    parent: dict[str, Any]
    next_link: int
    reason: str
    child: dict[str, Any] | None = None
    child_short: str = ""
    child_due: Any = None


def short_uuid(value: object) -> str:
    raw = str(value or "").strip()
    return raw[:8] if raw else ""


def int_or_default(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if not isinstance(value, (str, bytes, int, float)):
        return default
    try:
        return int(value)
    except Exception:
        return default


def is_nautical_recurrence(task: dict[str, Any]) -> bool:
    return any(str(task.get(field) or "").strip() for field in RECURRENCE_FIELDS)


def _is_unlinked_active_chain(task: dict[str, Any]) -> bool:
    if str(task.get("chain") or "").strip().lower() != "on":
        return False
    if not str(task.get("chainID") or "").strip():
        return False
    if str(task.get("nextLink") or "").strip():
        return False
    if not is_nautical_recurrence(task):
        return False
    return True


def is_orphan_completion_candidate(task: dict[str, Any]) -> bool:
    return str(task.get("status") or "").strip() == "completed" and _is_unlinked_active_chain(task)


def is_orphan_deleted_chain_candidate(task: dict[str, Any]) -> bool:
    return str(task.get("status") or "").strip() == "deleted" and _is_unlinked_active_chain(task)


def deleted_chain_disposition(
    task: dict[str, Any],
    *,
    safe_parse_datetime: Any,
) -> tuple[str, str]:
    """Classify an unlinked deleted chain as expiration, manual stop, or ambiguous."""
    if not is_orphan_deleted_chain_candidate(task):
        return "", ""
    if not str(task.get("until") or "").strip():
        return "manual", "deleted without native until"
    try:
        until_dt, until_err = safe_parse_datetime(task.get("until"))
        end_dt, end_err = safe_parse_datetime(task.get("end"))
    except Exception:
        return "ambiguous", "deleted task has no reliable native-until expiration evidence"
    if until_err or end_err or until_dt is None or end_dt is None:
        return "ambiguous", "deleted task has no reliable native-until expiration evidence"
    try:
        if until_dt <= end_dt:
            return "expiration", "native until elapsed"
        return "manual", "deleted before native until"
    except Exception:
        return "ambiguous", "deleted task has no reliable native-until expiration evidence"


def is_orphan_expiration_candidate(task: dict[str, Any], *, safe_parse_datetime: Any) -> bool:
    """Return whether a deleted link has strong evidence of native until expiration."""
    disposition, _reason = deleted_chain_disposition(
        task,
        safe_parse_datetime=safe_parse_datetime,
    )
    if not disposition:
        return False
    return disposition == "expiration"


def expiration_recurrence_parent(parent: dict[str, Any]) -> dict[str, Any]:
    """Return a computation-only parent that advances from due/scheduled, not deletion time."""
    target = parent.get("due") or parent.get("scheduled")
    if not str(target or "").strip():
        raise ValueError("expired recurrence has no due or scheduled timestamp")
    calculation_parent = dict(parent)
    calculation_parent["end"] = target
    return calculation_parent


def compute_expiration_child_due(parent: dict[str, Any], *, hook: Any) -> tuple[Any, dict[str, Any]]:
    """Compute the next recurrence target after an expired link without mutating it."""
    calculation_parent = expiration_recurrence_parent(parent)
    kind = recurrence_kind(parent)
    if kind in {"anchor", "anchor_file"}:
        child_due, meta, _dnf = hook._compute_anchor_child_due(calculation_parent)
    else:
        child_due, meta = hook._compute_cp_child_due(calculation_parent)
    target_field = "scheduled" if not parent.get("due") and parent.get("scheduled") else "due"
    result_meta = dict(meta or {})
    result_meta["basis"] = f"{target_field} recurrence target (expired)"
    result_meta["target_field"] = target_field
    return child_due, result_meta


def existing_child_short(
    parent: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    include_deleted: bool = False,
) -> str:
    chain_id = str(parent.get("chainID") or "").strip()
    next_link = int_or_default(parent.get("link"), 1) + 1
    for row in rows:
        if str(row.get("chainID") or "").strip() != chain_id:
            continue
        if int_or_default(row.get("link"), -1) != next_link:
            continue
        if not include_deleted and str(row.get("status") or "").strip() == "deleted":
            continue
        return short_uuid(row.get("uuid"))
    return ""


def recurrence_kind(task: dict[str, Any]) -> str:
    if str(task.get("anchor") or "").strip():
        return "anchor"
    if str(task.get("anchor_file") or "").strip():
        return "anchor_file"
    return "cp"


def describe_plan(plan: ReconcilePlan, *, fmt_dt_local: Any = None) -> dict[str, Any]:
    parent = plan.parent
    if plan.action == "manual_stop":
        trigger = "manual_deletion"
    elif str(parent.get("status") or "").strip() == "deleted":
        trigger = "expiration"
    else:
        trigger = "completion"
    evidence: dict[str, Any] = {
        "parent": short_uuid(parent.get("uuid")),
        "chainID": str(parent.get("chainID") or ""),
        "parent_link": int_or_default(parent.get("link"), 0),
        "next_link": plan.next_link,
        "kind": recurrence_kind(parent),
        "trigger": trigger,
        "reason": plan.reason,
    }
    if plan.child_due is not None:
        evidence["child_due"] = str(plan.child_due)
        if callable(fmt_dt_local):
            try:
                evidence["child_local"] = str(fmt_dt_local(plan.child_due))
            except Exception:
                pass
    if plan.child_short:
        evidence["existing_child"] = plan.child_short
    if plan.child:
        field = "due" if "due" in plan.child else "scheduled" if "scheduled" in plan.child else "due"
        evidence["child_field"] = field
        if plan.child.get(field) is not None:
            evidence["child_target"] = str(plan.child.get(field))
    return evidence


def _build_expiration_child_with_day_end(
    parent: dict[str, Any],
    *,
    child_due: Any,
    child_field: str,
    next_link: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt: Any,
    hook: Any,
) -> dict[str, Any]:
    target_raw = parent.get("due") or parent.get("scheduled")
    target_dt, target_err = hook._safe_parse_datetime(target_raw)
    if target_err or target_dt is None:
        raise ValueError(target_err or "expired recurrence has no due or scheduled timestamp")
    target_local = hook.core.to_local(target_dt)
    fallback_until = hook.core.build_local_datetime(target_local.date(), (23, 59)) + timedelta(seconds=59)
    fallback_parent = dict(parent)
    fallback_parent["until"] = hook.core.fmt_isoz(fallback_until)
    return hook._build_child_from_parent(
        fallback_parent,
        child_due,
        child_field,
        next_link,
        parent_short,
        kind,
        cpmax,
        until_dt,
    )


def build_reconcile_plan(
    parent: dict[str, Any],
    *,
    existing_children: list[dict[str, Any]],
    hook: Any,
) -> ReconcilePlan:
    link = int_or_default(parent.get("link"), 1)
    next_link = link + 1
    is_expiration = str(parent.get("status") or "").strip() == "deleted"
    if is_expiration:
        disposition, reason = deleted_chain_disposition(
            parent,
            safe_parse_datetime=hook._safe_parse_datetime,
        )
        if disposition == "manual":
            return ReconcilePlan("manual_stop", parent, next_link, reason)
        if disposition != "expiration":
            return ReconcilePlan(
                "error",
                parent,
                next_link,
                reason or "deleted task has no reliable native-until expiration evidence",
            )

    child_short = existing_child_short(parent, existing_children, include_deleted=is_expiration)
    if child_short:
        return ReconcilePlan("backfill_nextlink", parent, next_link, "next link already exists", child_short=child_short)

    kind = recurrence_kind(parent)
    until_dt, until_err = hook._safe_parse_datetime(parent.get("chainUntil"))
    if until_err:
        return ReconcilePlan("error", parent, next_link, f"invalid chainUntil: {until_err}")

    cpmax = hook.core.coerce_int(parent.get("chainMax"), 0)
    if cpmax and next_link > cpmax:
        return ReconcilePlan("legitimate_final", parent, next_link, "reached chainMax")

    try:
        if is_expiration:
            child_due, meta = compute_expiration_child_due(parent, hook=hook)
        elif kind in {"anchor", "anchor_file"}:
            child_due, meta, _dnf = hook._compute_anchor_child_due(parent)
        else:
            child_due, meta = hook._compute_cp_child_due(parent)
    except Exception as exc:
        return ReconcilePlan("error", parent, next_link, str(exc))

    if not child_due:
        return ReconcilePlan("error", parent, next_link, "could not compute next recurrence timestamp")
    if until_dt and child_due > until_dt:
        return ReconcilePlan("legitimate_final", parent, next_link, "reached chainUntil", child_due=child_due)

    child_field = "scheduled" if isinstance(meta, dict) and meta.get("target_field") == "scheduled" else "due"
    parent_short = short_uuid(parent.get("uuid"))
    try:
        child = hook._build_child_from_parent(parent, child_due, child_field, next_link, parent_short, kind, cpmax, until_dt)
    except Exception as exc:
        carry_conflict = (
            isinstance(exc, native_until.NativeUntilCarryError)
            and exc.code == native_until.CARRY_CONFLICT
        )
        if is_expiration and kind in {"anchor", "anchor_file"} and carry_conflict:
            try:
                child = _build_expiration_child_with_day_end(
                    parent,
                    child_due=child_due,
                    child_field=child_field,
                    next_link=next_link,
                    parent_short=parent_short,
                    kind=kind,
                    cpmax=cpmax,
                    until_dt=until_dt,
                    hook=hook,
                )
            except Exception as fallback_exc:
                return ReconcilePlan("error", parent, next_link, f"failed to build child: {fallback_exc}", child_due=child_due)
        else:
            return ReconcilePlan("error", parent, next_link, f"failed to build child: {exc}", child_due=child_due)
    reason = "expired link missing next link" if is_expiration else "missing next link"
    return ReconcilePlan("spawn", parent, next_link, reason, child=child, child_due=child_due)
