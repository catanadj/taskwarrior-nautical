from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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


def is_orphan_completion_candidate(task: dict[str, Any]) -> bool:
    if str(task.get("status") or "").strip() != "completed":
        return False
    if str(task.get("chain") or "").strip().lower() != "on":
        return False
    if not str(task.get("chainID") or "").strip():
        return False
    if str(task.get("nextLink") or "").strip():
        return False
    if not is_nautical_recurrence(task):
        return False
    return True


def existing_child_short(parent: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    chain_id = str(parent.get("chainID") or "").strip()
    next_link = int_or_default(parent.get("link"), 1) + 1
    for row in rows:
        if str(row.get("chainID") or "").strip() != chain_id:
            continue
        if int_or_default(row.get("link"), -1) != next_link:
            continue
        if str(row.get("status") or "").strip() == "deleted":
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
    evidence: dict[str, Any] = {
        "parent": short_uuid(parent.get("uuid")),
        "chainID": str(parent.get("chainID") or ""),
        "parent_link": int_or_default(parent.get("link"), 0),
        "next_link": plan.next_link,
        "kind": recurrence_kind(parent),
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


def build_reconcile_plan(
    parent: dict[str, Any],
    *,
    existing_children: list[dict[str, Any]],
    hook: Any,
) -> ReconcilePlan:
    link = int_or_default(parent.get("link"), 1)
    next_link = link + 1

    child_short = existing_child_short(parent, existing_children)
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
        if kind in {"anchor", "anchor_file"}:
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
        return ReconcilePlan("error", parent, next_link, f"failed to build child: {exc}", child_due=child_due)
    return ReconcilePlan("spawn", parent, next_link, "missing next link", child=child, child_due=child_due)
