from __future__ import annotations

import json
import uuid


def stable_child_uuid(
    parent_task: dict | None,
    child_task: dict | None,
    *,
    task_uuid_or_empty,
    coerce_int,
    stable_child_uuid_namespace,
) -> str:
    """Return a cross-device-stable UUID for a child slot when possible."""
    if not isinstance(parent_task, dict) or not isinstance(child_task, dict):
        return ""
    parent_uuid = task_uuid_or_empty(parent_task)
    if not parent_uuid:
        return ""
    link_no = coerce_int(child_task.get("link"), None)
    if link_no is None:
        return ""
    chain_id = (
        child_task.get("chainID")
        or child_task.get("chainid")
        or parent_task.get("chainID")
        or parent_task.get("chainid")
        or ""
    )
    kind = "anchor" if (parent_task.get("anchor") or "").strip() else "cp" if (parent_task.get("cp") or "").strip() else ""
    slot_key = json.dumps(
        {
            "chain_id": str(chain_id).strip().lower(),
            "kind": kind,
            "link": int(link_no),
            "parent_uuid": parent_uuid.lower(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return str(uuid.uuid5(stable_child_uuid_namespace, slot_key))


def child_uuid_for_spawn(
    parent_task: dict | None,
    child_task: dict | None,
    env: dict,
    *,
    stable_child_uuid,
    reserve_child_uuid,
) -> str:
    stable = stable_child_uuid(parent_task, child_task)
    if stable:
        return stable
    return reserve_child_uuid(env)


def prepare_spawn_child_payload(
    child_task: dict,
    parent_task: dict | None,
    env: dict,
    *,
    child_uuid_for_spawn,
    fmt_isoz,
    now_utc,
    strip_none_and_cast,
    normalise_datetime_fields,
) -> tuple[dict, str, str]:
    child_uuid = child_uuid_for_spawn(parent_task, child_task, env)
    child_obj = dict(child_task)
    child_obj["uuid"] = child_uuid
    if "entry" not in child_obj:
        child_obj["entry"] = fmt_isoz(now_utc())
    if "modified" not in child_obj:
        child_obj["modified"] = child_obj["entry"]

    child_short = child_uuid[:8]
    child_obj = strip_none_and_cast(child_obj)
    normalise_datetime_fields(child_obj)
    return child_obj, child_uuid, child_short


def build_child_from_parent(
    parent: dict,
    child_due_utc,
    child_field: str,
    next_link_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt,
    *,
    reserved_drop,
    reserved_override,
    debug_wait_sched: bool,
    clear_wait_sched_debug,
    fmt_isoz,
    now_utc,
    carry_relative_datetime,
    recurrence_anchor_field,
    configured_recurrence_uda_fields,
    short_uuid,
) -> dict:
    child = {k: v for k, v in parent.items() if k not in reserved_drop}
    if debug_wait_sched:
        clear_wait_sched_debug()
    for key in reserved_override:
        child.pop(key, None)
    child.update(
        {
            "status": "pending",
            "entry": fmt_isoz(now_utc()),
            "chain": "on",
            "prevLink": parent_short,
            "link": next_link_no,
        }
    )
    parent_anchor_field = recurrence_anchor_field(parent)
    if child_field == "scheduled":
        child.pop("due", None)
        child["scheduled"] = fmt_isoz(child_due_utc)
    else:
        child["due"] = fmt_isoz(child_due_utc)
    if kind in {"anchor", "anchor_file"}:
        child["anchor"] = parent.get("anchor")
        child["anchor_file"] = parent.get("anchor_file")
        child["anchor_mode"] = parent.get("anchor_mode") or "skip"
        child.pop("cp", None)
    else:
        child["cp"] = parent.get("cp")
        child.pop("anchor", None)
        child.pop("anchor_file", None)
        child.pop("anchor_mode", None)

    carry_relative_datetime(
        parent,
        child,
        child_due_utc,
        "wait",
        parent_anchor_field=parent_anchor_field,
        child_anchor_field=child_field,
    )
    if child_field != "scheduled":
        carry_relative_datetime(
            parent,
            child,
            child_due_utc,
            "scheduled",
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_field,
        )
    for uda_field in configured_recurrence_uda_fields(parent):
        carry_relative_datetime(
            parent,
            child,
            child_due_utc,
            uda_field,
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_field,
        )

    if cpmax:
        child["chainMax"] = int(cpmax)
    if until_dt:
        child["chainUntil"] = fmt_isoz(until_dt)

    try:
        parent_chain = (parent.get("chainID") or "").strip()
        if not parent_chain:
            parent_chain = short_uuid(parent.get("uuid"))
        child["chainID"] = parent_chain
    except Exception:
        pass

    return child
