from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class ModifyLifecycleRoute:
    is_deleted: bool
    has_nautical_fields: bool
    is_non_completion: bool


@dataclass(slots=True)
class ModifyNauticalTransition:
    state: str
    source: str = ""
    reason: str = ""


def task_has_nautical_recurrence_fields(task: dict[str, Any] | None) -> bool:
    if not isinstance(task, dict):
        return False
    keys = ("anchor", "anchor_file", "anchor_mode", "cp", "omit", "omit_file")
    for key in keys:
        val = task.get(key)
        if val is None:
            continue
        try:
            s = str(val).strip()
        except Exception:
            s = ""
        if s:
            return True
    return False


def task_has_nautical_chain_fields(task: dict[str, Any] | None) -> bool:
    if not isinstance(task, dict):
        return False
    keys = ("chainID", "nextLink", "prevLink", "link")
    for key in keys:
        val = task.get(key)
        if val is None:
            continue
        try:
            s = str(val).strip()
        except Exception:
            s = ""
        if s:
            return True
    return False


def task_has_nautical_fields(task: dict[str, Any] | None) -> bool:
    return task_has_nautical_recurrence_fields(task) or task_has_nautical_chain_fields(task)


def classify_modify_route(
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    *,
    is_non_completion_modify: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> ModifyLifecycleRoute:
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    is_deleted = (str(new.get("status") or "").lower() == "deleted")
    has_nautical_fields = task_has_nautical_fields(old) or task_has_nautical_fields(new)
    is_non_completion = bool(has_nautical_fields and not is_deleted and is_non_completion_modify(old, new))
    return ModifyLifecycleRoute(
        is_deleted=is_deleted,
        has_nautical_fields=has_nautical_fields,
        is_non_completion=is_non_completion,
    )


def promote_newly_nautical_task(
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    *,
    short_uuid: Callable[[Any], str],
) -> str | None:
    transition = apply_nautical_transition(old, new, short_uuid=short_uuid)
    return transition.source if transition.state == "enabled" else None


def apply_nautical_transition(
    old: dict[str, Any] | None,
    new: dict[str, Any] | None,
    *,
    short_uuid: Callable[[Any], str],
) -> ModifyNauticalTransition:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ModifyNauticalTransition(state="unchanged")

    old_has_recurrence = task_has_nautical_recurrence_fields(old)
    new_has_recurrence = task_has_nautical_recurrence_fields(new)
    new_chain = (new.get("chain") or "").strip().lower()

    if not old_has_recurrence and new_has_recurrence:
        if (new.get("anchor") or "").strip():
            source = "anchor"
        elif (new.get("anchor_file") or "").strip():
            source = "anchor_file"
        elif (new.get("cp") or "").strip():
            source = "cp"
        else:
            return ModifyNauticalTransition(state="unchanged")

        if new_chain != "on":
            new["chain"] = "on"

        already_chain = bool((new.get("chainID") or "").strip())
        linked_already = bool((new.get("prevLink") or new.get("nextLink") or "").strip())
        if not already_chain and not linked_already:
            new["chainID"] = short_uuid(new.get("uuid"))
        return ModifyNauticalTransition(
            state="enabled",
            source=source,
            reason="This task just gained Nautical recurrence and was promoted to chain:on.",
        )

    if old_has_recurrence and not new_has_recurrence:
        if new_chain != "off":
            new["chain"] = "off"
        return ModifyNauticalTransition(
            state="disabled",
            reason="This task no longer has Nautical recurrence fields.",
        )

    if new_has_recurrence and new_chain == "off":
        if (new.get("anchor") or "").strip():
            source = "anchor"
        elif (new.get("anchor_file") or "").strip():
            source = "anchor_file"
        elif (new.get("cp") or "").strip():
            source = "cp"
        else:
            source = ""
        return ModifyNauticalTransition(
            state="disabled",
            source=source,
            reason="This task's Nautical recurrence is disabled because chain:off is set.",
        )

    return ModifyNauticalTransition(state="unchanged")


__all__ = (
    "ModifyLifecycleRoute",
    "ModifyNauticalTransition",
    "apply_nautical_transition",
    "classify_modify_route",
    "promote_newly_nautical_task",
    "task_has_nautical_chain_fields",
    "task_has_nautical_fields",
    "task_has_nautical_recurrence_fields",
)
