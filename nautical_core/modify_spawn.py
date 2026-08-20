"""Deferred child-spawn orchestration for the completion lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(slots=True)
class SpawnServices:
    prepare_spawn_child_payload: Callable[..., tuple[dict, str, str]]
    child_uuid_for_spawn: Callable[..., str]
    fmt_isoz: Callable[[Any], str]
    now_utc: Callable[[], Any]
    strip_none_and_cast: Callable[[dict], dict]
    normalise_datetime_fields: Callable[[dict], None]
    lifecycle_models: Any
    lifecycle_spawn_identity: Callable[[dict, dict], Any]
    enqueue_spawn_intent: Callable[[Any], tuple[bool, str]]
    parse_datetime: Callable[[Any], Any]
    diag_count: Callable[[str], None]


def spawn_child_atomic(
    child_task: dict,
    parent_task_with_nextlink: dict,
    *,
    services: SpawnServices,
) -> tuple[str, set[str], bool, bool, str | None, str | None]:
    """Queue a child spawn intent for on-exit processing.

    The parent update is still applied by Taskwarrior through on-modify's JSON
    response; this function only stages the immutable child plan.
    """
    env = os.environ.copy()
    child_obj, _child_uuid, child_short = services.prepare_spawn_child_payload(
        child_task,
        parent_task_with_nextlink,
        env,
        child_uuid_for_spawn=services.child_uuid_for_spawn,
        fmt_isoz=services.fmt_isoz,
        now_utc=services.now_utc,
        strip_none_and_cast=services.strip_none_and_cast,
        normalise_datetime_fields=services.normalise_datetime_fields,
    )

    lifecycle_models = services.lifecycle_models
    lifecycle_identity = services.lifecycle_spawn_identity(parent_task_with_nextlink, child_obj)
    spawn_intent_id = lifecycle_identity.idempotency_key
    recurrence_guard = lifecycle_models.recurrence_fingerprint(
        parent_task_with_nextlink,
        parse_datetime=services.parse_datetime,
    )
    parent_guard = {
        "status": parent_task_with_nextlink.get("status") or "",
        "chain": parent_task_with_nextlink.get("chain") or "",
        "chainID": parent_task_with_nextlink.get("chainID") or "",
        "link": parent_task_with_nextlink.get("link") or "",
        "modified": parent_task_with_nextlink.get("modified") or "",
        "recurrence_fingerprint": recurrence_guard,
    }
    lifecycle_plan = lifecycle_models.LifecyclePlan.from_mappings(
        identity=lifecycle_identity,
        action=lifecycle_models.LifecycleAction.SPAWN_CHILD,
        parent_guard=lifecycle_models.ParentGuard.from_mapping(parent_guard),
        child_payload=child_obj,
        parent_patch={"nextLink": child_short},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )
    queued, queue_reason = services.enqueue_spawn_intent(lifecycle_plan)
    if not queued:
        return (
            child_short,
            set(),
            False,
            False,
            f"Spawn intent queue failed: {queue_reason}",
            spawn_intent_id,
        )
    services.diag_count("spawn_deferred")
    return (
        child_short,
        set(),
        False,
        True,
        "Spawn intent queued for on-exit processing",
        spawn_intent_id,
    )


__all__ = ("SpawnServices", "spawn_child_atomic")
