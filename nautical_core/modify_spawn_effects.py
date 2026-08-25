"""Lifecycle staging and atomic child-spawn effect assembly for on-modify."""

from __future__ import annotations

from typing import Any


def enqueue_spawn_intent(host: Any, plan) -> tuple[bool, str]:
    """Stage one immutable lifecycle plan without re-entering Taskwarrior."""
    context = getattr(host, "_INTEGRATION_CONTEXT", None)
    if context is None:
        return False, "validated integration context is unavailable"
    models = host._module("lifecycle_models")
    if not isinstance(plan, models.LifecyclePlan):
        return False, "invalid lifecycle plan"
    outbox = host._module("lifecycle_outbox").LifecycleOutboxRepository(host.TW_DATA_DIR)
    service = host._module("lifecycle_application").LifecycleApplicationService(
        outbox=outbox, owner="on-modify"
    )
    result = service.stage(
        plan,
        configuration_fingerprint=context.configuration.fingerprint,
        schedule_fingerprint=context.configuration.scheduler_fingerprint,
    )
    if result.ok:
        return True, ""
    return False, result.reason or "lifecycle outbox staging failed"


def lifecycle_spawn_identity(host: Any, parent: dict, child: dict):
    models = host._module("lifecycle_models")
    chain_id = str(parent.get("chainID") or "").strip()
    parent_uuid = str(parent.get("uuid") or "").strip()
    try:
        source_link = int(str(parent.get("link")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("lifecycle transition requires a numeric parent link") from exc
    try:
        target_link = int(str(child.get("link") or (source_link + 1)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("lifecycle transition requires a numeric child link") from exc
    event = (
        models.LifecycleEvent.EXPIRE
        if str(parent.get("status") or "").strip().lower() == "deleted"
        else models.LifecycleEvent.COMPLETE
    )
    return models.LifecycleIdentity(
        chain_id=chain_id,
        parent_uuid=parent_uuid,
        source_link=source_link,
        target_link=target_link,
        event=event,
    )


def spawn_child_atomic(host: Any, child_task, parent_task_with_nextlink: dict, *, lifecycle_plan=None):
    spawn = host._module("modify_spawn")
    if hasattr(child_task, "to_mapping"):
        child_task = child_task.to_mapping()
    return spawn.spawn_child_atomic(
        child_task,
        parent_task_with_nextlink,
        lifecycle_plan=lifecycle_plan,
        services=spawn.SpawnServices(
            prepare_spawn_child_payload=host._module("modify_spawn_prep").prepare_spawn_child_payload,
            child_uuid_for_spawn=host._child_uuid_for_spawn,
            fmt_isoz=host.core.fmt_isoz,
            now_utc=host.core.now_utc,
            lifecycle_models=host._module("lifecycle_models"),
            lifecycle_spawn_identity=lambda parent, child: lifecycle_spawn_identity(host, parent, child),
            enqueue_spawn_intent=lambda plan: enqueue_spawn_intent(host, plan),
            parse_datetime=getattr(host.core, "parse_dt_any", None),
            diag_count=host._diag_count,
        ),
    )


__all__ = ("enqueue_spawn_intent", "lifecycle_spawn_identity", "spawn_child_atomic")
