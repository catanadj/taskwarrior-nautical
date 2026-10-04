"""Lifecycle staging and atomic child-spawn effect assembly for on-modify."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID
from .task_datetime import datetime_value, parser_for_host
from dataclasses import dataclass
from .modify_models import DatetimeParserCallback
from .task_models import TaskPayload

if TYPE_CHECKING:
    from .modify_command_effects import CommandPorts
    from .modify_models import CoerceIntCallback
    from .integration_context import IntegrationContext
    from .lifecycle.application import LifecycleApplicationService
    from .lifecycle.models import LifecycleIdentity, LifecyclePlan
    from .lifecycle.outbox import LifecycleOutboxRepository


class SpawnPreparation(Protocol):
    def stable_child_uuid(
        self,
        parent_task: dict[str, Any] | None,
        child_task: dict[str, Any] | None,
        *,
        task_uuid_or_empty: Callable[[dict[str, Any] | None], str],
        coerce_int: CoerceIntCallback,
        stable_child_uuid_namespace: UUID,
    ) -> str: ...

    def child_uuid_for_spawn(
        self,
        parent_task: dict[str, Any] | None,
        child_task: dict[str, Any] | None,
        env: dict[str, Any],
        *,
        stable_child_uuid: Callable[
            [dict[str, Any] | None, dict[str, Any] | None], str
        ],
        generate_child_uuid_candidate: Callable[[Mapping[str, str]], str],
    ) -> str: ...


class SpawnCommandEffects(Protocol):
    def generate_child_uuid_candidate(
        self, ports: CommandPorts, env: Mapping[str, str]
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class SpawnIntentPorts:
    context: IntegrationContext | None
    outbox_factory: Callable[[str], LifecycleOutboxRepository]
    application_service: type[LifecycleApplicationService]
    data_dir: str


@dataclass(frozen=True, slots=True)
class ChildUuidPorts:
    prep: SpawnPreparation
    command: SpawnCommandEffects
    command_ports: CommandPorts
    task_uuid_or_empty: Callable[[dict[str, Any] | None], str]
    coerce_int: CoerceIntCallback
    namespace: UUID


@dataclass(frozen=True, slots=True)
class SpawnChildPorts:
    spawn: Any
    prepare_payload: Any
    child_uuid: Any
    format_datetime: Callable[[datetime], str]
    now_utc: Callable[[], datetime]
    spawn_identity: Callable[[TaskPayload, TaskPayload], LifecycleIdentity]
    enqueue_intent: Callable[[LifecyclePlan], tuple[bool, str]]
    parse_datetime: DatetimeParserCallback
    diag_count: Callable[[str], None]


def spawn_intent_ports_for(host: Any) -> SpawnIntentPorts:
    lifecycle_outbox = host._module("lifecycle_outbox")
    return SpawnIntentPorts(
        context=getattr(host, "_INTEGRATION_CONTEXT", None),
        outbox_factory=lifecycle_outbox.LifecycleOutboxRepository,
        application_service=host._module("lifecycle_application").LifecycleApplicationService,
        data_dir=host.TW_DATA_DIR,
    )


def child_uuid_ports_for(host: Any) -> ChildUuidPorts:
    command = host._module("modify_command_effects")
    return ChildUuidPorts(
        prep=host._module("modify_spawn_prep"),
        command=command,
        command_ports=command.command_ports_for(host),
        task_uuid_or_empty=host._module("modify_task_fields").task_uuid_or_empty,
        coerce_int=host.core.coerce_int,
        namespace=host._STABLE_CHILD_UUID_NAMESPACE,
    )


def spawn_child_ports_for(host: Any) -> SpawnChildPorts:
    intent_ports = spawn_intent_ports_for(host)
    uuid_ports = child_uuid_ports_for(host)
    return SpawnChildPorts(
        spawn=host._module("modify_spawn"),
        prepare_payload=host._module("modify_spawn_prep").prepare_spawn_child_payload,
        child_uuid=lambda parent, child, env: child_uuid_for_spawn(uuid_ports, parent, child, env),
        format_datetime=host.core.fmt_isoz,
        now_utc=host.core.now_utc,
        spawn_identity=lifecycle_spawn_identity,
        enqueue_intent=lambda plan: enqueue_spawn_intent(intent_ports, plan),
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        diag_count=host._diag_count,
    )


def enqueue_spawn_intent(ports: SpawnIntentPorts, plan: object) -> tuple[bool, str]:
    """Stage one immutable lifecycle plan without re-entering Taskwarrior."""
    context = ports.context
    if context is None:
        return False, "validated integration context is unavailable"
    from .lifecycle.models import LifecyclePlan

    if not isinstance(plan, LifecyclePlan):
        return False, "invalid lifecycle plan"
    outbox = ports.outbox_factory(ports.data_dir)
    # This hook runs while Taskwarrior holds its datastore lock; intentionally
    # compose a stage-only service with no command-capable execution port.
    service = ports.application_service(outbox=outbox, owner="on-modify")
    result = service.stage(
        plan,
        configuration_fingerprint=context.configuration.fingerprint,
        schedule_fingerprint=context.configuration.scheduler_fingerprint,
    )
    if result.ok:
        return True, ""
    reason = str(result.reason or "").strip()
    if reason:
        return False, reason
    kind = getattr(result.kind, "value", str(result.kind))
    return False, f"lifecycle outbox staging returned {kind}"


def lifecycle_spawn_identity(
    parent: Mapping[str, Any], child: Mapping[str, Any]
) -> LifecycleIdentity:
    from .lifecycle.models import LifecycleEvent, LifecycleIdentity

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
        LifecycleEvent.EXPIRE
        if str(parent.get("status") or "").strip().lower() == "deleted"
        else LifecycleEvent.COMPLETE
    )
    return LifecycleIdentity(
        chain_id=chain_id,
        parent_uuid=parent_uuid,
        source_link=source_link,
        target_link=target_link,
        event=event,
    )


def spawn_child_atomic(
    ports: SpawnChildPorts,
    child_task: Any,
    parent_task_with_nextlink: dict[str, Any],
    *,
    lifecycle_plan: Any = None,
) -> Any:
    if hasattr(child_task, "to_mapping"):
        child_task = child_task.to_mapping()
    return ports.spawn.spawn_child_atomic(
        child_task,
        parent_task_with_nextlink,
        lifecycle_plan=lifecycle_plan,
        services=ports.spawn.SpawnServices(
            prepare_spawn_child_payload=ports.prepare_payload,
            child_uuid_for_spawn=ports.child_uuid,
            fmt_isoz=ports.format_datetime,
            now_utc=ports.now_utc,
            lifecycle_spawn_identity=ports.spawn_identity,
            enqueue_spawn_intent=ports.enqueue_intent,
            parse_datetime=ports.parse_datetime,
            diag_count=ports.diag_count,
        ),
    )


def child_uuid_for_spawn(ports: ChildUuidPorts, parent_task: dict | None, child_task: dict | None, env: dict) -> str:
    return ports.prep.child_uuid_for_spawn(
        parent_task,
        child_task,
        env,
        stable_child_uuid=lambda parent, child: ports.prep.stable_child_uuid(
            parent,
            child,
            task_uuid_or_empty=ports.task_uuid_or_empty,
            coerce_int=ports.coerce_int,
            stable_child_uuid_namespace=ports.namespace,
        ),
        generate_child_uuid_candidate=lambda value: ports.command.generate_child_uuid_candidate(
            ports.command_ports, value
        ),
    )


__all__ = (
    "SpawnIntentPorts", "ChildUuidPorts", "SpawnChildPorts",
    "SpawnPreparation", "SpawnCommandEffects",
    "spawn_intent_ports_for", "child_uuid_ports_for", "spawn_child_ports_for",
    "enqueue_spawn_intent", "lifecycle_spawn_identity", "spawn_child_atomic",
    "child_uuid_for_spawn",
)
