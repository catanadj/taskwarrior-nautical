from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .integration_context import IntegrationContext
from .hook_workflow_context import BusinessCalendar, SnapshotLease, WorkflowInvocationContext
from .taskwarrior_uow import TaskwarriorUnitOfWork
from .task_models import TaskObservation, TaskPayload
from .task_changes import TaskTransition


@dataclass(slots=True)
class HookRuntimeContext:
    hook_name: str
    integration: IntegrationContext
    uow: TaskwarriorUnitOfWork
    hook_dir: str
    profile_level: int = 0
    import_ms: float | None = None
    lifecycle_result: Any | None = None
    workflow: WorkflowInvocationContext | None = None

    def close(self) -> None:
        if self.workflow is not None:
            self.workflow.close()


@dataclass(slots=True)
class OnAddRequest:
    runtime: HookRuntimeContext
    task: TaskPayload
    observation: TaskObservation | None = None
    prof: Any | None = None
    workflow_plan: Any | None = None


@dataclass(slots=True)
class OnModifyRequest:
    runtime: HookRuntimeContext
    old: TaskPayload
    new: TaskPayload
    old_observation: TaskObservation | None = None
    new_observation: TaskObservation | None = None
    transition: TaskTransition | None = None


@dataclass(slots=True)
class OnExitRequest:
    runtime: HookRuntimeContext


@dataclass(slots=True)
class OnAddContext:
    task: TaskPayload
    observation: TaskObservation | None
    now_utc: datetime
    now_local: datetime
    cp_str: str
    anchor_str: str
    anchor_file_str: str
    kind: str | None
    chain_state: str
    until_dt: datetime | None
    user_provided_due: bool
    recurrence_field: str
    due_dt: datetime
    past_due_warning: str | None
    due_day: Any
    due_hhmm: tuple[int, int]


def build_hook_runtime_context(
    *,
    hook_name: str,
    integration: IntegrationContext,
    uow: TaskwarriorUnitOfWork,
    hook_dir: str,
    profile_level: int = 0,
    import_ms: float | None = None,
    workflow: WorkflowInvocationContext | None = None,
    business_calendar: BusinessCalendar | None = None,
) -> HookRuntimeContext:
    if workflow is None:
        workflow = WorkflowInvocationContext.capture(
            integration,
            configuration_lease=SnapshotLease(integration.configuration.fingerprint),
            task_lease=SnapshotLease(f"{integration.taskdata_source}:{integration.taskdata}"),
            business_calendar=business_calendar,
        )
    return HookRuntimeContext(
        hook_name=hook_name,
        integration=integration,
        uow=uow,
        hook_dir=hook_dir,
        profile_level=int(profile_level or 0),
        import_ms=float(import_ms) if import_ms is not None else None,
        workflow=workflow,
    )


def build_on_add_context(
    task: TaskPayload,
    now_utc: datetime,
    now_local: datetime,
    *,
    validate_kind_not_conflicting: Callable[[str, str, str], tuple[bool, str]],
    kind_and_defaults_on_add: Callable[[TaskPayload, str, str, str], tuple[str | None, str]],
    validate_chain_limits_on_add: Callable[[TaskPayload, datetime], datetime | None],
    due_context_on_add: Callable[
        [TaskPayload, datetime],
        tuple[bool, str, datetime, str | None, Any, tuple[int, int]],
    ],
    observation: TaskObservation | None = None,
) -> OnAddContext:
    cp_str = (task.get('cp') or '').strip()
    anchor_str = (task.get('anchor') or '').strip()
    anchor_file_str = (task.get('anchor_file') or '').strip()
    is_valid, err = validate_kind_not_conflicting(cp_str, anchor_str, anchor_file_str)
    if not is_valid:
        raise ValueError(err)

    kind, chain_state = kind_and_defaults_on_add(task, cp_str, anchor_str, anchor_file_str)
    if not kind:
        until_dt = None
        user_provided_due = bool(task.get('due'))
        recurrence_field = "due" if task.get("due") else "scheduled" if task.get("scheduled") else "due"
        due_dt = now_utc
        past_due_warning = None
        due_local = now_local if isinstance(now_local, datetime) else now_utc
        due_day = due_local.date()
        due_hhmm = (due_local.hour, due_local.minute)
    else:
        until_dt = validate_chain_limits_on_add(task, now_utc)
        (
            user_provided_due,
            recurrence_field,
            due_dt,
            past_due_warning,
            due_day,
            due_hhmm,
        ) = due_context_on_add(task, now_utc)
    return OnAddContext(
        task=task,
        observation=observation,
        now_utc=now_utc,
        now_local=now_local,
        cp_str=cp_str,
        anchor_str=anchor_str,
        anchor_file_str=anchor_file_str,
        kind=kind,
        chain_state=chain_state,
        until_dt=until_dt,
        user_provided_due=user_provided_due,
        recurrence_field=recurrence_field,
        due_dt=due_dt,
        past_due_warning=past_due_warning,
        due_day=due_day,
        due_hhmm=due_hhmm,
    )


def build_on_add_request(
    *, runtime: HookRuntimeContext, task: TaskPayload, observation: TaskObservation | None = None, prof=None,
) -> OnAddRequest:
    return OnAddRequest(runtime=runtime, task=task, observation=observation, prof=prof)


def build_on_modify_request(
    *,
    runtime: HookRuntimeContext,
    old: TaskPayload,
    new: TaskPayload,
    old_observation: TaskObservation | None = None,
    new_observation: TaskObservation | None = None,
) -> OnModifyRequest:
    transition = (
        TaskTransition.from_observations(old_observation, new_observation)
        if old_observation is not None and new_observation is not None
        else None
    )
    return OnModifyRequest(
        runtime=runtime,
        old=old,
        new=new,
        old_observation=old_observation,
        new_observation=new_observation,
        transition=transition,
    )


def build_on_exit_request(*, runtime: HookRuntimeContext) -> OnExitRequest:
    return OnExitRequest(runtime=runtime)
