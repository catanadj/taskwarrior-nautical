from __future__ import annotations

from datetime import datetime
import importlib
from typing import Any, NoReturn, Protocol

from .task_models import TaskPayload


class OnAddServices(Protocol):
    """Typed services owned by the on-add implementation."""

    def result(self, task: TaskPayload, *, sanitize: bool, prof: Any) -> object: ...
    def has_nautical_fields(self, task: TaskPayload) -> bool: ...
    def load_core(self) -> None: ...
    def core(self) -> Any: ...
    def diag(self, message: str) -> None: ...
    def fail_and_exit(self, title: str, message: str) -> NoReturn: ...
    def build_context(self, task: TaskPayload, now_utc: datetime, now_local: datetime, *, observation: Any, prof: Any) -> Any: ...
    def stamp_chain_id(self, task: TaskPayload) -> None: ...
    def render_anchor_preview(self, context: Any, *, prof: Any) -> None: ...
    def render_cp_preview(self, context: Any, *, prof: Any) -> None: ...


class OnModifyServices(Protocol):
    """Typed services owned by the on-modify implementation."""

    def result(self, task: TaskPayload, *, sanitize: bool) -> object: ...
    def has_nautical_fields(self, task: TaskPayload) -> bool: ...
    def load_core(self) -> None: ...
    def diag(self, message: str) -> None: ...
    def fail_and_exit(self, title: str, message: str) -> NoReturn: ...
    def handle_non_completion(self, old: TaskPayload, new: TaskPayload, unit_of_work: Any) -> None: ...
    def handle_completion(self, old: TaskPayload, new: TaskPayload, unit_of_work: Any) -> Any: ...
    def handle_deleted(self, old: TaskPayload, new: TaskPayload, unit_of_work: Any) -> None: ...


class OnExitServices(Protocol):
    """Typed services owned by the on-exit implementation."""

    def redirect_stdout(self) -> None: ...
    def drain_outbox(self, unit_of_work: Any) -> dict[str, Any]: ...
    def strict_feedback(self, stats: dict[str, Any]) -> str | None: ...
    def result(self, *, exit_code: int, feedback_message: str | None, stats: dict[str, Any]) -> object: ...


def handle_on_add(
    request,
    services: OnAddServices,
) -> None:
    task = request.task
    prof = request.prof
    runtime = request.runtime
    if not services.has_nautical_fields(task):
        return services.result(task, sanitize=False, prof=prof)

    try:
        services.load_core()
    except Exception as exc:
        services.diag(f'core load failed: {exc}')
        services.fail_and_exit('Hook misconfigured', 'Failed to initialize nautical core')
    try:
        if getattr(prof, 'enabled', False) and runtime.import_ms is not None:
            prof.import_ms = runtime.import_ms
    except Exception:
        pass

    with prof.section('clock:now'):
        core = services.core()
        workflow = getattr(runtime, "workflow", None)
        if workflow is not None:
            now_utc = workflow.now_utc
            now_local = workflow.now_local
        else:
            now_utc = core.now_utc()
            now_local = core.to_local(now_utc)

    observation = getattr(request, "observation", None)
    plan_add = getattr(services, "plan_add", None)
    apply_add_plan = getattr(services, "apply_add_plan", None)
    planned_add = False
    workflow_plan = None
    if observation is not None and callable(plan_add) and callable(apply_add_plan):
        workflow_plan = plan_add(observation)
        apply_add_plan(task, workflow_plan)
        planned_add = True

    ctx = services.build_context(
        task,
        now_utc,
        now_local,
        observation=getattr(request, "observation", None),
        prof=prof,
    )
    if not ctx.kind:
        return services.result(task, sanitize=True, prof=prof)

    if not planned_add:
        services.stamp_chain_id(task)
    if ctx.kind in {'anchor', 'anchor_file'}:
        services.render_anchor_preview(ctx, prof=prof)
    else:
        services.render_cp_preview(ctx, prof=prof)
    record_schedule = getattr(services, "record_schedule", None)
    if workflow_plan is not None and callable(record_schedule):
        workflow_plan = record_schedule(workflow_plan, task, ctx.recurrence_field)
    record_limits = getattr(services, "record_limits", None)
    if workflow_plan is not None and callable(record_limits):
        workflow_plan = record_limits(workflow_plan, task, ctx)
    record_preview = getattr(services, "record_preview", None)
    if workflow_plan is not None and callable(record_preview):
        workflow_plan = record_preview(workflow_plan)
    if workflow_plan is not None:
        request.workflow_plan = workflow_plan
    return None



def handle_on_exit(
    request,
    services: OnExitServices,
):
    _ = request.runtime
    services.redirect_stdout()
    stats = services.drain_outbox(request.runtime.uow)
    strict_msg = services.strict_feedback(stats)
    if strict_msg:
        return services.result(exit_code=1, feedback_message=strict_msg, stats=stats)
    return services.result(exit_code=0, feedback_message=None, stats=stats)



def handle_on_modify(
    request,
    services: OnModifyServices,
):
    old, new = request.old, request.new
    transition = getattr(request, "transition", None)
    if transition is None:
        from .task_changes import TaskTransition
        from .task_models import TaskObservation
        transition = TaskTransition.from_observations(
            TaskObservation.from_mapping(old, source_query="modify workflow"),
            TaskObservation.from_mapping(new, source_query="modify workflow"),
        )
    workflow = importlib.import_module("nautical_core.modify_workflow")
    typed_route = workflow.classify_modify_transition(transition)
    terminal_decision = workflow.terminal_decision_for_route(typed_route)
    if terminal_decision is not None:
        request.terminal_decision = terminal_decision
    typed_handlers = bool(getattr(services, "typed_transition_handlers", False))

    def invoke_typed(handler_name):
        handler = getattr(services, handler_name)
        if typed_handlers:
            return handler(old, new, request.runtime.uow, transition)
        return handler(old, new, request.runtime.uow)

    if typed_route.kind is workflow.ModifyRouteKind.INVALID_IDENTITY_EDIT:
        services.fail_and_exit(
            "Invalid Nautical edit",
            "chain identity fields are immutable; restore chainID/link/prevLink/nextLink",
        )
    if typed_route.kind is workflow.ModifyRouteKind.DELETION:
        if typed_route.has_nautical_fields:
            services.load_core()
            invoke_typed("handle_deleted")
        return services.result(task=new, sanitize=False)
    if not typed_route.has_nautical_fields or typed_route.kind is workflow.ModifyRouteKind.ORDINARY:
        return services.result(task=new, sanitize=False)
    if typed_route.kind is workflow.ModifyRouteKind.IDEMPOTENT_COMPLETION:
        return services.result(task=new, sanitize=False)
    if typed_route.kind is workflow.ModifyRouteKind.COMPLETION:
        services.load_core()
        lifecycle_result = invoke_typed("handle_completion")
        request.runtime.lifecycle_result = lifecycle_result
        return None
    services.load_core()
    invoke_typed("handle_non_completion")
    return None
