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
    def is_non_completion(self, old: TaskPayload, new: TaskPayload) -> bool: ...
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
    modify_lifecycle = importlib.import_module("nautical_core.modify_lifecycle")

    route = modify_lifecycle.classify_modify_route(
        old,
        new,
        is_non_completion_modify=services.is_non_completion,
        transition=getattr(request, "transition", None),
    )
    typed_handlers = bool(getattr(services, "typed_transition_handlers", False))
    transition = getattr(request, "transition", None)

    def invoke(handler_name, *args):
        handler = getattr(services, handler_name)
        if typed_handlers and transition is not None:
            return handler(*args, transition)
        return handler(*args)

    if route.is_deleted and route.has_nautical_fields:
        try:
            services.load_core()
        except Exception as exc:
            services.diag(f'core load failed: {exc}')
            services.fail_and_exit('Hook misconfigured', 'Failed to initialize nautical core')
        invoke("handle_deleted", old, new, request.runtime.uow)
        return services.result(task=new, sanitize=False)

    if route.is_deleted:
        return services.result(task=new, sanitize=False)

    if not route.has_nautical_fields:
        return services.result(task=new, sanitize=False)

    try:
        services.load_core()
    except Exception as exc:
        services.diag(f'core load failed: {exc}')
        services.fail_and_exit('Hook misconfigured', 'Failed to initialize nautical core')

    if route.is_non_completion:
        invoke("handle_non_completion", old, new, request.runtime.uow)
        return None

    lifecycle_result = invoke("handle_completion", old, new, request.runtime.uow)
    try:
        request.runtime.lifecycle_result = lifecycle_result
    except Exception:
        # Minimal test/request adapters may not expose runtime state.
        pass
    return None
