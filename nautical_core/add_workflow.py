"""Pure add-workflow planning for recurrence activation."""

from __future__ import annotations

from dataclasses import dataclass

from .hook_workflow_models import AddWorkflowRequest, PatchOperation, TaskPatch, TaskPatchOperation, WorkflowRoute
from .task_models import TaskObservation, TaskTimestamp


@dataclass(frozen=True, slots=True)
class AddScheduleSelection:
    """Validated scheduler output consumed by the add planner."""

    target_field: str
    first_occurrence: TaskTimestamp | None
    status: str = "found"

    def __post_init__(self) -> None:
        field = str(self.target_field).strip()
        if field not in {"due", "scheduled"}:
            raise ValueError("add schedule target must be due or scheduled")
        status = str(self.status).strip().lower()
        if status not in {"found", "terminal", "unavailable"}:
            raise ValueError("invalid add schedule status")
        if status == "found" and self.first_occurrence is None:
            raise ValueError("found add schedule requires a first occurrence")
        if status != "found" and self.first_occurrence is not None:
            raise ValueError("non-found add schedule cannot carry an occurrence")
        object.__setattr__(self, "target_field", field)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True, slots=True)
class AddWorkflowPlan:
    """Typed add decision before patch application or presentation."""

    request: AddWorkflowRequest
    patch: TaskPatch
    recurrence_kind: str = ""
    target_field: str = "due"
    target_explicit: bool = False
    schedule: AddScheduleSelection | None = None

    @property
    def ordinary(self) -> bool:
        return self.request.route is WorkflowRoute.ORDINARY


def _text(task: TaskObservation, field: str) -> str:
    return str(task.get(field) or "").strip()


def classify_add_route(task: TaskObservation) -> WorkflowRoute:
    """Classify one typed add observation without loading scheduling modules."""
    if _text(task, "cp"):
        return WorkflowRoute.CP_ACTIVATION
    if _text(task, "anchor"):
        return WorkflowRoute.ANCHOR_ACTIVATION
    if _text(task, "anchor_file"):
        return WorkflowRoute.ANCHOR_FILE_ACTIVATION
    return WorkflowRoute.ORDINARY


def plan_add(task: TaskObservation) -> AddWorkflowPlan:
    """Return recurrence defaults and root identity as a typed patch."""
    route = classify_add_route(task)
    request = AddWorkflowRequest(task=task, route=route)
    has_due = bool(_text(task, "due"))
    has_scheduled = bool(_text(task, "scheduled"))
    target_field = "scheduled" if has_scheduled and not has_due else "due"
    if route is WorkflowRoute.ORDINARY:
        return AddWorkflowPlan(
            request=request,
            patch=TaskPatch(()),
            target_field=target_field,
            target_explicit=has_due or has_scheduled,
        )

    operations: list[TaskPatchOperation] = []
    if _text(task, "chain").lower() not in {"on", "true", "1"}:
        operations.append(TaskPatchOperation("chain", PatchOperation.SET, "on"))
    if not _text(task, "prevLink") and not _text(task, "nextLink") and not _text(task, "link"):
        operations.append(TaskPatchOperation("link", PatchOperation.SET, 1))
    if not _text(task, "chainID"):
        uuid = _text(task, "uuid")
        if uuid:
            operations.append(TaskPatchOperation("chainID", PatchOperation.SET, uuid.replace("-", "")[:8]))
    if route in {WorkflowRoute.ANCHOR_ACTIVATION, WorkflowRoute.ANCHOR_FILE_ACTIVATION} and not _text(task, "anchor_mode"):
        operations.append(TaskPatchOperation("anchor_mode", PatchOperation.SET, "skip"))
    return AddWorkflowPlan(
        request=request,
        patch=TaskPatch(tuple(operations)),
        recurrence_kind="cp" if route is WorkflowRoute.CP_ACTIVATION else "anchor",
        target_field=target_field,
        target_explicit=has_due or has_scheduled,
    )


def record_schedule(
    plan: AddWorkflowPlan,
    *,
    first_occurrence: TaskTimestamp | None,
    status: str = "found",
) -> AddWorkflowPlan:
    """Attach one scheduler result without mutating the task or doing I/O."""
    selection = AddScheduleSelection(
        target_field=plan.target_field,
        first_occurrence=first_occurrence,
        status=status,
    )
    return AddWorkflowPlan(
        request=plan.request,
        patch=plan.patch,
        recurrence_kind=plan.recurrence_kind,
        target_field=plan.target_field,
        target_explicit=plan.target_explicit,
        schedule=selection,
    )


__all__ = ("AddScheduleSelection", "AddWorkflowPlan", "classify_add_route", "plan_add", "record_schedule")
