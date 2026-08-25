"""Pure add-workflow planning for recurrence activation."""

from __future__ import annotations

from dataclasses import dataclass

from .hook_workflow_models import AddWorkflowRequest, PatchOperation, TaskPatch, TaskPatchOperation, WorkflowRoute
from .task_models import TaskObservation


@dataclass(frozen=True, slots=True)
class AddWorkflowPlan:
    """Typed add decision before patch application or presentation."""

    request: AddWorkflowRequest
    patch: TaskPatch
    recurrence_kind: str = ""

    @property
    def ordinary(self) -> bool:
        return self.request.route is WorkflowRoute.ORDINARY


def _text(task: TaskObservation, field: str) -> str:
    return str(task.get(field) or "").strip()


def classify_add_route(task: TaskObservation) -> WorkflowRoute:
    """Classify one typed add observation without loading scheduling modules."""
    if _text(task, "cp"):
        return WorkflowRoute.CP_ACTIVATION
    if _text(task, "anchor_file"):
        return WorkflowRoute.ANCHOR_FILE_ACTIVATION
    if _text(task, "anchor"):
        return WorkflowRoute.ANCHOR_ACTIVATION
    return WorkflowRoute.ORDINARY


def plan_add(task: TaskObservation) -> AddWorkflowPlan:
    """Return recurrence defaults and root identity as a typed patch."""
    route = classify_add_route(task)
    request = AddWorkflowRequest(task=task, route=route)
    if route is WorkflowRoute.ORDINARY:
        return AddWorkflowPlan(request=request, patch=TaskPatch(()))

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
    )


__all__ = ("AddWorkflowPlan", "classify_add_route", "plan_add")
