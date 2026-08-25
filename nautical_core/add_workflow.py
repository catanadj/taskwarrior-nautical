"""Pure add-workflow planning for recurrence activation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .hook_workflow_models import (
    AddWorkflowRequest,
    FeedbackFacts,
    PatchOperation,
    TaskPatch,
    TaskPatchOperation,
    WorkflowRoute,
)
from .task_models import TaskObservation, TaskTimestamp


class AddScheduleFailure(RuntimeError):
    """Fail-closed error for terminal or unavailable add scheduling."""

    def __init__(self, status: str) -> None:
        normalized = str(status).strip().lower()
        if normalized not in {"terminal", "unavailable"}:
            raise ValueError("invalid add schedule failure status")
        self.status = normalized
        super().__init__(f"add scheduler result is {normalized}")


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
class AddScheduleLimits:
    """Validated bounds produced alongside a recurrence selection."""

    native_until: TaskTimestamp | None = None
    chain_until: TaskTimestamp | None = None
    chain_max: int | None = None
    expiration_hops: int | None = None

    def __post_init__(self) -> None:
        for name in ("native_until", "chain_until"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, TaskTimestamp):
                raise TypeError(f"{name} must be a TaskTimestamp or None")
        for name in ("chain_max", "expiration_hops"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")


@dataclass(frozen=True, slots=True)
class AddPreviewPolicy:
    """Presentation-only occurrence request limits."""

    mode: str
    enabled: bool
    occurrence_limit: int

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower() or "rich"
        if self.occurrence_limit < 0:
            raise ValueError("preview occurrence limit cannot be negative")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "enabled", bool(self.enabled))


@dataclass(frozen=True, slots=True)
class AddWorkflowPlan:
    """Typed add decision before patch application or presentation."""

    request: AddWorkflowRequest
    patch: TaskPatch
    recurrence_kind: str = ""
    target_field: str = "due"
    target_explicit: bool = False
    schedule: AddScheduleSelection | None = None
    limits: AddScheduleLimits | None = None
    preview: AddPreviewPolicy | None = None
    feedback: FeedbackFacts = FeedbackFacts()

    @property
    def ordinary(self) -> bool:
        return self.request.route is WorkflowRoute.ORDINARY

    @property
    def deterministic_fingerprint(self) -> str:
        """Stable identity for the same request and planner decision."""
        payload = {
            "task": self.request.task.semantic_fingerprint,
            "route": self.request.route.value,
            "patch": [
                (item.field, item.operation.value, item.value)
                for item in self.patch.operations
            ],
            "target": (self.target_field, self.target_explicit),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


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
    warnings = () if status == "found" else (f"scheduler result unavailable: {status}",)
    feedback = FeedbackFacts(
        recurrence_kind=plan.recurrence_kind,
        first_occurrence=selection.first_occurrence,
        warnings=warnings,
    )
    return AddWorkflowPlan(
        request=plan.request,
        patch=plan.patch,
        recurrence_kind=plan.recurrence_kind,
        target_field=plan.target_field,
        target_explicit=plan.target_explicit,
        schedule=selection,
        limits=plan.limits,
        preview=plan.preview,
        feedback=feedback,
    )


def schedule_patch(
    plan: AddWorkflowPlan,
    *,
    first_occurrence: TaskTimestamp,
    encode_timestamp,
) -> TaskPatch:
    """Build the target-field patch after a successful scheduler decision.

    Explicit due/scheduled input is preserved.  Auto-assignment is the only
    path allowed to add a temporal field here.
    """
    if plan.target_explicit:
        return TaskPatch(())
    if not isinstance(first_occurrence, TaskTimestamp):
        raise TypeError("scheduler target must be a TaskTimestamp")
    encoded = encode_timestamp(first_occurrence)
    if not isinstance(encoded, str) or not encoded.strip():
        raise ValueError("scheduler timestamp encoder returned an empty value")
    return TaskPatch((TaskPatchOperation(plan.target_field, PatchOperation.SET, encoded),))


def record_limits(plan: AddWorkflowPlan, limits: AddScheduleLimits) -> AddWorkflowPlan:
    """Attach validated recurrence bounds without applying or rendering them."""
    if not isinstance(limits, AddScheduleLimits):
        raise TypeError("add schedule limits must be AddScheduleLimits")
    return AddWorkflowPlan(
        request=plan.request,
        patch=plan.patch,
        recurrence_kind=plan.recurrence_kind,
        target_field=plan.target_field,
        target_explicit=plan.target_explicit,
        schedule=plan.schedule,
        limits=limits,
        preview=plan.preview,
        feedback=plan.feedback,
    )


def preview_policy(
    *,
    panel_mode: str,
    requested_limit: int,
    hard_cap: int,
) -> AddPreviewPolicy:
    """Return a bounded presentation policy without touching scheduler state."""
    mode = str(panel_mode).strip().lower() or "rich"
    cap = max(0, int(hard_cap))
    requested = max(0, int(requested_limit))
    compact = mode in {"quiet", "minimal", "line", "text"}
    limit = min(cap, 1 if compact else requested)
    return AddPreviewPolicy(mode=mode, enabled=mode not in {"off", "none"}, occurrence_limit=limit)


def record_preview(plan: AddWorkflowPlan, policy: AddPreviewPolicy) -> AddWorkflowPlan:
    """Attach presentation policy without changing scheduling or task fields."""
    if not isinstance(policy, AddPreviewPolicy):
        raise TypeError("add preview policy must be AddPreviewPolicy")
    return AddWorkflowPlan(
        request=plan.request,
        patch=plan.patch,
        recurrence_kind=plan.recurrence_kind,
        target_field=plan.target_field,
        target_explicit=plan.target_explicit,
        schedule=plan.schedule,
        limits=plan.limits,
        preview=policy,
        feedback=plan.feedback,
    )


def require_schedule(plan: AddWorkflowPlan) -> AddScheduleSelection:
    """Return a usable schedule or reject an incomplete planner decision."""
    selection = plan.schedule
    if selection is None:
        raise AddScheduleFailure("unavailable")
    if selection.status != "found":
        raise AddScheduleFailure(selection.status)
    return selection


__all__ = (
    "AddPreviewPolicy",
    "AddScheduleFailure",
    "AddScheduleLimits",
    "AddScheduleSelection",
    "AddWorkflowPlan",
    "classify_add_route",
    "plan_add",
    "record_schedule",
    "schedule_patch",
    "record_limits",
    "preview_policy",
    "record_preview",
    "require_schedule",
)
