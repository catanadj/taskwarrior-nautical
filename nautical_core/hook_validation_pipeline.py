"""Shared staged validation contract for hook workflow consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, MutableMapping, Protocol, cast

from .hook_workflow_models import WorkflowRoute
from .task_changes import TaskTransition
from .task_models import TaskObservation


def normalize_description_uda_aliases(
    task: MutableMapping[str, object],
    *,
    previous: Mapping[str, object] | None = None,
    enabled: bool,
) -> bool:
    """Apply description aliases at the single typed-validation boundary.

    The parser remains responsible for the Taskwarrior-standard empty-value
    clear syntax.  This function only owns the enablement and error boundary
    shared by add and modify consumers.
    """
    if not enabled:
        return False
    from . import description_aliases

    task_dict = cast(dict[str, object], task)
    if previous is None:
        return bool(description_aliases.apply_description_aliases(task_dict))
    return bool(
        description_aliases.apply_description_aliases(
            task_dict,
            previous=cast(dict[str, object], dict(previous)),
        )
    )


class ValidationStage(str, Enum):
    SYNTAX = "syntax"
    DOMAIN = "domain"
    SATISFIABILITY = "satisfiability"
    TRANSITION = "transition"


class ValidationStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    """One actionable, presentation-free validation finding."""

    stage: ValidationStage
    code: str
    field: str
    reason: str
    retryable: bool = False
    correction: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", ValidationStage(self.stage))
        for name in ("code", "field", "reason"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"validation {name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "retryable", bool(self.retryable))
        object.__setattr__(self, "correction", str(self.correction or "").strip())


@dataclass(frozen=True, slots=True)
class ValidationInput:
    """Typed task evidence supplied to every validation stage."""

    current: TaskObservation
    previous: TaskObservation | None = None
    transition: TaskTransition | None = None
    route: WorkflowRoute = WorkflowRoute.ORDINARY

    def __post_init__(self) -> None:
        if not isinstance(self.current, TaskObservation):
            raise TypeError("validation input requires a current TaskObservation")
        if self.previous is not None and not isinstance(self.previous, TaskObservation):
            raise TypeError("validation previous value must be a TaskObservation")
        if self.transition is not None and not isinstance(self.transition, TaskTransition):
            raise TypeError("validation transition must be a TaskTransition")
        object.__setattr__(self, "route", WorkflowRoute(self.route))


class ValidationRule(Protocol):
    def __call__(self, value: ValidationInput) -> tuple[ValidationFinding, ...]: ...


@dataclass(frozen=True, slots=True)
class ValidationReport:
    status: ValidationStatus
    findings: tuple[ValidationFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ValidationStatus(self.status))
        findings = tuple(self.findings)
        if any(not isinstance(item, ValidationFinding) for item in findings):
            raise TypeError("validation findings must be typed values")
        if self.status is ValidationStatus.VALID and findings:
            raise ValueError("valid validation report cannot contain findings")
        object.__setattr__(self, "findings", findings)

    @classmethod
    def from_findings(cls, findings: tuple[ValidationFinding, ...]) -> "ValidationReport":
        findings = tuple(findings)
        if any(item.retryable for item in findings):
            status = ValidationStatus.UNAVAILABLE
        elif findings:
            status = ValidationStatus.INVALID
        else:
            status = ValidationStatus.VALID
        return cls(status, findings)


@dataclass(frozen=True, slots=True)
class ValidationPipeline:
    """Run ordered typed rules without mutation or presentation side effects."""

    rules: tuple[tuple[ValidationStage, ValidationRule], ...] = ()

    def __post_init__(self) -> None:
        normalized: list[tuple[ValidationStage, ValidationRule]] = []
        for stage, rule in self.rules:
            stage_value = ValidationStage(stage)
            if not callable(rule):
                raise TypeError("validation rule must be callable")
            normalized.append((stage_value, rule))
        object.__setattr__(self, "rules", tuple(normalized))

    def validate(self, value: ValidationInput) -> ValidationReport:
        if not isinstance(value, ValidationInput):
            raise TypeError("validation pipeline requires ValidationInput")
        findings: list[ValidationFinding] = []
        for _stage, rule in self.rules:
            findings.extend(rule(value))
        return ValidationReport.from_findings(tuple(findings))


__all__ = (
    "ValidationFinding",
    "ValidationInput",
    "ValidationPipeline",
    "ValidationReport",
    "ValidationRule",
    "ValidationStage",
    "ValidationStatus",
    "normalize_description_uda_aliases",
)
