"""Shared staged validation contract for hook workflow consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, MutableMapping, Protocol, cast, Any, Callable

from .hook_workflow_models import WorkflowRoute
from .task_changes import TaskTransition
from .task_models import ChainID, TaskLink, TaskObservation, TaskTimestamp


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


_STAGE_ORDER = {
    ValidationStage.SYNTAX: 0,
    ValidationStage.DOMAIN: 1,
    ValidationStage.SATISFIABILITY: 2,
    ValidationStage.TRANSITION: 3,
}


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


def _finding(code: str, field: str, reason: str, correction: str) -> ValidationFinding:
    return ValidationFinding(ValidationStage.DOMAIN, code, field, reason, correction=correction)


def validate_recurrence_exclusivity(value: ValidationInput) -> tuple[ValidationFinding, ...]:
    """Reject mixed recurrence sources before scheduler work begins."""
    task = value.current
    cp = str(task.get("cp") or "").strip()
    anchor = str(task.get("anchor") or "").strip()
    anchor_file = str(task.get("anchor_file") or "").strip()
    if cp and (anchor or anchor_file):
        return (_finding(
            "recurrence_kind_conflict", "cp",
            "cp cannot be combined with anchor or anchor_file",
            "Keep cp, or remove it before using anchor/anchor_file.",
        ),)
    return ()


def recurrence_kind_conflict(
    cp_value: object,
    anchor_value: object,
    anchor_file_value: object = None,
) -> tuple[bool, str | None]:
    """Validate recurrence source exclusivity for pre-observation callers."""
    cp = bool(str(cp_value or "").strip())
    anchor = bool(str(anchor_value or "").strip())
    anchor_file = bool(str(anchor_file_value or "").strip())
    if cp and anchor:
        return False, "Cannot set both 'cp' and 'anchor'. Choose one."
    if cp and anchor_file:
        return False, "Cannot set both 'cp' and 'anchor_file'. Choose one."
    return True, None


def reject_recurrence_kind_conflict(
    anchor_value: str,
    anchor_file_value: str,
    cp_value: str,
) -> None:
    """Raise the shared validation error used by modify route services."""
    valid, reason = recurrence_kind_conflict(cp_value, anchor_value, anchor_file_value)
    if not valid:
        raise ValueError(reason or "recurrence sources conflict")


def validate_recurrence_limits(
    cp_value: object,
    chain_max_value: object,
    chain_until_value: object,
    *,
    parse_cp_sequence: Callable[[str], Any],
    cp_sequence_parse_error: Callable[[str], str | None],
    parse_chain_max: Callable[[object], tuple[int | None, str | None]],
    parse_datetime: Callable[[object], Any],
) -> tuple[int | None, Any, tuple[ValidationFinding, ...]]:
    """Share CP/chain-limit parsing while leaving route policy to callers."""
    findings: list[ValidationFinding] = []
    # CP syntax retains its parser-specific diagnostics and is validated by
    # the route that owns the CP transition. This shared stage only handles
    # fields whose normalization contract is identical across routes.

    cpmax, chain_max_error = parse_chain_max(chain_max_value)
    if chain_max_error:
        findings.append(_finding(
            "chain_max_invalid", "chainMax", chain_max_error,
            "Set chainMax to a positive whole number or clear it.",
        ))

    until_dt = None
    chain_until = str(chain_until_value or "").strip()
    if chain_until:
        until_dt = parse_datetime(chain_until)
        if until_dt is None:
            findings.append(_finding(
                "chain_until_invalid", "chainUntil",
                f"Unrecognized datetime format '{chain_until}'",
                "Use Taskwarrior's datetime format or clear chainUntil.",
            ))
    return cpmax, until_dt, tuple(findings)


def validate_anchor_expression(
    expr: str | list[list[dict[str, Any]]],
    *,
    parse_anchor_expr: Callable[[str], Any],
    validate_anchor_expr: Callable[[str | list[list[dict[str, Any]]]], Any],
) -> None:
    """Validate one anchor expression for every workflow route."""
    if not str(expr or "").strip():
        raise ValueError("anchor is required if chaining by anchor")
    try:
        parse_anchor_expr(str(expr))
        validate_anchor_expr(expr)
    except Exception as exc:
        raise ValueError(f"anchor syntax error: {exc}") from exc


def validate_omit_expression(
    expr: str,
    *,
    validate_omit_expr: Callable[[str], Any],
) -> None:
    """Validate an optional omission expression at the shared boundary."""
    if not str(expr or "").strip():
        return
    try:
        validate_omit_expr(expr)
    except Exception as exc:
        raise ValueError(f"omit validation failed: {exc}") from exc


def validate_recurrence_files(
    anchor: object,
    anchor_file: object,
    omit: object,
    omit_file: object,
    *,
    load_anchor_file: Callable[[str], Any],
    load_omit_file: Callable[[str], Any],
) -> tuple[ValidationFinding, ...]:
    """Validate file-backed recurrence inputs without rendering or mutation."""
    anchor_text = str(anchor or "").strip()
    anchor_file_text = str(anchor_file or "").strip()
    omit_text = str(omit or "").strip()
    omit_file_text = str(omit_file or "").strip()
    findings: list[ValidationFinding] = []
    if omit_text and not (anchor_text or anchor_file_text):
        findings.append(_finding(
            "omit_without_anchor", "omit", "omit requires anchor or anchor_file",
            "Add an anchor/anchor_file or clear omit.",
        ))
    if omit_file_text and not (anchor_text or anchor_file_text):
        findings.append(_finding(
            "omit_file_without_anchor", "omit_file", "omit_file requires anchor or anchor_file",
            "Add an anchor/anchor_file or clear omit_file.",
        ))
    for field, value, loader in (
        ("anchor_file", anchor_file_text, load_anchor_file),
        ("omit_file", omit_file_text, load_omit_file),
    ):
        if not value:
            continue
        try:
            loader(value)
        except Exception as exc:
            findings.append(_finding(
                f"{field}_invalid", field, str(exc) or f"invalid {field}",
                f"Check the configured {field} path and file contents.",
            ))
    return tuple(findings)


def validate_anchor_mode_domain(value: ValidationInput) -> tuple[ValidationFinding, ...]:
    task = value.current
    if not str(task.get("anchor") or task.get("anchor_file") or "").strip():
        return ()
    mode = str(task.get("anchor_mode") or "skip").strip().lower()
    if mode not in {"skip", "all", "flex"}:
        return (_finding(
            "anchor_mode_invalid", "anchor_mode",
            f"unsupported anchor mode: {mode or '<empty>'}",
            "Use anchor_mode:skip, anchor_mode:all, or anchor_mode:flex.",
        ),)
    return ()


def validate_chain_limits_domain(value: ValidationInput) -> tuple[ValidationFinding, ...]:
    raw = value.current.get("chainMax")
    if raw in (None, ""):
        return ()
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        return (_finding(
            "chain_max_invalid", "chainMax", "chainMax must be a positive integer",
            "Set chainMax to a positive whole number or clear it.",
        ),)
    return ()


def validate_chain_identity_domain(value: ValidationInput) -> tuple[ValidationFinding, ...]:
    """Require identity once a task is an existing chain member.

    Activation routes intentionally validate the draft before generated
    identity exists; the add owner stamps it after this stage.
    """
    if value.previous is not None:
        return ()
    if value.route in {
        WorkflowRoute.CP_ACTIVATION,
        WorkflowRoute.ANCHOR_ACTIVATION,
        WorkflowRoute.ANCHOR_FILE_ACTIVATION,
    }:
        return ()
    task = value.current
    if str(task.get("chain") or "").strip().lower() not in {"on", "true", "1"}:
        return ()
    findings: list[ValidationFinding] = []
    if not isinstance(task.field("chainID").value, ChainID):
        findings.append(_finding(
            "chain_identity_missing", "chainID", "chainID is required for an enabled chain",
            "Restore the task's chainID or disable recurrence before retrying.",
        ))
    if not isinstance(task.field("link").value, TaskLink):
        findings.append(_finding(
            "chain_link_invalid", "link", "link must be a positive integer for an enabled chain",
            "Restore a positive chain link or disable recurrence before retrying.",
        ))
    return tuple(findings)


def validate_temporal_order_domain(value: ValidationInput) -> tuple[ValidationFinding, ...]:
    task = value.current
    due = task.field("due").value
    scheduled = task.field("scheduled").value
    wait = task.field("wait").value
    if not all(item is None or isinstance(item, TaskTimestamp) for item in (due, scheduled, wait)):
        return ()
    findings: list[ValidationFinding] = []
    if isinstance(due, TaskTimestamp) and isinstance(scheduled, TaskTimestamp) and due.value < scheduled.value:
        findings.append(_finding(
            "due_before_scheduled", "due", "due must not be earlier than scheduled",
            "Move due to scheduled or later, or adjust scheduled.",
        ))
    if isinstance(due, TaskTimestamp) and isinstance(wait, TaskTimestamp) and due.value < wait.value:
        findings.append(_finding(
            "due_before_wait", "due", "due must not be earlier than wait",
            "Move due to wait or later, or adjust wait.",
        ))
    return tuple(findings)


def validate_transition_policy(value: ValidationInput) -> tuple[ValidationFinding, ...]:
    """Reject user edits to identity fields before lifecycle planning."""
    transition = value.transition
    if transition is None:
        return ()
    identity_fields = tuple(
        field for field in ("chainID", "link", "prevLink", "nextLink")
        if transition.changed(field)
        and value.current.field(field).presence.value != "absent"
    )
    if not identity_fields:
        return ()
    return tuple(
        ValidationFinding(
            ValidationStage.TRANSITION,
            "chain_identity_edit",
            field,
            f"manual modification of {field} is not permitted",
            correction="Modify recurrence settings instead of chain identity fields.",
        )
        for field in identity_fields
    )


DEFAULT_DOMAIN_RULES: tuple[tuple[ValidationStage, ValidationRule], ...] = (
    (ValidationStage.DOMAIN, validate_recurrence_exclusivity),
    (ValidationStage.DOMAIN, validate_anchor_mode_domain),
    (ValidationStage.DOMAIN, validate_chain_limits_domain),
)


def build_default_validation_pipeline() -> "ValidationPipeline":
    return ValidationPipeline(DEFAULT_DOMAIN_RULES)


def validate_task_mapping(
    task: Mapping[str, object],
    *,
    route: WorkflowRoute,
    source_query: str,
) -> tuple[TaskObservation, "ValidationReport"]:
    """Decode and validate one hook task before domain work begins."""
    observation = TaskObservation.from_mapping(task, source_query=source_query)
    report = build_default_validation_pipeline().validate(
        ValidationInput(observation, route=route)
    )
    return observation, report


def validate_task_transition(
    old: TaskObservation,
    new: TaskObservation,
    *,
    route: WorkflowRoute,
    source_query: str,
) -> ValidationReport:
    """Validate a typed old/new pair without mutation or presentation."""
    transition = TaskTransition.from_observations(old, new)
    return ValidationPipeline(
        DEFAULT_DOMAIN_RULES
        + ((ValidationStage.DOMAIN, validate_chain_identity_domain),
           (ValidationStage.TRANSITION, validate_transition_policy))
    ).validate(ValidationInput(new, previous=old, transition=transition, route=route))


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
        findings = tuple(sorted(
            findings,
            key=lambda item: (_STAGE_ORDER[item.stage], item.code, item.field, item.reason),
        ))
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
            if normalized and _STAGE_ORDER[stage_value] < _STAGE_ORDER[normalized[-1][0]]:
                raise ValueError("validation stages must be declared in pipeline order")
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
    "DEFAULT_DOMAIN_RULES",
    "build_default_validation_pipeline",
    "validate_anchor_mode_domain",
    "validate_chain_identity_domain",
    "validate_chain_limits_domain",
    "validate_recurrence_exclusivity",
    "recurrence_kind_conflict",
    "reject_recurrence_kind_conflict",
    "validate_recurrence_limits",
    "validate_anchor_expression",
    "validate_omit_expression",
    "validate_recurrence_files",
    "validate_temporal_order_domain",
    "validate_task_mapping",
    "validate_task_transition",
    "normalize_description_uda_aliases",
)
