"""Typed carry and recurrence-transition effects for on-modify."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, ClassVar, NoReturn, Protocol

from .modify_carry_workflow import TemporalCarryDecision
from .modify_carry_workflow import NativeUntilDecision
from .native_until import NativeUntilCarryError
from .modify_validation import CompletionValidationServices
from .task_changes import TaskTransition
from .task_models import NauticalTask, TaskPayload, TaskTimestamp


@dataclass(frozen=True, slots=True)
class NativeCarryPorts:
    describe_carry: "NativeCarryDescription"
    parse_datetime: Callable[[object], datetime | None]
    to_local: Callable[[datetime], datetime]
    format_local: Callable[[datetime], str]
    anchor_field: Callable[[TaskPayload], str]
    panel: "NativeCarryPanel"
    abort: Callable[[int], NoReturn]


class NativeCarryDescription(Protocol):
    def __call__(
        self,
        until_dt: datetime | None,
        target_dt: datetime | None,
        *,
        to_local: Callable[[datetime], datetime],
    ) -> str | None: ...


class NativeCarryPanel(Protocol):
    def __call__(
        self,
        title: str,
        rows: list[tuple[str, str]],
        *,
        kind: str,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class CPCarryPorts:
    carry: "CPCarryOperation"
    field_changed: Callable[[TaskPayload, TaskPayload, str], bool]
    parse_datetime: Callable[[object], datetime | None]
    utc_to_local_naive: Callable[[datetime], datetime]
    local_naive_to_utc: Callable[[datetime], datetime]
    format_datetime: Callable[[datetime], str]
    carry_error: Callable[[str, str], Exception]
    workflow: "CPCarryWorkflow"


class CPCarryOperation(Protocol):
    def __call__(
        self,
        old: TaskPayload,
        new: TaskPayload,
        new_cp: str,
        *,
        field_changed: Callable[[TaskPayload, TaskPayload, str], bool],
        parse_datetime: Callable[[object], datetime | None],
        utc_to_local_naive: Callable[[datetime], datetime],
        local_naive_to_utc: Callable[[datetime], datetime],
        format_datetime: Callable[[datetime], str],
        carry_error: Callable[[str, str], Exception],
    ) -> tuple[datetime, datetime, list[tuple[str, datetime, datetime, timedelta]]] | None: ...


class CPCarryWorkflow(Protocol):
    def decision_from_cp_adjustments(
        self,
        result: tuple[datetime, datetime, list[tuple[str, datetime, datetime, timedelta]]] | None,
    ) -> TemporalCarryDecision: ...

    def apply_temporal_carry_patch(
        self,
        task: TaskPayload,
        decision: TemporalCarryDecision,
    ) -> None: ...

    def verify_temporal_carry_task(
        self,
        task: TaskPayload,
        decision: TemporalCarryDecision,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class NativePreservePorts:
    carry: "NativeUntilCarryOperation"
    field_changed: Callable[[TaskPayload, TaskPayload, str], bool]
    anchor_field: Callable[[TaskPayload], str]
    parse_datetime: Callable[[object], datetime | None]
    native_until: "NativeUntilPolicy"
    generation_service: Callable[[], "NativeUntilGenerationService"]
    reject_carry: Callable[
        [TaskPayload, TaskPayload, datetime | None, str, NativeUntilCarryError], None
    ]
    diagnostic: Callable[[str], None]
    workflow: "NativeUntilWorkflow"


class NativeUntilPolicy(Protocol):
    CARRY_INVALID: ClassVar[str]
    CARRY_FAILED: ClassVar[str]
    NativeUntilCarryError: ClassVar[type[NativeUntilCarryError]]


class NativeUntilGenerationService(Protocol):
    def carry_native_until(
        self,
        parent: NauticalTask,
        child: TaskPayload,
        child_due_utc: datetime,
        kind: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None: ...


class NativeUntilCarryOperation(Protocol):
    def __call__(
        self,
        old: TaskPayload,
        new: TaskPayload,
        kind: str,
        *,
        field_changed: Callable[[TaskPayload, TaskPayload, str], bool],
        recurrence_anchor_field: Callable[[TaskPayload], str],
        parse_datetime: Callable[[object], datetime | None],
        native_until: NativeUntilPolicy,
        generation_service: Callable[[], NativeUntilGenerationService],
        reject_carry: Callable[
            [TaskPayload, TaskPayload, datetime | None, str, NativeUntilCarryError], None
        ],
        diagnostic: Callable[[str], None],
    ) -> bool: ...


class NativeUntilWorkflow(Protocol):
    def apply_native_until_patch(self, task: TaskPayload, decision: NativeUntilDecision) -> None: ...

    def verify_native_until_task(self, task: TaskPayload, decision: NativeUntilDecision) -> None: ...


@dataclass(frozen=True, slots=True)
class CompletionValidationPorts:
    validate: "CompletionValidator"
    strip_quotes: Callable[[str], str]
    reject_conflicting_types: Callable[[str, str, str], None]
    validate_omit: Callable[[str, str, str, str], None]
    validate_chain_limits: Callable[[TaskPayload], None]
    parse_cp_sequence: Callable[[str], object]
    cp_sequence_parse_error: Callable[[str], str | None]
    field_changed: Callable[[TaskPayload, TaskPayload, str], bool]
    validate_anchor: Callable[[str], None]
    validate_cp: Callable[[str, object, object], None]
    apply_transition: Callable[[TaskPayload, TaskPayload], None]
    fail: Callable[[str, str], object]
    diagnostic: Callable[[str], None]


class CompletionValidator(Protocol):
    def __call__(
        self,
        old: TaskPayload,
        new: TaskPayload,
        *,
        services: CompletionValidationServices,
    ) -> tuple[str, str, str]: ...


def preserve_cp_relative_offsets_on_due_change(
    ports: CPCarryPorts,
    old: TaskPayload,
    new: TaskPayload,
    new_cp: str,
    *,
    transition: TaskTransition | None = None,
) -> TemporalCarryDecision:
    result = ports.carry(
        old,
        new,
        new_cp,
        field_changed=(
            (lambda _old, _new, field: transition.changed(field))
            if transition is not None
            else ports.field_changed
        ),
        parse_datetime=ports.parse_datetime,
        utc_to_local_naive=ports.utc_to_local_naive,
        local_naive_to_utc=ports.local_naive_to_utc,
        format_datetime=ports.format_datetime,
        carry_error=ports.carry_error,
    )
    decision = ports.workflow.decision_from_cp_adjustments(result)
    ports.workflow.apply_temporal_carry_patch(new, decision)
    ports.workflow.verify_temporal_carry_task(new, decision)
    return decision


def reject_native_until_carry(
    ports: NativeCarryPorts,
    old: TaskPayload,
    new: TaskPayload,
    new_target: datetime | None,
    old_target_field: str,
    exc: Exception,
) -> None:
    """Reject a target edit when its native expiration cannot be carried."""
    carry = None
    try:
        carry = ports.describe_carry(
            ports.parse_datetime(old.get("until")),
            ports.parse_datetime(old.get(old_target_field)),
            to_local=ports.to_local,
        )
    except Exception:
        # This explanation is optional; never let it replace the primary
        # rejection or prevent the user from seeing the required action.
        pass
    target_label = (
        ports.format_local(new_target)
        if isinstance(new_target, datetime)
        else str(ports.anchor_field(new) or "–")
    )
    rows = [("Target", target_label), ("Required", str(exc))]
    if carry:
        rows.insert(1, ("Carry", carry))
    ports.panel("❌ Invalid expiration window", rows, kind="error")
    ports.abort(1)


def preserve_native_until_on_target_change(
    ports: NativePreservePorts,
    old: TaskPayload,
    new: TaskPayload,
    kind: str,
    *,
    transition: TaskTransition | None = None,
) -> NativeUntilDecision:
    carried = ports.carry(
        old,
        new,
        kind,
        field_changed=(
            (lambda _old, _new, field: transition.changed(field))
            if transition is not None
            else ports.field_changed
        ),
        recurrence_anchor_field=ports.anchor_field,
        parse_datetime=ports.parse_datetime,
        native_until=ports.native_until,
        generation_service=ports.generation_service,
        reject_carry=ports.reject_carry,
        diagnostic=ports.diagnostic,
    )
    if not carried:
        return NativeUntilDecision("unchanged")
    value = ports.parse_datetime(new.get("until"))
    if value is None:
        return NativeUntilDecision(
            "rejected", reason="native-until carry produced no parseable value"
        )
    decision = NativeUntilDecision("carried", value=TaskTimestamp(value))
    ports.workflow.apply_native_until_patch(new, decision)
    ports.workflow.verify_native_until_task(new, decision)
    return decision


def validate_completion_cp_and_anchor(
    ports: CompletionValidationPorts,
    old: TaskPayload,
    new: TaskPayload,
    *,
    transition: TaskTransition | None = None,
) -> tuple[str, str, str]:
    return ports.validate(
        old,
        new,
        services=CompletionValidationServices(
            strip_quotes=ports.strip_quotes,
            reject_conflicting_types=ports.reject_conflicting_types,
            validate_omit=ports.validate_omit,
            validate_chain_limits=ports.validate_chain_limits,
            parse_cp_sequence=ports.parse_cp_sequence,
            cp_sequence_parse_error=ports.cp_sequence_parse_error,
            field_changed=(
                (lambda _old, _new, field: transition.changed(field))
                if transition is not None
                else ports.field_changed
            ),
            validate_anchor=ports.validate_anchor,
            validate_cp=ports.validate_cp,
            apply_transition=ports.apply_transition,
            fail=ports.fail,
            diagnostic=ports.diagnostic,
        ),
    )


__all__ = (
    "NativeCarryPorts",
    "CPCarryPorts",
    "NativePreservePorts",
    "CompletionValidationPorts",
    "preserve_cp_relative_offsets_on_due_change",
    "reject_native_until_carry",
    "preserve_native_until_on_target_change",
    "validate_completion_cp_and_anchor",
)
