"""Guarded, named Taskwarrior mutations for lifecycle transitions.

This module owns the write side of the integration boundary.  Callers supply
validated mutation requests; they never provide arbitrary Taskwarrior argv.
Every operation re-reads its target, applies a narrow selector, and verifies
the requested postcondition before reporting success.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Protocol, Sequence

from .integration_models import (
    Absent,
    ChainDisablePayload,
    ChildImportPayload,
    CommandFailureKind,
    FailureEvidence,
    Found,
    MetadataRepairPayload,
    MutationGuard,
    MutationOperation,
    MutationOutcome,
    MutationOutcomeKind,
    MutationPostcondition,
    MutationRequest,
    NativeUntilRepairPayload,
    ParentLinkPayload,
    TaskCommandResult,
    TaskRead,
    TaskwarriorMutationPort,
    Unavailable,
)
from .lifecycle_models import recurrence_fingerprint


class _TaskRepository(Protocol):
    def by_uuid(self, uuid_value: str, *, refresh: bool = False) -> TaskRead[Mapping[str, Any]]: ...


class _TaskClient(Protocol):
    def execute(
        self,
        args: Sequence[str],
        *,
        purpose: str,
        timeout: float,
        input_text: str | None = None,
        attempts: int = 1,
    ) -> TaskCommandResult: ...


class _UnitOfWork(Protocol):
    repository: _TaskRepository
    client: _TaskClient
    mutation_epoch: int

    def record_mutation(self, *, uncertain: bool = False) -> int: ...


_TIMESTAMP_FIELDS = {
    "modified": "modified",
    "due": "due",
    "until": "until",
    "end": "end",
}


def _text(value: object) -> str:
    return str(value or "").strip()


def _failure_from_command(result: TaskCommandResult, detail: str) -> FailureEvidence:
    kind = result.kind
    if kind in {CommandFailureKind.SUCCESS, CommandFailureKind.ABSENT}:
        kind = CommandFailureKind.REJECTED
    return FailureEvidence(
        result.command,
        kind,
        result.returncode,
        result.attempt,
        result.duration,
        kind in {CommandFailureKind.TIMEOUT, CommandFailureKind.BUSY, CommandFailureKind.EXECUTION_FAILURE},
        detail or result.stderr or result.kind.value,
    )


class TaskwarriorMutationService(TaskwarriorMutationPort):
    """Apply one typed request through one invocation-scoped unit of work."""

    def __init__(self, unit_of_work: _UnitOfWork, *, timeout: float = 30.0) -> None:
        self._uow = unit_of_work
        self._timeout = max(0.05, float(timeout))

    def _outcome(
        self,
        request: MutationRequest,
        kind: MutationOutcomeKind,
        *,
        postcondition: MutationPostcondition | None = None,
        reason: str = "",
        failure: FailureEvidence | None = None,
    ) -> MutationOutcome:
        return MutationOutcome(
            request.operation,
            kind,
            request.guard,
            () if postcondition is None else (postcondition,),
            reason,
            failure,
        )

    def _read_target(self, request: MutationRequest) -> tuple[Mapping[str, Any] | None, MutationOutcome | None]:
        if self._uow.mutation_epoch != request.guard.expected_mutation_epoch:
            return None, self._outcome(
                request,
                MutationOutcomeKind.CONFLICT,
                reason="mutation epoch changed before guard read",
            )
        result = self._uow.repository.by_uuid(request.guard.task_uuid, refresh=True)
        if isinstance(result, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if result.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return None, self._outcome(request, kind, reason=result.evidence.detail, failure=result.evidence)
        if isinstance(result, Absent):
            return None, self._outcome(request, MutationOutcomeKind.CONFLICT, reason="guard task is absent")
        if not isinstance(result, Found):
            return None, self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid guard read result")
        row = result.value
        mismatch = self._guard_mismatch(request.guard, row)
        if mismatch:
            if (
                request.operation is MutationOperation.CHAIN_DISABLE
                and request.guard.chain == "on"
                and _text(row.get("chain")).lower() == "off"
            ):
                return row, None
            return None, self._outcome(request, MutationOutcomeKind.CONFLICT, reason=mismatch)
        return row, None

    @staticmethod
    def _guard_mismatch(guard: MutationGuard, row: Mapping[str, Any]) -> str:
        values = {
            "status": guard.status,
            "chain": guard.chain,
            "chainID": guard.chain_id,
            "link": str(guard.link),
        }
        for field, expected in values.items():
            actual = _text(row.get(field))
            if field == "link":
                try:
                    actual = str(int(float(actual)))
                except (TypeError, ValueError, OverflowError):
                    pass
            if actual.casefold() != expected.casefold():
                return f"guard {field} changed (expected {expected}, found {actual or '-'})"
        try:
            actual_identity = recurrence_fingerprint(row)
        except Exception as exc:
            return f"guard recurrence identity unavailable: {exc}"
        if actual_identity != guard.recurrence_identity:
            return "guard recurrence identity changed"
        for timestamp in guard.timestamps:
            field = _TIMESTAMP_FIELDS.get(timestamp.field.value)
            if field is not None and _text(row.get(field)) != timestamp.value:
                return f"guard {field} changed"
        return ""

    def _command_failure(self, request: MutationRequest, result: TaskCommandResult) -> MutationOutcome:
        evidence = _failure_from_command(result, f"{request.operation.value} command failed")
        kind = MutationOutcomeKind.RETRYABLE if evidence.retryable else MutationOutcomeKind.REJECTED
        return self._outcome(request, kind, reason=evidence.detail, failure=evidence)

    def _run_modify(self, request: MutationRequest, selectors: Sequence[str], updates: Sequence[str]) -> MutationOutcome | None:
        result = self._uow.client.execute(
            [*selectors, "modify", *updates],
            purpose=f"lifecycle {request.operation.value}",
            timeout=self._timeout,
            attempts=1,
        )
        if not result.ok:
            self._uow.record_mutation(uncertain=True)
            return self._command_failure(request, result)
        self._uow.record_mutation(uncertain=False)
        return None

    def _verify(
        self,
        request: MutationRequest,
        postcondition: MutationPostcondition,
        predicate,
        *,
        target_uuid: str | None = None,
    ) -> MutationOutcome:
        result = self._uow.repository.by_uuid(target_uuid or request.guard.task_uuid, refresh=True)
        if isinstance(result, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if result.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return self._outcome(request, kind, reason=result.evidence.detail, failure=result.evidence)
        if isinstance(result, Absent):
            return self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="postcondition target is absent")
        if not isinstance(result, Found):
            return self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid postcondition read result")
        if not predicate(result.value):
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="postcondition does not match")
        return self._outcome(request, MutationOutcomeKind.APPLIED, postcondition=postcondition)

    def apply(self, request: MutationRequest) -> MutationOutcome:
        if not isinstance(request, MutationRequest):
            raise TypeError("mutation service requires a MutationRequest")
        handlers = {
            MutationOperation.CHILD_IMPORT: self.import_child,
            MutationOperation.PARENT_LINK: self.link_parent,
            MutationOperation.CHAIN_DISABLE: self.disable_chain,
            MutationOperation.NATIVE_UNTIL_REPAIR: self.repair_native_until,
            MutationOperation.METADATA_REPAIR: self.repair_metadata,
        }
        return handlers[request.operation](request)

    def import_child(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.CHILD_IMPORT or not isinstance(request.payload, ChildImportPayload):
            raise TypeError("child import requires a child-import request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        if _text(parent.get("chain")).lower() != "on":
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="parent chain is no longer active")
        existing = self._uow.repository.by_uuid(request.payload.child_uuid, refresh=True)
        if isinstance(existing, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if existing.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return self._outcome(request, kind, reason=existing.evidence.detail, failure=existing.evidence)
        if isinstance(existing, Found):
            row = existing.value
            if (
                _text(row.get("chainID")) == request.payload.chain_id
                and _text(row.get("prevLink")).lower() == request.guard.task_uuid[:8].lower()
                and _text(row.get("link")) == str(request.payload.target_link)
            ):
                return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED)
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="unrelated task already owns child UUID")
        if not isinstance(existing, Absent):
            return self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid child lookup result")
        result = self._uow.client.execute(
            ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            purpose="lifecycle child import",
            timeout=self._timeout,
            input_text=json.dumps(request.payload.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n",
            attempts=1,
        )
        if not result.ok:
            self._uow.record_mutation(uncertain=True)
            return self._command_failure(request, result)
        self._uow.record_mutation(uncertain=True)
        return self._verify(
            request,
            MutationPostcondition.CHILD_IMPORTED,
            lambda row: (
                _text(row.get("uuid")).lower() == request.payload.child_uuid.lower()
                and _text(row.get("chainID")) == request.payload.chain_id
                and _text(row.get("link")) == str(request.payload.target_link)
            ),
            target_uuid=request.payload.child_uuid,
        )

    def link_parent(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.PARENT_LINK or not isinstance(request.payload, ParentLinkPayload):
            raise TypeError("parent linking requires a parent-link request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        if _text(parent.get("chain")).lower() != "on":
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="parent chain is no longer active")
        current = _text(parent.get("nextLink"))
        if current.casefold() == request.payload.child_short_uuid.casefold():
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.PARENT_LINKED)
        if current != request.payload.expected_next_link:
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="parent nextLink changed")
        failure = self._run_modify(
            request,
            self._selectors(request.guard, extra=(f"nextLink:{current}",)),
            (f"nextLink:{request.payload.child_short_uuid}",),
        )
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.PARENT_LINKED,
            lambda row: _text(row.get("nextLink")).casefold() == request.payload.child_short_uuid.casefold(),
        )

    def disable_chain(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.CHAIN_DISABLE or not isinstance(request.payload, ChainDisablePayload):
            raise TypeError("chain disablement requires a chain-disable request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        if _text(parent.get("chain")).lower() == "off":
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.CHAIN_DISABLED)
        failure = self._run_modify(request, self._selectors(request.guard), ("chain:off",))
        if failure is not None:
            return failure
        return self._verify(request, MutationPostcondition.CHAIN_DISABLED, lambda row: _text(row.get("chain")).lower() == "off")

    def repair_native_until(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.NATIVE_UNTIL_REPAIR or not isinstance(request.payload, NativeUntilRepairPayload):
            raise TypeError("native-until repair requires a native-until request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        current = _text(parent.get("until"))
        if current == request.payload.replacement_until:
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.NATIVE_UNTIL_REPAIRED)
        if current != request.payload.expected_until:
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="native until changed")
        failure = self._run_modify(
            request,
            self._selectors(request.guard, extra=(f"until:{current}",)),
            (f"until:{request.payload.replacement_until}",),
        )
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.NATIVE_UNTIL_REPAIRED,
            lambda row: _text(row.get("until")) == request.payload.replacement_until,
        )

    def repair_metadata(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.METADATA_REPAIR or not isinstance(request.payload, MetadataRepairPayload):
            raise TypeError("metadata repair requires a metadata request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        updates = request.payload.to_dict()
        if all(_text(parent.get(key)) == _text(value) for key, value in updates.items()):
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.METADATA_REPAIRED)
        failure = self._run_modify(
            request,
            self._selectors(request.guard),
            tuple(f"{key}:{value}" for key, value in updates.items()),
        )
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.METADATA_REPAIRED,
            lambda row: all(_text(row.get(key)) == _text(value) for key, value in updates.items()),
        )

    @staticmethod
    def _selectors(guard: MutationGuard, *, extra: Sequence[str] = ()) -> tuple[str, ...]:
        selectors = (
            f"uuid:{guard.task_uuid}",
            f"status:{guard.status}",
            f"chainID:{guard.chain_id}",
            f"link:{guard.link}",
        )
        modified = next(
            (timestamp.value for timestamp in guard.timestamps if timestamp.field.value == "modified"),
            "",
        )
        if modified:
            selectors = (*selectors, f"modified:{modified}")
        return (*selectors, *tuple(extra))


__all__ = ("TaskwarriorMutationService",)
