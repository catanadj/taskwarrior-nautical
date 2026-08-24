"""Guarded, named Taskwarrior mutations for lifecycle transitions.

This module owns the write side of the integration boundary.  Callers supply
validated mutation requests; they never provide arbitrary Taskwarrior argv.
Every operation re-reads its target, applies a narrow selector, and verifies
the requested postcondition before reporting success.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

from .integration_models import (
    Absent,
    ChainDisablePayload,
    ChildCompensationPayload,
    ChildImportPayload,
    CommandFailureKind,
    FailureEvidence,
    GuardTimestamp,
    GuardTimestampField,
    Found,
    IntegrationContractError,
    MetadataRepairPayload,
    MutationGuard,
    MutationOperation,
    MutationOutcome,
    MutationOutcomeKind,
    MutationPostcondition,
    MutationRequest,
    NativeUntilRepairPayload,
    ParentLinkPayload,
    ParentLinkClearPayload,
    TaskCommandResult,
    TaskRead,
    TaskwarriorMutationPort,
    Unavailable,
)
from .task_codec import TaskCodec
from .lifecycle_models import recurrence_fingerprint
from .task_codec import DEFAULT_TASK_CODEC
from .task_models import FieldPresence, TaskObservation
from .task_changes import timestamp_equal
from .task_set_reads import SetReadResult, SetReadStatus, UUIDSetRequest


class _TaskRepository(Protocol):
    def by_uuid(self, uuid_value: str, *, refresh: bool = False) -> TaskRead[TaskObservation]: ...

    def read_uuid_set(self, request: UUIDSetRequest) -> SetReadResult: ...


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


def _observed_value(row: TaskObservation | Mapping[str, Any], field: str) -> object:
    """Read a postcondition field without thawing an observation."""
    if isinstance(row, TaskObservation):
        state = row.field(field)
        return None if state.presence is FieldPresence.ABSENT else state.raw_value()
    return row.get(field)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse the compact and ISO UTC forms emitted by Taskwarrior/Nautical."""
    value_text = _text(value)
    if not value_text:
        return None
    try:
        parsed = datetime.fromisoformat(value_text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = datetime.strptime(value_text, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _link_text(value: object) -> str:
    text = _text(value)
    try:
        return str(int(float(text)))
    except (TypeError, ValueError, OverflowError):
        return text


def _child_import_matches(
    row: TaskObservation | Mapping[str, Any],
    payload: ChildImportPayload,
    parent_uuid: str,
) -> bool:
    """Return whether a Taskwarrior row is a complete imported child.

    Identity alone is insufficient: a partially imported or repurposed UUID
    must never be treated as an idempotent child.  Keep this predicate shared
    by the existing-child fast path and the post-import verification path.
    """
    fields = payload.to_dict()
    if _text(_observed_value(row, "uuid")).lower() != payload.child_uuid.lower():
        return False
    if _text(_observed_value(row, "chainID")) != payload.chain_id:
        return False
    if _link_text(_observed_value(row, "link")) != str(payload.target_link):
        return False
    expected_prev = _text(fields.get("prevLink"))
    if not expected_prev or _text(_observed_value(row, "prevLink")).lower() != expected_prev.lower():
        return False
    if _text(_observed_value(row, "prevLink")).lower() != _text(parent_uuid)[:8].lower():
        return False
    status = _text(_observed_value(row, "status")).lower()
    if status != "pending":
        # Taskwarrior may immediately expire a child whose carried native
        # until is already in the past. That is a valid imported occurrence;
        # the reconciler can continue from its deleted slot on the next hop.
        until = _parse_timestamp(fields.get("until"))
        if status != "deleted" or until is None or until > datetime.now(timezone.utc):
            return False
    if _text(_observed_value(row, "chain")).lower() != "on":
        return False

    # A child must retain the recurrence mode that generated it.  Compare
    # only mode-defining fields; Taskwarrior may add derived metadata during
    # import (modified, urgency, and so on).
    mode_fields = ("cp", "anchor", "anchor_file", "anchor_mode")
    if not any(_text(fields.get(field)) for field in mode_fields[:3]):
        return False
    for field in mode_fields:
        if (
            field in fields
            and TaskCodec.normalize_text(_observed_value(row, field)) != TaskCodec.normalize_text(fields.get(field))
        ):
            return False
    return True


def _failure_from_command(result: TaskCommandResult, detail: str) -> FailureEvidence:
    kind = result.kind
    if kind in {CommandFailureKind.SUCCESS, CommandFailureKind.ABSENT}:
        kind = CommandFailureKind.REJECTED
    command_detail = str(result.stderr or result.stdout or "").strip()
    if command_detail and detail:
        detail = f"{detail}: {command_detail}"
    elif command_detail:
        detail = command_detail
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
        self._prefetched_children: dict[str, TaskRead[TaskObservation]] = {}
        self._prefetched_parents: dict[str, TaskObservation] = {}

    def preflight_lifecycle_batch(
        self,
        payloads: Sequence[ChildImportPayload],
        *,
        parent_expectations: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Preload targeted child/parent evidence for one drain phase.

        A cached absence can only skip a redundant pre-import UUID read: the
        import command remains authoritative if another process creates the
        child.  Present rows are deliberately *not* cached because linking a
        stale or deleted child would weaken idempotency; those paths retain a
        fresh UUID read and postcondition verification.
        """
        self._prefetched_children.clear()
        self._prefetched_parents.clear()
        wanted = tuple(payloads)
        parent_wanted = tuple(
            (str(uuid_value or "").strip().lower(), str(expected or "").strip())
            for uuid_value, expected in parent_expectations
            if str(uuid_value or "").strip()
        )
        if not wanted and not parent_wanted:
            return
        requested = tuple(
            dict.fromkeys(
                [payload.child_uuid.lower() for payload in wanted]
                + [uuid_value for uuid_value, _ in parent_wanted]
            )
        )
        try:
            result = self._uow.repository.read_uuid_set(
                UUIDSetRequest(requested, refresh=True, expected_mutation_epoch=self._uow.mutation_epoch)
            )
        except (TypeError, ValueError):
            return
        if result.status is not SetReadStatus.COMPLETE:
            return
        for payload in wanted:
            identity = payload.child_uuid.lower()
            cached = result.found.get(identity)
            if cached is not None:
                self._prefetched_children[identity] = Found(
                    cached,
                    f"preflight:uuid:{identity}",
                )
            else:
                self._prefetched_children[payload.child_uuid.lower()] = Absent(
                    f"preflight:uuid:{payload.child_uuid.lower()}", "preflight snapshot contains no matching UUID"
                )
        for parent_uuid, expected_next_link in parent_wanted:
            row = result.found.get(parent_uuid)
            if row is not None:
                # The snapshot is the batch guard for both child import and
                # parent link.  Child import does not mutate the parent; the
                # eventual guarded link command and phase verification still
                # detect concurrent edits and trigger compensation/retry.
                self._prefetched_parents[parent_uuid] = row

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

    def _read_target(self, request: MutationRequest) -> tuple[TaskObservation | None, MutationOutcome | None]:
        # Reuse the authoritative batch parent snapshot for child import. The
        # import itself cannot change the parent; its guarded parent-link
        # command and postcondition phase remain the concurrency boundary.
        if request.operation is MutationOperation.CHILD_IMPORT:
            cached_parent = self._prefetched_parents.get(request.guard.task_uuid.lower())
            if cached_parent is not None:
                mismatch = self._guard_mismatch(request.guard, cached_parent)
                if mismatch:
                    return None, self._outcome(request, MutationOutcomeKind.CONFLICT, reason=mismatch)
                return cached_parent, None
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
            if request.operation is MutationOperation.CHILD_COMPENSATION:
                return None, self._outcome(
                    request,
                    MutationOutcomeKind.ALREADY_APPLIED,
                    postcondition=MutationPostcondition.CHILD_COMPENSATED,
                )
            return None, self._outcome(request, MutationOutcomeKind.CONFLICT, reason="guard task is absent")
        if not isinstance(result, Found):
            return None, self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid guard read result")
        row = result.value
        if not isinstance(row, TaskObservation):
            return None, self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid guard task shape")
        mismatch = self._guard_mismatch(request.guard, row)
        if mismatch:
            # A successful parent link changes Taskwarrior's ``modified``
            # timestamp.  If the process crashed after that external
            # mutation but before the outbox stage was persisted, the next
            # recovery read must recognize the requested link as converged.
            # Keep all other guard fields strict: only the timestamp may have
            # changed, and only for the parent-link operation.
            if (
                request.operation is MutationOperation.PARENT_LINK
                and isinstance(request.payload, ParentLinkPayload)
                and _text(_observed_value(row, "nextLink")).casefold()
                == request.payload.child_short_uuid.casefold()
                and mismatch.startswith("guard modified changed")
            ):
                expected_modified = next(
                    (
                        timestamp.value
                        for timestamp in request.guard.timestamps
                        if timestamp.field.value == "modified"
                    ),
                    None,
                )
                if expected_modified is not None:
                    if not self._guard_mismatch(request.guard, row, modified_override=expected_modified):
                        return row, None
            if (
                request.operation is MutationOperation.NATIVE_UNTIL_REPAIR
                and isinstance(request.payload, NativeUntilRepairPayload)
                and mismatch.startswith("guard modified changed")
            ):
                expected_modified = next(
                    (
                        timestamp.value
                        for timestamp in request.guard.timestamps
                        if timestamp.field.value == "modified"
                    ),
                    None,
                )
                if expected_modified is not None:
                    if not self._guard_mismatch(request.guard, row, modified_override=expected_modified):
                        return row, None
            if (
                request.operation is MutationOperation.METADATA_REPAIR
                and isinstance(request.payload, MetadataRepairPayload)
                and request.payload.expected_dict().get("link") == ""
                and not _text(_observed_value(row, "link"))
                and mismatch.startswith("guard link changed")
            ):
                return row, None
            if (
                request.operation is MutationOperation.CHAIN_DISABLE
                and request.guard.chain == "on"
                and _text(_observed_value(row, "chain")).lower() == "off"
            ):
                return row, None
            if (
                request.operation is MutationOperation.CHILD_COMPENSATION
                and _text(_observed_value(row, "status")).lower() == "deleted"
            ):
                return row, None
            return None, self._outcome(request, MutationOutcomeKind.CONFLICT, reason=mismatch)
        return row, None

    @staticmethod
    def _guard_mismatch(
        guard: MutationGuard,
        row: TaskObservation,
        *,
        modified_override: str | None = None,
    ) -> str:
        values = {
            "status": guard.status,
            "chain": guard.chain,
            "chainID": guard.chain_id,
            "link": str(guard.link),
        }
        for field, expected in values.items():
            actual = _text(_observed_value(row, field))
            if field == "link":
                try:
                    actual = str(int(float(actual)))
                except (TypeError, ValueError, OverflowError):
                    pass
            if actual.casefold() != expected.casefold():
                return f"guard {field} changed (expected {expected}, found {actual or '-'})"
        try:
            actual_identity = recurrence_fingerprint(row.to_mapping())
        except Exception as exc:
            return f"guard recurrence identity unavailable: {exc}"
        if actual_identity != guard.recurrence_identity:
            return "guard recurrence identity changed"
        for timestamp in guard.timestamps:
            ts_field = _TIMESTAMP_FIELDS.get(timestamp.field.value)
            actual_value: object = (
                modified_override
                if ts_field == "modified" and modified_override is not None
                else _observed_value(row, ts_field) if ts_field is not None else None
            )
            if ts_field is not None and _text(actual_value) != timestamp.value:
                return f"guard {ts_field} changed"
        return ""

    def _command_failure(self, request: MutationRequest, result: TaskCommandResult) -> MutationOutcome:
        evidence = _failure_from_command(result, f"{request.operation.value} command failed")
        kind = MutationOutcomeKind.RETRYABLE if evidence.retryable else MutationOutcomeKind.REJECTED
        return self._outcome(request, kind, reason=evidence.detail, failure=evidence)

    def _run_modify(self, request: MutationRequest, selectors: Sequence[str], updates: Sequence[str]) -> MutationOutcome | None:
        result = self._uow.client.execute(
            ["rc.hooks=off", *selectors, "modify", *updates],
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
        predicate: Any,
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
        row = result.value
        if not isinstance(row, TaskObservation) or not predicate(row):
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="postcondition does not match")
        return self._outcome(request, MutationOutcomeKind.APPLIED, postcondition=postcondition)

    def apply(self, request: MutationRequest) -> MutationOutcome:
        if not isinstance(request, MutationRequest):
            raise TypeError("mutation service requires a MutationRequest")
        handlers: dict[MutationOperation, Callable[[MutationRequest], MutationOutcome]] = {
            MutationOperation.CHILD_IMPORT: self.import_child,
            MutationOperation.CHILD_COMPENSATION: self.compensate_child,
            MutationOperation.PARENT_LINK: self.link_parent,
            MutationOperation.PARENT_LINK_CLEAR: self.clear_parent_link,
            MutationOperation.CHAIN_DISABLE: self.disable_chain,
            MutationOperation.NATIVE_UNTIL_REPAIR: self.repair_native_until,
            MutationOperation.METADATA_REPAIR: self.repair_metadata,
        }
        return handlers[request.operation](request)

    def compensate_child(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.CHILD_COMPENSATION or not isinstance(request.payload, ChildCompensationPayload):
            raise TypeError("child compensation requires a child-compensation request")
        child, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert child is not None
        if _text(_observed_value(child, "status")).lower() == "deleted":
            return self._outcome(
                request,
                MutationOutcomeKind.ALREADY_APPLIED,
                postcondition=MutationPostcondition.CHILD_COMPENSATED,
            )
        if _text(_observed_value(child, "status")).lower() != request.payload.expected_status:
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="child status changed before compensation")
        failure = self._run_modify(request, self._selectors(request.guard), ("status:deleted",))
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.CHILD_COMPENSATED,
            lambda row: _text(_observed_value(row, "status")).lower() == "deleted",
        )

    def compensate_imported_child(self, request: MutationRequest) -> MutationOutcome:
        """Build a child-scoped guard before compensating an imported child.

        The import request is guarded by the parent, so it cannot be reused
        directly for deletion.  Re-read the child, prove it still matches the
        deterministic import, then delegate to the normal guarded
        compensation mutation.  A changed or repurposed child is refused.
        """
        if request.operation is not MutationOperation.CHILD_IMPORT or not isinstance(request.payload, ChildImportPayload):
            raise TypeError("imported-child compensation requires a child-import request")
        child_result = self._uow.repository.by_uuid(request.payload.child_uuid, refresh=True)
        if isinstance(child_result, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if child_result.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return self._outcome(request, kind, reason=child_result.evidence.detail, failure=child_result.evidence)
        if isinstance(child_result, Absent):
            return self._outcome(
                request,
                MutationOutcomeKind.ALREADY_APPLIED,
                postcondition=MutationPostcondition.CHILD_COMPENSATED,
                reason="child is already absent",
            )
        if not isinstance(child_result, Found) or not isinstance(child_result.value, TaskObservation):
            return self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid child compensation read result")
        child = child_result.value
        if not _child_import_matches(child, request.payload, request.guard.task_uuid):
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="child changed before compensation")
        mapping = child.to_mapping()
        status = _text(_observed_value(child, "status")).lower()
        chain_id = _text(_observed_value(child, "chainID"))
        link_text = _link_text(_observed_value(child, "link"))
        try:
            link = int(link_text)
            timestamp_value = _text(_observed_value(child, "modified")) or _text(_observed_value(child, "end"))
            guard = MutationGuard(
                task_uuid=request.payload.child_uuid,
                status=status,
                chain_id=chain_id,
                link=link,
                recurrence_identity=recurrence_fingerprint(mapping),
                timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, timestamp_value),),
                expected_mutation_epoch=self._uow.mutation_epoch,
                chain=_text(_observed_value(child, "chain")).lower() or "on",
            )
            compensation = MutationRequest(
                MutationOperation.CHILD_COMPENSATION,
                guard,
                ChildCompensationPayload(request.payload.child_uuid, expected_status=status),
            )
        except (IntegrationContractError, TypeError, ValueError) as exc:
            return self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason=f"child compensation guard unavailable: {exc}")
        return self.compensate_child(compensation)

    def import_child(self, request: MutationRequest, *, verify: bool = True) -> MutationOutcome:
        if request.operation is not MutationOperation.CHILD_IMPORT or not isinstance(request.payload, ChildImportPayload):
            raise TypeError("child import requires a child-import request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        if _text(_observed_value(parent, "chain")).lower() != "on":
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="parent chain is no longer active")
        existing = self._prefetched_children.pop(request.payload.child_uuid.lower(), None)
        prefetched = existing is not None
        if existing is None:
            existing = self._uow.repository.by_uuid(request.payload.child_uuid, refresh=True)
        if isinstance(existing, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if existing.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return self._outcome(request, kind, reason=existing.evidence.detail, failure=existing.evidence)
        if isinstance(existing, Found):
            row = existing.value
            if _child_import_matches(row, request.payload, request.guard.task_uuid):
                # A present child from the batch preflight must still pass
                # the phase-wide authoritative verification. This avoids a
                # per-child reread during recovery without trusting stale
                # identity evidence across the mutation boundary.
                if prefetched and not verify:
                    return self._outcome(request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED)
                return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED)
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="unrelated task already owns child UUID")
        if not isinstance(existing, Absent):
            return self._outcome(request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid child lookup result")
        result = self._uow.client.execute(
            ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            purpose="lifecycle child import",
            timeout=self._timeout,
            input_text=DEFAULT_TASK_CODEC.encode_task_import_mapping(request.payload.to_dict()) + "\n",
            attempts=1,
        )
        if not result.ok:
            self._uow.record_mutation(uncertain=True)
            return self._command_failure(request, result)
        self._uow.record_mutation(uncertain=True)
        if not verify:
            return self._outcome(request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED)
        return self._verify(
            request,
            MutationPostcondition.CHILD_IMPORTED,
            lambda row: _child_import_matches(row, request.payload, request.guard.task_uuid),
            target_uuid=request.payload.child_uuid,
        )

    def link_parent(self, request: MutationRequest, *, verify: bool = True) -> MutationOutcome:
        if request.operation is not MutationOperation.PARENT_LINK or not isinstance(request.payload, ParentLinkPayload):
            raise TypeError("parent linking requires a parent-link request")
        cached_parent = self._prefetched_parents.pop(request.guard.task_uuid.lower(), None)
        parent: TaskObservation | None
        failure: MutationOutcome | None
        if cached_parent is not None:
            mismatch = self._guard_mismatch(request.guard, cached_parent)
            if mismatch:
                parent, failure = None, self._outcome(request, MutationOutcomeKind.CONFLICT, reason=mismatch)
            else:
                parent, failure = cached_parent, None
        else:
            parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        if _text(_observed_value(parent, "chain")).lower() != "on":
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="parent chain is no longer active")
        current = _text(_observed_value(parent, "nextLink"))
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
        if not verify:
            return self._outcome(request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.PARENT_LINKED)
        return self._verify(
            request,
            MutationPostcondition.PARENT_LINKED,
            lambda row: _text(_observed_value(row, "nextLink")).casefold() == request.payload.child_short_uuid.casefold(),
        )

    def apply_lifecycle_unverified(self, request: MutationRequest) -> MutationOutcome:
        """Apply a child-import or parent-link command for batch verification.

        Guards and mutation commands remain per-intent.  Only the successful
        postcondition read is deferred until the lifecycle batch has completed
        its mutation phase; a crash leaves the durable stage unchanged and the
        next drain re-checks the task idempotently.
        """
        if request.operation is MutationOperation.CHILD_IMPORT:
            return self.import_child(request, verify=False)
        if request.operation is MutationOperation.PARENT_LINK:
            return self.link_parent(request, verify=False)
        raise TypeError("batch lifecycle mutation supports child import and parent link only")

    def apply_lifecycle_children_unverified(
        self,
        requests: Sequence[MutationRequest],
    ) -> dict[str, MutationOutcome]:
        """Import one lifecycle wave with one Taskwarrior command.

        Parent guards and deterministic child identities are still checked per
        request.  Only the external import command is combined; the following
        authoritative child-set verification remains per-wave in
        ``verify_lifecycle_children``.
        """
        pending = tuple(requests)
        if any(
            request.operation is not MutationOperation.CHILD_IMPORT
            or not isinstance(request.payload, ChildImportPayload)
            for request in pending
        ):
            raise TypeError("batch child import requires child-import requests")
        outcomes: dict[str, MutationOutcome] = {}
        imports: list[MutationRequest] = []
        for request in pending:
            payload = cast(ChildImportPayload, request.payload)
            parent, failure = self._read_target(request)
            if failure is not None:
                outcomes[payload.child_uuid.lower()] = failure
                continue
            assert parent is not None
            if _text(_observed_value(parent, "chain")).lower() != "on":
                outcomes[payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.CONFLICT, reason="parent chain is no longer active"
                )
                continue
            existing = self._prefetched_children.pop(payload.child_uuid.lower(), None)
            if existing is None:
                existing = self._uow.repository.by_uuid(payload.child_uuid, refresh=True)
            if isinstance(existing, Unavailable):
                kind = MutationOutcomeKind.RETRYABLE if existing.retryable else MutationOutcomeKind.MANUAL_REVIEW
                outcomes[payload.child_uuid.lower()] = self._outcome(
                    request, kind, reason=existing.evidence.detail, failure=existing.evidence
                )
            elif isinstance(existing, Found):
                if _child_import_matches(existing.value, payload, request.guard.task_uuid):
                    outcomes[payload.child_uuid.lower()] = self._outcome(
                        request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED
                    )
                else:
                    outcomes[payload.child_uuid.lower()] = self._outcome(
                        request, MutationOutcomeKind.CONFLICT, reason="unrelated task already owns child UUID"
                    )
            elif isinstance(existing, Absent):
                imports.append(request)
            else:
                outcomes[payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid child lookup result"
                )
        if not imports:
            return outcomes
        input_text = "".join(
            DEFAULT_TASK_CODEC.encode_task_import_mapping(cast(ChildImportPayload, request.payload).to_dict()) + "\n"
            for request in imports
        )
        result = self._uow.client.execute(
            ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            purpose="lifecycle child import batch",
            timeout=self._timeout,
            input_text=input_text,
            attempts=1,
        )
        if not result.ok:
            self._uow.record_mutation(uncertain=True)
            for request in imports:
                outcomes[cast(ChildImportPayload, request.payload).child_uuid.lower()] = self._command_failure(
                    request, result
                )
            return outcomes
        self._uow.record_mutation(uncertain=True)
        for request in imports:
            payload = cast(ChildImportPayload, request.payload)
            outcomes[payload.child_uuid.lower()] = self._outcome(
                request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED
            )
        return outcomes

    def verify_lifecycle_children(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        """Verify imported children with one authoritative UUID-set snapshot."""
        pending = tuple(
            request
            for request in requests
            if request.operation is MutationOperation.CHILD_IMPORT
            and isinstance(request.payload, ChildImportPayload)
        )
        if not pending:
            return {}
        identities = tuple(cast(ChildImportPayload, request.payload).child_uuid for request in pending)
        try:
            result = self._uow.repository.read_uuid_set(
                UUIDSetRequest(identities, refresh=True, expected_mutation_epoch=self._uow.mutation_epoch)
            )
        except (TypeError, ValueError) as exc:
            result = None
            failure_reason = str(exc)
        if result is None or result.status is not SetReadStatus.COMPLETE:
            evidence = result.failures[0] if result is not None and result.failures else None
            reason = (evidence.detail if evidence is not None else ("child postcondition set read unavailable" if result is not None else failure_reason))
            kind = MutationOutcomeKind.RETRYABLE if evidence is not None and evidence.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return {
                cast(ChildImportPayload, request.payload).child_uuid.lower(): self._outcome(
                    request, kind, reason=reason, failure=evidence
                )
                for request in pending
            }
        outcomes: dict[str, MutationOutcome] = {}
        for request in pending:
            assert isinstance(request.payload, ChildImportPayload)
            match = result.found.get(request.payload.child_uuid.lower())
            if match is not None and _child_import_matches(match, request.payload, request.guard.task_uuid):
                outcomes[request.payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED
                )
            else:
                reason = "child postcondition target is absent" if match is None else "child postcondition does not match"
                outcomes[request.payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason=reason
                )
        return outcomes

    def verify_lifecycle_parents(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        """Verify parent links with one authoritative UUID-set snapshot."""
        pending = tuple(
            request
            for request in requests
            if request.operation is MutationOperation.PARENT_LINK
            and isinstance(request.payload, ParentLinkPayload)
        )
        if not pending:
            return {}
        identities = tuple(request.guard.task_uuid for request in pending)
        try:
            result = self._uow.repository.read_uuid_set(
                UUIDSetRequest(identities, refresh=True, expected_mutation_epoch=self._uow.mutation_epoch)
            )
        except (TypeError, ValueError) as exc:
            result = None
            failure_reason = str(exc)
        if result is None or result.status is not SetReadStatus.COMPLETE:
            evidence = result.failures[0] if result is not None and result.failures else None
            reason = (evidence.detail if evidence is not None else ("parent postcondition set read unavailable" if result is not None else failure_reason))
            kind = MutationOutcomeKind.RETRYABLE if evidence is not None and evidence.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return {
                request.guard.task_uuid.lower(): self._outcome(
                    request, kind, reason=reason, failure=evidence
                )
                for request in pending
            }
        outcomes: dict[str, MutationOutcome] = {}
        for request in pending:
            assert isinstance(request.payload, ParentLinkPayload)
            match = result.found.get(request.guard.task_uuid.lower())
            if match is not None and _text(_observed_value(match, "nextLink")).casefold() == request.payload.child_short_uuid.casefold():
                outcomes[request.guard.task_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.PARENT_LINKED
                )
            else:
                reason = "parent postcondition target is absent" if match is None else "parent postcondition does not match"
                outcomes[request.guard.task_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason=reason
                )
        return outcomes

    def disable_chain(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.CHAIN_DISABLE or not isinstance(request.payload, ChainDisablePayload):
            raise TypeError("chain disablement requires a chain-disable request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        if _text(_observed_value(parent, "chain")).lower() == "off":
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.CHAIN_DISABLED)
        failure = self._run_modify(request, self._selectors(request.guard), ("chain:off",))
        if failure is not None:
            return failure
        return self._verify(request, MutationPostcondition.CHAIN_DISABLED, lambda row: _text(_observed_value(row, "chain")).lower() == "off")

    def clear_parent_link(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.PARENT_LINK_CLEAR or not isinstance(request.payload, ParentLinkClearPayload):
            raise TypeError("parent-link clearing requires a parent-link-clear request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        current = _text(_observed_value(parent, "nextLink"))
        if not current:
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.PARENT_LINK_CLEARED)
        if current.casefold() != request.payload.expected_next_link.casefold():
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="parent nextLink changed")
        failure = self._run_modify(
            request,
            self._selectors(request.guard, extra=(f"nextLink:{current}",)),
            ("nextLink:",),
        )
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.PARENT_LINK_CLEARED,
            lambda row: not _text(_observed_value(row, "nextLink")),
        )

    def repair_native_until(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.NATIVE_UNTIL_REPAIR or not isinstance(request.payload, NativeUntilRepairPayload):
            raise TypeError("native-until repair requires a native-until request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        current = _text(_observed_value(parent, "until"))
        if timestamp_equal(current, request.payload.replacement_until):
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.NATIVE_UNTIL_REPAIRED)
        if not timestamp_equal(current, request.payload.expected_until):
            return self._outcome(request, MutationOutcomeKind.CONFLICT, reason="native until changed")
        failure = self._run_modify(
            request,
            self._selectors(request.guard, extra=(f"until:{current}",), include_modified=False),
            (f"until:{request.payload.replacement_until}",),
        )
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.NATIVE_UNTIL_REPAIRED,
            lambda row: timestamp_equal(_observed_value(row, "until"), request.payload.replacement_until),
        )

    def repair_metadata(self, request: MutationRequest) -> MutationOutcome:
        if request.operation is not MutationOperation.METADATA_REPAIR or not isinstance(request.payload, MetadataRepairPayload):
            raise TypeError("metadata repair requires a metadata request")
        parent, failure = self._read_target(request)
        if failure is not None:
            return failure
        assert parent is not None
        updates = request.payload.to_dict()
        if all(_text(_observed_value(parent, key)) == _text(value) for key, value in updates.items()):
            return self._outcome(request, MutationOutcomeKind.ALREADY_APPLIED, postcondition=MutationPostcondition.METADATA_REPAIRED)
        expected = request.payload.expected_dict()
        for key, value in expected.items():
            if _text(_observed_value(parent, key)) != _text(value):
                return self._outcome(request, MutationOutcomeKind.CONFLICT, reason=f"metadata field {key} changed")
        failure = self._run_modify(
            request,
            self._selectors(
                request.guard,
                include_link=not (
                    "link" in request.payload.expected_dict()
                    and request.payload.expected_dict().get("link") == ""
                ),
            ),
            tuple(f"{key}:{value}" for key, value in updates.items()),
        )
        if failure is not None:
            return failure
        return self._verify(
            request,
            MutationPostcondition.METADATA_REPAIRED,
            lambda row: all(_text(_observed_value(row, key)) == _text(value) for key, value in updates.items()),
        )

    @staticmethod
    def _selectors(
        guard: MutationGuard,
        *,
        extra: Sequence[str] = (),
        include_link: bool = True,
        include_modified: bool = True,
    ) -> tuple[str, ...]:
        selectors: tuple[str, ...] = (
            f"uuid:{guard.task_uuid}",
            f"status:{guard.status}",
            f"chain:{guard.chain}",
            f"chainID:{guard.chain_id}",
        )
        if include_link:
            selectors = (*selectors, f"link:{guard.link}")
        modified = next(
            (timestamp.value for timestamp in guard.timestamps if timestamp.field.value == "modified"),
            "",
        )
        # Native-until repairs already include an exact ``until`` selector.
        # Taskwarrior rewrites ``modified`` as part of that same update, so a
        # stale timestamp selector can reject an otherwise safe repair after a
        # fresh guard read. Identity plus expected-until remains the atomic
        # mutation boundary for this operation.
        if modified and include_modified:
            selectors = (*selectors, f"modified:{modified}")
        end = next(
            (timestamp.value for timestamp in guard.timestamps if timestamp.field.value == "end"),
            "",
        )
        if end:
            selectors = (*selectors, f"end:{end}")
        return (*selectors, *tuple(extra))


__all__ = ("TaskwarriorMutationService",)
