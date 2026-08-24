"""Guarded, named Taskwarrior mutations for lifecycle transitions.

This module owns the write side of the integration boundary.  Callers supply
validated mutation requests; they never provide arbitrary Taskwarrior argv.
Every operation re-reads its target, applies a narrow selector, and verifies
the requested postcondition before reporting success.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, Sequence, cast

from .integration_models import (
    Absent,
    ChainDisablePayload,
    ChildCompensationPayload,
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


class _TaskRepository(Protocol):
    def by_uuid(self, uuid_value: str, *, refresh: bool = False) -> TaskRead[TaskObservation]: ...


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

    def prefetch_lifecycle_batch(
        self,
        payloads: Sequence[ChildImportPayload],
        *,
        parent_expectations: Sequence[tuple[str, str]] = (),
    ) -> None:
        """Preload safe child-absence decisions for one drain.

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
        broad_snapshot = getattr(self._uow.repository, "broad_snapshot", None)
        if not callable(broad_snapshot):
            return
        try:
            read = broad_snapshot(
                identity="lifecycle-child-prefetch",
                filters=("chain:on",),
                statuses=("completed", "deleted", "pending", "recurring", "waiting"),
                complete_chain_history=False,
                refresh=True,
            )
        except Exception:
            return
        if isinstance(read, Absent):
            for payload in wanted:
                self._prefetched_children[payload.child_uuid.lower()] = read
            return
        if not isinstance(read, Found):
            return
        snapshot = read.value
        uuid_matches = getattr(snapshot, "uuid_matches", None)
        if not callable(uuid_matches):
            return
        for payload in wanted:
            try:
                matches = tuple(uuid_matches(payload.child_uuid))
            except Exception:
                continue
            if not matches:
                self._prefetched_children[payload.child_uuid.lower()] = Absent(
                    f"prefetch:uuid:{payload.child_uuid.lower()}", "prefetch snapshot contains no matching UUID"
                )
        for parent_uuid, expected_next_link in parent_wanted:
            try:
                matches = tuple(uuid_matches(parent_uuid))
            except Exception:
                continue
            if len(matches) == 1:
                row = matches[0]
                # A cached row that already has the requested link is not
                # safe to trust across the batch; retain a fresh read for
                # idempotent classification.
                if _text(_observed_value(row, "nextLink")).casefold() != expected_next_link.casefold():
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
        handlers = {
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
        if existing is None:
            existing = self._uow.repository.by_uuid(request.payload.child_uuid, refresh=True)
        if isinstance(existing, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if existing.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return self._outcome(request, kind, reason=existing.evidence.detail, failure=existing.evidence)
        if isinstance(existing, Found):
            row = existing.value
            if _child_import_matches(row, request.payload, request.guard.task_uuid):
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
        if cached_parent is not None and _text(_observed_value(cached_parent, "nextLink")).casefold() != request.payload.child_short_uuid.casefold():
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

    def verify_lifecycle_children(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        """Verify imported children with one authoritative broad snapshot."""
        pending = tuple(
            request
            for request in requests
            if request.operation is MutationOperation.CHILD_IMPORT
            and isinstance(request.payload, ChildImportPayload)
        )
        if not pending:
            return {}
        broad_snapshot = getattr(self._uow.repository, "broad_snapshot", None)
        if not callable(broad_snapshot):
            return {
                cast(ChildImportPayload, request.payload).child_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="child batch verification is unavailable"
                )
                for request in pending
            }
        read = broad_snapshot(
            identity="lifecycle-child-postverify",
            filters=("chain:on",),
            statuses=("completed", "deleted", "pending", "recurring", "waiting"),
            complete_chain_history=False,
            refresh=True,
        )
        if isinstance(read, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if read.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return {
                cast(ChildImportPayload, request.payload).child_uuid.lower(): self._outcome(
                    request, kind, reason=read.evidence.detail, failure=read.evidence
                )
                for request in pending
            }
        if isinstance(read, Absent):
            return {
                cast(ChildImportPayload, request.payload).child_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="child postcondition snapshot is absent"
                )
                for request in pending
            }
        if not isinstance(read, Found):
            return {
                cast(ChildImportPayload, request.payload).child_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid child postcondition snapshot"
                )
                for request in pending
            }
        snapshot = read.value
        uuid_matches = getattr(snapshot, "uuid_matches", None)
        if not callable(uuid_matches):
            return {
                cast(ChildImportPayload, request.payload).child_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="malformed child postcondition snapshot"
                )
                for request in pending
            }
        outcomes: dict[str, MutationOutcome] = {}
        for request in pending:
            assert isinstance(request.payload, ChildImportPayload)
            try:
                matches = tuple(uuid_matches(request.payload.child_uuid))
            except Exception as exc:
                outcomes[request.payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason=f"malformed child postcondition snapshot: {exc}"
                )
                continue
            if len(matches) == 1 and _child_import_matches(matches[0], request.payload, request.guard.task_uuid):
                outcomes[request.payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.CHILD_IMPORTED
                )
            else:
                reason = "child postcondition target is absent" if not matches else "child postcondition does not match"
                outcomes[request.payload.child_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason=reason
                )
        return outcomes

    def verify_lifecycle_parents(self, requests: Sequence[MutationRequest]) -> dict[str, MutationOutcome]:
        """Verify parent links with one authoritative broad snapshot."""
        pending = tuple(
            request
            for request in requests
            if request.operation is MutationOperation.PARENT_LINK
            and isinstance(request.payload, ParentLinkPayload)
        )
        if not pending:
            return {}
        broad_snapshot = getattr(self._uow.repository, "broad_snapshot", None)
        if not callable(broad_snapshot):
            return {
                request.guard.task_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="parent batch verification is unavailable"
                )
                for request in pending
            }
        read = broad_snapshot(
            identity="lifecycle-parent-postverify",
            filters=("chain:on",),
            statuses=("completed", "deleted", "pending", "recurring", "waiting"),
            complete_chain_history=False,
            refresh=True,
        )
        if isinstance(read, Unavailable):
            kind = MutationOutcomeKind.RETRYABLE if read.retryable else MutationOutcomeKind.MANUAL_REVIEW
            return {
                request.guard.task_uuid.lower(): self._outcome(
                    request, kind, reason=read.evidence.detail, failure=read.evidence
                )
                for request in pending
            }
        if isinstance(read, Absent):
            return {
                request.guard.task_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="parent postcondition snapshot is absent"
                )
                for request in pending
            }
        if not isinstance(read, Found):
            return {
                request.guard.task_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="invalid parent postcondition snapshot"
                )
                for request in pending
            }
        snapshot = read.value
        uuid_matches = getattr(snapshot, "uuid_matches", None)
        if not callable(uuid_matches):
            return {
                request.guard.task_uuid.lower(): self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason="malformed parent postcondition snapshot"
                )
                for request in pending
            }
        outcomes: dict[str, MutationOutcome] = {}
        for request in pending:
            assert isinstance(request.payload, ParentLinkPayload)
            try:
                matches = tuple(uuid_matches(request.guard.task_uuid))
            except Exception as exc:
                outcomes[request.guard.task_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.MANUAL_REVIEW, reason=f"malformed parent postcondition snapshot: {exc}"
                )
                continue
            if len(matches) == 1 and _text(_observed_value(matches[0], "nextLink")).casefold() == request.payload.child_short_uuid.casefold():
                outcomes[request.guard.task_uuid.lower()] = self._outcome(
                    request, MutationOutcomeKind.APPLIED, postcondition=MutationPostcondition.PARENT_LINKED
                )
            else:
                reason = "parent postcondition target is absent" if not matches else "parent postcondition does not match"
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
        return (*selectors, *tuple(extra))


__all__ = ("TaskwarriorMutationService",)
