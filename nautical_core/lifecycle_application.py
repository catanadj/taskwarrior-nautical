"""One production lifecycle application service.

This module owns the single staging path and the single execution path for
lifecycle transitions produced by the scheduler and lifecycle planner. It
does not schedule or plan anything itself; it only accepts already-validated
``LifecyclePlan`` objects and applies them through the guarded Taskwarrior
mutation gateway (``taskwarrior_mutations``).

Two shapes of transition exist, and they are handled differently on purpose:

* ``SPAWN_CHILD`` plans require two ordered, externally-visible mutations
  (import the child, then link the parent to it). These go through the
  durable lifecycle outbox (``lifecycle_outbox``) so a crash between the two
  mutations resumes from the last verified stage instead of repeating or
  losing work.
* ``DISABLE_CHAIN``, ``FINALIZE_CHAIN``, and ``UPDATE_PARENT`` plans are one
  guarded, self-verifying mutation. They apply immediately against the
  mutation gateway and report their outcome inline; there is no meaningful
  intermediate stage to persist, so they do not use outbox staging.

Both paths are driven by the same invocation-scoped unit of work and the
same ``TaskwarriorMutationPort``, so on-exit and reconcile calling this
service see identical outcomes from identical Taskwarrior state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import time
from typing import Any, Callable, Mapping, Protocol, cast

from .integration_models import (
    ChainDisablePayload,
    ChildImportPayload,
    GuardTimestamp,
    GuardTimestampField,
    IntegrationContractError,
    MetadataRepairPayload,
    MutationGuard,
    MutationOperation,
    MutationOutcome,
    MutationOutcomeKind,
    MutationRequest,
    ParentLinkPayload,
    TaskwarriorMutationPort,
)
from .lifecycle_models import (
    ExecutionStage,
    LifecycleAction,
    LifecycleDrainProgress,
    LifecycleDrainProgressCallback,
    LifecycleDrainStage,
    LifecycleIdentity,
    LifecyclePlan,
)
from .task_codec import TaskCodec
from .lifecycle_outbox import (
    LifecycleOutboxRecord,
    LifecycleOutboxRepository,
    OutboxFailure,
    OutboxResult,
    OutboxResultKind,
)


class LifecycleApplicationError(RuntimeError):
    """Raised when the lifecycle application service is misused."""


class LifecycleApplicationOutcomeKind(str, Enum):
    APPLIED = "applied"
    ALREADY_APPLIED = "already_applied"
    RETRYABLE = "retryable"
    CONFLICT = "conflict"
    MANUAL_REVIEW = "manual_review"
    QUARANTINED = "quarantined"
    NOOP = "noop"


@dataclass(frozen=True, slots=True)
class LifecycleApplicationOutcome:
    """Tagged result of one staging or execution call."""

    kind: LifecycleApplicationOutcomeKind
    identity: LifecycleIdentity
    reason: str = ""
    intent_id: str = ""
    mutations: tuple[MutationOutcome, ...] = ()

    def __post_init__(self) -> None:
        try:
            kind = LifecycleApplicationOutcomeKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise LifecycleApplicationError("invalid lifecycle application outcome kind") from exc
        if not isinstance(self.identity, LifecycleIdentity):
            raise LifecycleApplicationError("lifecycle application outcome requires a LifecycleIdentity")
        try:
            mutations = tuple(self.mutations)
        except TypeError as exc:
            raise LifecycleApplicationError("lifecycle application outcome requires typed mutation outcomes") from exc
        if any(not isinstance(item, MutationOutcome) for item in mutations):
            raise LifecycleApplicationError("lifecycle application outcome requires typed mutation outcomes")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        object.__setattr__(self, "mutations", mutations)

    @property
    def ok(self) -> bool:
        return self.kind in {
            LifecycleApplicationOutcomeKind.APPLIED,
            LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
            LifecycleApplicationOutcomeKind.NOOP,
        }


@dataclass(frozen=True, slots=True)
class DrainResult:
    """Outcome of one bounded drain pass: what was claimed, and how it went."""

    claim: OutboxResult
    outcomes: tuple[LifecycleApplicationOutcome, ...] = ()


@dataclass(slots=True)
class _BatchState:
    record: LifecycleOutboxRecord
    plan: LifecyclePlan
    child_payload: ChildImportPayload
    link_payload: ParentLinkPayload
    stage: ExecutionStage
    mutations: list[MutationOutcome]
    terminal: LifecycleApplicationOutcome | None = None
    progress_completed: int = 0


class _UnitOfWork(Protocol):
    mutation_epoch: int


# Durable stage progression used only for SPAWN_CHILD outbox intents. Mirrors
# lifecycle_outbox._STAGE_ORDER; kept local because that mapping is private
# to the outbox repository and this module only needs the ordering, not the
# repository's own bookkeeping.
_SPAWN_STAGE_ORDER = {
    ExecutionStage.PLANNED: 0,
    ExecutionStage.PERSISTED: 0,
    ExecutionStage.CHILD_PRESENT: 1,
    ExecutionStage.PARENT_LINKED: 2,
    ExecutionStage.VERIFIED: 3,
    ExecutionStage.FINALIZED: 4,
}


def _remaining_drain_work(stage: ExecutionStage) -> int:
    """Return observable work units remaining for one claimed spawn intent."""
    stage_order = _SPAWN_STAGE_ORDER[stage]
    if stage_order < _SPAWN_STAGE_ORDER[ExecutionStage.CHILD_PRESENT]:
        return 6
    if stage_order < _SPAWN_STAGE_ORDER[ExecutionStage.PARENT_LINKED]:
        return 4
    if stage_order < _SPAWN_STAGE_ORDER[ExecutionStage.VERIFIED]:
        return 2
    return 1

_TERMINAL_ACTIONS = (LifecycleAction.DISABLE_CHAIN, LifecycleAction.FINALIZE_CHAIN)

_OUTBOX_TO_APPLICATION = {
    OutboxResultKind.APPLIED: LifecycleApplicationOutcomeKind.APPLIED,
    OutboxResultKind.ALREADY_APPLIED: LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
    OutboxResultKind.RETRYABLE: LifecycleApplicationOutcomeKind.RETRYABLE,
    OutboxResultKind.CONFLICT: LifecycleApplicationOutcomeKind.CONFLICT,
    OutboxResultKind.REJECTED: LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
}

_MUTATION_TO_APPLICATION = {
    MutationOutcomeKind.APPLIED: LifecycleApplicationOutcomeKind.APPLIED,
    MutationOutcomeKind.ALREADY_APPLIED: LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
    MutationOutcomeKind.RETRYABLE: LifecycleApplicationOutcomeKind.RETRYABLE,
    MutationOutcomeKind.CONFLICT: LifecycleApplicationOutcomeKind.CONFLICT,
    MutationOutcomeKind.REJECTED: LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
    MutationOutcomeKind.MANUAL_REVIEW: LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
}


def _mutation_guard(plan: LifecyclePlan, *, mutation_epoch: int) -> MutationGuard | None:
    """Build a fresh, invocation-scoped guard from a plan's durable ParentGuard."""
    guard = plan.parent_guard
    if not guard.modified:
        return None
    try:
        return MutationGuard(
            task_uuid=plan.identity.parent_uuid,
            status=guard.status,
            chain_id=guard.chain_id,
            link=guard.link,
            recurrence_identity=guard.recurrence_fingerprint,
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, guard.modified),),
            expected_mutation_epoch=mutation_epoch,
            chain=guard.chain,
        )
    except IntegrationContractError:
        return None


def _child_import_payload(plan: LifecyclePlan) -> ChildImportPayload | None:
    from .task_codec import DEFAULT_TASK_CODEC
    from .task_models import NauticalTask, TaskDraft

    child = plan.child_dict()
    if not child:
        return None
    for field in ("anchor", "anchor_file", "omit", "omit_file", "cp", "chainMax", "chainUntil", "bc"):
        if field not in child:
            continue
        value = TaskCodec.normalize_text(child.get(field))
        if value:
            child[field] = value
        else:
            child.pop(field, None)
    if "anchor_mode" in child:
        child["anchor_mode"] = TaskCodec.normalize_text(child.get("anchor_mode")) or "skip"
    try:
        task = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(child, source_query="lifecycle child import")
        )
        draft = TaskDraft.from_task(task)
        return ChildImportPayload.from_draft(draft, parent_uuid=plan.identity.parent_uuid)
    except (IntegrationContractError, TypeError, ValueError):
        return None


def _parent_link_payload(plan: LifecyclePlan) -> ParentLinkPayload | None:
    next_link = str(plan.parent_patch_dict().get("nextLink") or "").strip()
    if not next_link:
        return None
    try:
        return ParentLinkPayload(parent_uuid=plan.identity.parent_uuid, child_short_uuid=next_link)
    except IntegrationContractError:
        return None


def _chain_disable_payload(plan: LifecyclePlan) -> ChainDisablePayload | None:
    try:
        return ChainDisablePayload(task_uuid=plan.identity.parent_uuid)
    except IntegrationContractError:
        return None


def _update_parent_payload(plan: LifecyclePlan) -> MetadataRepairPayload | None:
    updates = plan.parent_patch_dict()
    if not updates:
        return None
    try:
        from .integration_models import _freeze_pairs

        return MetadataRepairPayload(plan.identity.parent_uuid, _freeze_pairs(updates))
    except IntegrationContractError:
        return None


def _validate_spawn_plan(plan: LifecyclePlan) -> str:
    """Return an empty string if a SPAWN_CHILD plan is well-formed, else why not."""
    if not plan.parent_guard.modified:
        return "lifecycle plan is missing its parent modified guard timestamp"
    if _child_import_payload(plan) is None:
        return "lifecycle plan child payload is missing or malformed"
    if _parent_link_payload(plan) is None:
        return "lifecycle plan is missing its parent nextLink patch"
    return ""


def _plan_difference_summary(
    existing: LifecyclePlan,
    requested: LifecyclePlan,
    *,
    existing_config: str = "",
    requested_config: str = "",
    existing_schedule: str = "",
    requested_schedule: str = "",
) -> str:
    """Describe immutable intent differences without dumping full task JSON."""
    differences: list[str] = []
    existing_payload = existing.compatibility_payload()
    requested_payload = requested.compatibility_payload()
    sections = (
        ("identity", existing_payload.get("identity", {}), requested_payload.get("identity", {})),
        ("child", existing_payload.get("child_payload", {}), requested_payload.get("child_payload", {})),
        ("parent", existing_payload.get("parent_patch", {}), requested_payload.get("parent_patch", {})),
        ("guard", existing_payload.get("parent_guard", {}), requested_payload.get("parent_guard", {})),
    )
    for section, old_values, new_values in sections:
        keys = sorted(set(old_values) | set(new_values))
        for key in keys:
            old = old_values.get(key, "<absent>")
            new = new_values.get(key, "<absent>")
            if type(old) is not type(new) or old != new:
                differences.append(f"{section}.{key}={old!r} -> {new!r}")
    for field in ("action", "expected_postconditions"):
        old = existing_payload.get(field, "<absent>")
        new = requested_payload.get(field, "<absent>")
        if type(old) is not type(new) or old != new:
            differences.append(f"{field}={old!r} -> {new!r}")
    if existing_config != requested_config:
        differences.append(f"configuration_fingerprint={existing_config!r} -> {requested_config!r}")
    if existing_schedule != requested_schedule:
        differences.append(f"schedule_fingerprint={existing_schedule!r} -> {requested_schedule!r}")
    return "; ".join(differences) or "no immutable field difference was found"


def _from_outbox_result(
    result: OutboxResult,
    identity: LifecycleIdentity,
    *,
    requested: LifecyclePlan | None = None,
    requested_config: str = "",
    requested_schedule: str = "",
) -> LifecycleApplicationOutcome:
    kind = _OUTBOX_TO_APPLICATION.get(result.kind, LifecycleApplicationOutcomeKind.MANUAL_REVIEW)
    intent_id = result.record.intent_id if result.record is not None else ""
    reason = result.reason
    if result.record is not None and requested is not None and not result.ok:
        changed = _plan_difference_summary(
            result.record.plan,
            requested,
            existing_config=result.record.configuration_fingerprint,
            requested_config=requested_config,
            existing_schedule=result.record.schedule_fingerprint,
            requested_schedule=requested_schedule,
        )
        reason = f"{reason}; changed fields: {changed}"
    return LifecycleApplicationOutcome(kind, identity, reason=reason, intent_id=intent_id)


class LifecycleApplicationService:
    """Stage typed SPAWN_CHILD plans into the outbox; execute every plan kind
    through one guarded mutation gateway and one invocation-scoped unit of
    work. This is the only production lifecycle staging and execution path;
    on-exit and reconcile are both expected to call this service rather than
    maintaining their own copies of this orchestration.

    ``unit_of_work`` and ``mutations`` are optional because staging has a
    legitimate caller that must not have them: on-modify builds and stages
    plans while Taskwarrior still holds its datastore lock for the task
    being modified, and deliberately avoids constructing a command-capable
    unit of work to reduce the risk of re-entering Taskwarrior from inside
    the hook. A service constructed without them can still call ``stage()``;
    calling ``drain()`` or ``apply_immediate()`` without them raises.
    """

    def __init__(
        self,
        *,
        unit_of_work: _UnitOfWork | None = None,
        mutations: TaskwarriorMutationPort | None = None,
        outbox: LifecycleOutboxRepository,
        owner: str = "",
        lease_seconds: float = 30.0,
    ) -> None:
        self._uow = unit_of_work
        self._mutations = mutations
        self._outbox = outbox
        self._owner = str(owner or "").strip() or f"pid-{os.getpid()}"
        self._lease_seconds = max(1.0, float(lease_seconds))

    def _require_execution_deps(self) -> None:
        if self._uow is None or self._mutations is None:
            raise LifecycleApplicationError(
                "executing lifecycle intents requires a unit_of_work and a mutation gateway; "
                "this service was constructed for staging only"
            )

    # -- staging: SPAWN_CHILD plans only -------------------------------

    def stage(
        self,
        plan: LifecyclePlan,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
    ) -> LifecycleApplicationOutcome:
        """Persist one typed lifecycle plan into the durable outbox.

        Only SPAWN_CHILD plans use durable staging; NOOP is reported inline
        with no side effect, and every other action is a single guarded
        mutation that belongs in ``apply_immediate`` instead.
        """
        if not isinstance(plan, LifecyclePlan):
            raise LifecycleApplicationError("staging requires a validated LifecyclePlan")
        if plan.action is LifecycleAction.NOOP:
            return LifecycleApplicationOutcome(LifecycleApplicationOutcomeKind.NOOP, plan.identity)
        if plan.action is not LifecycleAction.SPAWN_CHILD:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
                plan.identity,
                reason=f"{plan.action.value} does not use durable outbox staging; call apply_immediate",
            )
        reason = _validate_spawn_plan(plan)
        if reason:
            return LifecycleApplicationOutcome(LifecycleApplicationOutcomeKind.MANUAL_REVIEW, plan.identity, reason=reason)
        try:
            result = self._outbox.enqueue(
                plan,
                configuration_fingerprint=str(configuration_fingerprint or "").strip(),
                schedule_fingerprint=str(schedule_fingerprint or "").strip(),
            )
        except Exception as exc:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.RETRYABLE,
                plan.identity,
                reason=f"outbox persistence failed: {str(exc).strip() or type(exc).__name__}",
            )
        return _from_outbox_result(
            result,
            plan.identity,
            requested=plan,
            requested_config=str(configuration_fingerprint or "").strip(),
            requested_schedule=str(schedule_fingerprint or "").strip(),
        )

    # -- execution: claimed SPAWN_CHILD intents ------------------------

    def drain(
        self,
        *,
        limit: int,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
        progress: LifecycleDrainProgressCallback | None = None,
    ) -> DrainResult:
        """Claim a bounded batch of ready spawn intents and execute each one.

        Bounded by ``limit`` so one invocation cannot be pinned to an
        unbounded recovery pass; whatever is not claimed this round remains
        durable and eligible for the next drain.
        """
        self._require_execution_deps()
        try:
            claim, records = self._outbox.claim_batch(
                owner=self._owner, lease_seconds=self._lease_seconds, limit=max(1, int(limit))
            )
        except Exception as exc:
            claim = OutboxResult(
                OutboxResultKind.RETRYABLE,
                reason=f"outbox claim failed: {str(exc).strip() or type(exc).__name__}",
            )
            return DrainResult(claim=claim, outcomes=())
        if not records:
            return DrainResult(claim=claim, outcomes=())
        total = sum(_remaining_drain_work(record.stage) for record in records)
        self._report_drain_progress(
            progress,
            LifecycleDrainProgress(
                stage=LifecycleDrainStage.CLAIMED,
                completed=0,
                total=total,
                elapsed_seconds=0.0,
            ),
        )
        config = str(configuration_fingerprint or "").strip()
        schedule = str(schedule_fingerprint or "").strip()
        prefetch = getattr(self._mutations, "prefetch_lifecycle_batch", None)
        if callable(prefetch):
            payloads = tuple(
                payload
                for record in records
                if _SPAWN_STAGE_ORDER[record.stage] < _SPAWN_STAGE_ORDER[ExecutionStage.CHILD_PRESENT]
                for payload in (_child_import_payload(record.plan),)
                if payload is not None
            )
            parent_expectations = tuple(
                (record.plan.identity.parent_uuid, str(record.plan.parent_patch_dict().get("nextLink") or "").strip())
                for record in records
                if _SPAWN_STAGE_ORDER[record.stage] < _SPAWN_STAGE_ORDER[ExecutionStage.PARENT_LINKED]
                and str(record.plan.parent_patch_dict().get("nextLink") or "").strip()
            )
            try:
                prefetch(payloads, parent_expectations=parent_expectations)
            except Exception:
                # Prefetch is an optimization only; normal authoritative
                # UUID reads remain the correctness fallback.
                pass
        batch_apply = getattr(self._mutations, "apply_lifecycle_unverified", None)
        batch_children = getattr(self._mutations, "verify_lifecycle_children", None)
        batch_parents = getattr(self._mutations, "verify_lifecycle_parents", None)
        if len(records) > 1 and all(callable(item) for item in (batch_apply, batch_children, batch_parents)):
            return self._drain_batched(
                claim,
                records,
                configuration_fingerprint=config,
                schedule_fingerprint=schedule,
                apply_unverified=batch_apply,
                verify_children=batch_children,
                verify_parents=batch_parents,
                progress=progress,
            )
        outcomes: list[LifecycleApplicationOutcome] = []
        drain_started = time.monotonic()
        completed = 0
        for record in records:
            record_total = _remaining_drain_work(record.stage)
            record_completed = 0

            def report_action(detail: str, units: int = 1) -> None:
                nonlocal completed, record_completed
                advance = min(max(0, int(units)), record_total - record_completed)
                if not advance:
                    return
                completed += advance
                record_completed += advance
                self._report_drain_progress(
                    progress,
                    LifecycleDrainProgress(
                        stage=LifecycleDrainStage.PROCESSING,
                        completed=completed,
                        total=total,
                        intent_id=record.intent_id,
                        detail=detail,
                        elapsed_seconds=time.monotonic() - drain_started,
                    ),
                )

            self._report_drain_progress(
                progress,
                LifecycleDrainProgress(
                    stage=LifecycleDrainStage.PROCESSING,
                    completed=completed,
                    total=total,
                    intent_id=record.intent_id,
                    detail="starting intent",
                    elapsed_seconds=time.monotonic() - drain_started,
                ),
            )
            outcome = self._execute_claimed(
                record,
                configuration_fingerprint=config,
                schedule_fingerprint=schedule,
                progress_action=report_action,
            )
            outcomes.append(outcome)
            report_action("intent finished", record_total - record_completed)
            self._report_drain_progress(
                progress,
                LifecycleDrainProgress(
                    stage=LifecycleDrainStage.COMPLETE,
                    completed=completed,
                    total=total,
                    intent_id=record.intent_id,
                    outcome=outcome.kind.value,
                    detail="intent finished",
                    elapsed_seconds=time.monotonic() - drain_started,
                ),
            )
        return DrainResult(claim=claim, outcomes=tuple(outcomes))

    def _drain_batched(
        self,
        claim: OutboxResult,
        records: tuple[LifecycleOutboxRecord, ...],
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
        apply_unverified: Any,
        verify_children: Any,
        verify_parents: Any,
        progress: LifecycleDrainProgressCallback | None = None,
    ) -> DrainResult:
        """Run multi-intent spawn work in two mutation/verification phases.

        Child imports are followed by one authoritative child snapshot, then
        parent links are followed by one authoritative parent snapshot. A
        successful mutation is not staged until its batch verification passes;
        interruption therefore leaves the durable record recoverable.
        """
        states: list[_BatchState] = []
        outcomes: list[LifecycleApplicationOutcome] = []
        total = sum(_remaining_drain_work(record.stage) for record in records)
        completed = 0
        drain_started = time.monotonic()

        def report_action(state: _BatchState, detail: str, units: int = 1) -> None:
            nonlocal completed
            remaining = _remaining_drain_work(state.record.stage) - state.progress_completed
            advance = min(max(0, int(units)), remaining)
            if not advance:
                return
            state.progress_completed += advance
            completed += advance
            self._report_drain_progress(
                progress,
                LifecycleDrainProgress(
                    stage=LifecycleDrainStage.PROCESSING,
                    completed=completed,
                    total=total,
                    intent_id=state.record.intent_id,
                    detail=detail,
                    elapsed_seconds=time.monotonic() - drain_started,
                ),
            )

        def report_outcome(
            record: LifecycleOutboxRecord,
            outcome: LifecycleApplicationOutcome,
            state: _BatchState | None = None,
        ) -> None:
            nonlocal completed
            record_total = _remaining_drain_work(record.stage)
            accounted = state.progress_completed if state is not None else 0
            completed += max(0, record_total - accounted)
            if state is not None:
                state.progress_completed = record_total
            self._report_drain_progress(
                progress,
                LifecycleDrainProgress(
                    stage=LifecycleDrainStage.COMPLETE,
                    completed=completed,
                    total=total,
                    intent_id=record.intent_id,
                    outcome=outcome.kind.value,
                    detail="intent finished",
                    elapsed_seconds=time.monotonic() - drain_started,
                ),
            )

        for record in records:
            plan = record.plan
            child_payload = _child_import_payload(plan)
            link_payload = _parent_link_payload(plan)
            if (
                record.configuration_fingerprint != configuration_fingerprint
                or record.schedule_fingerprint != schedule_fingerprint
                or plan.action is not LifecycleAction.SPAWN_CHILD
                or child_payload is None
                or link_payload is None
            ):
                outcome = self._manual_review(record, "outbox record is invalid for batched lifecycle execution")
                outcomes.append(outcome)
                report_outcome(record, outcome)
                continue
            states.append(_BatchState(record, plan, child_payload, link_payload, record.stage, []))

        pending_children: list[tuple[_BatchState, MutationRequest]] = []
        for state in states:
            if state.terminal is not None or _SPAWN_STAGE_ORDER[state.stage] >= _SPAWN_STAGE_ORDER[ExecutionStage.CHILD_PRESENT]:
                continue
            lease_failure = self._renew_before_step(state.record, "child import", tuple(state.mutations))
            if lease_failure is not None:
                state.terminal = lease_failure
                continue
            request = self._request_for(MutationOperation.CHILD_IMPORT, state.plan, state.child_payload)
            if request is None:
                state.terminal = self._manual_review(state.record, "could not construct child-import mutation")
                continue
            outcome = apply_unverified(request)
            state.mutations.append(outcome)
            report_action(state, "child mutation")
            if outcome.kind is MutationOutcomeKind.APPLIED:
                pending_children.append((state, request))
            elif outcome.kind is MutationOutcomeKind.ALREADY_APPLIED:
                report_action(state, "child verified")
                settled = self._settle_step(state.record, outcome, ExecutionStage.CHILD_PRESENT)
                state.terminal = settled
                if settled is None:
                    state.stage = ExecutionStage.CHILD_PRESENT
            else:
                state.terminal = self._settle_step(state.record, outcome, ExecutionStage.CHILD_PRESENT)

        verified_children: list[tuple[_BatchState, MutationRequest]] = []
        for state, request in pending_children:
            lease_failure = self._renew_before_step(state.record, "child verification", tuple(state.mutations))
            if lease_failure is not None:
                state.terminal = lease_failure
                continue
            verified_children.append((state, request))
        child_results = verify_children(tuple(request for _, request in verified_children))
        for state, request in verified_children:
            payload = cast(ChildImportPayload, request.payload)
            outcome = child_results.get(payload.child_uuid.lower())
            if outcome is None:
                outcome = MutationOutcome(
                    request.operation,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    request.guard,
                    (),
                    "child batch verification returned no result",
                )
            state.mutations.append(outcome)
            report_action(state, "child verified")
            settled = self._settle_step(state.record, outcome, ExecutionStage.CHILD_PRESENT)
            state.terminal = settled
            if settled is None:
                state.stage = ExecutionStage.CHILD_PRESENT

        pending_parents: list[tuple[_BatchState, MutationRequest]] = []
        for state in states:
            if state.terminal is not None or _SPAWN_STAGE_ORDER[state.stage] >= _SPAWN_STAGE_ORDER[ExecutionStage.PARENT_LINKED]:
                continue
            lease_failure = self._renew_before_step(state.record, "parent link", tuple(state.mutations))
            if lease_failure is not None:
                state.terminal = lease_failure
                continue
            request = self._request_for(MutationOperation.PARENT_LINK, state.plan, state.link_payload)
            if request is None:
                state.terminal = self._manual_review(state.record, "could not construct parent-link mutation")
                continue
            outcome = apply_unverified(request)
            state.mutations.append(outcome)
            report_action(state, "parent mutation")
            if outcome.kind is MutationOutcomeKind.APPLIED:
                pending_parents.append((state, request))
            elif outcome.kind is MutationOutcomeKind.ALREADY_APPLIED:
                report_action(state, "parent verified")
                settled = self._settle_step(state.record, outcome, ExecutionStage.PARENT_LINKED)
                state.terminal = settled
                if settled is None:
                    state.stage = ExecutionStage.PARENT_LINKED
            else:
                state.terminal = self._settle_step(state.record, outcome, ExecutionStage.PARENT_LINKED)

        verified_parents: list[tuple[_BatchState, MutationRequest]] = []
        for state, request in pending_parents:
            lease_failure = self._renew_before_step(state.record, "parent verification", tuple(state.mutations))
            if lease_failure is not None:
                state.terminal = lease_failure
                continue
            verified_parents.append((state, request))
        parent_results = verify_parents(tuple(request for _, request in verified_parents))
        for state, request in verified_parents:
            outcome = parent_results.get(request.guard.task_uuid.lower())
            if outcome is None:
                outcome = MutationOutcome(
                    request.operation,
                    MutationOutcomeKind.MANUAL_REVIEW,
                    request.guard,
                    (),
                    "parent batch verification returned no result",
                )
            state.mutations.append(outcome)
            report_action(state, "parent verified")
            settled = self._settle_step(state.record, outcome, ExecutionStage.PARENT_LINKED)
            state.terminal = settled
            if settled is None:
                state.stage = ExecutionStage.PARENT_LINKED

        for state in states:
            if state.terminal is not None:
                outcome = LifecycleApplicationOutcome(
                    state.terminal.kind,
                    state.terminal.identity,
                    reason=state.terminal.reason,
                    intent_id=state.terminal.intent_id,
                    mutations=tuple(state.mutations),
                )
                outcomes.append(outcome)
                report_outcome(state.record, outcome, state)
                continue
            lease_failure = self._renew_before_step(state.record, "verification", tuple(state.mutations))
            if lease_failure is not None:
                outcomes.append(lease_failure)
                report_outcome(state.record, lease_failure, state)
                continue
            advance = self._outbox.advance_stage(
                intent_id=state.record.intent_id, owner=self._owner, stage=ExecutionStage.VERIFIED
            )
            report_action(state, "intent verified")
            if not advance.ok:
                outcome = self._retry_or_review(state.record, advance, "could not persist verified lifecycle stage", tuple(state.mutations))
                outcomes.append(outcome)
                report_outcome(state.record, outcome, state)
                continue
            lease_failure = self._renew_before_step(state.record, "acknowledgement", tuple(state.mutations))
            if lease_failure is not None:
                outcomes.append(lease_failure)
                report_outcome(state.record, lease_failure, state)
                continue
            ack = self._outbox.acknowledge(intent_id=state.record.intent_id, owner=self._owner)
            report_action(state, "intent acknowledged")
            if not ack.ok:
                outcome = self._retry_or_review(state.record, ack, "could not acknowledge finalized lifecycle intent", tuple(state.mutations))
                outcomes.append(outcome)
                report_outcome(state.record, outcome, state)
                continue
            outcome = LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.APPLIED,
                state.plan.identity,
                intent_id=state.record.intent_id,
                mutations=tuple(state.mutations),
            )
            outcomes.append(outcome)
            report_outcome(state.record, outcome, state)
        return DrainResult(claim=claim, outcomes=tuple(outcomes))

    @staticmethod
    def _report_drain_progress(
        progress: LifecycleDrainProgressCallback | None,
        event: LifecycleDrainProgress,
    ) -> None:
        """Notify an observer without allowing presentation to affect safety."""
        if progress is None:
            return
        try:
            progress(event)
        except Exception:
            return

    def execute_staged(
        self,
        plan: LifecyclePlan,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
    ) -> LifecycleApplicationOutcome:
        """Claim and execute exactly the intent for one already-staged plan.

        Unlike ``drain()``, which claims whatever is next in FIFO order from
        the shared outbox, this targets the specific intent derived from
        ``plan``'s identity. For a caller that stages one plan and must
        execute exactly that one under its own external lock (e.g. reconcile,
        which holds a per-parent lock and cannot risk acting on an unrelated
        intent that happened to be claimed instead).
        """
        self._require_execution_deps()
        if not isinstance(plan, LifecyclePlan):
            raise LifecycleApplicationError("execute_staged requires a validated LifecyclePlan")
        intent_id = plan.identity.idempotency_key
        claim = self._outbox.claim_intent(owner=self._owner, lease_seconds=self._lease_seconds, intent_id=intent_id)
        if claim.kind is OutboxResultKind.ALREADY_APPLIED:
            record = claim.record
            if record is None:
                return LifecycleApplicationOutcome(
                    LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
                    plan.identity,
                    reason="acknowledged lifecycle intent has no durable record",
                    intent_id=intent_id,
                )
            config = str(configuration_fingerprint or "").strip()
            schedule = str(schedule_fingerprint or "").strip()
            if (
                record.plan.compatibility_key() != plan.compatibility_key()
                or record.configuration_fingerprint != config
                or record.schedule_fingerprint != schedule
            ):
                return LifecycleApplicationOutcome(
                    LifecycleApplicationOutcomeKind.CONFLICT,
                    plan.identity,
                    reason="acknowledged lifecycle intent does not match requested immutable inputs",
                    intent_id=intent_id,
                )
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
                plan.identity,
                intent_id=intent_id,
            )
        if not claim.ok or claim.record is None:
            kind = (
                LifecycleApplicationOutcomeKind.RETRYABLE
                if claim.kind is OutboxResultKind.RETRYABLE
                else LifecycleApplicationOutcomeKind.MANUAL_REVIEW
            )
            reason = claim.reason or "could not claim the staged lifecycle intent"
            if claim.record is not None:
                changed = _plan_difference_summary(
                    claim.record.plan,
                    plan,
                    existing_config=claim.record.configuration_fingerprint,
                    requested_config=str(configuration_fingerprint or "").strip(),
                    existing_schedule=claim.record.schedule_fingerprint,
                    requested_schedule=str(schedule_fingerprint or "").strip(),
                )
                reason = (
                    f"{reason}; changed fields: {changed}"
                )
                if claim.record.failure is not None:
                    reason += f"; previous failure: {claim.record.failure.code}: {claim.record.failure.message}"
            return LifecycleApplicationOutcome(
                kind,
                plan.identity,
                reason=reason,
                intent_id=intent_id,
            )
        config = str(configuration_fingerprint or "").strip()
        schedule = str(schedule_fingerprint or "").strip()
        return self._execute_claimed(claim.record, configuration_fingerprint=config, schedule_fingerprint=schedule)

    def _execute_claimed(
        self,
        record: LifecycleOutboxRecord,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
        progress_action: Callable[[str, int], None] | None = None,
    ) -> LifecycleApplicationOutcome:
        def report_action(detail: str, units: int = 1) -> None:
            if progress_action is not None:
                progress_action(detail, units)

        plan = record.plan
        if (
            record.configuration_fingerprint != configuration_fingerprint
            or record.schedule_fingerprint != schedule_fingerprint
        ):
            return self._manual_review(record, "lifecycle plan no longer matches current configuration or schedule")
        if plan.action is not LifecycleAction.SPAWN_CHILD:
            return self._manual_review(record, f"outbox record has an unsupported action: {plan.action.value}")

        child_payload = _child_import_payload(plan)
        link_payload = _parent_link_payload(plan)
        if child_payload is None or link_payload is None:
            return self._manual_review(record, "lifecycle plan is missing its child import or parent link payload")

        mutations: list[MutationOutcome] = []
        stage = record.stage

        lease_failure = self._renew_before_step(record, "starting lifecycle execution", tuple(mutations))
        if lease_failure is not None:
            return lease_failure

        if _SPAWN_STAGE_ORDER[stage] < _SPAWN_STAGE_ORDER[ExecutionStage.CHILD_PRESENT]:
            lease_failure = self._renew_before_step(record, "child import", tuple(mutations))
            if lease_failure is not None:
                return lease_failure
            outcome = self._apply(MutationOperation.CHILD_IMPORT, plan, child_payload)
            if outcome is None:
                return self._manual_review(record, "could not construct a guarded child-import mutation request")
            mutations.append(outcome)
            report_action("child mutation and verification", 2)
            lease_failure = self._renew_before_step(record, "child-import progress", tuple(mutations))
            if lease_failure is not None:
                return lease_failure
            settled = self._settle_step(record, outcome, ExecutionStage.CHILD_PRESENT)
            if settled is not None:
                return LifecycleApplicationOutcome(
                    settled.kind, settled.identity, reason=settled.reason, intent_id=settled.intent_id, mutations=tuple(mutations)
                )
            stage = ExecutionStage.CHILD_PRESENT

        if _SPAWN_STAGE_ORDER[stage] < _SPAWN_STAGE_ORDER[ExecutionStage.PARENT_LINKED]:
            lease_failure = self._renew_before_step(record, "parent link", tuple(mutations))
            if lease_failure is not None:
                return lease_failure
            outcome = self._apply(MutationOperation.PARENT_LINK, plan, link_payload)
            if outcome is None:
                return self._manual_review(record, "could not construct a guarded parent-link mutation request")
            mutations.append(outcome)
            report_action("parent mutation and verification", 2)
            lease_failure = self._renew_before_step(record, "parent-link progress", tuple(mutations))
            if lease_failure is not None:
                return lease_failure
            settled = self._settle_step(record, outcome, ExecutionStage.PARENT_LINKED)
            if settled is not None:
                return LifecycleApplicationOutcome(
                    settled.kind, settled.identity, reason=settled.reason, intent_id=settled.intent_id, mutations=tuple(mutations)
                )
            stage = ExecutionStage.PARENT_LINKED

        if _SPAWN_STAGE_ORDER[stage] < _SPAWN_STAGE_ORDER[ExecutionStage.VERIFIED]:
            # The sequential path verifies each mutation inline. The batched
            # path reaches this method only after its phase snapshot has
            # already proven both postconditions.
            lease_failure = self._renew_before_step(record, "verification", tuple(mutations))
            if lease_failure is not None:
                return lease_failure
            advance = self._outbox.advance_stage(intent_id=record.intent_id, owner=self._owner, stage=ExecutionStage.VERIFIED)
            report_action("intent verified")
            if not advance.ok:
                return self._retry_or_review(record, advance, "could not persist verified lifecycle stage", tuple(mutations))

        lease_failure = self._renew_before_step(record, "acknowledgement", tuple(mutations))
        if lease_failure is not None:
            return lease_failure
        ack = self._outbox.acknowledge(intent_id=record.intent_id, owner=self._owner)
        report_action("intent acknowledged")
        if not ack.ok:
            return self._retry_or_review(record, ack, "could not acknowledge finalized lifecycle intent", tuple(mutations))
        return LifecycleApplicationOutcome(
            LifecycleApplicationOutcomeKind.APPLIED, plan.identity, intent_id=record.intent_id, mutations=tuple(mutations)
        )

    def _renew_before_step(
        self,
        record: LifecycleOutboxRecord,
        step: str,
        mutations: tuple[MutationOutcome, ...],
    ) -> LifecycleApplicationOutcome | None:
        """Refresh ownership immediately before each claimed operation.

        A batch is claimed at one instant, but execution is sequential.  A
        later record must not be mutated using an expired lease.  The atomic
        repository check also turns lease loss into a typed outcome before
        the next external mutation can begin.
        """
        renewed = self._outbox.renew_lease(
            intent_id=record.intent_id,
            owner=self._owner,
            lease_seconds=self._lease_seconds,
        )
        if renewed.kind is OutboxResultKind.APPLIED:
            return None
        return self._retry_or_review(record, renewed, f"could not renew lifecycle lease before {step}", mutations)

    def _settle_step(
        self,
        record: LifecycleOutboxRecord,
        outcome: MutationOutcome,
        next_stage: ExecutionStage,
    ) -> LifecycleApplicationOutcome | None:
        """Persist progress for one successful mutation, or fail the intent.

        Returns None to continue the sequence, or a terminal outcome for
        this drain pass.
        """
        plan = record.plan
        if outcome.kind in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}:
            advance = self._outbox.advance_stage(intent_id=record.intent_id, owner=self._owner, stage=next_stage)
            if not advance.ok:
                return self._retry_or_review(record, advance, "could not persist lifecycle stage progress", ())
            return None
        if outcome.kind is MutationOutcomeKind.RETRYABLE:
            release = self._outbox.release_retry(
                intent_id=record.intent_id,
                owner=self._owner,
                failure=OutboxFailure("mutation_retryable", outcome.reason or "mutation is retryable"),
            )
            if release.kind is OutboxResultKind.REJECTED:
                return LifecycleApplicationOutcome(
                    LifecycleApplicationOutcomeKind.QUARANTINED,
                    plan.identity,
                    reason=release.reason or "lifecycle retry budget exhausted",
                    intent_id=record.intent_id,
                )
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.RETRYABLE, plan.identity, reason=outcome.reason, intent_id=record.intent_id
            )
        return self._manual_review(
            record,
            outcome.reason or outcome.kind.value,
            failure=OutboxFailure(f"mutation_{outcome.kind.value}", outcome.reason or outcome.kind.value),
        )

    def _retry_or_review(
        self,
        record: LifecycleOutboxRecord,
        outbox_result: OutboxResult,
        fallback: str,
        mutations: tuple[MutationOutcome, ...],
    ) -> LifecycleApplicationOutcome:
        plan = record.plan
        reason = outbox_result.reason or fallback
        if outbox_result.kind is OutboxResultKind.RETRYABLE:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.RETRYABLE, plan.identity, reason=reason, intent_id=record.intent_id, mutations=mutations
            )
        return LifecycleApplicationOutcome(
            LifecycleApplicationOutcomeKind.MANUAL_REVIEW, plan.identity, reason=reason, intent_id=record.intent_id, mutations=mutations
        )

    def _manual_review(
        self,
        record: LifecycleOutboxRecord,
        reason: str,
        *,
        failure: OutboxFailure | None = None,
    ) -> LifecycleApplicationOutcome:
        try:
            persisted = self._outbox.manual_review(
                intent_id=record.intent_id,
                owner=self._owner,
                failure=failure or OutboxFailure("invalid_intent", reason),
            )
        except Exception as exc:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.RETRYABLE,
                record.plan.identity,
                reason=(
                    "manual-review persistence failed: "
                    f"{str(exc).strip() or type(exc).__name__}; {reason}"
                ),
                intent_id=record.intent_id,
            )
        if not persisted.ok:
            persistence_reason = persisted.reason or persisted.kind.value
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.RETRYABLE,
                record.plan.identity,
                reason=f"manual-review persistence failed: {persistence_reason}; {reason}",
                intent_id=record.intent_id,
            )
        return LifecycleApplicationOutcome(
            LifecycleApplicationOutcomeKind.MANUAL_REVIEW, record.plan.identity, reason=reason, intent_id=record.intent_id
        )

    # -- execution: single-mutation plans (no outbox) ------------------

    def apply_immediate(self, plan: LifecyclePlan) -> LifecycleApplicationOutcome:
        """Apply one DISABLE_CHAIN, FINALIZE_CHAIN, or UPDATE_PARENT plan.

        These are one guarded, self-verifying mutation; they run inline
        against the mutation gateway with no durable outbox involvement.
        """
        self._require_execution_deps()
        if not isinstance(plan, LifecyclePlan):
            raise LifecycleApplicationError("immediate application requires a validated LifecyclePlan")
        if plan.action is LifecycleAction.NOOP:
            return LifecycleApplicationOutcome(LifecycleApplicationOutcomeKind.NOOP, plan.identity)
        if plan.action is LifecycleAction.SPAWN_CHILD:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
                plan.identity,
                reason="spawn_child transitions require durable outbox staging; call stage/drain",
            )

        payload: Any
        if plan.action in _TERMINAL_ACTIONS:
            operation = MutationOperation.CHAIN_DISABLE
            payload = _chain_disable_payload(plan)
        elif plan.action is LifecycleAction.UPDATE_PARENT:
            operation = MutationOperation.METADATA_REPAIR
            payload = _update_parent_payload(plan)
        else:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
                plan.identity,
                reason=f"unsupported lifecycle action: {plan.action.value}",
            )
        if payload is None:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
                plan.identity,
                reason=f"{plan.action.value} plan has no usable mutation payload",
            )

        outcome = self._apply(operation, plan, payload)
        if outcome is None:
            return LifecycleApplicationOutcome(
                LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
                plan.identity,
                reason="could not construct a guarded mutation request",
            )
        kind = _MUTATION_TO_APPLICATION.get(outcome.kind, LifecycleApplicationOutcomeKind.MANUAL_REVIEW)
        return LifecycleApplicationOutcome(kind, plan.identity, reason=outcome.reason, mutations=(outcome,))

    # -- shared mutation application -----------------------------------

    def _request_for(self, operation: MutationOperation, plan: LifecyclePlan, payload: Any = None) -> MutationRequest | None:
        self._require_execution_deps()
        assert self._uow is not None
        guard = _mutation_guard(plan, mutation_epoch=self._uow.mutation_epoch)
        if guard is None:
            return None
        if payload is None and operation is MutationOperation.CHILD_IMPORT:
            payload = _child_import_payload(plan)
        elif payload is None and operation is MutationOperation.PARENT_LINK:
            payload = _parent_link_payload(plan)
        elif payload is None:
            return None
        if payload is None:
            return None
        try:
            return MutationRequest(operation=operation, guard=guard, payload=payload)
        except IntegrationContractError:
            return None

    def _apply(self, operation: MutationOperation, plan: LifecyclePlan, payload: Any) -> MutationOutcome | None:
        self._require_execution_deps()
        assert self._mutations is not None
        request = self._request_for(operation, plan, payload)
        if request is None:
            return None
        return self._mutations.apply(request)


__all__ = (
    "DrainResult",
    "LifecycleApplicationError",
    "LifecycleApplicationOutcome",
    "LifecycleApplicationOutcomeKind",
    "LifecycleApplicationService",
)
