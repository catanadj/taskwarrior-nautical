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
from typing import Any, Protocol

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
    LifecycleIdentity,
    LifecyclePlan,
)
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
    child = plan.child_dict()
    if not child:
        return None
    try:
        return ChildImportPayload.from_mapping(child, parent_uuid=plan.identity.parent_uuid)
    except IntegrationContractError:
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
        return MetadataRepairPayload.from_mapping(plan.identity.parent_uuid, updates)
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


def _from_outbox_result(result: OutboxResult, identity: LifecycleIdentity) -> LifecycleApplicationOutcome:
    kind = _OUTBOX_TO_APPLICATION.get(result.kind, LifecycleApplicationOutcomeKind.MANUAL_REVIEW)
    intent_id = result.record.intent_id if result.record is not None else ""
    return LifecycleApplicationOutcome(kind, identity, reason=result.reason, intent_id=intent_id)


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
        result = self._outbox.enqueue(
            plan,
            configuration_fingerprint=str(configuration_fingerprint or "").strip(),
            schedule_fingerprint=str(schedule_fingerprint or "").strip(),
        )
        return _from_outbox_result(result, plan.identity)

    # -- execution: claimed SPAWN_CHILD intents ------------------------

    def drain(
        self,
        *,
        limit: int,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
    ) -> DrainResult:
        """Claim a bounded batch of ready spawn intents and execute each one.

        Bounded by ``limit`` so one invocation cannot be pinned to an
        unbounded recovery pass; whatever is not claimed this round remains
        durable and eligible for the next drain.
        """
        self._require_execution_deps()
        claim, records = self._outbox.claim_batch(
            owner=self._owner, lease_seconds=self._lease_seconds, limit=max(1, int(limit))
        )
        if not records:
            return DrainResult(claim=claim, outcomes=())
        config = str(configuration_fingerprint or "").strip()
        schedule = str(schedule_fingerprint or "").strip()
        outcomes = tuple(
            self._execute_claimed(record, configuration_fingerprint=config, schedule_fingerprint=schedule)
            for record in records
        )
        return DrainResult(claim=claim, outcomes=outcomes)

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
        if not claim.ok or claim.record is None:
            kind = (
                LifecycleApplicationOutcomeKind.RETRYABLE
                if claim.kind is OutboxResultKind.RETRYABLE
                else LifecycleApplicationOutcomeKind.MANUAL_REVIEW
            )
            return LifecycleApplicationOutcome(
                kind,
                plan.identity,
                reason=claim.reason or "could not claim the staged lifecycle intent",
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
    ) -> LifecycleApplicationOutcome:
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

        if _SPAWN_STAGE_ORDER[stage] < _SPAWN_STAGE_ORDER[ExecutionStage.CHILD_PRESENT]:
            outcome = self._apply(MutationOperation.CHILD_IMPORT, plan, child_payload)
            if outcome is None:
                return self._manual_review(record, "could not construct a guarded child-import mutation request")
            mutations.append(outcome)
            settled = self._settle_step(record, outcome, ExecutionStage.CHILD_PRESENT)
            if settled is not None:
                return LifecycleApplicationOutcome(
                    settled.kind, settled.identity, reason=settled.reason, intent_id=settled.intent_id, mutations=tuple(mutations)
                )
            stage = ExecutionStage.CHILD_PRESENT

        if _SPAWN_STAGE_ORDER[stage] < _SPAWN_STAGE_ORDER[ExecutionStage.PARENT_LINKED]:
            outcome = self._apply(MutationOperation.PARENT_LINK, plan, link_payload)
            if outcome is None:
                return self._manual_review(record, "could not construct a guarded parent-link mutation request")
            mutations.append(outcome)
            settled = self._settle_step(record, outcome, ExecutionStage.PARENT_LINKED)
            if settled is not None:
                return LifecycleApplicationOutcome(
                    settled.kind, settled.identity, reason=settled.reason, intent_id=settled.intent_id, mutations=tuple(mutations)
                )
            stage = ExecutionStage.PARENT_LINKED

        if _SPAWN_STAGE_ORDER[stage] < _SPAWN_STAGE_ORDER[ExecutionStage.VERIFIED]:
            # Both mutations already verify their own postcondition before
            # reporting success, so reaching here means the transition is
            # verified; only the durable stage marker needs to catch up.
            advance = self._outbox.advance_stage(intent_id=record.intent_id, owner=self._owner, stage=ExecutionStage.VERIFIED)
            if not advance.ok:
                return self._retry_or_review(record, advance, "could not persist verified lifecycle stage", tuple(mutations))

        ack = self._outbox.acknowledge(intent_id=record.intent_id, owner=self._owner)
        if not ack.ok:
            return self._retry_or_review(record, ack, "could not acknowledge finalized lifecycle intent", tuple(mutations))
        return LifecycleApplicationOutcome(
            LifecycleApplicationOutcomeKind.APPLIED, plan.identity, intent_id=record.intent_id, mutations=tuple(mutations)
        )

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
        self._outbox.manual_review(
            intent_id=record.intent_id,
            owner=self._owner,
            failure=OutboxFailure(f"mutation_{outcome.kind.value}", outcome.reason or outcome.kind.value),
        )
        return LifecycleApplicationOutcome(
            LifecycleApplicationOutcomeKind.MANUAL_REVIEW, plan.identity, reason=outcome.reason, intent_id=record.intent_id
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

    def _manual_review(self, record: LifecycleOutboxRecord, reason: str) -> LifecycleApplicationOutcome:
        self._outbox.manual_review(
            intent_id=record.intent_id, owner=self._owner, failure=OutboxFailure("invalid_intent", reason)
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

    def _apply(self, operation: MutationOperation, plan: LifecyclePlan, payload: Any) -> MutationOutcome | None:
        self._require_execution_deps()
        assert self._uow is not None and self._mutations is not None
        guard = _mutation_guard(plan, mutation_epoch=self._uow.mutation_epoch)
        if guard is None:
            return None
        try:
            request = MutationRequest(operation=operation, guard=guard, payload=payload)
        except IntegrationContractError:
            return None
        return self._mutations.apply(request)


__all__ = (
    "DrainResult",
    "LifecycleApplicationError",
    "LifecycleApplicationOutcome",
    "LifecycleApplicationOutcomeKind",
    "LifecycleApplicationService",
)
