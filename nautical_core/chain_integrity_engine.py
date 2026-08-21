"""Orchestration owner for chain integrity audit, planning, and drain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .chain_graph import ChainGraph
from .chain_integrity_application import (
    IntegrityApplicationResult,
    IntegrityApplicationService,
    IntegrityMutationRequestFactory,
    IntegrityOutboxPersistResult,
    RepositoryIntegrityOutboxSink,
)
from .chain_integrity_context import IntegrityContext, OutboxSnapshot, load_outbox_snapshot
from .chain_integrity_models import (
    ChainSnapshot,
    FindingStatus,
    IntegrityFinding,
    IntegrityRepairPlan,
    IntegrityReportStatus,
)
from .chain_repair_planner import IntegrityPlanningResult, IntegrityRepairPlanner, PlannerRefusal
from .chain_snapshot import IntegritySnapshotRequest
from .integration_models import Found, MutationOutcomeKind, TaskRead, Unavailable
from .lifecycle_outbox import LifecycleOutboxRepository, OutboxFailure


class _SnapshotProvider(Protocol):
    def collect(self, request: IntegritySnapshotRequest) -> TaskRead[ChainSnapshot]: ...


class _MutationExecutor(Protocol):
    def repair_metadata(self, request: object) -> object: ...


class _NoSnapshotProvider:
    def collect(self, _request: IntegritySnapshotRequest) -> TaskRead[ChainSnapshot]:
        raise RuntimeError("this integrity engine instance is lifecycle-planning only")


@dataclass(frozen=True, slots=True)
class IntegrityEngineResult:
    status: IntegrityReportStatus
    snapshot: ChainSnapshot | None = None
    findings: tuple[IntegrityFinding, ...] = ()
    plans: tuple[IntegrityRepairPlan, ...] = ()
    refusals: tuple[PlannerRefusal, ...] = ()
    applications: tuple[IntegrityApplicationResult, ...] = ()
    reason: str = ""


class _AlreadyPersistedSink:
    def persist(self, _plan) -> IntegrityOutboxPersistResult:
        return IntegrityOutboxPersistResult(True)


class ChainIntegrityEngine:
    """One owner for integrity evidence, planning, and durable work drain."""

    def __init__(
        self,
        snapshots: _SnapshotProvider,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str = "integrity",
    ) -> None:
        self._snapshots = snapshots
        self._configuration_fingerprint = str(configuration_fingerprint or "").strip()
        self._schedule_fingerprint = str(schedule_fingerprint or "").strip()
        if not self._configuration_fingerprint:
            raise ValueError("integrity engine requires a configuration fingerprint")
        if not self._schedule_fingerprint:
            raise ValueError("integrity engine requires a schedule fingerprint")
        self._planner = IntegrityRepairPlanner()
        self._application = IntegrityApplicationService()

    @classmethod
    def lifecycle_only(
        cls,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str = "integrity",
    ) -> "ChainIntegrityEngine":
        return cls(
            _NoSnapshotProvider(),
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )

    def plan_recovery(
        self,
        parent: dict[str, object],
        *,
        existing_children: list[dict[str, object]],
        hook: object,
        generation: object = None,
    ) -> object:
        """Build one successor/expiration decision through the engine owner."""
        from .chain_integrity_lifecycle import build_reconcile_plan

        return build_reconcile_plan(
            parent,
            existing_children=existing_children,
            hook=hook,
            generation=generation,
        )

    def audit(
        self,
        request: IntegritySnapshotRequest,
        *,
        outbox_repository: LifecycleOutboxRepository,
        mutation_epoch: int = 0,
    ) -> IntegrityEngineResult:
        read = self._snapshots.collect(request)
        if isinstance(read, Unavailable):
            return IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=read.evidence.detail)
        if not isinstance(read, Found):
            return IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason="integrity snapshot is absent")
        return self.audit_snapshot(
            read.value,
            outbox_repository=outbox_repository,
            mutation_epoch=mutation_epoch,
        )

    def audit_snapshot(
        self,
        snapshot: ChainSnapshot,
        *,
        outbox_repository: LifecycleOutboxRepository,
        mutation_epoch: int = 0,
    ) -> IntegrityEngineResult:
        """Audit one already-authoritative snapshot without another export."""
        if not isinstance(snapshot, ChainSnapshot):
            raise TypeError("integrity engine requires a ChainSnapshot")
        try:
            graph = ChainGraph.from_snapshot(snapshot)
        except Exception as exc:
            return IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, snapshot, reason=str(exc))
        context = IntegrityContext(
            graph,
            load_outbox_snapshot(outbox_repository),
            self._configuration_fingerprint,
            mutation_epoch,
        )
        from .chain_invariants import evaluate_context

        findings = evaluate_context(context)
        planning: IntegrityPlanningResult = self._planner.plan(context, findings)
        status = self._status(findings, planning)
        return IntegrityEngineResult(status, snapshot, findings, planning.plans, planning.refusals)

    def apply(
        self,
        result: IntegrityEngineResult,
        *,
        executor: _MutationExecutor,
        request_factory: IntegrityMutationRequestFactory,
        outbox_repository: LifecycleOutboxRepository,
        owner: str,
        drain: bool = True,
    ) -> IntegrityEngineResult:
        if result.snapshot is None:
            return result
        applications: list[IntegrityApplicationResult] = []
        sink = RepositoryIntegrityOutboxSink(
            outbox_repository,
            configuration_fingerprint=self._configuration_fingerprint,
            schedule_fingerprint=self._schedule_fingerprint,
        )
        for plan in result.plans:
            if len(plan.operations) == 1:
                applications.extend(self._application.apply(plan, executor, request_factory))
            else:
                persisted = sink.persist(plan)
                if not persisted.accepted:
                    applications.extend(IntegrityApplicationResult(
                        plan.plan_id, operation.operation_id,
                        MutationOutcomeKind.MANUAL_REVIEW,
                        persisted.reason,
                    ) for operation in plan.operations)
        if drain:
            applications.extend(self.drain(
                outbox_repository, owner=owner, executor=executor, request_factory=request_factory,
            ))
        return IntegrityEngineResult(
            self._application_status(applications, result.status), result.snapshot,
            result.findings, result.plans, result.refusals, tuple(applications), result.reason,
        )

    def drain(
        self,
        repository: LifecycleOutboxRepository,
        *,
        owner: str,
        executor: _MutationExecutor,
        request_factory: IntegrityMutationRequestFactory,
        limit: int = 20,
        lease_seconds: float = 120.0,
    ) -> tuple[IntegrityApplicationResult, ...]:
        claim, records = repository.claim_integrity_batch(owner=owner, lease_seconds=lease_seconds, limit=limit)
        if not claim.ok:
            return ()
        applications: list[IntegrityApplicationResult] = []
        for record in records:
            results = self._application.apply(record.envelope.plan, executor, request_factory, _AlreadyPersistedSink())
            applications.extend(results)
            if results and all(item.kind in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED} for item in results):
                repository.acknowledge_integrity(intent_id=record.envelope.intent_id, owner=owner)
            else:
                repository.manual_review_integrity(
                    intent_id=record.envelope.intent_id,
                    owner=owner,
                    failure=OutboxFailure("integrity_application", results[-1].reason if results else "no application result"),
                )
        return tuple(applications)

    @staticmethod
    def _status(findings: tuple[IntegrityFinding, ...], planning: IntegrityPlanningResult) -> IntegrityReportStatus:
        if any(item.status is FindingStatus.UNAVAILABLE for item in findings):
            return IntegrityReportStatus.UNAVAILABLE
        if planning.refusals or any(item.status is FindingStatus.MANUAL_REVIEW for item in findings):
            return IntegrityReportStatus.MANUAL_REVIEW
        if planning.plans or findings:
            return IntegrityReportStatus.REPAIRABLE
        return IntegrityReportStatus.HEALTHY

    @staticmethod
    def _application_status(
        applications: list[IntegrityApplicationResult],
        fallback: IntegrityReportStatus,
    ) -> IntegrityReportStatus:
        if any(item.kind in {MutationOutcomeKind.MANUAL_REVIEW, MutationOutcomeKind.CONFLICT} for item in applications):
            return IntegrityReportStatus.MANUAL_REVIEW
        if any(item.kind is MutationOutcomeKind.RETRYABLE for item in applications):
            return IntegrityReportStatus.UNAVAILABLE
        return fallback


__all__ = ["ChainIntegrityEngine", "IntegrityEngineResult"]
