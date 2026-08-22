"""Orchestration owner for chain integrity audit, planning, and drain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
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
    ReferenceState,
    SnapshotCoverage,
)
from .chain_repair_planner import IntegrityPlanningResult, IntegrityRepairPlanner, PlannerRefusal
from .chain_integrity_recovery import IntegrityRecoveryService, RecoveryAudit
from .chain_snapshot import IntegritySnapshotKind, IntegritySnapshotRequest
from .integration_models import (
    CommandFailureKind,
    FailureEvidence,
    Found,
    MutationOutcomeKind,
    TaskCommand,
    TaskRead,
    Unavailable,
)
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
    chain_statuses: tuple[tuple[str, IntegrityReportStatus], ...] = ()
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
        max_hydrated_chains: int = 32,
    ) -> None:
        self._snapshots = snapshots
        self._configuration_fingerprint = str(configuration_fingerprint or "").strip()
        self._schedule_fingerprint = str(schedule_fingerprint or "").strip()
        if not self._configuration_fingerprint:
            raise ValueError("integrity engine requires a configuration fingerprint")
        if not self._schedule_fingerprint:
            raise ValueError("integrity engine requires a schedule fingerprint")
        if isinstance(max_hydrated_chains, bool) or not isinstance(max_hydrated_chains, int) or max_hydrated_chains < 1:
            raise ValueError("max_hydrated_chains must be a positive integer")
        self._max_hydrated_chains = max_hydrated_chains
        self._planner = IntegrityRepairPlanner()
        self._application = IntegrityApplicationService()
        self._recovery = IntegrityRecoveryService()

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
        from .chain_integrity_lifecycle import plan_recovery_decision

        return plan_recovery_decision(
            parent,
            existing_children=existing_children,
            hook=hook,
            generation=generation,
        )

    def audit_native_until(self, rows, *, predecessor, safe_parse_datetime, fmt_isoz, utc_to_local_naive, local_naive_to_utc) -> RecoveryAudit:
        """Delegate recovery evidence through the single integrity owner."""
        return self._recovery.audit_native_until(
            rows,
            predecessor=predecessor,
            safe_parse_datetime=safe_parse_datetime,
            fmt_isoz=fmt_isoz,
            utc_to_local_naive=utc_to_local_naive,
            local_naive_to_utc=local_naive_to_utc,
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
        hydrated = self._hydrate_required(request, read.value)
        if isinstance(hydrated, Unavailable):
            return IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=hydrated.evidence.detail)
        snapshot, hydrated_chains = hydrated
        return self.audit_snapshot(
            snapshot,
            outbox_repository=outbox_repository,
            mutation_epoch=mutation_epoch,
            hydrated_chains=hydrated_chains,
        )

    def _hydrate_required(
        self,
        request: IntegritySnapshotRequest,
        snapshot: ChainSnapshot,
    ) -> tuple[ChainSnapshot, frozenset[str]] | Unavailable:
        """Hydrate only chains whose candidate evidence has unresolved edges."""
        if request.kind is not IntegritySnapshotKind.CANDIDATES or request.complete_chain_history:
            return snapshot, frozenset()
        graph = ChainGraph.from_snapshot(snapshot)
        required: set[str] = set()
        for node in graph.nodes:
            if not node.chain_id:
                continue
            if any(
                graph.reference(node.task_uuid, field).state is ReferenceState.OUTSIDE_COVERAGE
                for field in ("prevLink", "nextLink")
            ):
                required.add(node.chain_id)
            elif node.link is not None and node.link > 1 and not str(node.field("prevLink", "") or "").strip():
                required.add(node.chain_id)
        if not required:
            return snapshot, frozenset()
        if len(required) > self._max_hydrated_chains:
            return Unavailable(
                "integrity:bounded-hydration",
                self._failure_evidence(
                    "bounded hydration requires "
                    f"{len(required)} chains, limit is {self._max_hydrated_chains}"
                ),
            )
        rows_by_uuid = {node.task_uuid.lower(): node for node in snapshot.rows}
        for chain_id in sorted(required):
            read = self._snapshots.collect(IntegritySnapshotRequest.chain(
                chain_id,
                statuses=request.statuses,
                complete_chain_history=True,
                refresh=request.refresh,
            ))
            if isinstance(read, Unavailable):
                return read
            if not isinstance(read, Found) or not read.value.rows:
                return Unavailable(
                    f"integrity:chain:{chain_id}",
                    self._failure_evidence(f"required chain hydration returned no rows for {chain_id}"),
                )
            for node in read.value.rows:
                rows_by_uuid[node.task_uuid.lower()] = node
        rows = tuple(sorted(rows_by_uuid.values(), key=lambda node: (node.chain_id, node.link is None, node.link or 0, node.task_uuid)))
        encoded = json.dumps(
            {"base": snapshot.snapshot_id, "hydrated": sorted(required), "rows": [node.to_dict() for node in rows]},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        hydrated_snapshot = ChainSnapshot(
            "cis1-hydrated-" + hashlib.sha256(encoded).hexdigest()[:24],
            snapshot.coverage,
            "taskwarrior.authoritative_export+bounded_hydration",
            rows,
            snapshot.configuration_fingerprint,
            snapshot.complete_chain_history,
        )
        return hydrated_snapshot, frozenset(required)

    @staticmethod
    def _failure_evidence(detail: str) -> FailureEvidence:
        command = TaskCommand(("task", "export"), "integrity bounded hydration", 1.0)
        return FailureEvidence(command, CommandFailureKind.INVALID_RESPONSE, 0, 1, 0.0, False, detail)

    def audit_snapshot(
        self,
        snapshot: ChainSnapshot,
        *,
        outbox_repository: LifecycleOutboxRepository,
        mutation_epoch: int = 0,
        hydrated_chains: frozenset[str] = frozenset(),
    ) -> IntegrityEngineResult:
        """Audit one already-authoritative snapshot without another export."""
        if not isinstance(snapshot, ChainSnapshot):
            raise TypeError("integrity engine requires a ChainSnapshot")
        outbox = load_outbox_snapshot(outbox_repository)
        groups: dict[str, list] = {}
        for row in snapshot.rows:
            groups.setdefault(row.chain_id, []).append(row)
        if not groups:
            groups[""] = []
        findings: list[IntegrityFinding] = []
        plans: list[IntegrityRepairPlan] = []
        refusals: list[PlannerRefusal] = []
        statuses: list[tuple[str, IntegrityReportStatus]] = []
        for chain_id in sorted(groups):
            scoped = ChainSnapshot(
                f"{snapshot.snapshot_id}:{chain_id or 'unassigned'}",
                SnapshotCoverage.CHAIN if chain_id in hydrated_chains else snapshot.coverage,
                snapshot.source,
                tuple(groups[chain_id]),
                snapshot.configuration_fingerprint,
                snapshot.complete_chain_history or chain_id in hydrated_chains,
                snapshot.reason,
            )
            try:
                graph = ChainGraph.from_snapshot(scoped)
                context = IntegrityContext(
                    graph, outbox, self._configuration_fingerprint, mutation_epoch,
                )
                from .chain_invariants import evaluate_context

                local_findings = evaluate_context(context)
                planning: IntegrityPlanningResult = self._planner.plan(context, local_findings)
                local_status = self._status(local_findings, planning)
            except Exception as exc:
                local_findings = ()
                planning = IntegrityPlanningResult((), ())
                local_status = IntegrityReportStatus.UNAVAILABLE
                reason = str(exc).strip() or type(exc).__name__
                return IntegrityEngineResult(
                    IntegrityReportStatus.UNAVAILABLE,
                    snapshot,
                    tuple(findings),
                    tuple(plans),
                    tuple(refusals),
                    chain_statuses=tuple(statuses) + ((chain_id, local_status),),
                    reason=reason,
                )
            findings.extend(local_findings)
            plans.extend(planning.plans)
            refusals.extend(planning.refusals)
            statuses.append((chain_id, local_status))
        status = self._aggregate_status(tuple(statuses), tuple(plans), tuple(findings))
        return IntegrityEngineResult(
            status,
            snapshot,
            tuple(findings),
            tuple(plans),
            tuple(refusals),
            chain_statuses=tuple(statuses),
        )

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
            # Keep single- and multi-operation plans on the same application
            # boundary.  The service persists multi-step work before any
            # external mutation and owns the refusal result on failure.
            applications.extend(self._application.apply(
                plan, executor, request_factory, sink,
            ))
        if drain:
            applications.extend(self.drain(
                outbox_repository, owner=owner, executor=executor, request_factory=request_factory,
            ))
        return IntegrityEngineResult(
            self._application_status(applications, result.status), result.snapshot,
            result.findings,
            result.plans,
            result.refusals,
            tuple(applications),
            result.chain_statuses,
            result.reason,
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
            envelope = record.envelope
            if (
                envelope.configuration_fingerprint != self._configuration_fingerprint
                or envelope.schedule_fingerprint != self._schedule_fingerprint
            ):
                reason = (
                    "integrity configuration drift: "
                    f"configuration={envelope.configuration_fingerprint!r}->{self._configuration_fingerprint!r}, "
                    f"schedule={envelope.schedule_fingerprint!r}->{self._schedule_fingerprint!r}"
                )
                review = repository.manual_review_integrity(
                    intent_id=envelope.intent_id,
                    owner=owner,
                    failure=OutboxFailure("configuration_drift", reason),
                )
                applications.append(IntegrityApplicationResult(
                    envelope.plan.plan_id,
                    envelope.plan.operations[0].operation_id,
                    MutationOutcomeKind.MANUAL_REVIEW if review.ok else MutationOutcomeKind.RETRYABLE,
                    reason if review.ok else f"manual-review persistence failed: {review.reason or review.kind.value}",
                ))
                continue
            results = self._application.apply(record.envelope.plan, executor, request_factory, _AlreadyPersistedSink())
            applications.extend(results)
            if results and all(item.kind in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED} for item in results):
                repository.acknowledge_integrity(intent_id=record.envelope.intent_id, owner=owner)
            else:
                review = repository.manual_review_integrity(
                    intent_id=record.envelope.intent_id,
                    owner=owner,
                    failure=OutboxFailure("integrity_application", results[-1].reason if results else "no application result"),
                )
                if not review.ok:
                    # Do not claim durable review when the state transition
                    # itself failed; retain an explicit unavailable outcome.
                    reason = review.reason or review.kind.value
                    applications.append(IntegrityApplicationResult(
                        record.envelope.plan.plan_id,
                        results[-1].operation_id if results else record.envelope.intent_id,
                        MutationOutcomeKind.RETRYABLE,
                        f"manual-review persistence failed: {reason}",
                    ))
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
    def _aggregate_status(
        statuses: tuple[tuple[str, IntegrityReportStatus], ...],
        plans: tuple[IntegrityRepairPlan, ...],
        findings: tuple[IntegrityFinding, ...],
    ) -> IntegrityReportStatus:
        values = {status for _chain, status in statuses}
        if values and values <= {IntegrityReportStatus.HEALTHY, IntegrityReportStatus.REPAIRABLE}:
            return IntegrityReportStatus.REPAIRABLE if plans or findings else IntegrityReportStatus.HEALTHY
        if IntegrityReportStatus.REPAIRABLE in values or plans:
            return IntegrityReportStatus.REPAIRABLE
        if IntegrityReportStatus.MANUAL_REVIEW in values:
            return IntegrityReportStatus.MANUAL_REVIEW
        if IntegrityReportStatus.UNAVAILABLE in values:
            return IntegrityReportStatus.UNAVAILABLE
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
