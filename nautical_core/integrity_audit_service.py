"""Shared integrity auditing for authoritative task snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from .taskwarrior_uow import TaskwarriorUnitOfWork

from .chain_integrity_engine import ChainIntegrityEngine, IntegrityEngineResult
from .chain_integrity_models import IntegrityReportStatus, SnapshotCoverage
from .chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
from .integrity_report import doctor_findings
from .integration_models import Found, Unavailable
from .lifecycle_outbox import LifecycleOutboxRepository
from .operator_context import OperatorInvocationContext
from .operator_models import OperatorFailure, OperatorOperation, OperatorRequest, OperatorScope, OperatorScopeKind
from .operator_snapshot import ChainSnapshotReader, SnapshotReadRequest
from .task_models import TaskObservation


@dataclass(frozen=True, slots=True)
class IntegrityAuditBundle:
    """Audited snapshot plus the engine used for a subsequent guarded apply."""

    engine: ChainIntegrityEngine
    result: IntegrityEngineResult
    findings: list[dict[str, object]]


def audit_authoritative_rows(
    unit_of_work: TaskwarriorUnitOfWork,
    rows: Sequence[TaskObservation],
    *,
    source: str = "doctor.authoritative_export",
    coverage: SnapshotCoverage = SnapshotCoverage.COMPLETE,
) -> tuple[IntegrityEngineResult | None, list[dict[str, object]]]:
    """Audit one already-exported Taskwarrior snapshot through the engine."""
    configuration = getattr(getattr(unit_of_work, "context", None), "configuration", None)
    if configuration is None:
        return None, []
    snapshots = ChainSnapshotService(unit_of_work, configuration_fingerprint=configuration.fingerprint)
    engine = ChainIntegrityEngine(
        snapshots,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    request = IntegritySnapshotRequest.candidates(complete_chain_history=True)
    normalized = snapshots.from_rows(request, tuple(rows), source=source, coverage=coverage)
    if isinstance(normalized, Unavailable):
        return IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=normalized.evidence.detail), []
    scope = OperatorScope(OperatorScopeKind.SYSTEM)
    context = OperatorInvocationContext.from_unit_of_work(
        OperatorRequest(OperatorOperation.INSPECT, scope), unit_of_work,
    )
    projected = ChainSnapshotReader(lambda _request: Found(normalized, "doctor authoritative export")).read_chain_snapshot(
        context, SnapshotReadRequest(scope),
    )
    if isinstance(projected, OperatorFailure):
        return IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=projected.message), []
    result = engine.audit_snapshot(
        projected,
        outbox_repository=LifecycleOutboxRepository(unit_of_work.outbox.taskdata),
        mutation_epoch=unit_of_work.mutation_epoch,
    )
    return result, doctor_findings(result)


def audit_authoritative_rows_with_engine(
    unit_of_work: TaskwarriorUnitOfWork,
    rows: Sequence[TaskObservation],
    *,
    source: str,
    coverage: SnapshotCoverage,
    outbox_repository: LifecycleOutboxRepository | None = None,
) -> IntegrityAuditBundle | None:
    """Return the shared audit result and engine for an operator apply path."""
    configuration = getattr(getattr(unit_of_work, "context", None), "configuration", None)
    if configuration is None:
        return None
    snapshots = ChainSnapshotService(unit_of_work, configuration_fingerprint=configuration.fingerprint)
    engine = ChainIntegrityEngine(
        snapshots,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    normalized = snapshots.from_rows(
        IntegritySnapshotRequest.candidates(complete_chain_history=True),
        tuple(rows), source=source, coverage=coverage,
    )
    if isinstance(normalized, Unavailable):
        result = IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=normalized.evidence.detail)
        return IntegrityAuditBundle(engine, result, [])
    scope = OperatorScope(OperatorScopeKind.SYSTEM)
    context = OperatorInvocationContext.from_unit_of_work(
        OperatorRequest(OperatorOperation.INSPECT, scope), unit_of_work,
    )
    projected = ChainSnapshotReader(lambda _request: Found(normalized, source)).read_chain_snapshot(
        context, SnapshotReadRequest(scope),
    )
    if isinstance(projected, OperatorFailure):
        result = IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=projected.message)
        return IntegrityAuditBundle(engine, result, [])
    result = engine.audit_snapshot(
        projected,
        outbox_repository=outbox_repository or LifecycleOutboxRepository(unit_of_work.outbox.taskdata),
        mutation_epoch=unit_of_work.mutation_epoch,
    )
    return IntegrityAuditBundle(engine, result, doctor_findings(result))


__all__ = ["IntegrityAuditBundle", "audit_authoritative_rows", "audit_authoritative_rows_with_engine"]
