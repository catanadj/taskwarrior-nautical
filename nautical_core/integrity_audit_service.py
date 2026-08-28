"""Shared integrity auditing for authoritative task snapshots."""

from __future__ import annotations

from collections.abc import Sequence

from .chain_integrity_engine import ChainIntegrityEngine, IntegrityEngineResult
from .chain_integrity_models import IntegrityReportStatus
from .chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
from .integrity_report import doctor_findings
from .integration_models import Found, Unavailable
from .lifecycle_outbox import LifecycleOutboxRepository
from .operator_context import OperatorInvocationContext
from .operator_models import OperatorFailure, OperatorOperation, OperatorRequest, OperatorScope, OperatorScopeKind
from .operator_snapshot import ChainSnapshotReader, SnapshotReadRequest
from .task_models import TaskObservation


def audit_authoritative_rows(unit_of_work: object, rows: Sequence[TaskObservation]) -> tuple[IntegrityEngineResult | None, list[dict[str, object]]]:
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
    normalized = snapshots.from_rows(request, tuple(rows), source="doctor.authoritative_export")
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


__all__ = ["audit_authoritative_rows"]
