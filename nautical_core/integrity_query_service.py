"""Read-only integrity query orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, cast

from .chain_integrity_engine import ChainIntegrityEngine, IntegrityEngineResult
from .chain_integrity_models import IntegrityReportStatus
from .chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
from .integrity_report import public_payload
from .integration_context import IntegrationAccess
from .lifecycle_outbox import LifecycleOutboxRepository
from .operator_context import OperatorInvocationContext
from .operator_models import (
    OperatorFailure,
    OperatorOperation,
    OperatorRequest,
    OperatorScope,
)
from .query_models import QueryContractError
from .operator_snapshot import ChainSnapshotReader, SnapshotReadRequest
from .taskwarrior_uow import build_operator_uow


@dataclass(frozen=True, slots=True)
class IntegrityQueryService:
    """Own the complete read-only integrity query pipeline for one runtime."""

    core: Any
    task_binary: str
    env: Mapping[str, str]
    uow_builder: Callable[..., Any] = build_operator_uow

    @staticmethod
    def request_from_selector(
        *, uuids: list[str] | None = None, chain_id: str | None = None, all_tasks: bool = False
    ) -> IntegritySnapshotRequest:
        """Compile exactly one CLI selector into the canonical snapshot request."""
        selected = sum(bool(value) for value in (uuids, chain_id, all_tasks))
        if selected != 1:
            raise QueryContractError("integrity query requires exactly one of --uuid, --chain-id, or --all")
        if uuids:
            if len(uuids) != 1:
                raise QueryContractError("integrity query accepts one --uuid")
            return IntegritySnapshotRequest.uuid(uuids[0], complete_chain_history=True)
        if chain_id:
            return IntegritySnapshotRequest.chain(chain_id)
        # Whole-system audits request authoritative history directly rather
        # than bounded hydration of every individual chain.
        return IntegritySnapshotRequest.candidates(complete_chain_history=True)

    def query(self, request: IntegritySnapshotRequest) -> tuple[dict[str, Any], int]:
        if not isinstance(request, IntegritySnapshotRequest):
            raise TypeError("integrity query requires an IntegritySnapshotRequest")

        unit_of_work = self.uow_builder(
            core=self.core,
            task_binary=self.task_binary,
            env=self.env,
            access=IntegrationAccess.READ_ONLY,
        )
        configuration = unit_of_work.context.configuration
        snapshots = ChainSnapshotService(
            unit_of_work,
            configuration_fingerprint=configuration.fingerprint,
        )
        engine = ChainIntegrityEngine(
            snapshots,
            configuration_fingerprint=configuration.fingerprint,
            schedule_fingerprint=configuration.scheduler_fingerprint,
        )
        scope = OperatorScope.from_selector(
            chain_id=request.chain_id or None,
            uuid=request.task_uuid or None,
            all_tasks=request.kind.value == "candidates",
        )
        operator_request = OperatorRequest(OperatorOperation.INTEGRITY, scope)
        operator_context = OperatorInvocationContext.from_unit_of_work(operator_request, unit_of_work)
        read_result = ChainSnapshotReader(cast(Any, snapshots.collect)).read_chain_snapshot(
            operator_context,
            SnapshotReadRequest(scope),
        )
        if isinstance(read_result, OperatorFailure):
            result = IntegrityEngineResult(IntegrityReportStatus.UNAVAILABLE, reason=read_result.message)
        else:
            result = engine.audit_snapshot(
                read_result,
                outbox_repository=LifecycleOutboxRepository(unit_of_work.outbox.taskdata),
                mutation_epoch=unit_of_work.mutation_epoch,
            )
        payload = public_payload(
            result,
            query={
                "kind": request.kind.value,
                "chainID": request.chain_id or None,
                "uuid": request.task_uuid or None,
            },
            configuration_fingerprint=configuration.fingerprint,
        )
        return payload, 3 if result.status.value == "unavailable" else 0


__all__ = ["IntegrityQueryService"]
