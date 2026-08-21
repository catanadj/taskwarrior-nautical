"""Pure deterministic repair planning for chain integrity findings."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .chain_graph import ChainGraph
from .chain_integrity_context import IntegrityContext
from .chain_integrity_models import (
    FindingStatus,
    IntegrityFinding,
    IntegrityRepairPlan,
    IntegrityOperation,
    RepairOperationKind,
    RepairSafety,
    SnapshotCoverage,
)


@dataclass(frozen=True, slots=True)
class PlannerRefusal:
    """A finding that cannot safely become an automatic repair plan."""

    invariant_id: str
    reason_code: str
    reason: str
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class IntegrityPlanningResult:
    plans: tuple[IntegrityRepairPlan, ...] = ()
    refusals: tuple[PlannerRefusal, ...] = ()


def _pairs(finding: IntegrityFinding) -> dict[str, object]:
    return {key: value for key, value in finding.observed}


def _plan_id(snapshot_id: str, chain_id: str, operation: IntegrityOperation, configuration: str) -> str:
    payload = {
        "snapshot": snapshot_id,
        "chain": chain_id,
        "configuration": configuration,
        "operation": {
            "kind": operation.kind.value,
            "target": operation.target_uuid,
            "guard": operation.guard,
            "payload": operation.payload,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "cirp1-" + hashlib.sha256(encoded).hexdigest()[:24]


class IntegrityRepairPlanner:
    """Convert uniquely derivable findings into guarded typed plans."""

    def plan(self, context: IntegrityContext, findings: tuple[IntegrityFinding, ...]) -> IntegrityPlanningResult:
        if not isinstance(context, IntegrityContext):
            raise TypeError("repair planner requires an IntegrityContext")
        graph = context.graph
        plans: list[IntegrityRepairPlan] = []
        refusals: list[PlannerRefusal] = []
        operations_by_field: dict[tuple[str, str], IntegrityOperation] = {}
        for finding in sorted(findings, key=lambda item: (
            item.chain_id, item.subject_uuids, item.invariant_id, item.reason_code,
        )):
            operation = self._operation_for(context, finding)
            if operation is None:
                refusals.append(PlannerRefusal(
                    finding.invariant_id,
                    finding.reason_code,
                    self._refusal_reason(context, finding),
                    finding.snapshot_id,
                ))
                continue
            payload_fields = tuple(key for key, _value in operation.payload)
            for field in payload_fields:
                conflict_key = (operation.target_uuid, field)
                previous = operations_by_field.get(conflict_key)
                if previous is not None:
                    if previous.payload == operation.payload:
                        operation = None
                        break
                    refusals.append(PlannerRefusal(
                        finding.invariant_id,
                        "conflicting_repair",
                        "incompatible repairs target the same task field",
                        finding.snapshot_id,
                    ))
                    operation = None
                    break
                operations_by_field[conflict_key] = operation
            if operation is None:
                continue
            plan_id = _plan_id(graph.snapshot.snapshot_id, operation.chain_id, operation, context.configuration_fingerprint)
            plans.append(IntegrityRepairPlan(
                plan_id,
                graph.snapshot.snapshot_id,
                operation.chain_id,
                RepairSafety.SAFE,
                "reciprocal_link",
                "Restore one uniquely resolved reciprocal chain link.",
                (operation,),
                context.configuration_fingerprint,
            ))
        return IntegrityPlanningResult(tuple(plans), tuple(refusals))

    def _operation_for(self, context: IntegrityContext, finding: IntegrityFinding) -> IntegrityOperation | None:
        if finding.status is not FindingStatus.REPAIRABLE or finding.reason_code != "non_reciprocal_reference":
            return None
        graph = context.graph
        if not context.configuration_fingerprint:
            return None
        if graph.snapshot.coverage not in {SnapshotCoverage.CHAIN, SnapshotCoverage.COMPLETE}:
            return None
        if len(finding.subject_uuids) != 1:
            return None
        source_matches = graph.uuid_matches(finding.subject_uuids[0])
        if len(source_matches) != 1 or not source_matches[0].has_complete_identity:
            return None
        source = source_matches[0]
        observed = _pairs(finding)
        fields = [
            field for field in ("prevLink", "nextLink")
            if observed.get(field) == graph.reference(source.task_uuid, field).target_uuid
        ]
        if len(fields) != 1:
            return None
        field = fields[0]
        reference = graph.reference(source.task_uuid, field)
        if reference.state.value != "resolved" or not reference.target_uuid:
            return None
        target_matches = graph.uuid_matches(reference.target_uuid)
        if len(target_matches) != 1 or not target_matches[0].has_complete_identity:
            return None
        target = target_matches[0]
        opposite = "prevLink" if field == "nextLink" else "nextLink"
        operation_id = "ciop1-" + hashlib.sha256(
            f"{graph.snapshot.snapshot_id}:{target.task_uuid}:{opposite}:{source.task_uuid}".encode("utf-8")
        ).hexdigest()[:24]
        return IntegrityOperation(
            operation_id,
            RepairOperationKind.LINK_REPAIR,
            target.chain_id,
            target.task_uuid,
            (
                ("snapshot_id", graph.snapshot.snapshot_id),
                ("target_uuid", target.task_uuid),
                ("target_link", target.link),
            ),
            ("target remains present", f"{opposite} is absent or stale on {target.task_uuid}"),
            (f"{opposite} points to {source.task_uuid}",),
            ((opposite, source.task_uuid),),
        )

    @staticmethod
    def _refusal_reason(context: IntegrityContext, finding: IntegrityFinding) -> str:
        if context.graph.snapshot.coverage not in {SnapshotCoverage.CHAIN, SnapshotCoverage.COMPLETE}:
            return "repair requires complete chain coverage"
        if finding.status is not FindingStatus.REPAIRABLE:
            return "finding is not automatically repairable"
        if finding.invariant_id.startswith("lifecycle."):
            return "lifecycle successor decisions belong to LifecyclePlanner"
        return "no unique guarded operation is defined for this finding"


__all__ = ["IntegrityPlanningResult", "IntegrityRepairPlanner", "PlannerRefusal"]
