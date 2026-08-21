"""Pure invariant registry for immutable chain graphs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from .chain_graph import ChainGraph
from .chain_integrity_context import IntegrityContext, OutboxCoverage
from .chain_integrity_models import (
    ChainNode,
    FindingSeverity,
    FindingStatus,
    IntegrityFinding,
    ReferenceState,
    SnapshotCoverage,
)


InvariantEvaluator = Callable[[ChainGraph], tuple[IntegrityFinding, ...]]


@dataclass(frozen=True, slots=True)
class InvariantRule:
    """One named, pure rule and the coverage it needs to be meaningful."""

    invariant_id: str
    required_coverage: SnapshotCoverage
    evaluate: InvariantEvaluator

    def __post_init__(self) -> None:
        invariant_id = str(self.invariant_id or "").strip()
        if not invariant_id:
            raise ValueError("invariant rule requires an ID")
        if not callable(self.evaluate):
            raise TypeError("invariant rule evaluator must be callable")
        object.__setattr__(self, "invariant_id", invariant_id)
        object.__setattr__(self, "required_coverage", SnapshotCoverage(self.required_coverage))


def _finding(
    graph: ChainGraph,
    invariant_id: str,
    status: FindingStatus,
    severity: FindingSeverity,
    node: ChainNode | None,
    reason_code: str,
    message: str,
    *,
    observed: tuple[tuple[str, object], ...] = (),
    expected: tuple[tuple[str, object], ...] = (),
    evidence: tuple[tuple[str, object], ...] = (),
) -> IntegrityFinding:
    return IntegrityFinding(
        invariant_id,
        status,
        severity,
        graph.snapshot.snapshot_id,
        node.chain_id if node is not None else "",
        (node.task_uuid,) if node is not None else (),
        reason_code,
        message,
        observed,
        expected,
        evidence + (("coverage", graph.snapshot.coverage.value),),
    )


def _identity_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        if not node.chain_id:
            findings.append(_finding(
                graph, "identity.chain_id_required", FindingStatus.REPAIRABLE, FindingSeverity.ERROR, node,
                "missing_chain_id", "Chain node has no chainID.",
                observed=(("chainID", ""),), expected=(("chainID", "required"),),
            ))
        if node.link is None:
            findings.append(_finding(
                graph, "identity.link_required", FindingStatus.REPAIRABLE, FindingSeverity.ERROR, node,
                "missing_link", "Chain node has no positive numeric link.",
                observed=(("link", None),), expected=(("link", "positive integer"),),
            ))
    return tuple(findings)


def _duplicate_slot_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for (chain_id, link), nodes in sorted(graph.by_slot.items(), key=lambda item: item[0]):
        if len(nodes) < 2:
            continue
        subjects = tuple(node.task_uuid for node in nodes)
        findings.append(IntegrityFinding(
            "slot.duplicate_occupant",
            FindingStatus.MANUAL_REVIEW,
            FindingSeverity.ERROR,
            graph.snapshot.snapshot_id,
            chain_id,
            subjects,
            "duplicate_slot",
            f"Chain slot {chain_id}:{link} has multiple occupants.",
            (("link", link), ("occupants", subjects)),
            (("occupants", 1),),
            (("coverage", graph.snapshot.coverage.value),),
        ))
    return tuple(findings)


def _edge_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        for field, opposite, direction in (("prevLink", "nextLink", -1), ("nextLink", "prevLink", 1)):
            reference = graph.reference(node.task_uuid, field)
            if reference.state is not ReferenceState.RESOLVED:
                continue
            target_matches = graph.uuid_matches(reference.target_uuid)
            if len(target_matches) != 1:
                continue
            target = target_matches[0]
            if node.chain_id and target.chain_id and node.chain_id != target.chain_id:
                findings.append(_finding(
                    graph, "edge.same_chain", FindingStatus.MANUAL_REVIEW, FindingSeverity.ERROR, node,
                    "cross_chain_reference", f"{field} points to a different chain.",
                    observed=((field, target.chain_id),), expected=((field, node.chain_id),),
                ))
            if node.link is not None and target.link is not None and target.link != node.link + direction:
                findings.append(_finding(
                    graph, "edge.adjacent_slot", FindingStatus.MANUAL_REVIEW, FindingSeverity.ERROR, node,
                    "non_adjacent_reference", f"{field} does not point to an adjacent link.",
                    observed=((field, target.link),), expected=((field, node.link + direction),),
                ))
            reciprocal = graph.reference(target.task_uuid, opposite)
            if reciprocal.state is not ReferenceState.RESOLVED or reciprocal.target_uuid != node.task_uuid:
                findings.append(_finding(
                    graph, "edge.reciprocal", FindingStatus.REPAIRABLE, FindingSeverity.ERROR, node,
                    "non_reciprocal_reference", f"{field} is not reciprocated by {opposite}.",
                    observed=((field, target.task_uuid), (opposite, reciprocal.state.value)),
                    expected=((opposite, node.task_uuid),),
                ))
    return tuple(findings)


def _topology_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    """Detect forks and cycles without mutating or re-reading the graph."""
    findings: list[IntegrityFinding] = []
    incoming: dict[str, list[str]] = {}
    for node in graph.nodes:
        reference = graph.reference(node.task_uuid, "nextLink")
        if reference.state is ReferenceState.RESOLVED:
            incoming.setdefault(reference.target_uuid, []).append(node.task_uuid)
    for target_uuid, sources in sorted(incoming.items()):
        if len(sources) > 1:
            target = graph.uuid_matches(target_uuid)
            chain_id = target[0].chain_id if len(target) == 1 else ""
            findings.append(IntegrityFinding(
                "edge.fork",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                graph.snapshot.snapshot_id,
                chain_id,
                tuple(sorted((*sources, target_uuid))),
                "multiple_predecessors",
                "Multiple nodes point to the same successor.",
                (("target", target_uuid), ("sources", tuple(sorted(sources)))),
                (("sources", 1),),
                (("coverage", graph.snapshot.coverage.value),),
            ))
    reported: set[tuple[str, ...]] = set()
    for start in graph.nodes:
        path: list[str] = []
        current = start.task_uuid
        while current:
            if current in path:
                cycle = tuple(sorted(path[path.index(current):]))
                if cycle not in reported:
                    reported.add(cycle)
                    findings.append(IntegrityFinding(
                        "edge.cycle",
                        FindingStatus.MANUAL_REVIEW,
                        FindingSeverity.ERROR,
                        graph.snapshot.snapshot_id,
                        start.chain_id,
                        cycle,
                        "cycle_detected",
                        "Chain nextLink references form a cycle.",
                        (("cycle", cycle),),
                        (("cycle", ()),),
                        (("coverage", graph.snapshot.coverage.value),),
                    ))
                break
            path.append(current)
            reference = graph.reference(current, "nextLink")
            if reference.state is not ReferenceState.RESOLVED:
                break
            current = reference.target_uuid
    return tuple(findings)


def _lifecycle_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        status = node.status.lower()
        if status not in {"completed", "deleted"} or str(node.field("chain", "on") or "on").lower() != "on":
            continue
        if graph.reference(node.task_uuid, "nextLink").state is not ReferenceState.ABSENT:
            continue
        if node.field("chainMax") not in (None, "", "null") or node.field("chainUntil") not in (None, "", "null"):
            continue
        findings.append(_finding(
            graph,
            "lifecycle.successor_expected",
            FindingStatus.REPAIRABLE,
            FindingSeverity.ERROR,
            node,
            "missing_successor",
            "Completed chain-on node has no successor or terminal bound.",
            observed=(("status", status), ("nextLink", "")),
            expected=(("nextLink", "successor or terminal bound"),),
        ))
    return tuple(findings)


def _recurrence_identity_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        if node.link != 1:
            continue
        anchor = str(node.field("anchor", "") or "").strip()
        cp = str(node.field("cp", "") or "").strip()
        if anchor and cp:
            findings.append(_finding(
                graph,
                "identity.recurrence_exclusive",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                node,
                "multiple_recurrence_modes",
                "Chain root contains both anchor and cp recurrence identities.",
                observed=(("anchor", anchor), ("cp", cp)),
                expected=(("anchor_or_cp", "exactly one"),),
            ))
        elif not anchor and not cp and str(node.field("chain", "on") or "on").lower() == "on":
            findings.append(_finding(
                graph,
                "identity.recurrence_required",
                FindingStatus.REPAIRABLE,
                FindingSeverity.ERROR,
                node,
                "missing_recurrence_identity",
                "Chain root has no anchor or cp recurrence identity.",
                observed=(("anchor", ""), ("cp", "")),
                expected=(("anchor_or_cp", "required"),),
            ))
    return tuple(findings)


def _canonical_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        return None
    try:
        if text.endswith("Z") and "T" in text and len(text) >= 16 and text[0:8].isdigit():
            text = f"{text[:4]}-{text[4:6]}-{text[6:8]}{text[8:]}"
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _temporal_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        due_raw = node.field("due")
        due = _canonical_timestamp(due_raw)
        until_raw = node.field("until")
        until = _canonical_timestamp(until_raw)
        scheduled_raw = node.field("scheduled")
        scheduled = _canonical_timestamp(scheduled_raw)
        if due is not None and until is not None and until < due:
            findings.append(_finding(
                graph,
                "carry.until_after_due",
                FindingStatus.REPAIRABLE,
                FindingSeverity.ERROR,
                node,
                "until_before_due",
                "Native until is earlier than the task due timestamp.",
                observed=(("due", str(due_raw)), ("until", str(until_raw))),
                expected=(("until", "at or after due"),),
            ))
        if due is not None and scheduled is not None and scheduled > due:
            findings.append(_finding(
                graph,
                "carry.scheduled_before_due",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                node,
                "scheduled_after_due",
                "Scheduled timestamp is later than due.",
                observed=(("scheduled", str(scheduled_raw)), ("due", str(due_raw))),
                expected=(("scheduled", "at or before due"),),
            ))
    return tuple(findings)


DEFAULT_INVARIANTS: tuple[InvariantRule, ...] = (
    InvariantRule("identity", SnapshotCoverage.CANDIDATES, _identity_rule),
    InvariantRule("slot.duplicate_occupant", SnapshotCoverage.CANDIDATES, _duplicate_slot_rule),
    InvariantRule("edge", SnapshotCoverage.CANDIDATES, _edge_rule),
    InvariantRule("edge.topology", SnapshotCoverage.CANDIDATES, _topology_rule),
    InvariantRule("lifecycle", SnapshotCoverage.CANDIDATES, _lifecycle_rule),
    InvariantRule("identity.recurrence", SnapshotCoverage.CANDIDATES, _recurrence_identity_rule),
    InvariantRule("carry.temporal", SnapshotCoverage.CANDIDATES, _temporal_rule),
)


def evaluate_invariants(
    graph: ChainGraph,
    rules: Iterable[InvariantRule] = DEFAULT_INVARIANTS,
) -> tuple[IntegrityFinding, ...]:
    """Evaluate rules in stable order and deduplicate identical evidence."""
    findings: list[IntegrityFinding] = []
    for rule in sorted(tuple(rules), key=lambda item: item.invariant_id):
        if graph.snapshot.coverage in {SnapshotCoverage.UNAVAILABLE, SnapshotCoverage.TRUNCATED}:
            findings.append(_finding(
                graph,
                rule.invariant_id,
                FindingStatus.UNAVAILABLE,
                FindingSeverity.ERROR,
                None,
                "snapshot_coverage_unavailable",
                f"Cannot evaluate {rule.invariant_id}: snapshot coverage is {graph.snapshot.coverage.value}.",
                expected=(("coverage", rule.required_coverage.value),),
            ))
            continue
        findings.extend(rule.evaluate(graph))
    unique: dict[tuple[str, str, tuple[str, ...], str], IntegrityFinding] = {}
    for finding in findings:
        key = (finding.invariant_id, finding.chain_id, finding.subject_uuids, finding.reason_code)
        unique.setdefault(key, finding)
    return tuple(sorted(unique.values(), key=lambda item: (
        item.chain_id, item.subject_uuids, item.invariant_id, item.reason_code,
    )))


def _outbox_rule(context: IntegrityContext) -> tuple[IntegrityFinding, ...]:
    graph = context.graph
    if context.outbox.coverage is OutboxCoverage.UNAVAILABLE:
        return (IntegrityFinding(
            "outbox.snapshot_available",
            FindingStatus.UNAVAILABLE,
            FindingSeverity.ERROR,
            graph.snapshot.snapshot_id,
            "",
            (),
            "outbox_unavailable",
            f"Lifecycle intent evidence is unavailable: {context.outbox.reason}",
            (),
            (("outbox", "available"),),
            (("outbox_snapshot", context.outbox.snapshot_id),),
        ),)
    findings: list[IntegrityFinding] = []
    for record in context.outbox.records:
        identity = record.plan.identity
        if record.stage is not record.plan.stage:
            findings.append(IntegrityFinding(
                "outbox.stage_agreement",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                graph.snapshot.snapshot_id,
                identity.chain_id,
                (identity.parent_uuid,),
                "stage_mismatch",
                "Outbox lifecycle stage differs from the persisted plan stage.",
                (("intent_id", record.intent_id), ("record_stage", record.stage.value)),
                (("plan_stage", record.plan.stage.value),),
                (("outbox_snapshot", context.outbox.snapshot_id),),
            ))
        if record.state.value in {"manual_review", "quarantined"} and record.failure is None:
            findings.append(IntegrityFinding(
                "outbox.failure_evidence",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                graph.snapshot.snapshot_id,
                identity.chain_id,
                (identity.parent_uuid,),
                "terminal_without_failure",
                "Outbox terminal review state has no durable failure evidence.",
                (("intent_id", record.intent_id), ("state", record.state.value)),
                (("failure", "present"),),
                (("outbox_snapshot", context.outbox.snapshot_id),),
            ))
        parent_matches = graph.uuid_matches(identity.parent_uuid)
        chain_nodes = graph.chain_nodes(identity.chain_id)
        if not parent_matches or not chain_nodes:
            if graph.snapshot.coverage is not SnapshotCoverage.COMPLETE:
                findings.append(IntegrityFinding(
                    "outbox.parent_coverage",
                    FindingStatus.UNAVAILABLE,
                    FindingSeverity.ERROR,
                    graph.snapshot.snapshot_id,
                    identity.chain_id,
                    (identity.parent_uuid,),
                    "parent_outside_coverage",
                    "Outbox intent references task evidence outside the graph coverage.",
                    (("intent_id", record.intent_id),),
                    (("parent", "covered"),),
                    (("outbox_snapshot", context.outbox.snapshot_id),),
                ))
            else:
                findings.append(IntegrityFinding(
                    "outbox.parent_present",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    graph.snapshot.snapshot_id,
                    identity.chain_id,
                    (identity.parent_uuid,),
                    "outbox_parent_missing",
                    "Outbox intent references a missing parent task.",
                    (("intent_id", record.intent_id),),
                    (("parent", "present"),),
                    (("outbox_snapshot", context.outbox.snapshot_id),),
                ))
        if context.configuration_fingerprint and record.configuration_fingerprint != context.configuration_fingerprint:
            findings.append(IntegrityFinding(
                "outbox.configuration_fingerprint",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                graph.snapshot.snapshot_id,
                identity.chain_id,
                (identity.parent_uuid,),
                "configuration_drift",
                "Outbox intent was created under a different configuration fingerprint.",
                (("intent_id", record.intent_id), ("observed", record.configuration_fingerprint)),
                (("configuration", context.configuration_fingerprint),),
                (("outbox_snapshot", context.outbox.snapshot_id),),
            ))
    return tuple(findings)


def evaluate_context(context: IntegrityContext) -> tuple[IntegrityFinding, ...]:
    """Evaluate graph rules plus separated outbox evidence in stable order."""
    findings = (*evaluate_invariants(context.graph), *_outbox_rule(context))
    unique: dict[tuple[str, str, tuple[str, ...], str], IntegrityFinding] = {}
    for finding in findings:
        unique.setdefault((finding.invariant_id, finding.chain_id, finding.subject_uuids, finding.reason_code), finding)
    return tuple(sorted(unique.values(), key=lambda item: (
        item.chain_id, item.subject_uuids, item.invariant_id, item.reason_code,
    )))


__all__ = ["DEFAULT_INVARIANTS", "InvariantRule", "evaluate_context", "evaluate_invariants"]
