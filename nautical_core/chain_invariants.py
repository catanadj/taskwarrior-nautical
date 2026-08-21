"""Pure invariant registry for immutable chain graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .chain_graph import ChainGraph
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


DEFAULT_INVARIANTS: tuple[InvariantRule, ...] = (
    InvariantRule("identity", SnapshotCoverage.CANDIDATES, _identity_rule),
    InvariantRule("slot.duplicate_occupant", SnapshotCoverage.CANDIDATES, _duplicate_slot_rule),
    InvariantRule("edge", SnapshotCoverage.CANDIDATES, _edge_rule),
)


def evaluate_invariants(
    graph: ChainGraph,
    rules: Iterable[InvariantRule] = DEFAULT_INVARIANTS,
) -> tuple[IntegrityFinding, ...]:
    """Evaluate rules in stable order and deduplicate identical evidence."""
    findings: list[IntegrityFinding] = []
    for rule in sorted(tuple(rules), key=lambda item: item.invariant_id):
        if graph.snapshot.coverage in {SnapshotCoverage.UNAVAILABLE, SnapshotCoverage.TRUNCATED}:
            continue
        findings.extend(rule.evaluate(graph))
    unique: dict[tuple[str, str, tuple[str, ...], str], IntegrityFinding] = {}
    for finding in findings:
        key = (finding.invariant_id, finding.chain_id, finding.subject_uuids, finding.reason_code)
        unique.setdefault(key, finding)
    return tuple(sorted(unique.values(), key=lambda item: (
        item.chain_id, item.subject_uuids, item.invariant_id, item.reason_code,
    )))


__all__ = ["DEFAULT_INVARIANTS", "InvariantRule", "evaluate_invariants"]
