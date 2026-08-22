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
    LifecycleIntent,
    ReferenceState,
    SnapshotCoverage,
)
from .lifecycle_models import recurrence_fingerprint


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
                graph, "identity.chain_id_required", FindingStatus.MANUAL_REVIEW, FindingSeverity.ERROR, node,
                "missing_chain_id", "Chain node has no chainID.",
                observed=(("chainID", ""),), expected=(("chainID", "required"),),
            ))
        if node.link is None:
            findings.append(_finding(
                graph, "identity.link_required", FindingStatus.MANUAL_REVIEW, FindingSeverity.ERROR, node,
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


def _missing_link_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        if node.link is not None or not node.chain_id:
            continue
        previous = graph.reference(node.task_uuid, "prevLink")
        following = graph.reference(node.task_uuid, "nextLink")
        if previous.state is not ReferenceState.RESOLVED or following.state is not ReferenceState.RESOLVED:
            continue
        previous_nodes = graph.uuid_matches(previous.target_uuid)
        following_nodes = graph.uuid_matches(following.target_uuid)
        if len(previous_nodes) != 1 or len(following_nodes) != 1:
            continue
        previous_node, following_node = previous_nodes[0], following_nodes[0]
        if previous_node.chain_id != node.chain_id or following_node.chain_id != node.chain_id:
            continue
        if previous_node.link is None or following_node.link is None:
            continue
        expected = previous_node.link + 1
        if following_node.link != expected + 1 or (node.chain_id, expected) in graph.by_slot:
            continue
        findings.append(_finding(
            graph,
            "slot.missing_link",
            FindingStatus.REPAIRABLE,
            FindingSeverity.ERROR,
            node,
            "missing_link_between_neighbors",
            "Adjacent resolved links uniquely determine the missing link number.",
            observed=(("prevLink", previous_node.link), ("nextLink", following_node.link)),
            expected=(("link", expected),),
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
        intent = node.lifecycle_intent
        if intent not in {LifecycleIntent.COMPLETED, LifecycleIntent.DELETED}:
            continue
        status = intent.value
        if graph.reference(node.task_uuid, "nextLink").state is not ReferenceState.ABSENT:
            continue
        if intent is LifecycleIntent.DELETED:
            until = _canonical_timestamp(node.field("until"))
            ended = _canonical_timestamp(node.field("end"))
            if until is None or ended is None:
                findings.append(_finding(
                    graph,
                    "lifecycle.deleted_disposition",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    node,
                    "deleted_expiration_evidence_unavailable",
                    "Deleted chain-on tip lacks reliable expiration evidence.",
                    observed=(("until", node.field("until")), ("end", node.field("end"))),
                    expected=(("until", "parseable"), ("end", "parseable")),
                ))
                continue
            if ended < until:
                findings.append(_finding(
                    graph,
                    "lifecycle.deleted_disposition",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    node,
                    "deleted_before_expiration",
                    "Deleted chain-on tip ended before native until and must not auto-spawn.",
                    observed=(("until", node.field("until")), ("end", node.field("end"))),
                    expected=(("end", "at or after until"),),
                ))
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


def _terminal_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    """Validate terminal bounds before they suppress successor recovery."""
    findings: list[IntegrityFinding] = []
    for node in graph.nodes:
        if node.lifecycle_intent not in {LifecycleIntent.COMPLETED, LifecycleIntent.DELETED}:
            continue
        if graph.reference(node.task_uuid, "nextLink").state is not ReferenceState.ABSENT:
            continue
        chain_max = node.field("chainMax")
        if chain_max not in (None, "", "null"):
            valid_max = not isinstance(chain_max, bool)
            try:
                valid_max = valid_max and int(float(str(chain_max).strip())) > 0
            except (TypeError, ValueError, OverflowError):
                valid_max = False
            if not valid_max:
                findings.append(_finding(
                    graph,
                    "terminal.chain_max_valid",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    node,
                    "invalid_chain_max_terminal_bound",
                    "Terminal chainMax is malformed and cannot justify stopping recovery.",
                    observed=(("chainMax", chain_max),),
                    expected=(("chainMax", "positive integer"),),
                ))
        chain_until = node.field("chainUntil")
        if chain_until not in (None, "", "null") and _canonical_timestamp(chain_until) is None:
            findings.append(_finding(
                graph,
                "terminal.chain_until_valid",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                node,
                "invalid_chain_until_terminal_bound",
                "Terminal chainUntil is malformed and cannot justify stopping recovery.",
                observed=(("chainUntil", chain_until),),
                expected=(("chainUntil", "timezone-aware timestamp"),),
            ))
    return tuple(findings)


def _child_continuity_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    """Validate temporal continuity only across one resolved parent edge."""
    findings: list[IntegrityFinding] = []
    for parent in graph.nodes:
        reference = graph.reference(parent.task_uuid, "nextLink")
        if reference.state is not ReferenceState.RESOLVED:
            continue
        children = graph.uuid_matches(reference.target_uuid)
        if len(children) != 1:
            continue
        child = children[0]
        parent_raw = parent.field("due") or parent.field("scheduled")
        child_raw = child.field("due") or child.field("scheduled")
        parent_dt = _canonical_timestamp(parent_raw)
        child_dt = _canonical_timestamp(child_raw)
        if parent_dt is not None and child_dt is not None and child_dt <= parent_dt:
            findings.append(IntegrityFinding(
                    "continuity.child_temporal_order",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    graph.snapshot.snapshot_id,
                    parent.chain_id,
                    (parent.task_uuid, child.task_uuid),
                    "child_not_after_parent",
                    "Resolved child recurrence target is not later than its parent target.",
                    (("parent_target", str(parent_raw)), ("child_target", str(child_raw))),
                    (("child_target", "after parent_target"),),
                    (("parent_link", parent.link), ("child_link", child.link), ("coverage", graph.snapshot.coverage.value)),
            ))
        parent_kind = next((field for field in ("anchor", "anchor_file", "cp") if str(parent.field(field, "") or "").strip()), "")
        child_kind = next((field for field in ("anchor", "anchor_file", "cp") if str(child.field(field, "") or "").strip()), "")
        if parent_kind and parent_kind != child_kind:
            findings.append(IntegrityFinding(
                "continuity.child_recurrence_identity",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                graph.snapshot.snapshot_id,
                parent.chain_id,
                (parent.task_uuid, child.task_uuid),
                "child_recurrence_identity_mismatch",
                "Resolved child does not retain the parent recurrence kind.",
                (("parent_kind", parent_kind), ("child_kind", child_kind or "<missing>")),
                (("child_kind", parent_kind),),
                (("parent_link", parent.link), ("child_link", child.link)),
            ))
    return tuple(findings)


def _carry_continuity_rule(graph: ChainGraph) -> tuple[IntegrityFinding, ...]:
    """Check relative carried timing only across one resolved edge."""
    findings: list[IntegrityFinding] = []
    for parent in graph.nodes:
        reference = graph.reference(parent.task_uuid, "nextLink")
        if reference.state is not ReferenceState.RESOLVED:
            continue
        children = graph.uuid_matches(reference.target_uuid)
        if len(children) != 1:
            continue
        child = children[0]
        parent_due = _canonical_timestamp(parent.field("due"))
        child_due = _canonical_timestamp(child.field("due"))
        if parent_due is None or child_due is None:
            continue
        for field in ("scheduled", "wait", "until"):
            parent_value = _canonical_timestamp(parent.field(field))
            child_value = _canonical_timestamp(child.field(field))
            if parent_value is None or child_value is None:
                continue
            parent_delta = (parent_value - parent_due).total_seconds()
            child_delta = (child_value - child_due).total_seconds()
            if parent_delta == child_delta:
                continue
            findings.append(IntegrityFinding(
                "carry.child_relative_offset",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                graph.snapshot.snapshot_id,
                parent.chain_id,
                (parent.task_uuid, child.task_uuid),
                "child_carry_offset_changed",
                f"Child {field} offset does not preserve the parent recurrence carry.",
                (("field", field), ("parent_offset_seconds", parent_delta), ("child_offset_seconds", child_delta)),
                (("offset_seconds", parent_delta),),
                (("parent_link", parent.link), ("child_link", child.link)),
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
                FindingStatus.MANUAL_REVIEW,
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
        wait_raw = node.field("wait")
        wait = _canonical_timestamp(wait_raw)
        chain_until_raw = node.field("chainUntil")
        chain_until = _canonical_timestamp(chain_until_raw)
        if due is not None and until is not None and until < due:
            findings.append(_finding(
                graph,
                "carry.until_after_due",
                FindingStatus.MANUAL_REVIEW,
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
        if due is not None and wait is not None and wait > due:
            findings.append(_finding(
                graph,
                "carry.wait_before_due",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                node,
                "wait_after_due",
                "Wait timestamp is later than due.",
                observed=(("wait", str(wait_raw)), ("due", str(due_raw))),
                expected=(("wait", "at or before due"),),
            ))
        if due is not None and chain_until is not None and chain_until < due:
            findings.append(_finding(
                graph,
                "carry.chain_until_after_due",
                FindingStatus.MANUAL_REVIEW,
                FindingSeverity.ERROR,
                node,
                "chain_until_before_due",
                "Chain end point is earlier than the current due timestamp.",
                observed=(("chainUntil", str(chain_until_raw)), ("due", str(due_raw))),
                expected=(("chainUntil", "at or after due"),),
            ))
    return tuple(findings)


DEFAULT_INVARIANTS: tuple[InvariantRule, ...] = (
    InvariantRule("identity", SnapshotCoverage.CANDIDATES, _identity_rule),
    InvariantRule("slot.duplicate_occupant", SnapshotCoverage.CANDIDATES, _duplicate_slot_rule),
    InvariantRule("slot.missing_link", SnapshotCoverage.CHAIN, _missing_link_rule),
    InvariantRule("edge", SnapshotCoverage.CANDIDATES, _edge_rule),
    InvariantRule("edge.topology", SnapshotCoverage.CANDIDATES, _topology_rule),
    InvariantRule("lifecycle", SnapshotCoverage.CANDIDATES, _lifecycle_rule),
    InvariantRule("terminal", SnapshotCoverage.CANDIDATES, _terminal_rule),
    InvariantRule("continuity.child_temporal_order", SnapshotCoverage.CANDIDATES, _child_continuity_rule),
    InvariantRule("carry.child_relative_offset", SnapshotCoverage.CANDIDATES, _carry_continuity_rule),
    InvariantRule("identity.recurrence", SnapshotCoverage.CANDIDATES, _recurrence_identity_rule),
    InvariantRule("carry.temporal", SnapshotCoverage.CANDIDATES, _temporal_rule),
)

# Explicit ownership for checks that historically lived in operator tools.
# This is intentionally data, not executable coupling: front ends may render
# these owners, but the registry remains the only source of invariant logic.
INVARIANT_OWNERSHIP: dict[str, tuple[str, ...]] = {
    "doctor.chain_identity": ("identity",),
    "doctor.chain_slots": ("slot.duplicate_occupant", "slot.missing_link"),
    "doctor.chain_links": ("edge", "edge.topology"),
    "chain_repair.link_inference": ("slot.missing_link", "edge"),
    "native_until.predecessor_and_order": ("carry.temporal",),
    "reconcile.lifecycle_recovery": ("lifecycle", "terminal", "continuity.child_temporal_order"),
    "reconcile.recurrence_identity": ("identity.recurrence",),
}

PRESENTATION_ONLY_CHECKS: frozenset[str] = frozenset({
    "doctor.installation",
    "doctor.configuration",
    "doctor.dependencies",
    "reconcile.rendering",
})


def validate_ownership_map() -> None:
    """Fail fast if a front-end ownership entry references no registry rule."""
    known = {rule.invariant_id for rule in DEFAULT_INVARIANTS}
    missing = sorted({owner for owners in INVARIANT_OWNERSHIP.values() for owner in owners} - known)
    if missing:
        raise RuntimeError("invariant ownership map references unknown rules: " + ", ".join(missing))


def evaluate_invariants(
    graph: ChainGraph,
    rules: Iterable[InvariantRule] = DEFAULT_INVARIANTS,
) -> tuple[IntegrityFinding, ...]:
    """Evaluate rules in stable order and deduplicate identical evidence."""
    findings: list[IntegrityFinding] = []
    for rule in sorted(tuple(rules), key=lambda item: item.invariant_id):
        coverage_ok = (
            rule.required_coverage is SnapshotCoverage.CANDIDATES
            and graph.snapshot.coverage in {SnapshotCoverage.CANDIDATES, SnapshotCoverage.CHAIN, SnapshotCoverage.COMPLETE}
        ) or (
            rule.required_coverage is SnapshotCoverage.CHAIN
            and graph.snapshot.coverage in {SnapshotCoverage.CHAIN, SnapshotCoverage.COMPLETE}
        ) or graph.snapshot.coverage is rule.required_coverage
        if not coverage_ok:
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


def _finalization_rule(context: IntegrityContext) -> tuple[IntegrityFinding, ...]:
    """Check acknowledged terminal plans against the persisted parent tip."""
    graph = context.graph
    findings: list[IntegrityFinding] = []
    for chain_id in sorted({record.plan.identity.chain_id for record in context.outbox.records}):
        for record in context.outbox.terminal_records(chain_id):
            identity = record.plan.identity
            matches = graph.uuid_matches(identity.parent_uuid)
            if len(matches) != 1:
                if graph.snapshot.coverage is not SnapshotCoverage.COMPLETE:
                    findings.append(IntegrityFinding(
                        "lifecycle.finalization_parent_coverage",
                        FindingStatus.UNAVAILABLE,
                        FindingSeverity.ERROR,
                        graph.snapshot.snapshot_id,
                        identity.chain_id,
                        (identity.parent_uuid,),
                        "terminal_parent_outside_coverage",
                        "Acknowledged terminal plan cannot be verified outside snapshot coverage.",
                        (("intent_id", record.intent_id),),
                        (("parent", "covered"),),
                        (("outbox_snapshot", context.outbox.snapshot_id),),
                    ))
                continue
            parent = matches[0]
            guard = record.plan.parent_guard
            observed_identity = {
                "status": parent.status,
                "chain": str(parent.field("chain", "on") or "on"),
                "chainID": parent.chain_id,
                "link": parent.link,
            }
            guard_identity = {
                "status": guard.status,
                "chain": guard.chain,
                "chainID": guard.chain_id,
                "link": guard.link,
            }
            if observed_identity != guard_identity:
                findings.append(_finding(
                    graph,
                    "lifecycle.finalization_guard",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    parent,
                    "terminal_guard_mismatch",
                    "Terminal exhaustion evidence was created for different parent identity facts.",
                    observed=tuple(sorted(observed_identity.items())),
                    expected=tuple(sorted(guard_identity.items())),
                    evidence=(("intent_id", record.intent_id),),
                ))
            observed_fingerprint = recurrence_fingerprint(parent.to_dict())
            if guard.recurrence_fingerprint and guard.recurrence_fingerprint != observed_fingerprint:
                findings.append(_finding(
                    graph,
                    "lifecycle.finalization_recurrence_guard",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    parent,
                    "terminal_recurrence_guard_mismatch",
                    "Terminal exhaustion evidence is stale for the persisted recurrence inputs.",
                    observed=(("recurrence_fingerprint", observed_fingerprint),),
                    expected=(("recurrence_fingerprint", guard.recurrence_fingerprint),),
                    evidence=(("intent_id", record.intent_id),),
                ))
            next_ref = graph.reference(parent.task_uuid, "nextLink")
            chain_state = str(parent.field("chain", "on") or "on").strip().lower()
            if next_ref.state is not ReferenceState.ABSENT or chain_state == "on":
                findings.append(_finding(
                    graph,
                    "lifecycle.finalization_postcondition",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    parent,
                    "terminal_postcondition_mismatch",
                    "Acknowledged terminal plan does not match the persisted parent tip.",
                    observed=(("nextLink", next_ref.state.value), ("chain", chain_state)),
                    expected=(("nextLink", "absent"), ("chain", "off")),
                    evidence=(("intent_id", record.intent_id), ("terminal_kind", record.plan.terminal_kind or identity.event.value)),
                ))
    return tuple(findings)


def _acknowledged_postcondition_rule(context: IntegrityContext) -> tuple[IntegrityFinding, ...]:
    """Verify durable acknowledged lifecycle postconditions against the graph."""
    graph = context.graph
    findings: list[IntegrityFinding] = []
    for record in context.outbox.records:
        state = getattr(record, "state", None)
        if not hasattr(record, "plan") or getattr(state, "value", "") != "acknowledged":
            continue
        plan = record.plan
        parent_matches = graph.uuid_matches(plan.identity.parent_uuid)
        if len(parent_matches) != 1:
            continue
        parent = parent_matches[0]
        child = plan.child_dict()
        expected_child = str(child.get("uuid") or "").strip().lower()
        expected_link = str(plan.parent_patch_dict().get("nextLink") or "").strip().lower()
        for postcondition in plan.expected_postconditions:
            satisfied = True
            if postcondition in {"child_exists", "child_present"}:
                satisfied = bool(expected_child and len(graph.uuid_matches(expected_child)) == 1)
            elif postcondition in {"parent_linked", "parent_next_linked"}:
                actual = str(parent.field("nextLink", "") or "").strip().lower()
                satisfied = bool(expected_link and actual == expected_link)
            elif postcondition in {"chain_off", "parent_chain_off"}:
                satisfied = str(parent.field("chain", "on") or "on").strip().lower() == "off"
            if not satisfied:
                findings.append(_finding(
                    graph,
                    "outbox.acknowledged_postcondition",
                    FindingStatus.MANUAL_REVIEW,
                    FindingSeverity.ERROR,
                    parent,
                    "acknowledged_postcondition_mismatch",
                    "Acknowledged lifecycle intent does not satisfy its persisted postcondition.",
                    observed=(("postcondition", postcondition),),
                    expected=(("postcondition", "satisfied"),),
                    evidence=(("intent_id", record.intent_id),),
                ))
    return tuple(findings)


def evaluate_context(context: IntegrityContext) -> tuple[IntegrityFinding, ...]:
    """Evaluate graph rules plus separated outbox evidence in stable order."""
    findings = (
        *evaluate_invariants(context.graph),
        *_outbox_rule(context),
        *_finalization_rule(context),
        *_acknowledged_postcondition_rule(context),
    )
    unique: dict[tuple[str, str, tuple[str, ...], str], IntegrityFinding] = {}
    for finding in findings:
        unique.setdefault((finding.invariant_id, finding.chain_id, finding.subject_uuids, finding.reason_code), finding)
    return tuple(sorted(unique.values(), key=lambda item: (
        item.chain_id, item.subject_uuids, item.invariant_id, item.reason_code,
    )))


__all__ = ["DEFAULT_INVARIANTS", "InvariantRule", "evaluate_context", "evaluate_invariants"]
