from __future__ import annotations

import unittest

from nautical_core.chain_graph import ChainGraph
from nautical_core.chain_integrity_context import IntegrityContext, OutboxSnapshot
from nautical_core.chain_integrity_models import (
    ChainNode,
    ChainSnapshot,
    LifecycleIntent,
    ReferenceState,
    SnapshotCoverage,
)
from nautical_core.chain_invariants import evaluate_invariants
from nautical_core.chain_repair_planner import IntegrityRepairPlanner


def _node(
    task_uuid: str,
    *,
    chain_id: str = "chain-a",
    link: int | None = None,
    status: str = "pending",
    **fields: object,
) -> ChainNode:
    return ChainNode(
        task_uuid,
        chain_id,
        link,
        status,
        tuple(fields.items()),
    )


class ChainGraphAndRepairPlannerTests(unittest.TestCase):
    def test_graph_indexes_and_short_references_are_deterministic(self) -> None:
        first = _node(
            "aaaaaaaa-0000-0000-0000-000000000911",
            link=1,
            nextLink="bbbbbbbb",
        )
        second = _node(
            "bbbbbbbb-0000-0000-0000-000000000912",
            link=2,
            status="completed",
            prevLink=first.task_uuid,
        )
        snapshot = ChainSnapshot(
            "graph-test", SnapshotCoverage.CHAIN, "unit", (second, first)
        )
        graph = ChainGraph.from_snapshot(snapshot)
        reordered = ChainGraph.from_snapshot(
            ChainSnapshot("graph-test", SnapshotCoverage.CHAIN, "unit", (first, second))
        )

        self.assertEqual(graph.nodes, reordered.nodes)
        self.assertEqual(graph.to_dict(), reordered.to_dict())
        self.assertEqual(graph.slot_nodes("chain-a", 1), (first,))
        self.assertEqual(graph.status_nodes("completed"), (second,))
        self.assertEqual(
            graph.reference(first.task_uuid, "nextLink").state,
            ReferenceState.RESOLVED,
        )
        self.assertEqual(
            graph.reference(second.task_uuid, "prevLink").target_uuid,
            first.task_uuid,
        )

    def test_graph_preserves_ambiguous_and_out_of_coverage_edges(self) -> None:
        outside = _node(
            "cccccccc-0000-0000-0000-000000000913",
            link=3,
            nextLink="dddddddd",
        )
        graph = ChainGraph.from_snapshot(
            ChainSnapshot("partial", SnapshotCoverage.CANDIDATES, "unit", (outside,))
        )
        self.assertEqual(
            graph.reference(outside.task_uuid, "nextLink").state,
            ReferenceState.OUTSIDE_COVERAGE,
        )

        duplicate_a = _node("dddddddd-0000-0000-0000-000000000914", link=4)
        duplicate_b = _node("dddddddd-0000-0000-0000-000000000915", link=5)
        ambiguous = ChainGraph.from_snapshot(
            ChainSnapshot(
                "ambiguous",
                SnapshotCoverage.CANDIDATES,
                "unit",
                (outside, duplicate_a, duplicate_b),
            )
        )
        self.assertEqual(
            ambiguous.reference(outside.task_uuid, "nextLink").state,
            ReferenceState.AMBIGUOUS,
        )
        self.assertEqual(ambiguous.orphan_candidates(), (outside,))

    def test_graph_lifecycle_and_topology_queries_use_semantic_intent(self) -> None:
        root = _node(
            "aaaaaaaa-0000-0000-0000-000000000916",
            link=1,
            nextLink="bbbbbbbb",
        )
        child = _node(
            "bbbbbbbb-0000-0000-0000-000000000917",
            link=2,
            status="completed",
            prevLink=root.task_uuid,
        )
        disabled = _node(
            "cccccccc-0000-0000-0000-000000000918",
            chain_id="chain-b",
            link=1,
            chain="off",
        )
        graph = ChainGraph.from_snapshot(
            ChainSnapshot(
                "topology", SnapshotCoverage.CHAIN, "unit", (child, disabled, root)
            )
        )

        self.assertEqual(graph.roots("chain-a"), (root,))
        self.assertEqual(graph.tips("chain-a"), (child,))
        self.assertEqual(graph.referenced_children(), (child,))
        self.assertEqual(
            graph.lifecycle_nodes(LifecycleIntent.DISABLED.value), (disabled,)
        )

    def test_repair_planner_is_repeatable_and_refuses_partial_coverage(self) -> None:
        source = _node(
            "aaaaaaaa-0000-0000-0000-000000000931",
            link=1,
            anchor="w:mon",
            nextLink="bbbbbbbb",
        )
        target = _node("bbbbbbbb-0000-0000-0000-000000000932", link=2)
        snapshot = ChainSnapshot(
            "planner", SnapshotCoverage.CHAIN, "unit", (target, source)
        )
        graph = ChainGraph.from_snapshot(snapshot)
        context = IntegrityContext(
            graph, OutboxSnapshot.from_records(()), "cfg-planner"
        )
        findings = evaluate_invariants(graph)

        planner = IntegrityRepairPlanner()
        result = planner.plan(context, findings)
        self.assertEqual(len(result.plans), 1)
        self.assertEqual(
            result.plans[0].operations[0].payload,
            (("prevLink", source.task_uuid),),
        )
        self.assertEqual(result.plans, planner.plan(context, findings).plans)
        self.assertEqual(
            len(planner.plan(context, findings + findings).plans), 1
        )

        partial_graph = ChainGraph.from_snapshot(
            ChainSnapshot(
                "partial-planner",
                SnapshotCoverage.CANDIDATES,
                "unit",
                (target, source),
            )
        )
        partial_context = IntegrityContext(
            partial_graph, OutboxSnapshot.from_records(())
        )
        partial = planner.plan(partial_context, findings)
        self.assertFalse(partial.plans)
        self.assertTrue(partial.refusals)

        predecessor = _node(
            "cccccccc-0000-0000-0000-000000000933",
            chain_id="slot-chain",
            link=1,
            nextLink="eeeeeeee",
        )
        missing = _node(
            "eeeeeeee-0000-0000-0000-000000000934",
            chain_id="slot-chain",
            prevLink="cccccccc",
            nextLink="ffffffff",
        )
        successor = _node(
            "ffffffff-0000-0000-0000-000000000935",
            chain_id="slot-chain",
            link=3,
            prevLink="eeeeeeee",
        )
        slot_graph = ChainGraph.from_snapshot(
            ChainSnapshot(
                "slot-planner",
                SnapshotCoverage.CHAIN,
                "unit",
                (successor, missing, predecessor),
            )
        )
        slot_context = IntegrityContext(
            slot_graph, OutboxSnapshot.from_records(()), "cfg-planner"
        )
        slot = planner.plan(slot_context, evaluate_invariants(slot_graph))
        self.assertTrue(any(plan.reason_code == "missing_link" for plan in slot.plans))

    def test_reciprocal_repair_refuses_ambiguous_source_slot(self) -> None:
        root = _node(
            "aaaaaaaa-0000-0000-0000-000000000941",
            link=19,
            nextLink="bbbbbbbb",
        )
        proposed = _node(
            "bbbbbbbb-0000-0000-0000-000000000942",
            link=20,
            prevLink=root.task_uuid,
            nextLink="dddddddd",
        )
        competing = _node(
            "cccccccc-0000-0000-0000-000000000943",
            link=20,
            prevLink=root.task_uuid,
            nextLink="eeeeeeee",
        )
        graph = ChainGraph.from_snapshot(
            ChainSnapshot(
                "ambiguous-repair",
                SnapshotCoverage.CHAIN,
                "unit",
                (root, proposed, competing),
            )
        )
        context = IntegrityContext(graph, OutboxSnapshot.from_records(()), "cfg-planner")

        result = IntegrityRepairPlanner().plan(context, evaluate_invariants(graph))

        self.assertFalse(result.plans)
        self.assertTrue(any(
            refusal.reason == "ambiguous_slot_occupancy"
            for refusal in result.refusals
        ))


if __name__ == "__main__":
    unittest.main()
