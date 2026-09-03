import unittest
from types import SimpleNamespace

from nautical_core.chain_invariants import _child_continuity_rule, _outbox_rule
from nautical_core.chain_integrity_context import OutboxSnapshot
from nautical_core.chain_integrity_models import ReferenceState, SnapshotCoverage
from nautical_core.lifecycle_models import (
    ExecutionStage, LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard,
)
from nautical_core.lifecycle_outbox import LifecycleOutboxRecord, OutboxProcessingState


class _Node:
    def __init__(self, uuid, values, *, chain="chain", link=1):
        self.task_uuid = uuid
        self.chain_id = chain
        self.link = link
        self._values = values

    def field(self, name, default=None):
        return self._values.get(name, default)


class _Graph:
    def __init__(self, parent, child):
        self.nodes = (parent, child)
        self.snapshot = SimpleNamespace(snapshot_id="snapshot", coverage=SnapshotCoverage.CHAIN)
        self._child = child

    def reference(self, _node, _field):
        if _node == "parent":
            return SimpleNamespace(state=ReferenceState.RESOLVED, target_uuid=self._child.task_uuid)
        return SimpleNamespace(state=ReferenceState.ABSENT, target_uuid=None)

    def uuid_matches(self, uuid):
        return (self._child,) if uuid == self._child.task_uuid else ()


class ChildTemporalInvariantTests(unittest.TestCase):
    def test_cp_uses_completion_reference_when_due_values_match(self):
        parent = _Node(
            "parent",
            {"cp": "1d", "due": "20260903T025500Z", "end": "20260902T060826Z"},
        )
        child = _Node("child", {"cp": "1d", "due": "20260903T025500Z"}, link=2)
        self.assertEqual(_child_continuity_rule(_Graph(parent, child)), ())

    def test_anchor_still_requires_strictly_later_target(self):
        parent = _Node("parent", {"anchor": "w:mon", "due": "20260903T025500Z"})
        child = _Node("child", {"anchor": "w:mon", "due": "20260903T025500Z"}, link=2)
        findings = _child_continuity_rule(_Graph(parent, child))
        self.assertEqual([finding.reason_code for finding in findings], ["child_not_after_parent"])

    def test_acknowledged_finalized_intent_is_not_stage_mismatch(self):
        identity = LifecycleIdentity("chain", "parent", 1, 2, LifecycleEvent.COMPLETE)
        guard = ParentGuard("completed", "on", "chain", 1)
        plan = LifecyclePlan(identity, LifecycleAction.SPAWN_CHILD, guard)
        record = LifecycleOutboxRecord(
            identity.idempotency_key, plan, "config", "schedule",
            OutboxProcessingState.ACKNOWLEDGED, ExecutionStage.FINALIZED,
        )
        graph = SimpleNamespace(
            snapshot=SimpleNamespace(snapshot_id="snapshot"),
            uuid_matches=lambda _uuid: (SimpleNamespace(),),
            chain_nodes=lambda _chain: (SimpleNamespace(),),
        )
        context = SimpleNamespace(
            graph=graph,
            outbox=OutboxSnapshot.from_records((record,)),
            configuration_fingerprint="config",
            schedule_fingerprint="schedule",
        )
        self.assertEqual(_outbox_rule(context), ())


if __name__ == "__main__":
    unittest.main()
