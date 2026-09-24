from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nautical_core.chain_integrity_engine import ChainIntegrityEngine, IntegrityEngineResult
from nautical_core.chain_integrity_application import IntegrityApplicationResult, IntegrityApplicationService
from nautical_core.chain_integrity_context import (
    IntegrityContext,
    OutboxCoverage,
    OutboxSnapshot,
    load_outbox_snapshot,
)
from nautical_core.chain_graph import ChainGraph
from nautical_core.chain_invariants import (
    DEFAULT_INVARIANTS,
    INVARIANT_OWNERSHIP,
    evaluate_context,
    evaluate_invariants,
    validate_ownership_map,
)
from nautical_core.chain_integrity_models import (
    ChainNode,
    ChainSnapshot,
    IntegrityOperation,
    IntegrityReportStatus,
    IntegrityRepairPlan,
    RepairOperationKind,
    RepairSafety,
    SnapshotCoverage,
)
from nautical_core.chain_snapshot import IntegritySnapshotKind, IntegritySnapshotRequest
from nautical_core.integration_models import (
    CommandFailureKind,
    FailureEvidence,
    Found,
    TaskCommand,
    MutationOutcomeKind,
    Unavailable,
)
from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
from nautical_core.task_models import NauticalTask, TaskDraft, TaskObservation
from nautical_core.lifecycle_models import (
    LifecycleAction,
    LifecycleEvent,
    LifecycleIdentity,
    LifecyclePlan,
    ParentGuard,
)
from nautical_core.lifecycle_outbox import ExecutionStage, LifecycleOutboxRecord, OutboxProcessingState


def node(row: dict[str, object]) -> ChainNode:
    return ChainNode.from_observation(TaskObservation.from_mapping(row, source_query="chain-engine-test"))


class ChainIntegrityEngineTests(unittest.TestCase):
    def test_integrity_context_keeps_graph_and_outbox_provenance_separate(self) -> None:
        graph = ChainGraph.from_snapshot(
            ChainSnapshot("context-graph", SnapshotCoverage.CANDIDATES, "taskwarrior", ())
        )
        context = IntegrityContext(
            graph, OutboxSnapshot.from_records(()), "cfg-context", mutation_epoch=3,
            metadata={"source": "test"},
        )
        self.assertEqual(context.snapshot_id, "context-graph")
        self.assertIs(context.outbox.coverage, OutboxCoverage.COMPLETE)
        self.assertEqual(context.outbox.records, ())
        self.assertIsNone(context.outbox.by_intent("missing"))
        self.assertEqual(context.metadata["source"], "test")

        failed = IntegrityContext(graph, OutboxSnapshot.unavailable("sqlite unavailable"))
        self.assertFalse(failed.outbox_available)
        self.assertEqual(failed.outbox.reason, "sqlite unavailable")
        unavailable_findings = evaluate_context(failed)
        self.assertTrue(any(
            item.invariant_id == "outbox.snapshot_available" and item.status.value == "unavailable"
            for item in unavailable_findings
        ))

        class Repository:
            def snapshot_records(self):
                return type("Result", (), {"ok": True, "reason": ""})(), ()

        loaded = load_outbox_snapshot(Repository())
        self.assertIs(loaded.coverage, OutboxCoverage.COMPLETE)
        self.assertEqual(loaded.records, ())

        class FailedRepository:
            def snapshot_records(self):
                return type("Result", (), {"ok": False, "reason": "database busy"})(), ()

        self.assertIs(load_outbox_snapshot(FailedRepository()).coverage, OutboxCoverage.UNAVAILABLE)

    def test_application_refuses_unregistered_unsafe_and_misrouted_repairs(self) -> None:
        operation = IntegrityOperation(
            "application-op", RepairOperationKind.LINK_REPAIR, "application-chain",
            "aaaaaaaa-0000-0000-0000-000000000941", (("snapshot_id", "application-snapshot"),),
            ("target remains present",), ("link is reciprocal",),
        )
        plan = IntegrityRepairPlan(
            "application-plan", "application-snapshot", "application-chain", RepairSafety.SAFE,
            "reciprocal_link", "test application", (operation,), "cfg-application",
        )
        results = IntegrityApplicationService().apply(plan, object(), lambda _operation: object())
        self.assertIs(results[0].kind, MutationOutcomeKind.MANUAL_REVIEW)
        stale = IntegrityApplicationResult(
            "stale-plan", "stale-op", MutationOutcomeKind.CONFLICT, "guard modified changed"
        )
        self.assertTrue(stale.stale)

        unsafe_plan = IntegrityRepairPlan(
            "unsafe-application-plan", "application-snapshot", "application-chain", RepairSafety.MANUAL,
            "manual_only", "must not be applied automatically", (operation,), "cfg-application",
        )

        def forbidden(_request):
            raise AssertionError("unsafe plan reached metadata mutation")

        unsafe = IntegrityApplicationService().apply(
            unsafe_plan, type("Executor", (), {"repair_metadata": forbidden})(), lambda _operation: object()
        )
        self.assertIs(unsafe[0].kind, MutationOutcomeKind.MANUAL_REVIEW)
        self.assertIn("SAFE", unsafe[0].reason)

        lifecycle_operation = IntegrityOperation(
            "lifecycle-op", RepairOperationKind.LIFECYCLE_TRANSITION, "application-chain",
            operation.target_uuid, (("snapshot_id", "application-snapshot"),),
            ("target remains present",), ("lifecycle transition applied",), (("action", "complete"),),
        )
        lifecycle_plan = IntegrityRepairPlan(
            "lifecycle-plan", "application-snapshot", "application-chain", RepairSafety.SAFE,
            "lifecycle", "lifecycle work belongs to the lifecycle service", (lifecycle_operation,),
            "cfg-application",
        )
        misrouted = IntegrityApplicationService().apply(
            lifecycle_plan,
            type("Executor", (), {"repair_metadata": forbidden})(),
            lambda _operation: object(),
        )
        self.assertIs(misrouted[0].kind, MutationOutcomeKind.MANUAL_REVIEW)
        self.assertIn("no application adapter", misrouted[0].reason)

        metadata_operation = IntegrityOperation(
            "metadata-op", RepairOperationKind.METADATA_REPAIR, "application-chain",
            operation.target_uuid, (("snapshot_id", "application-snapshot"),),
            ("target remains present",), ("link is 2",), (("link", 2),),
        )
        metadata_plan = IntegrityRepairPlan(
            "metadata-plan", "application-snapshot", "application-chain", RepairSafety.SAFE,
            "missing_link", "test metadata application", (metadata_operation,), "cfg-application",
        )
        accepted_but_unvalidated = IntegrityApplicationService().apply(
            metadata_plan,
            type("Executor", (), {"repair_metadata": lambda _self, _request: type(
                "Mutation", (), {"kind": MutationOutcomeKind.APPLIED, "reason": ""}
            )()})(),
            lambda _operation: object(),
        )
        self.assertIs(accepted_but_unvalidated[0].kind, MutationOutcomeKind.MANUAL_REVIEW)

        second_operation = IntegrityOperation(
            "metadata-op-2", RepairOperationKind.METADATA_REPAIR, "application-chain",
            "bbbbbbbb-0000-0000-0000-000000000942", (("snapshot_id", "application-snapshot"),),
            ("target remains present",), ("link is 3",), (("link", 3),),
        )
        multi_plan = IntegrityRepairPlan(
            "multi-plan", "application-snapshot", "application-chain", RepairSafety.SAFE,
            "structural_batch", "test multi-operation application", (metadata_operation, second_operation),
            "cfg-application",
        )
        multi = IntegrityApplicationService().apply(multi_plan, object(), lambda _operation: object())
        self.assertTrue(multi)
        self.assertTrue(all(item.kind is MutationOutcomeKind.MANUAL_REVIEW for item in multi))

    def test_acknowledged_lifecycle_postconditions_are_checked_against_current_graph(self) -> None:
        parent_uuid = "11111111-0000-0000-0000-000000000927"
        identity = LifecycleIdentity("final-chain", parent_uuid, 2, None, LifecycleEvent.CHAIN_UNTIL)
        guard = ParentGuard("completed", "on", "final-chain", 2, "fp-final")
        plan = LifecyclePlan(
            identity,
            LifecycleAction.FINALIZE_CHAIN,
            guard,
            parent_patch=(("chain", "off"),),
            expected_postconditions=("chain_off", "no_successor"),
            terminal_kind="date_limit",
        )
        record = LifecycleOutboxRecord(
            identity.idempotency_key,
            plan,
            "cfg-final",
            "sched-final",
            OutboxProcessingState.ACKNOWLEDGED,
            ExecutionStage.FINALIZED,
        )
        enabled_parent = node(
            {"uuid": parent_uuid, "status": "completed", "chain": "on", "chainID": "final-chain", "link": 2}
        )
        graph = ChainGraph.from_snapshot(
            ChainSnapshot("finalization-test", SnapshotCoverage.CHAIN, "test", (enabled_parent,))
        )
        context = IntegrityContext(graph, OutboxSnapshot.from_records((record,)), "cfg-final")
        findings = evaluate_context(context)
        reasons = {item.reason_code for item in findings}
        self.assertIn("terminal_postcondition_mismatch", reasons)
        self.assertIn("terminal_recurrence_guard_mismatch", reasons)

        disabled_parent = node(
            {**enabled_parent.to_dict(), "chain": "off"}
        )
        disabled_graph = ChainGraph.from_snapshot(
            ChainSnapshot("finalization-ok", SnapshotCoverage.CHAIN, "test", (disabled_parent,))
        )
        disabled_findings = evaluate_context(
            IntegrityContext(disabled_graph, OutboxSnapshot.from_records((record,)), "cfg-final")
        )
        self.assertNotIn("terminal_postcondition_mismatch", {item.reason_code for item in disabled_findings})
        self.assertEqual(LifecyclePlan.from_dict(plan.to_dict()).terminal_kind, "date_limit")

        spawn_identity = LifecycleIdentity("final-chain", parent_uuid, 2, 3, LifecycleEvent.COMPLETE)
        child_observation = TaskObservation.from_mapping(
            {
                "uuid": "22222222-0000-4000-8000-000000000930", "description": "next", "status": "pending",
                "chain": "on", "chainID": "final-chain", "link": 3, "prevLink": parent_uuid[:8],
                "cp": "1d", "due": "20260824T090000Z",
            },
            source_query="integrity-child-draft",
        )
        draft = TaskDraft.from_task(NauticalTask.from_observation(child_observation))
        spawn_plan = LifecyclePlan.from_draft(
            identity=spawn_identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            draft=draft,
            parent_patch={"nextLink": "22222222"},
            expected_postconditions=("child_exists", "parent_linked"),
        )
        spawn_record = LifecycleOutboxRecord(
            spawn_identity.idempotency_key,
            spawn_plan,
            "cfg-final",
            "sched-final",
            OutboxProcessingState.ACKNOWLEDGED,
            ExecutionStage.FINALIZED,
        )
        spawn_findings = evaluate_context(
            IntegrityContext(graph, OutboxSnapshot.from_records((spawn_record,)), "cfg-final")
        )
        self.assertIn("acknowledged_postcondition_mismatch", {item.reason_code for item in spawn_findings})

    def test_invariant_ownership_map_references_registered_rules(self) -> None:
        validate_ownership_map()
        known = {rule.invariant_id for rule in DEFAULT_INVARIANTS}
        self.assertTrue(INVARIANT_OWNERSHIP)
        self.assertTrue(all(owner in known for owners in INVARIANT_OWNERSHIP.values() for owner in owners))

    def test_invariant_registry_reports_boundaries_deterministically(self) -> None:
        first = {
            "uuid": "aaaaaaaa-0000-4000-8000-000000000921", "status": "pending",
            "chainID": "invariant-chain", "link": 1, "anchor": "w:mon", "nextLink": "bbbbbbbb",
        }
        second = {
            "uuid": "bbbbbbbb-0000-4000-8000-000000000922", "status": "pending",
            "chainID": "invariant-chain", "link": 2, "anchor": "w:mon", "prevLink": "aaaaaaaa",
        }

        def graph(name: str, coverage: SnapshotCoverage, *rows: dict[str, object]) -> ChainGraph:
            snapshot = ChainSnapshot(name, coverage, "test", tuple(node(row) for row in rows))
            return ChainGraph.from_snapshot(snapshot)

        healthy = graph("invariant-healthy", SnapshotCoverage.CHAIN, second, first)
        self.assertEqual(evaluate_invariants(healthy), ())

        broken = graph(
            "invariant-broken", SnapshotCoverage.CANDIDATES,
            {"uuid": "dddddddd-0000-0000-0000-000000000924", "status": "pending", "chainID": "invariant-chain", "link": 1},
            second,
            {"uuid": "cccccccc-0000-0000-0000-000000000923", "status": "pending", "link": 1, "nextLink": "bbbbbbbb"},
            first,
        )
        findings = evaluate_invariants(broken)
        ids = {(item.invariant_id, item.reason_code) for item in findings}
        self.assertTrue({
            ("identity.chain_id_required", "missing_chain_id"),
            ("identity.recurrence_required", "missing_recurrence_identity"),
            ("slot.duplicate_occupant", "duplicate_slot"),
            ("edge.reciprocal", "non_reciprocal_reference"),
        } <= ids)
        self.assertEqual(findings, evaluate_invariants(broken))

        truncated = evaluate_invariants(graph("invariant-truncated", SnapshotCoverage.TRUNCATED, first))
        self.assertTrue(truncated)
        self.assertTrue(all(item.status.value == "unavailable" for item in truncated))

        temporal = graph(
            "invariant-temporal", SnapshotCoverage.CHAIN,
            {"uuid": "eeeeeeee-0000-0000-0000-000000000925", "status": "pending",
             "chainID": "temporal-chain", "link": 1, "anchor": "w:mon",
             "due": "20260821T120000Z", "until": "20260821T110000Z", "wait": "20260821T130000Z"},
        )
        temporal_ids = {(item.invariant_id, item.reason_code) for item in evaluate_invariants(temporal)}
        self.assertTrue({("carry.until_after_due", "until_before_due"), ("carry.wait_before_due", "wait_after_due")} <= temporal_ids)

        terminal = graph(
            "invariant-terminal", SnapshotCoverage.CANDIDATES,
            {"uuid": "ffffffff-0000-0000-0000-000000000926", "status": "completed",
             "chain": "on", "chainID": "terminal-chain", "link": 4,
             "chainMax": "not-a-number", "chainUntil": "not-a-date"},
        )
        terminal_ids = {(item.invariant_id, item.reason_code) for item in evaluate_invariants(terminal)}
        self.assertTrue({
            ("terminal.chain_max_valid", "invalid_chain_max_terminal_bound"),
            ("terminal.chain_until_valid", "invalid_chain_until_terminal_bound"),
        } <= terminal_ids)

        deleted = graph(
            "invariant-deleted", SnapshotCoverage.CANDIDATES,
            {"uuid": "abababab-0000-0000-0000-000000000931", "status": "deleted",
             "chain": "on", "chainID": "deleted-chain", "link": 3},
        )
        deleted_ids = {(item.invariant_id, item.reason_code) for item in evaluate_invariants(deleted)}
        self.assertIn(("lifecycle.deleted_disposition", "deleted_expiration_evidence_unavailable"), deleted_ids)

        backward = graph(
            "invariant-continuity", SnapshotCoverage.CANDIDATES,
            {"uuid": "12121212-0000-0000-0000-000000000928", "status": "completed",
             "chain": "on", "chainID": "continuity-chain", "link": 1, "anchor": "w:mon",
             "due": "20260822T120000Z", "scheduled": "20260822T110000Z",
             "until": "20260822T230000Z", "nextLink": "13131313"},
            {"uuid": "13131313-0000-0000-0000-000000000929", "status": "pending",
             "chain": "on", "chainID": "continuity-chain", "link": 2, "due": "20260822T110000Z",
             "scheduled": "20260822T103000Z", "until": "20260823T230000Z", "prevLink": "12121212"},
        )
        continuity = evaluate_invariants(backward)
        continuity_ids = {(item.invariant_id, item.reason_code) for item in continuity}
        self.assertTrue({
            ("continuity.child_temporal_order", "child_not_after_parent"),
            ("continuity.child_recurrence_identity", "child_recurrence_identity_mismatch"),
            ("carry.child_relative_offset", "child_carry_offset_changed"),
        } <= continuity_ids)
        self.assertFalse([item for item in continuity if dict(item.observed).get("field") == "until"])

        exact_until = graph(
            "invariant-exact-until", SnapshotCoverage.CANDIDATES,
            {"uuid": "14141414-0000-0000-0000-000000000930", "status": "completed",
             "chain": "on", "chainID": "exact-until", "link": 1, "cp": "1d",
             "due": "20260822T100000Z", "until": "20260822T230001Z", "nextLink": "15151515"},
            {"uuid": "15151515-0000-0000-0000-000000000931", "status": "pending",
             "chain": "on", "chainID": "exact-until", "link": 2, "cp": "1d",
             "due": "20260823T100000Z", "until": "20260823T220001Z", "prevLink": "14141414"},
        )
        exact_findings = [item for item in evaluate_invariants(exact_until) if dict(item.observed).get("field") == "until"]
        self.assertTrue(exact_findings)

    def test_engine_owns_empty_audit_replay_and_noop_drain(self) -> None:
        class Provider:
            def collect(self, _request):
                return Found(
                    ChainSnapshot(
                        "engine-snapshot", SnapshotCoverage.CHAIN, "test.provider",
                        complete_chain_history=True,
                    ),
                    "engine snapshot",
                )

        with TemporaryDirectory() as directory:
            outbox = _LifecycleOutboxRepository(Path(directory))
            self.assertTrue(outbox.open().ok)
            engine = ChainIntegrityEngine(Provider(), configuration_fingerprint="cfg-engine")
            audited = engine.audit(IntegritySnapshotRequest.chain("engine-chain"), outbox_repository=outbox)
            self.assertIs(audited.status, IntegrityReportStatus.HEALTHY)
            self.assertEqual(engine.audit_snapshot(audited.snapshot, outbox_repository=outbox), audited)
            applied = engine.apply(
                audited,
                executor=object(),
                request_factory=lambda _operation: None,
                outbox_repository=outbox,
                owner="engine-test",
            )
            self.assertIs(applied.status, IntegrityReportStatus.HEALTHY)
            self.assertEqual(applied.applications, ())

    def test_audit_and_snapshot_front_ends_produce_identical_report_payloads(self) -> None:
        node_value = node(
            {
                "uuid": "aaaaaaaa-0000-0000-0000-000000000936",
                "status": "pending", "chainID": "parity-chain", "link": 1,
                "chain": "on", "anchor": "w:mon",
            }
        )

        class Provider:
            def collect(self, _request):
                return Found(
                    ChainSnapshot("parity-snapshot", SnapshotCoverage.CHAIN, "test", (node_value,)),
                    "test",
                )

        with TemporaryDirectory() as directory:
            outbox = _LifecycleOutboxRepository(Path(directory))
            self.assertTrue(outbox.open().ok)
            engine = ChainIntegrityEngine(Provider(), configuration_fingerprint="cfg-parity")
            first = engine.audit(IntegritySnapshotRequest.chain("parity-chain"), outbox_repository=outbox)
            second = engine.audit_snapshot(first.snapshot, outbox_repository=outbox)

        self.assertEqual(
            tuple(item.to_dict() for item in first.findings),
            tuple(item.to_dict() for item in second.findings),
        )
        self.assertEqual(
            tuple(item.to_dict() for item in first.plans),
            tuple(item.to_dict() for item in second.plans),
        )

    def test_bounded_hydration_uses_chain_scope_and_fails_closed(self) -> None:
        parent = node(
            {
                "uuid": "aaaaaaaa-0000-0000-0000-000000000931", "status": "completed",
                "chain": "on", "chainID": "hydrate-chain", "link": 2, "prevLink": "outside0",
            }
        )
        predecessor = node(
            {
                "uuid": "bbbbbbbb-0000-0000-0000-000000000932", "status": "deleted",
                "chain": "on", "chainID": "hydrate-chain", "link": 1,
                "nextLink": parent.task_uuid,
            }
        )
        broad = ChainSnapshot("hydrate-candidates", SnapshotCoverage.CANDIDATES, "test", (parent,))

        class Provider:
            def __init__(self) -> None:
                self.requests = []

            def collect(self, request):
                self.requests.append(request)
                if request.kind is IntegritySnapshotKind.CANDIDATES:
                    return Found(broad, "candidates")
                return Found(
                    ChainSnapshot("hydrate-chain-full", SnapshotCoverage.CHAIN, "test", (parent, predecessor)),
                    "chain:hydrate-chain",
                )

        provider = Provider()
        engine = ChainIntegrityEngine(provider, configuration_fingerprint="cfg-hydrate", max_hydrated_chains=1)
        with TemporaryDirectory() as directory:
            outbox = _LifecycleOutboxRepository(Path(directory))
            self.assertTrue(outbox.open().ok)
            result = engine.audit(IntegritySnapshotRequest.candidates(), outbox_repository=outbox)
        self.assertNotEqual(result.status.value, "unavailable")
        self.assertEqual(len(result.snapshot.rows), 2)
        self.assertTrue(any(item.kind is IntegritySnapshotKind.CHAIN for item in provider.requests))

        class BusyProvider(Provider):
            def collect(self, request):
                if request.kind is IntegritySnapshotKind.CHAIN:
                    command = TaskCommand(("task", "export"), "hydration test", 1.0)
                    return Unavailable(
                        "chain:hydrate-chain",
                        FailureEvidence(
                            command, CommandFailureKind.BUSY, 1, 1, 0.001, True, "database busy"
                        ),
                    )
                return super().collect(request)

        busy_provider = BusyProvider()
        busy_engine = ChainIntegrityEngine(
            busy_provider, configuration_fingerprint="cfg-hydrate", max_hydrated_chains=1
        )
        with TemporaryDirectory() as directory:
            outbox = _LifecycleOutboxRepository(Path(directory))
            self.assertTrue(outbox.open().ok)
            unavailable = busy_engine.audit(
                IntegritySnapshotRequest.candidates(), outbox_repository=outbox
            )
        self.assertIs(unavailable.status, IntegrityReportStatus.UNAVAILABLE)

        with self.assertRaises(ValueError):
            ChainIntegrityEngine(provider, configuration_fingerprint="cfg-hydrate", max_hydrated_chains=0)

    def test_engine_apply_surfaces_durable_outbox_failure_before_mutation(self) -> None:
        operation = IntegrityOperation(
            "engine-outbox-op", RepairOperationKind.METADATA_REPAIR, "engine-outbox-chain",
            "aaaaaaaa-0000-0000-0000-000000000950", (("snapshot_id", "engine-outbox-snapshot"),),
            ("target remains present",), ("link is 2",), (("link", 2),),
        )
        second = IntegrityOperation(
            "engine-outbox-op-2", RepairOperationKind.METADATA_REPAIR, "engine-outbox-chain",
            "bbbbbbbb-0000-0000-0000-000000000951", (("snapshot_id", "engine-outbox-snapshot"),),
            ("target remains present",), ("link is 3",), (("link", 3),),
        )
        plan = IntegrityRepairPlan(
            "engine-outbox-plan", "engine-outbox-snapshot", "engine-outbox-chain", RepairSafety.SAFE,
            "structural_batch", "durable failure", (operation, second), "cfg-engine-outbox",
        )

        class FailingRepository:
            def enqueue_integrity(self, _envelope):
                return type("Result", (), {"ok": False, "reason": "disk full"})()

            def claim_integrity_batch(self, **_kwargs):
                raise AssertionError("drain must not run after persistence failure")

        class ForbiddenExecutor:
            def repair_metadata(self, _request):
                raise AssertionError("mutation must not run before durable persistence")

        engine = ChainIntegrityEngine.lifecycle_only(configuration_fingerprint="cfg-engine-outbox")
        result = engine.apply(
            IntegrityEngineResult(
                IntegrityReportStatus.REPAIRABLE,
                snapshot=ChainSnapshot("engine-outbox-snapshot", SnapshotCoverage.CHAIN, "test", ()),
                plans=(plan,),
            ),
            executor=ForbiddenExecutor(),
            request_factory=lambda _operation: None,
            outbox_repository=FailingRepository(),
            owner="engine-test",
            drain=False,
        )
        self.assertEqual(result.status, IntegrityReportStatus.MANUAL_REVIEW)
        self.assertEqual(len(result.applications), 2)
        self.assertTrue(all(item.kind is MutationOutcomeKind.MANUAL_REVIEW for item in result.applications))
        self.assertTrue(all("disk full" in item.reason for item in result.applications))
