from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from nautical_core.chain_integrity_models import (
    ChainNode,
    ChainReference,
    ChainSnapshot,
    FindingSeverity,
    FindingStatus,
    IntegrityContractError,
    IntegrityFinding,
    IntegrityOperation,
    IntegrityReport,
    IntegrityReportStatus,
    IntegrityRepairPlan,
    ReferenceState,
    RepairOperationKind,
    RepairSafety,
    SnapshotCoverage,
)
from nautical_core.chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
from nautical_core.integration_models import (
    Absent,
    CommandFailureKind,
    FailureEvidence,
    Found,
    TaskCommand,
    TaskCommandResult,
    Unavailable,
)
from nautical_core.task_read_repository import (
    AuthoritativeTaskSnapshot,
    TaskQueryKind,
    TaskSnapshotScope,
)
from nautical_core.task_models import TaskObservation


def node(row: dict[str, object]) -> ChainNode:
    return ChainNode.from_observation(TaskObservation.from_mapping(row, source_query="chain-integrity-test"))


class ChainIntegrityModelTests(unittest.TestCase):
    def test_observations_may_be_incomplete_but_repair_plans_may_not(self) -> None:
        incomplete = node(
            {
                "uuid": "00000000-0000-4000-8000-000000000991",
                "status": "pending",
                "link": "7.000000",
                "nextLink": "00000000",
            }
        )
        self.assertEqual(incomplete.link, 7)
        self.assertFalse(incomplete.has_complete_identity)
        self.assertEqual(incomplete.field("nextLink"), "00000000")

        complete = node(
            {
                "uuid": "00000000-0000-4000-8000-000000000992",
                "status": "pending",
                "chainID": "chain-contract",
                "link": 8,
            }
        )
        reference = ChainReference(
            "nextLink", "00000000", ReferenceState.RESOLVED,
            target_uuid=complete.task_uuid, target_link=8,
        )
        self.assertIs(reference.state, ReferenceState.RESOLVED)

        snapshot = ChainSnapshot(
            "snapshot-contract", SnapshotCoverage.CANDIDATES, "task export",
            (incomplete, complete), "cfg-contract",
        )
        finding = IntegrityFinding(
            "identity.chain_id_required", FindingStatus.REPAIRABLE, FindingSeverity.ERROR,
            snapshot.snapshot_id, "chain-contract", (incomplete.task_uuid,), "missing_chain_id",
            "Nautical recurrence has no chain identity.",
            observed=(("chainID", ""),), expected=(("chainID", "required"),),
            evidence=(("coverage", snapshot.coverage.value),),
        )
        operation = IntegrityOperation(
            "operation-contract", RepairOperationKind.LINK_REPAIR, "chain-contract",
            complete.task_uuid, (("modified", "20260821T120000Z"),),
            ("target remains present",), ("nextLink is reciprocal",),
        )
        plan = IntegrityRepairPlan(
            "plan-contract", snapshot.snapshot_id, "chain-contract", RepairSafety.SAFE,
            "reciprocal_link", "Repair one reciprocal chain link.", (operation,), "cfg-contract",
        )
        report = IntegrityReport(snapshot, IntegrityReportStatus.REPAIRABLE, (finding,), (plan,))
        self.assertIs(report.plans[0].operations[0].kind, RepairOperationKind.LINK_REPAIR)

        invalid_cases = (
            lambda: ChainReference("nextLink", "00000000", ReferenceState.RESOLVED),
            lambda: ChainSnapshot("snapshot-unavailable", SnapshotCoverage.UNAVAILABLE, "task export"),
            lambda: IntegrityRepairPlan(
                "plan-dependency", snapshot.snapshot_id, "chain-contract", RepairSafety.SAFE,
                "reciprocal_link", "invalid dependency",
                (
                    IntegrityOperation(
                        "operation-dependency", RepairOperationKind.LINK_REPAIR,
                        "chain-contract", complete.task_uuid, (("modified", "20260821T120000Z"),),
                        ("target remains present",), ("nextLink is reciprocal",),
                        depends_on=("missing-operation",),
                    ),
                ),
            ),
        )
        for make_invalid in invalid_cases:
            with self.subTest(make_invalid=make_invalid), self.assertRaises(IntegrityContractError):
                make_invalid()

        with self.assertRaises(FrozenInstanceError):
            plan.plan_id = "changed"  # type: ignore[misc]

    def test_snapshot_service_reuses_authoritative_evidence_and_fails_closed(self) -> None:
        command = TaskCommand(("task", "export"), "chain integrity test", 1.0)
        result = TaskCommandResult(command, 0, "[]", "", CommandFailureKind.SUCCESS, 1, 0.001)
        scope = TaskSnapshotScope(TaskQueryKind.BROAD, "chain:on", ("completed", "pending"))
        rows = (
            {"uuid": "00000000-0000-4000-8000-000000000901", "status": "pending", "chainID": "snap-chain", "link": 1},
            {"uuid": "00000000-0000-4000-8000-000000000902", "status": "completed", "chainID": "snap-chain", "link": 2},
        )
        observations = tuple(TaskObservation.from_mapping(row, source_query="chain-snapshot-test") for row in rows)
        authoritative = AuthoritativeTaskSnapshot(scope, observations, result)

        class Repository:
            def __init__(self) -> None:
                self.calls = 0
                self.response = Found(authoritative, "broad:chain:on")

            def broad_snapshot(self, **_kwargs):
                self.calls += 1
                return self.response

        class Unit:
            mutation_epoch = 0

            def __init__(self) -> None:
                self.repository = Repository()

        unit = Unit()
        service = ChainSnapshotService(unit, configuration_fingerprint="cfg-snapshot")
        request = IntegritySnapshotRequest.chain("snap-chain")
        first = service.collect(request)
        second = service.collect(request)
        self.assertIsInstance(first, Found)
        self.assertIsInstance(second, Found)
        self.assertEqual(first.value.rows, second.value.rows)
        self.assertEqual(len(first.value.rows), 2)
        self.assertEqual(unit.repository.calls, 1)

        unit.mutation_epoch = 1
        service.collect(request)
        self.assertEqual(unit.repository.calls, 2)

        unit.repository.response = Absent("broad:chain:on", "no matching chain")
        empty = service.collect(IntegritySnapshotRequest.candidates())
        self.assertIsInstance(empty, Found)
        self.assertEqual(empty.value.rows, ())

        unavailable = Unavailable(
            "broad:chain:on",
            FailureEvidence(command, CommandFailureKind.INVALID_RESPONSE, 1, 1, 0.001, False, "malformed export"),
        )
        unit.repository.response = unavailable
        self.assertIsInstance(service.collect(IntegritySnapshotRequest.candidates(refresh=True)), Unavailable)

        malformed = AuthoritativeTaskSnapshot(
            scope,
            (TaskObservation.from_mapping({"status": "pending"}, source_query="malformed-chain-row"),),
            result,
        )
        unit.repository.response = Found(malformed, "broad:chain:on")
        self.assertIsInstance(service.collect(IntegritySnapshotRequest.candidates(refresh=True)), Unavailable)

        truncated = AuthoritativeTaskSnapshot(scope, observations, result, truncated=True)
        unit.repository.response = Found(truncated, "broad:chain:on")
        self.assertIsInstance(service.collect(IntegritySnapshotRequest.candidates(refresh=True)), Unavailable)

        duplicate_rows = (
            observations[0],
            TaskObservation.from_mapping(
                {**rows[0], "status": "completed", "link": 2}, source_query="duplicate-chain-row"
            ),
        )
        duplicate = AuthoritativeTaskSnapshot(scope, duplicate_rows, result)
        unit.repository.response = Found(duplicate, "broad:chain:on")
        duplicate_read = service.collect(IntegritySnapshotRequest.candidates(refresh=True))
        self.assertIsInstance(duplicate_read, Unavailable)
        self.assertIn("duplicate full UUID", duplicate_read.evidence.detail)

        wrong_chain = AuthoritativeTaskSnapshot(
            scope,
            (TaskObservation.from_mapping({**rows[0], "chainID": "other-chain"}, source_query="wrong-chain"),),
            result,
        )
        unit.repository.response = Found(wrong_chain, "broad:chain:on")
        self.assertIsInstance(
            service.collect(IntegritySnapshotRequest.chain("snap-chain", refresh=True)), Unavailable
        )

        unit.repository.response = Found(authoritative, "broad:uuid")
        uuid_read = service.collect(IntegritySnapshotRequest.uuid(rows[0]["uuid"], refresh=True))
        self.assertIsInstance(uuid_read, Found)
        self.assertEqual(unit.repository.calls, 9)
