import unittest
import json
from datetime import datetime, timezone
from pathlib import Path

from nautical_core.operator_application import DomainApplicationRegistry
from nautical_core.operator_control_plane import OperatorControlPlane
from nautical_core.operator_inspectors import inspect_operator_snapshot, standard_inspector_bundle, run_inspectors
from nautical_core.operator_models import (CoverageKind, CoverageRequirement, OperatorCapabilities, OperatorCoverage, OperatorFailure, OperatorLimits,
    OperatorScope, OperatorScopeKind, OperatorOperation, OperatorRequest, OperatorV2Result, OperatorV2Status)
from nautical_core.operator_findings import FindingActionability, FindingSeverity, OperatorFinding
from nautical_core.operator_presentation import ordered_findings, ordered_records, render_contract_json
from nautical_core.operator_snapshot import OperatorSnapshot
from nautical_core.operator_snapshot import ChainSnapshotReader, SnapshotReadRequest
from nautical_core.integration_context import (
    IntegrationAccess, IntegrationContext, SilentDiagnostics,
    SystemClock, ValidatedNauticalConfiguration,
)
from nautical_core.integration_models import (
    CommandFailureKind, FailureEvidence, TaskCommand, Unavailable,
)
from nautical_core.operator_models import OperatorFailure
from nautical_core.operator_plans import OperatorPlan
from nautical_core.operator_context import OperatorInvocationContext


class OperatorConformanceTests(unittest.TestCase):
    def test_control_plane_and_direct_inspection_share_one_projection(self) -> None:
        snapshot = OperatorSnapshot(
            "conformance-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
        )
        requirement = CoverageRequirement(CoverageKind.COMPLETE)
        limits = OperatorLimits()
        expected = inspect_operator_snapshot(snapshot, requirement, limits, scope=OperatorScope.system())

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        actual = control_plane.inspect(snapshot, requirement, limits, scope=OperatorScope.system())
        self.assertEqual(actual, expected)
        self.assertEqual(
            tuple(item.to_dict() for item in actual),
            tuple(item.to_dict() for item in control_plane.inspect(snapshot, requirement, limits, scope=OperatorScope.system())),
        )

    def test_shuffled_findings_have_one_stable_order(self) -> None:
        findings = [
            OperatorFinding("b", "chain", FindingSeverity.WARNING, FindingActionability.INFORMATIONAL, "b", affected=("z",), guidance="inspect"),
            OperatorFinding("a", "chain", FindingSeverity.ERROR, FindingActionability.BLOCKING, "a", affected=("a",), guidance="inspect"),
        ]
        records = [item.to_dict() for item in findings]
        reversed_records = list(reversed(records))
        self.assertEqual(ordered_findings(records), ordered_findings(reversed_records))

    def test_shuffled_operator_domains_have_stable_projection(self) -> None:
        """Task, chain, outbox, and plan records remain deterministic when input order varies."""
        records = [
            {"chain_id": "chain-b", "link": 2, "uuid": "task-b", "status": "pending"},
            {"chain_id": "chain-a", "link": 1, "uuid": "task-a", "status": "completed"},
        ]
        self.assertEqual(ordered_records(records), ordered_records(list(reversed(records))))

        plan = OperatorPlan(
            action="repair",
            snapshot_id="snapshot-1",
            configuration_fingerprint="config-1",
            scope=OperatorScope.system(),
            coverage=OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snapshot-1"),
            operations=({"kind": "link", "uuid": "task-a", "fields": {"nextLink": "task-b"}},),
        )
        equivalent = OperatorPlan.from_mapping(json.loads(json.dumps(plan.to_dict(), ensure_ascii=False)))
        self.assertEqual(plan.fingerprint, equivalent.fingerprint)
        self.assertEqual(render_contract_json(plan), render_contract_json(equivalent))

    def test_cross_interface_projection_matrix_preserves_snapshot_facts(self) -> None:
        """The operator clients must project one snapshot identically for each scope."""
        snapshot = OperatorSnapshot(
            "conformance-matrix-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
        )
        requirement = CoverageRequirement(CoverageKind.COMPLETE)
        limits = OperatorLimits()

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        scopes = (
            OperatorScope.system(),
            OperatorScope(OperatorScopeKind.CHAIN, ("chain-a",)),
            OperatorScope(OperatorScopeKind.UUID, ("task-a",)),
        )
        for scope in scopes:
            canonical = inspect_operator_snapshot(snapshot, requirement, limits, scope=scope)
            facade = control_plane.inspect(snapshot, requirement, limits, scope=scope)
            inspector_client = run_inspectors(snapshot, standard_inspector_bundle(scope=scope))
            # The facade and direct clients must agree on the complete finding
            # records, not merely their count or presentation order.
            self.assertEqual(tuple(item.to_dict() for item in facade), tuple(item.to_dict() for item in canonical))
            self.assertEqual(tuple(item.to_dict() for item in inspector_client), tuple(item.to_dict() for item in canonical[1:]))

    def test_versioned_result_json_round_trip_preserves_envelope(self) -> None:
        result = OperatorV2Result(
            "nautical.query.test", "query", OperatorV2Status.OK,
            payload={"message": "héllo", "items": [1, 2]},
        )
        encoded = render_contract_json(result)
        decoded = OperatorV2Result.from_mapping(json.loads(encoded))
        self.assertEqual(decoded, result)

    def test_public_operator_contracts_round_trip_through_json(self) -> None:
        """Representative request, finding, plan, and snapshot documents stay decodable."""
        request = OperatorRequest(OperatorOperation.INTEGRITY, OperatorScope.system())
        finding = OperatorFinding(
            "snapshot.test", "snapshot", FindingSeverity.WARNING,
            FindingActionability.INFORMATIONAL, "test finding",
        )
        snapshot = OperatorSnapshot(
            "snapshot-roundtrip",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1", "config-1",
        )
        plan = OperatorPlan(
            "noop", "snapshot-roundtrip", "config-1", OperatorScope.system(),
            snapshot.coverage,
        )
        for value, decoder in (
            (request, OperatorRequest.from_mapping),
            (finding, OperatorFinding.from_mapping),
            (snapshot, OperatorSnapshot.from_mapping),
            (plan, OperatorPlan.from_mapping),
        ):
            encoded = render_contract_json(value)
            self.assertEqual(decoder(json.loads(encoded)), value)

        result = OperatorV2Result(
            "nautical.query.integrity", "integrity", OperatorV2Status.UNAVAILABLE,
            payload={"snapshot": None},
            failure=OperatorFailure("snapshot_unavailable", "snapshot unavailable", True),
        )
        decoded_result = OperatorV2Result.from_mapping(json.loads(render_contract_json(result)))
        self.assertEqual(decoded_result, result)

        capabilities = OperatorCapabilities(taskwarrior_version="3.4.2")
        decoded_capabilities = OperatorCapabilities.from_mapping(
            json.loads(render_contract_json(capabilities))
        )
        self.assertEqual(decoded_capabilities, capabilities)

    def test_snapshot_unavailable_evidence_is_retryable(self) -> None:
        configuration = ValidatedNauticalConfiguration(
            source="test", fingerprint="config-1", scheduler_fingerprint="schedule-1",
            timezone_name="UTC", values=(),
        )
        integration = IntegrationContext(
            Path("/tmp/operator-conformance"), "test", ("task",), configuration,
            timezone.utc, SilentDiagnostics(), SystemClock(), "conformance", 8,
            IntegrationAccess.READ_ONLY,
        )
        request = OperatorRequest(OperatorOperation.INTEGRITY, OperatorScope.system())
        context = OperatorInvocationContext.from_integration(request, integration)
        command = TaskCommand(("task", "export"), "snapshot", 1.0)
        evidence = FailureEvidence(command, CommandFailureKind.TIMEOUT, 124, 1, 1.0, True, "timed out")
        reader = ChainSnapshotReader(lambda _request: Unavailable("snapshot", evidence))
        result = reader.read(context, SnapshotReadRequest(OperatorScope.system()))
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "snapshot_unavailable")
        self.assertTrue(result.retryable)

    def test_invalid_snapshot_collector_result_fails_closed(self) -> None:
        configuration = ValidatedNauticalConfiguration(
            source="test", fingerprint="config-1", scheduler_fingerprint="schedule-1",
            timezone_name="UTC", values=(),
        )
        integration = IntegrationContext(
            Path("/tmp/operator-conformance"), "test", ("task",), configuration,
            timezone.utc, SilentDiagnostics(), SystemClock(), "conformance", 8,
            IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(
            OperatorRequest(OperatorOperation.INTEGRITY, OperatorScope.system()), integration,
        )
        reader = ChainSnapshotReader(lambda _request: object())
        result = reader.read(context, SnapshotReadRequest(OperatorScope.system()))
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "invalid_snapshot_read")


if __name__ == "__main__":
    unittest.main()
