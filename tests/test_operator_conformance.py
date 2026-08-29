import unittest
import json
from unittest.mock import patch
from datetime import datetime, timezone
from pathlib import Path

from nautical_core.operator_application import DomainApplicationRegistry
from nautical_core.operator_control_plane import OperatorControlPlane
from nautical_core.operator_inspectors import inspect_operator_snapshot, standard_inspector_bundle, run_inspectors
from nautical_core.operator_models import (CoverageKind, CoverageRequirement, OperatorCapabilities, OperatorCoverage, OperatorCursor, OperatorFailure, OperatorLimits,
    OperatorPage, OperatorResult, OperatorScope, OperatorScopeKind, OperatorOperation, OperatorRequest, OperatorStatus, OperatorV2Result, OperatorV2Status,
    OperatorExitCode, OperatorContractError, OperatorPhase, OperatorPhaseResult, exit_code_for_v2_status)
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
    def test_operator_phase_result_is_typed_and_fail_closed(self) -> None:
        successful = OperatorPhaseResult(OperatorPhase.ACQUIRE_SNAPSHOT, value={"snapshot": "s1"})
        self.assertEqual(successful.phase, OperatorPhase.ACQUIRE_SNAPSHOT)
        failure = OperatorFailure("snapshot_unavailable", "snapshot unavailable", retryable=True)
        failed = OperatorPhaseResult(OperatorPhase.ACQUIRE_SNAPSHOT, failure=failure)
        self.assertIs(failed.failure, failure)
        with self.assertRaises(OperatorContractError):
            OperatorPhaseResult(OperatorPhase.PLAN)
        with self.assertRaises(OperatorContractError):
            OperatorPhaseResult(OperatorPhase.PLAN, value={}, failure=failure)

    def test_v2_status_exit_code_matrix_is_exhaustive(self) -> None:
        expected = {
            OperatorV2Status.OK: OperatorExitCode.SUCCESS,
            OperatorV2Status.FOUND: OperatorExitCode.SUCCESS,
            OperatorV2Status.EMPTY: OperatorExitCode.SUCCESS,
            OperatorV2Status.ABSENT: OperatorExitCode.SUCCESS,
            OperatorV2Status.EXHAUSTED: OperatorExitCode.PARTIAL,
            OperatorV2Status.ATTENTION: OperatorExitCode.FINDINGS,
            OperatorV2Status.REPAIRABLE: OperatorExitCode.FINDINGS,
            OperatorV2Status.DEFERRED: OperatorExitCode.FINDINGS,
            OperatorV2Status.INVALID: OperatorExitCode.INVALID_REQUEST,
            OperatorV2Status.UNAVAILABLE: OperatorExitCode.UNAVAILABLE,
            OperatorV2Status.PARTIAL: OperatorExitCode.PARTIAL,
            OperatorV2Status.MANUAL_REVIEW: OperatorExitCode.MANUAL_REVIEW,
            OperatorV2Status.ERROR: OperatorExitCode.INTERNAL_FAILURE,
        }
        self.assertEqual(set(expected), set(OperatorV2Status))
        self.assertEqual(
            {status: exit_code_for_v2_status(status) for status in OperatorV2Status},
            expected,
        )
        with self.assertRaises(OperatorContractError):
            exit_code_for_v2_status("not-a-v2-status")

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

    def test_control_plane_inspection_exposes_ordered_typed_phases(self) -> None:
        snapshot = OperatorSnapshot(
            "phase-snapshot", OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-1", "config-1",
        )

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        phases = control_plane.inspect_phases(snapshot, CoverageRequirement(CoverageKind.COMPLETE), OperatorLimits())
        self.assertEqual(
            tuple(phase.phase for phase in phases),
            (OperatorPhase.VALIDATE_REQUEST, OperatorPhase.COMPILE_SCOPE, OperatorPhase.INSPECT, OperatorPhase.RESULT),
        )
        self.assertTrue(phases[0].value)

    def test_control_plane_inspection_stops_when_findings_budget_is_exceeded(self) -> None:
        snapshot = OperatorSnapshot(
            "phase-snapshot", OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-1", "config-1",
        )

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        finding = OperatorFinding(
            code="test.finding", domain="test", severity=FindingSeverity.WARNING,
            actionability=FindingActionability.INFORMATIONAL, message="finding",
        )
        with patch.object(OperatorControlPlane, "inspect", return_value=(finding, finding)):
            phases = control_plane.inspect_phases(
                snapshot,
                CoverageRequirement(CoverageKind.COMPLETE),
                OperatorLimits(findings=1),
            )
        self.assertEqual(phases[-1].phase, OperatorPhase.INSPECT)
        self.assertEqual(phases[-1].failure.code, "inspection_limit_exceeded")

    def test_control_plane_application_rejects_untyped_authorization_before_owner(self) -> None:
        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        phases = control_plane.apply_domain_phases("repair", object())  # type: ignore[arg-type]
        self.assertEqual(tuple(phase.phase for phase in phases), (OperatorPhase.AUTHORIZE,))
        self.assertEqual(phases[0].failure.code, "invalid_authorization")

    def test_control_plane_request_pipeline_rejects_invalid_context_before_read(self) -> None:
        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        phases = control_plane.inspect_request_phases(object(), object(), object())  # type: ignore[arg-type]
        self.assertEqual(phases[0].phase, OperatorPhase.VALIDATE_REQUEST)
        self.assertEqual(phases[0].failure.code, "invalid_request")

    def test_control_plane_domain_application_emits_ordered_effect_phases(self) -> None:
        from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
        from nautical_core.operator_domain_plans import DomainApplicationAuthorization

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        class Owner:
            def apply(self, authorization):
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)

        control_plane = OperatorControlPlane.from_configuration(
            Configuration(), DomainApplicationRegistry({"lifecycle": Owner()}),
        )
        scope = OperatorScope.system()
        request = OperatorRequest(OperatorOperation.APPLY, scope, apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))
        coverage = OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snap-1")
        plan = LifecyclePlan(
            identity=LifecycleIdentity("chain-1", "task-1", 1, None, LifecycleEvent.DISABLE),
            action=LifecycleAction.DISABLE_CHAIN,
            parent_guard=ParentGuard("completed", "on", "chain-1", 1),
        )
        authorization = DomainApplicationAuthorization(
            plan, request, "snap-1", "config-1", scope, coverage, "schedule-1",
        )
        phases = control_plane.apply_domain_phases("lifecycle", authorization)
        self.assertEqual(
            tuple(phase.phase for phase in phases),
            (OperatorPhase.AUTHORIZE, OperatorPhase.APPLY, OperatorPhase.VERIFY, OperatorPhase.RESULT),
        )
        self.assertEqual(phases[-1].value.status, OperatorStatus.OK)

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

    def test_public_result_serializes_supported_nested_values(self) -> None:
        """The public encoder keeps all supported evidence JSON-native and Unicode-safe."""
        from datetime import date
        from zoneinfo import ZoneInfo

        class EvidenceDocument:
            def to_dict(self):
                return {
                    "timezone": ZoneInfo("Europe/Bucharest"),
                    "path": Path("/tmp/nautical-ă"),
                    "captured_at": datetime(2026, 8, 29, 9, 30, tzinfo=timezone.utc),
                    "date": date(2026, 8, 29),
                    "status_value": OperatorV2Status.FOUND,
                    "nested": {"message": "héllo", "values": [1, True, None]},
                }

        encoded = render_contract_json(EvidenceDocument())
        self.assertIn("héllo", encoded)
        document = json.loads(encoded)
        payload = document
        self.assertEqual(payload["timezone"], "Europe/Bucharest")
        self.assertEqual(payload["path"], "/tmp/nautical-ă")
        self.assertEqual(payload["captured_at"], "2026-08-29T09:30:00Z")
        self.assertEqual(payload["date"], "2026-08-29")
        self.assertEqual(payload["status_value"], "found")
        self.assertEqual(payload["nested"]["values"], [1, True, None])

    def test_operator_contracts_reject_nested_mutation(self) -> None:
        """Frozen dataclasses must also protect nested evidence containers."""
        coverage = OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snapshot-immutable")
        snapshot = OperatorSnapshot(
            "snapshot-immutable", coverage, datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1", "config-1", components={"nested": {"value": 1}},
        )
        finding = OperatorFinding(
            "test", "snapshot", FindingSeverity.INFO, FindingActionability.INFORMATIONAL,
            "immutable", observed={"nested": {"value": 1}},
        )
        page = OperatorPage(items=({"nested": {"value": 1}},))
        plan = OperatorPlan(
            "inspect", "snapshot-immutable", "config-1", OperatorScope.system(), coverage,
            immutable_inputs={"nested": {"value": 1}},
        )
        for value, field in ((snapshot, "components"), (finding, "observed"), (page, "items"), (plan, "immutable_inputs")):
            with self.assertRaises(TypeError):
                if field == "items":
                    value.items[0]["nested"]["value"] = 2
                else:
                    getattr(value, field)["nested"]["value"] = 2

    def test_operator_contracts_reject_cycles_and_order_sets(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        with self.assertRaises(OperatorContractError):
            OperatorV2Result("nautical.query.test", "query", OperatorV2Status.OK, payload={"cycle": cyclic})

        first = OperatorV2Result(
            "nautical.query.test", "query", OperatorV2Status.OK,
            payload={"values": {"b", "a"}},
        )
        second = OperatorV2Result(
            "nautical.query.test", "query", OperatorV2Status.OK,
            payload={"values": {"a", "b"}},
        )
        self.assertEqual(first, second)
        self.assertEqual(first.to_dict()["values"], ["a", "b"])

    def test_plan_fingerprint_is_detached_from_caller_inputs(self) -> None:
        """Caller-owned nested mappings cannot change a constructed plan."""
        coverage = OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snapshot-inputs")
        inputs = {"nested": {"value": 1}}
        plan = OperatorPlan(
            "inspect", "snapshot-inputs", "config-1", OperatorScope.system(), coverage,
            immutable_inputs=inputs,
        )
        fingerprint = plan.fingerprint
        inputs["nested"]["value"] = 99
        self.assertEqual(plan.fingerprint, fingerprint)
        self.assertEqual(plan.immutable_inputs["nested"]["value"], 1)

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

        cursor = OperatorCursor("snapshot-roundtrip", "config-1", "epoch-1", position=2, page_size=2)
        page = OperatorPage(items=({"uuid": "task-2"},), cursor=cursor, complete=False)
        legacy = OperatorResult(
            OperatorOperation.INSPECT, OperatorStatus.OK,
            data={"count": 1}, page=page,
        )
        self.assertEqual(
            OperatorResult.from_mapping(json.loads(render_contract_json(legacy))),
            legacy,
        )

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
