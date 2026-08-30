"""Cross-stage fail-closed coverage for the operator control plane."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable
from nautical_core.integration_context import (
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.operator_application import apply_authorized
from nautical_core.operator_models import (
    CoverageKind,
    CoverageRequirement,
    OperatorContractError,
    OperatorCoverage,
    OperatorFailure,
    OperatorOperation,
    OperatorPhase,
    OperatorRequest,
    OperatorLimits,
    OperatorScope,
    OperatorScopeKind,
    OperatorStatus,
)
from nautical_core.operator_plans import OperatorPlan
from nautical_core.operator_snapshot import ChainSnapshotReader, SnapshotReadRequest
from nautical_core.operator_inspectors import inspect_operator_snapshot
from nautical_core.operator_context import OperatorInvocationContext
from nautical_core.operator_snapshot import OperatorSnapshot


class OperatorFailureMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.configuration = ValidatedNauticalConfiguration(
            source="test",
            fingerprint="cfg-failure-matrix",
            scheduler_fingerprint="schedule-failure-matrix",
            timezone_name="UTC",
            values=(),
        )
        self.integration = IntegrationContext(
            "/tmp/operator-failure-matrix",
            "test",
            ("task",),
            self.configuration,
            timezone.utc,
            SilentDiagnostics(),
            SystemClock(),
            "failure-matrix",
            8,
            IntegrationAccess.READ_ONLY,
        )
        self.request = OperatorRequest(OperatorOperation.INTEGRITY, OperatorScope.system())
        self.context = OperatorInvocationContext.from_integration(self.request, self.integration)

    def test_scope_snapshot_inspection_and_planning_failures_stop_progression(self) -> None:
        """Each unavailable/insufficient stage must block later operator stages."""
        # Scope resolution is validated before any provider is called.
        with self.assertRaises(OperatorContractError):
            OperatorScope.from_mapping({"kind": "system", "values": ["unexpected"]})

        command = TaskCommand(("task", "export"), "failure-matrix", 1.0)
        evidence = FailureEvidence(command, CommandFailureKind.TIMEOUT, 124, 1, 1.0, True, "timed out")
        reader = ChainSnapshotReader(lambda _request: Unavailable("snapshot", evidence))
        snapshot_failure = reader.read(self.context, SnapshotReadRequest(OperatorScope.system()))
        self.assertIsInstance(snapshot_failure, OperatorFailure)
        self.assertEqual(snapshot_failure.code, "snapshot_unavailable")
        self.assertTrue(snapshot_failure.retryable)

        snapshot = OperatorSnapshot(
            "failure-matrix-snapshot",
            OperatorCoverage(CoverageKind.BOUNDED, "taskwarrior", omitted_count=1),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            self.configuration.fingerprint,
        )
        findings = inspect_operator_snapshot(
            snapshot,
            CoverageRequirement(CoverageKind.COMPLETE),
            limits=self._limits(),
            scope=OperatorScope.system(),
        )
        self.assertEqual(findings[0].actionability.value, "blocking")

        with self.assertRaises(OperatorContractError):
            incomplete = OperatorPlan(
                "apply",
                snapshot.snapshot_id,
                self.configuration.fingerprint,
                OperatorScope.system(),
                snapshot.coverage,
                operations=({"kind": "repair"},),
            )
            incomplete.validate_for_request(
                OperatorRequest(
                    OperatorOperation.APPLY,
                    OperatorScope.system(),
                    apply=True,
                    coverage=CoverageRequirement(CoverageKind.COMPLETE),
                )
            )

    def test_interrupted_snapshot_budget_is_typed_unavailable(self) -> None:
        """A budget interruption must stop before the collector and retain evidence."""
        from nautical_core.operator_context import OperatorInvocationContext
        from nautical_core.operator_models import OperatorLimits

        request = OperatorRequest(
            OperatorOperation.INTEGRITY,
            OperatorScope(OperatorScopeKind.CHAIN, ("chain-x",)),
            limits=OperatorLimits(taskwarrior_calls=1),
        )
        context = OperatorInvocationContext.from_integration(request, self.integration)
        self.assertTrue(context.budget is not None and context.budget.consume("taskwarrior_calls"))
        calls = []

        def collector(_request):
            calls.append(_request)
            raise AssertionError("collector ran after the Taskwarrior budget was exhausted")

        result = ChainSnapshotReader(collector).read(
            context,
            SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-x",))),
        )
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "snapshot_limit_exceeded")
        self.assertFalse(calls)

    def test_delegation_and_verification_failures_never_report_success(self) -> None:
        plan = OperatorPlan(
            "apply",
            "snapshot-1",
            self.configuration.fingerprint,
            OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},),
        )
        request = OperatorRequest(
            OperatorOperation.APPLY,
            OperatorScope.system(),
            apply=True,
            coverage=CoverageRequirement(CoverageKind.COMPLETE),
        )
        calls: list[str] = []

        class Guard:
            def verify(self, _authorization):
                return None

        class Owner:
            def apply(self, _authorization):
                calls.append("apply")
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)

        class FailingPostcondition:
            def verify(self, _authorization, _result):
                raise OperatorContractError("injected postcondition failure")

        from nautical_core.operator_models import OperatorResult

        with self.assertRaisesRegex(OperatorContractError, "postcondition failure"):
            apply_authorized(plan, request, Guard(), Owner(), FailingPostcondition())
        self.assertEqual(calls, ["apply"])

    def test_effect_budget_crossing_after_boundary_remains_visible_and_retryable(self) -> None:
        """Post-boundary budget crossings do not hide an effect or claim clean accounting."""
        from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
        from nautical_core.operator_control_plane import OperatorControlPlane
        from nautical_core.operator_domain_plans import DomainApplicationAuthorization
        from nautical_core.operator_control_plane import DomainApplicationRegistry
        from nautical_core.operator_context import OperatorInvocationBudget
        from nautical_core.operator_models import OperatorCoverage, CoverageKind, CoverageRequirement, OperatorResult, OperatorStatus

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        class Owner:
            def __init__(self) -> None:
                self.calls = 0

            def apply(self, _authorization):
                self.calls += 1
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)

        owner = Owner()
        control_plane = OperatorControlPlane.from_configuration(
            Configuration(), DomainApplicationRegistry({"lifecycle": owner}),
        )
        plan = LifecyclePlan(
            identity=LifecycleIdentity("chain-1", "task-1", 1, None, LifecycleEvent.DISABLE),
            action=LifecycleAction.DISABLE_CHAIN,
            parent_guard=ParentGuard("completed", "on", "chain-1", 1),
        )
        request = OperatorRequest(
            OperatorOperation.APPLY,
            OperatorScope.system(),
            apply=True,
            coverage=CoverageRequirement(CoverageKind.COMPLETE),
        )
        authorization = DomainApplicationAuthorization(
            plan, request, "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snap-1"), "schedule-1",
        )
        budget = OperatorInvocationBudget(OperatorLimits(taskwarrior_calls=1))
        budget.begin_effect()
        budget.consume("taskwarrior_calls", 2)
        phases = control_plane.apply_domain_phases("lifecycle", authorization, budget=budget)
        self.assertEqual(phases[-1].phase, OperatorPhase.RESULT)
        self.assertEqual(phases[-1].value.status, OperatorStatus.OK)
        self.assertTrue(budget.report()["exceeded"])
        self.assertEqual(owner.calls, 1)

    @staticmethod
    def _limits():
        from nautical_core.operator_models import OperatorLimits

        return OperatorLimits()


if __name__ == "__main__":
    unittest.main()
