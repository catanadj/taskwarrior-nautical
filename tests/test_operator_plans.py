import json
import unittest

from nautical_core.operator_models import CoverageKind, CoverageRequirement, OperatorContractError, OperatorCoverage, OperatorOperation, OperatorRequest, OperatorResult, OperatorScope, OperatorScopeKind, OperatorStatus
from nautical_core.operator_plans import OperatorPlan
from nautical_core.operator_application import ApplicationReceipt, MappingGuardVerifier, MappingPostconditionVerifier, OperatorApplicationRegistry, apply_authorized, authorize_application
from nautical_core.operator_domain_plans import DomainApplicationAuthorization, require_domain_effect_plan
from nautical_core.operator_domain_planner import OperatorDomainPlanner
from nautical_core.operator_control_plane import OperatorControlPlane
from nautical_core.lifecycle_recovery_models import RecoveryPlanResult, RecoveryRefusal, RecoveryStatus
from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
from nautical_core.task_models import TaskObservation


class OperatorPlanTests(unittest.TestCase):
    def test_recovery_result_variants_are_typed_and_immutable(self) -> None:
        parent = TaskObservation.from_mapping(
            {
                "uuid": "00000000-0000-4000-8000-000000000901",
                "status": "completed",
                "chain": "on",
                "chainID": "recovery-test",
                "link": 1,
            },
            source_query="recovery-model-test",
        )
        guard = ParentGuard("completed", "on", "recovery-test", 1)
        parent_uuid = str(parent.field("uuid").raw_value())
        identity = LifecycleIdentity("recovery-test", parent_uuid, 1, None, LifecycleEvent.COMPLETE)
        plan = LifecyclePlan(identity=identity, action=LifecycleAction.NOOP, parent_guard=guard)
        result = RecoveryPlanResult(parent, plan)
        self.assertIs(result.plan, plan)
        self.assertFalse(result.applied)
        refusal = RecoveryRefusal(parent, RecoveryStatus.MANUAL_REVIEW, "requires review", {"chain": "recovery-test"})
        self.assertEqual(refusal.status, RecoveryStatus.MANUAL_REVIEW)
        with self.assertRaises(TypeError):
            refusal.evidence["chain"] = "other"  # type: ignore[index]

    def test_recovery_refusal_requires_reason(self) -> None:
        parent = TaskObservation.from_mapping(
            {"uuid": "00000000-0000-4000-8000-000000000902", "status": "pending"},
            source_query="recovery-model-test",
        )
        with self.assertRaises(ValueError):
            RecoveryRefusal(parent, RecoveryStatus.ERROR, "")

    def test_plan_round_trip_is_evidence_bound(self) -> None:
        plan = OperatorPlan(
            "inspect", "snap-1", "config-1",
            OperatorScope(OperatorScopeKind.CHAIN, ("abc",)),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "noop", "link": 1},),
            immutable_inputs={"plan_version": 1},
            expected_guards={"epoch": 3},
        )
        self.assertEqual(OperatorPlan.from_mapping(plan.to_dict()), plan)
        json.dumps(plan.to_dict(), ensure_ascii=False)
        self.assertEqual(plan.fingerprint, OperatorPlan.from_mapping(plan.to_dict()).fingerprint)
        request = OperatorRequest(OperatorOperation.INSPECT, plan.scope, coverage=CoverageRequirement(CoverageKind.COMPLETE))
        plan.validate_for_request(request)
        with self.assertRaises(OperatorContractError):
            plan.validate_for_request(OperatorRequest(OperatorOperation.INSPECT, OperatorScope.system()))

    def test_effectful_plan_requires_complete_coverage(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorPlan(
                "apply", "snap-1", "config-1",
                OperatorScope(OperatorScopeKind.SYSTEM),
                OperatorCoverage(CoverageKind.BOUNDED, "taskwarrior", omitted_count=1),
            )

    def test_plan_rejects_opaque_operations(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorPlan(
                "inspect", "snap-1", "config-1",
                OperatorScope(OperatorScopeKind.SYSTEM),
                OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
                operations=({"link": 1},),
            )

    def test_noop_classification_is_explicit_and_deterministic(self) -> None:
        terminal = OperatorPlan(
            "terminal", "snap-1", "config-1",
            OperatorScope(OperatorScopeKind.SYSTEM),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
        )
        self.assertTrue(terminal.is_noop)
        applied = OperatorPlan(
            "apply", "snap-1", "config-1",
            OperatorScope(OperatorScopeKind.SYSTEM),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair", "uuid": "abc"},),
        )
        self.assertFalse(applied.is_noop)

    def test_plan_rejects_untyped_guard_inputs(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorPlan(
                "inspect", "snap-1", "config-1",
                OperatorScope(OperatorScopeKind.SYSTEM),
                OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
                immutable_inputs=[("uuid", "abc")],  # type: ignore[arg-type]
            )

    def test_plan_rejects_non_json_values(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorPlan(
                "inspect", "snap-1", "config-1",
                OperatorScope(OperatorScopeKind.SYSTEM),
                OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
                expected_guards={"timezone": object()},
            )

    def test_plan_rejects_mismatched_coverage_snapshot(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorPlan(
                "inspect", "snap-1", "config-1",
                OperatorScope(OperatorScopeKind.SYSTEM),
                OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", snapshot_id="snap-2"),
            )

    def test_application_boundary_requires_explicit_effectful_apply(self) -> None:
        plan = OperatorPlan(
            "apply", "snap-1", "config-1",
            OperatorScope(OperatorScopeKind.SYSTEM),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair", "uuid": "abc"},),
        )
        request = OperatorRequest(
            OperatorOperation.APPLY, plan.scope,
            apply=True, coverage=CoverageRequirement(CoverageKind.COMPLETE),
        )
        self.assertIs(authorize_application(plan, request).plan, plan)
        with self.assertRaises(OperatorContractError):
            authorize_application(plan, OperatorRequest(OperatorOperation.PLAN, plan.scope))

        class Owner:
            def apply(self, authorization):
                self.authorization = authorization
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)

        owner = Owner()
        class Verifier:
            def verify(self, authorization):
                self.authorization = authorization

        verifier = Verifier()
        class Postcondition:
            def verify(self, authorization, result):
                self.result = result

        postcondition = Postcondition()
        result = apply_authorized(plan, request, verifier, owner, postcondition)
        self.assertEqual(result.status, OperatorStatus.OK)
        self.assertIs(owner.authorization.plan, plan)
        self.assertIs(verifier.authorization.plan, plan)
        self.assertIs(postcondition.result, result)
        with self.assertRaises(OperatorContractError):
            apply_authorized(plan, request, verifier, object(), postcondition)
        with self.assertRaises(OperatorContractError):
            apply_authorized(plan, request, object(), owner, postcondition)
        with self.assertRaises(OperatorContractError):
            apply_authorized(plan, request, verifier, owner, object())
        class WrongOwner:
            def apply(self, authorization):
                return OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK)

        with self.assertRaises(OperatorContractError):
            apply_authorized(plan, request, verifier, WrongOwner(), postcondition)

    def test_failure_injection_guard_blocks_effect_owner(self) -> None:
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},),
        )
        request = OperatorRequest(OperatorOperation.APPLY, plan.scope, apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))
        calls = []

        class Guard:
            def verify(self, authorization):
                raise OperatorContractError("injected stale guard")

        class Owner:
            def apply(self, authorization):
                calls.append(authorization)
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)

        with self.assertRaisesRegex(OperatorContractError, "stale guard"):
            apply_authorized(plan, request, Guard(), Owner(), object())
        self.assertEqual(calls, [])

    def test_failure_injection_postcondition_rejects_unverified_effect(self) -> None:
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},),
        )
        request = OperatorRequest(OperatorOperation.APPLY, plan.scope, apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))

        class Guard:
            def verify(self, authorization):
                return None

        class Owner:
            def apply(self, authorization):
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)

        class Postcondition:
            def verify(self, authorization, result):
                raise OperatorContractError("injected verification failure")

        with self.assertRaisesRegex(OperatorContractError, "verification failure"):
            apply_authorized(plan, request, Guard(), Owner(), Postcondition())

    def test_application_registry_requires_one_owner_per_action(self) -> None:
        registry = OperatorApplicationRegistry()
        class Owner:
            def apply(self, authorization):
                return OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK)
        owner = Owner()
        registry.register("apply", owner)
        self.assertIs(registry.owner_for(OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},),
        )), owner)
        with self.assertRaises(OperatorContractError):
            registry.register("apply", owner)
        with self.assertRaises(OperatorContractError):
            registry.owner_for(OperatorPlan(
                "repair", "snap-1", "config-1", OperatorScope.system(),
                OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
                operations=({"kind": "repair"},),
            ))

        request = OperatorRequest(
            OperatorOperation.APPLY, OperatorScope.system(), apply=True,
            coverage=CoverageRequirement(CoverageKind.COMPLETE),
        )
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},),
        )
        class Verifier:
            def verify(self, authorization): pass
        class Postcondition:
            def verify(self, authorization, result): pass
        self.assertEqual(registry.apply(plan, request, Verifier(), Postcondition()).status, OperatorStatus.OK)
        receipt = registry.apply_with_receipt(plan, request, Verifier(), Postcondition())
        self.assertEqual(receipt.plan_fingerprint, plan.fingerprint)
        self.assertTrue(receipt.to_dict()["verified"])
        with self.assertRaises(OperatorContractError):
            ApplicationReceipt(plan.fingerprint, OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK), verified=False)

    def test_mapping_guard_verifier_rejects_stale_evidence(self) -> None:
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},), expected_guards={"epoch": 2},
        )
        request = OperatorRequest(OperatorOperation.APPLY, OperatorScope.system(), apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))
        verifier = MappingGuardVerifier(lambda: {"epoch": 1})
        with self.assertRaisesRegex(OperatorContractError, "epoch"):
            authorize_application(plan, request)
            verifier.verify(authorize_application(plan, request))

    def test_mapping_postcondition_verifier_checks_fresh_evidence(self) -> None:
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},), expected_postconditions={"status": "done"},
        )
        request = OperatorRequest(OperatorOperation.APPLY, OperatorScope.system(), apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))
        authorization = authorize_application(plan, request)
        verifier = MappingPostconditionVerifier(lambda: {"status": "pending"})
        with self.assertRaisesRegex(OperatorContractError, "status"):
            verifier.verify(authorization, OperatorResult(OperatorOperation.APPLY, OperatorStatus.OK))

    def test_domain_effect_boundary_rejects_generic_operator_payloads(self) -> None:
        with self.assertRaises(TypeError):
            require_domain_effect_plan({"kind": "repair"})

    def test_domain_authorization_binds_evidence_to_typed_plan(self) -> None:
        request = OperatorRequest(OperatorOperation.APPLY, OperatorScope.system(), apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"), operations=({"kind": "repair"},),
        )
        with self.assertRaises(TypeError):
            DomainApplicationAuthorization(plan, request, "snap-1", "config-1", request.scope,
                                            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"))

    def test_domain_planner_rejects_untyped_lifecycle_output(self) -> None:
        class Planner:
            def plan(self, *args, **kwargs):
                return {"action": "apply"}
        with self.assertRaises(TypeError):
            OperatorDomainPlanner(Planner(), object()).plan_lifecycle(object(), "complete")

    def test_domain_planner_propagates_planning_failure(self) -> None:
        class Planner:
            def plan(self, *args, **kwargs):
                raise RuntimeError("injected planning failure")

        with self.assertRaisesRegex(RuntimeError, "planning failure"):
            OperatorDomainPlanner(Planner(), object()).plan_lifecycle(object(), "complete")

    def test_application_owner_failure_never_reaches_postcondition(self) -> None:
        plan = OperatorPlan(
            "apply", "snap-1", "config-1", OperatorScope.system(),
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            operations=({"kind": "repair"},),
        )
        request = OperatorRequest(OperatorOperation.APPLY, plan.scope, apply=True,
                                  coverage=CoverageRequirement(CoverageKind.COMPLETE))

        class Guard:
            def verify(self, authorization):
                return None

        class Owner:
            def apply(self, authorization):
                raise RuntimeError("injected delegation failure")

        class Postcondition:
            def verify(self, authorization, result):
                raise AssertionError("postcondition must not run")

        with self.assertRaisesRegex(RuntimeError, "delegation failure"):
            apply_authorized(plan, request, Guard(), Owner(), Postcondition())

    def test_control_plane_factory_requires_validated_dependencies(self) -> None:
        with self.assertRaises(ValueError):
            OperatorControlPlane.from_configuration(None, OperatorApplicationRegistry())

if __name__ == "__main__":
    unittest.main()
