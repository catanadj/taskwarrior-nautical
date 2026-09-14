from __future__ import annotations

from types import SimpleNamespace
import unittest

from nautical_core.lifecycle_application import (
    DrainResult,
    LifecycleApplicationError,
    LifecycleApplicationOutcome,
    LifecycleApplicationOutcomeKind,
    LifecycleApplicationService,
)
from nautical_core.lifecycle_outbox import OutboxResult, OutboxResultKind
from nautical_core.lifecycle_models import (
    LifecycleAction,
    LifecycleEvent,
    LifecycleIdentity,
    LifecyclePlan,
    ParentGuard,
)
from nautical_core.lifecycle_operator_owner import LifecycleOperatorOwner
from nautical_core.operator_domain_plans import DomainApplicationAuthorization
from nautical_core.operator_models import (
    CoverageKind,
    CoverageRequirement,
    OperatorCoverage,
    OperatorOperation,
    OperatorRequest,
    OperatorScope,
    OperatorStatus,
)


_EXECUTION_METHODS = (
    "apply_lifecycle_unverified",
    "apply_lifecycle_children_unverified",
    "verify_lifecycle_children",
    "verify_lifecycle_parents",
    "preflight_lifecycle_batch",
)


class _CompleteExecutionPort:
    def apply_lifecycle_unverified(self, _request):
        raise AssertionError("not called by construction test")

    def apply_lifecycle_children_unverified(self, _requests):
        raise AssertionError("not called by construction test")

    def verify_lifecycle_children(self, _requests):
        raise AssertionError("not called by construction test")

    def verify_lifecycle_parents(self, _requests):
        raise AssertionError("not called by construction test")

    def preflight_lifecycle_batch(self, _payloads, *, parent_expectations=()):
        raise AssertionError("not called by construction test")


class _CompleteMutationGateway:
    def apply(self, _request):
        raise AssertionError("not called by construction test")

    def compensate_imported_child(self, _request):
        raise AssertionError("not called by construction test")


class LifecycleExecutionCapabilityTests(unittest.TestCase):
    def test_explicit_execution_protocol_names_all_required_operations(self) -> None:
        from nautical_core.lifecycle_application import LifecycleExecutionPort

        self.assertEqual(
            tuple(name for name in _EXECUTION_METHODS if hasattr(LifecycleExecutionPort, name)),
            _EXECUTION_METHODS,
        )

    def test_complete_execution_provider_is_accepted(self) -> None:
        provider = _CompleteExecutionPort()
        service = LifecycleApplicationService(
            mutations=_CompleteMutationGateway(), execution=provider,
            outbox=SimpleNamespace(enqueue=lambda *_args, **_kwargs: None),
        )

        self.assertIs(service._execution, provider)

    def test_incomplete_execution_provider_reports_sorted_missing_capabilities(self) -> None:
        provider = SimpleNamespace(apply_lifecycle_unverified=lambda _request: None)
        with self.assertRaises(LifecycleApplicationError) as raised:
            LifecycleApplicationService(
                mutations=_CompleteMutationGateway(), execution=provider, outbox=object()
            )

        missing = [
            "apply_lifecycle_children_unverified",
            "preflight_lifecycle_batch",
            "verify_lifecycle_children",
            "verify_lifecycle_parents",
        ]
        self.assertEqual(
            str(raised.exception),
            "lifecycle execution capability is incomplete; missing: " + ", ".join(missing),
        )

    def test_incomplete_mutation_gateway_is_rejected_at_composition(self) -> None:
        cases = (
            (SimpleNamespace(), "apply, compensate_imported_child"),
            (SimpleNamespace(apply=lambda _request: None), "compensate_imported_child"),
            (SimpleNamespace(compensate_imported_child=lambda _request: None), "apply"),
        )
        for gateway, missing in cases:
            with self.subTest(missing=missing), self.assertRaises(LifecycleApplicationError) as raised:
                LifecycleApplicationService(
                    mutations=gateway,
                    execution=_CompleteExecutionPort(),
                    outbox=object(),
                )
            self.assertEqual(
                str(raised.exception),
                "lifecycle mutation gateway is incomplete; missing: " + missing,
            )

    def test_incomplete_execution_outbox_is_rejected_at_composition(self) -> None:
        with self.assertRaises(LifecycleApplicationError) as raised:
            LifecycleApplicationService(
                unit_of_work=SimpleNamespace(mutation_epoch=0),
                mutations=_CompleteMutationGateway(),
                execution=_CompleteExecutionPort(),
                outbox=SimpleNamespace(),
            )

        self.assertEqual(
            str(raised.exception),
            "lifecycle outbox capability is incomplete; missing: "
            "acknowledge, acknowledge_many, advance_stage, advance_stages, "
            "claim_batch, claim_intent, claim_intents, enqueue, enqueue_many, "
            "manual_review, release_retry, renew_lease, renew_leases, session",
        )

    def test_execution_outbox_validates_single_record_recovery_operations(self) -> None:
        available = (
            "enqueue", "enqueue_many", "claim_intent", "claim_intents",
            "renew_lease", "renew_leases", "advance_stage", "advance_stages",
            "acknowledge", "acknowledge_many", "release_retry", "manual_review", "session",
        )
        outbox = SimpleNamespace(**{name: lambda **_kwargs: None for name in available})

        with self.assertRaises(LifecycleApplicationError) as raised:
            LifecycleApplicationService(
                unit_of_work=SimpleNamespace(mutation_epoch=0),
                mutations=_CompleteMutationGateway(),
                execution=_CompleteExecutionPort(),
                outbox=outbox,
            )

        self.assertEqual(
            str(raised.exception),
            "lifecycle outbox capability is incomplete; missing: claim_batch",
        )

    def test_stage_only_service_requires_enqueue_at_composition(self) -> None:
        with self.assertRaises(LifecycleApplicationError) as raised:
            LifecycleApplicationService(outbox=SimpleNamespace())

        self.assertEqual(
            str(raised.exception),
            "lifecycle outbox capability is incomplete; missing: enqueue",
        )

    def test_stage_only_service_rejects_execution_before_claiming_work(self) -> None:
        class Outbox:
            claim_calls = 0

            def enqueue(self, *_args, **_kwargs):
                raise AssertionError("execution must be rejected before staging")

            def claim_batch(self, **_kwargs):
                self.claim_calls += 1
                raise AssertionError("stage-only service must reject before claiming")

        outbox = Outbox()
        service = LifecycleApplicationService(outbox=outbox)
        actions = (
            lambda: service.drain(
                limit=1, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ),
            lambda: service.drain_claimed(
                (), configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ),
            lambda: service.apply_immediate(None),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaisesRegex(
                    LifecycleApplicationError,
                    "^lifecycle execution capability is unavailable$",
                ):
                    action()
        self.assertEqual(outbox.claim_calls, 0)

    def test_operator_owner_uses_single_bounded_drain_with_identical_fingerprints(self) -> None:
        identity = LifecycleIdentity(
            "owner-chain", "00000000-0000-4000-8000-000000000111", 1, 2,
            LifecycleEvent.COMPLETE,
        )
        plan = LifecyclePlan(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "owner-chain", 1),
        )
        scope = OperatorScope.system()
        request = OperatorRequest(
            OperatorOperation.APPLY,
            scope,
            apply=True,
            coverage=CoverageRequirement(CoverageKind.COMPLETE),
        )
        authorization = DomainApplicationAuthorization(
            plan,
            request,
            "snapshot-1",
            "config-1",
            scope,
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snapshot-1"),
            "schedule-1",
        )

        class Application:
            def __init__(self):
                self.calls = []

            def stage(self, plan, *, configuration_fingerprint, schedule_fingerprint):
                self.calls.append(("stage", configuration_fingerprint, schedule_fingerprint))
                return LifecycleApplicationOutcome(
                    LifecycleApplicationOutcomeKind.APPLIED,
                    plan.identity,
                    intent_id="intent-1",
                )

            def drain(self, *, limit, configuration_fingerprint, schedule_fingerprint):
                self.calls.append(("drain", limit, configuration_fingerprint, schedule_fingerprint))
                outcome = LifecycleApplicationOutcome(
                    LifecycleApplicationOutcomeKind.APPLIED,
                    plan.identity,
                    intent_id="intent-1",
                )
                return DrainResult(OutboxResult(OutboxResultKind.APPLIED), (outcome,))

        application = Application()
        result = LifecycleOperatorOwner(application).apply(authorization)

        self.assertEqual(result.status, OperatorStatus.OK)
        self.assertEqual(
            application.calls,
            [
                ("stage", "config-1", "schedule-1"),
                ("drain", 1, "config-1", "schedule-1"),
            ],
        )

    def test_operator_owner_does_not_report_staged_success_when_drain_has_no_outcome(self) -> None:
        identity = LifecycleIdentity(
            "owner-chain", "00000000-0000-4000-8000-000000000111", 1, 2,
            LifecycleEvent.COMPLETE,
        )
        plan = LifecyclePlan(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "owner-chain", 1),
        )
        scope = OperatorScope.system()
        request = OperatorRequest(
            OperatorOperation.APPLY,
            scope,
            apply=True,
            coverage=CoverageRequirement(CoverageKind.COMPLETE),
        )
        authorization = DomainApplicationAuthorization(
            plan,
            request,
            "snapshot-1",
            "config-1",
            scope,
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", "snapshot-1"),
            "schedule-1",
        )

        class Application:
            def stage(self, plan, *, configuration_fingerprint, schedule_fingerprint):
                return LifecycleApplicationOutcome(
                    LifecycleApplicationOutcomeKind.APPLIED,
                    plan.identity,
                    intent_id="intent-1",
                )

            def drain(self, *, limit, configuration_fingerprint, schedule_fingerprint):
                return DrainResult(OutboxResult(OutboxResultKind.APPLIED), ())

        result = LifecycleOperatorOwner(Application()).apply(authorization)

        self.assertEqual(result.status, OperatorStatus.UNAVAILABLE)
        self.assertEqual(result.failure.code, "lifecycle_retryable")


if __name__ == "__main__":
    unittest.main()
