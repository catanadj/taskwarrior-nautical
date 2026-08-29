"""Promoted lifecycle failure-injection coverage.

The golden suite owns the broad integration fixtures. These focused wrappers
make the mutation-boundary harness part of the normal unit-test contract too:
outbox failures, persisted stage failures, and crash/resume behavior must stay
green even when the full golden dispatcher is not run.
"""

from __future__ import annotations

import unittest
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from nautical_core.lifecycle_application import LifecycleApplicationService
from nautical_core.lifecycle_models import LifecycleDrainProgress, LifecycleDrainStage
from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
from nautical_core.lifecycle_outbox import LifecycleOutboxRepository
from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits
from nautical_core.integration_models import MutationOperation, MutationOutcome, MutationOutcomeKind, MutationPostcondition

from dev_tools.nautical_golden_tests import (
    test_lifecycle_application_conflict_and_retry_budget_outcomes,
    test_lifecycle_application_crash_at_each_stage_resumes_without_remutation,
    test_lifecycle_application_outbox_faults_are_retryable,
    test_lifecycle_application_stage_failure_matrix_resumes_idempotently,
    test_lifecycle_application_idempotency_and_duplicate_staging,
)


class LifecycleFailureInjectionTests(unittest.TestCase):
    def test_outbox_failures_are_retryable(self) -> None:
        test_lifecycle_application_outbox_faults_are_retryable()

    def test_each_persisted_stage_failure_resumes_safely(self) -> None:
        test_lifecycle_application_stage_failure_matrix_resumes_idempotently()

    def test_crash_resume_does_not_remutate_completed_stages(self) -> None:
        test_lifecycle_application_crash_at_each_stage_resumes_without_remutation()

    def test_conflicts_and_retry_budgets_are_explicit(self) -> None:
        test_lifecycle_application_conflict_and_retry_budget_outcomes()

    def test_replay_and_duplicate_staging_are_idempotent(self) -> None:
        test_lifecycle_application_idempotency_and_duplicate_staging()

    def test_progress_observer_failure_is_contained(self) -> None:
        event = LifecycleDrainProgress(LifecycleDrainStage.PROCESSING, 1, 2, intent_id="intent-1")

        def failing_observer(_event):
            raise RuntimeError("injected progress failure")

        # Progress is presentation-only and must never interrupt lifecycle
        # application or turn a successful drain into an error.
        LifecycleApplicationService._report_drain_progress(failing_observer, event)

    def test_budget_interrupt_after_child_import_resumes_at_parent_link(self) -> None:
        class Uow:
            mutation_epoch = 0

        class Gateway:
            def __init__(self):
                self.calls = []

            def apply(self, request):
                self.calls.append(request.operation)
                postcondition = (
                    MutationPostcondition.CHILD_IMPORTED
                    if request.operation is MutationOperation.CHILD_IMPORT
                    else MutationPostcondition.PARENT_LINKED
                )
                return MutationOutcome(request.operation, MutationOutcomeKind.APPLIED, request.guard, (postcondition,))

        parent = "00000000-0000-4000-8000-000000000201"
        child = "00000000-0000-4000-8000-000000000202"
        from dev_tools.nautical_golden_tests import _task_draft

        plan = LifecyclePlan.from_draft(
            identity=LifecycleIdentity("budget-chain", parent, 1, 2, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "budget-chain", 1, "rf1-budget", "20260101T000000Z"),
            draft=_task_draft({"uuid": child, "description": "budget child", "status": "pending", "chain": "on",
                               "chainID": "budget-chain", "link": 2, "prevLink": parent[:8], "cp": "1d",
                               "due": "20260102T000000Z"}),
            parent_patch={"nextLink": child[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )
        with TemporaryDirectory() as td:
            outbox = LifecycleOutboxRepository(Path(td))
            first_gateway = Gateway()
            first = LifecycleApplicationService(
                unit_of_work=Uow(), mutations=first_gateway, outbox=outbox,
                budget=OperatorInvocationBudget(OperatorLimits(taskwarrior_calls=1)), owner="budget-a",
            )
            first.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            drained = first.drain(limit=1, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            self.assertEqual(drained.outcomes[0].kind.value, "retryable")
            self.assertEqual(first_gateway.calls, [MutationOperation.CHILD_IMPORT])
            _, status = outbox.status()
            self.assertEqual(status["records"][0]["stage"], "child_present")
            time.sleep(1.1)

            second_gateway = Gateway()
            second = LifecycleApplicationService(
                unit_of_work=Uow(), mutations=second_gateway, outbox=outbox, owner="budget-b",
            )
            resumed = second.drain(limit=1, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            self.assertTrue(resumed.outcomes[0].ok)
            self.assertEqual(second_gateway.calls, [MutationOperation.PARENT_LINK])


if __name__ == "__main__":
    unittest.main()
