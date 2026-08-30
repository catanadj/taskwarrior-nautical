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
from nautical_core.lifecycle_outbox import LifecycleOutboxRecord, LifecycleOutboxRepository
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
    @staticmethod
    def _bulk_plan(chain: str, parent: str, child: str, link: int) -> LifecyclePlan:
        from dev_tools.nautical_golden_tests import _task_draft

        return LifecyclePlan.from_draft(
            identity=LifecycleIdentity(chain, parent, link, link + 1, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", chain, link, f"rf-{chain}", "20260101T000000Z"),
            draft=_task_draft({
                "uuid": child, "description": "bulk child", "status": "pending", "chain": "on",
                "chainID": chain, "link": link + 1, "prevLink": parent[:8], "cp": "1d",
                "due": "20260102T000000Z",
            }),
            parent_patch={"nextLink": child[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )

    def test_bulk_enqueue_and_exact_claim_are_scoped_and_idempotent(self) -> None:
        with TemporaryDirectory() as td:
            outbox = LifecycleOutboxRepository(Path(td))
            plans = tuple(
                self._bulk_plan(
                    f"bulk-{idx}",
                    f"00000000-0000-4000-8000-0000000002{idx:02d}",
                    f"00000000-0000-4000-8000-0000000003{idx:02d}",
                    1,
                )
                for idx in range(1, 4)
            )
            overall, results = outbox.enqueue_many(
                plans, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            )
            self.assertEqual(overall.kind.value, "applied")
            self.assertEqual(set(results), {plan.identity.idempotency_key for plan in plans})
            self.assertTrue(all(result.kind.value == "applied" for result in results.values()))

            repeat, repeated = outbox.enqueue_many(
                plans, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            )
            self.assertEqual(repeat.kind.value, "applied")
            self.assertTrue(all(result.kind.value == "already_applied" for result in repeated.values()))

            target_ids = [plans[0].identity.idempotency_key, plans[2].identity.idempotency_key]
            claim, claimed = outbox.claim_intents(intent_ids=target_ids, owner="bulk-owner", lease_seconds=30)
            self.assertEqual(claim.kind.value, "applied")
            self.assertEqual(set(claimed), set(target_ids))
            self.assertTrue(all(result.kind.value == "applied" for result in claimed.values()))
            batch, records = outbox.claim_batch(owner="other-owner", lease_seconds=30, limit=5)
            self.assertEqual(batch.kind.value, "applied")
            self.assertEqual([record.intent_id for record in records], [plans[1].identity.idempotency_key])

    def test_execute_wave_claims_only_its_staged_intents(self) -> None:
        with TemporaryDirectory() as td:
            outbox = LifecycleOutboxRepository(Path(td))
            plans = tuple(
                self._bulk_plan(
                    f"wave-{idx}",
                    f"00000000-0000-4000-8000-0000000004{idx:02d}",
                    f"00000000-0000-4000-8000-0000000005{idx:02d}",
                    1,
                )
                for idx in range(1, 3)
            )

            class WaveService(LifecycleApplicationService):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.claimed_records: tuple[LifecycleOutboxRecord, ...] = ()

                def drain_claimed(self, records, **kwargs):
                    self.claimed_records = tuple(records)
                    return DrainResult(claim=OutboxResult(OutboxResultKind.APPLIED), outcomes=())

            from nautical_core.lifecycle_application import DrainResult
            from nautical_core.lifecycle_outbox import OutboxResult, OutboxResultKind

            service = WaveService(outbox=outbox, owner="wave-owner")
            result = service.execute_wave(plans, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            self.assertEqual(result.claim.kind.value, "applied")
            claimed_ids: set[str] = {record.intent_id for record in service.claimed_records}
            self.assertEqual(claimed_ids, {plan.identity.idempotency_key for plan in plans})

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
