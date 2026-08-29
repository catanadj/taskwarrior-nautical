"""Promoted lifecycle failure-injection coverage.

The golden suite owns the broad integration fixtures. These focused wrappers
make the mutation-boundary harness part of the normal unit-test contract too:
outbox failures, persisted stage failures, and crash/resume behavior must stay
green even when the full golden dispatcher is not run.
"""

from __future__ import annotations

import unittest

from nautical_core.lifecycle_application import LifecycleApplicationService
from nautical_core.lifecycle_models import LifecycleDrainProgress, LifecycleDrainStage

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


if __name__ == "__main__":
    unittest.main()
