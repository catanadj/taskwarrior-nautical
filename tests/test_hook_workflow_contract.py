from __future__ import annotations

import unittest

from nautical_core.hook_workflow_models import (
    HOOK_OUTPUT_CONTRACTS,
    HookKind,
    OUTCOME_DISPOSITION_RULES,
    OutcomeDisposition,
    ROUTE_PRECEDENCE,
    WorkflowFailureCategory,
    WorkflowOutcome,
    WorkflowOutcomeKind,
    WorkflowRoute,
)


class HookWorkflowContractTests(unittest.TestCase):
    def test_route_precedence_is_closed_and_unique(self) -> None:
        self.assertEqual(len(ROUTE_PRECEDENCE), len(set(ROUTE_PRECEDENCE)))
        self.assertNotIn(WorkflowRoute.EXIT_DRAIN, ROUTE_PRECEDENCE)

    def test_output_contract_preserves_unicode_and_is_strict(self) -> None:
        self.assertEqual(set(HOOK_OUTPUT_CONTRACTS), set(HookKind))
        self.assertFalse(HOOK_OUTPUT_CONTRACTS[HookKind.ADD].ensure_ascii)
        self.assertEqual(HOOK_OUTPUT_CONTRACTS[HookKind.EXIT].stdout, "empty")

    def test_deferred_failure_preserves_input(self) -> None:
        result = WorkflowOutcome(
            kind=WorkflowOutcomeKind.RETRYABLE_UNAVAILABLE,
            disposition=OutcomeDisposition.DEFER_RECOVERY,
            route=WorkflowRoute.COMPLETION,
            preserve_input=True,
            failure=WorkflowFailureCategory.EVIDENCE_UNAVAILABLE,
        )
        self.assertTrue(result.preserve_input)

    def test_all_outcomes_have_one_disposition_rule(self) -> None:
        self.assertEqual(set(OUTCOME_DISPOSITION_RULES), set(WorkflowOutcomeKind))

    def test_success_cannot_carry_failure(self) -> None:
        with self.assertRaises(ValueError):
            WorkflowOutcome(
                kind=WorkflowOutcomeKind.ACCEPTED_PATCH,
                disposition=OutcomeDisposition.EMIT_TASK,
                route=WorkflowRoute.ORDINARY,
                failure=WorkflowFailureCategory.PROGRAMMING_ERROR,
            )

    def test_rejection_requires_nonzero_exit(self) -> None:
        with self.assertRaises(ValueError):
            WorkflowOutcome(
                kind=WorkflowOutcomeKind.REJECTED_INPUT,
                disposition=OutcomeDisposition.REJECT_OPERATION,
                route=WorkflowRoute.ANCHOR_ACTIVATION,
                failure=WorkflowFailureCategory.INVALID_INPUT,
            )


if __name__ == "__main__":
    unittest.main()
