from __future__ import annotations

import unittest

from nautical_core.hook_workflow_models import (
    HOOK_OUTPUT_CONTRACTS,
    HookKind,
    OUTCOME_DISPOSITION_RULES,
    FeedbackFacts,
    LifecycleEffectRef,
    OutcomeDisposition,
    ROUTE_PRECEDENCE,
    WorkflowFailureCategory,
    WorkflowOutcome,
    WorkflowOutcomeKind,
    WorkflowRoute,
    WorkflowOperationalResult,
    TaskPatchEffect,
    TerminalStateEffect,
    TaskPatch,
    TaskPatchOperation,
    PatchOperation,
)
from nautical_core.lifecycle_models import TaskLifecycleState
from nautical_core.task_models import TaskObservation


class HookWorkflowContractTests(unittest.TestCase):
    @staticmethod
    def _observation() -> TaskObservation:
        return TaskObservation.from_mapping(
            {
                "uuid": "11111111-1111-4111-8111-111111111111",
                "status": "pending",
                "chainID": "abcd1234",
                "link": 1,
                "chain": "on",
                "cp": "P1D",
            },
            source_query="contract-test",
        )

    def test_route_precedence_is_closed_and_unique(self) -> None:
        self.assertEqual(len(ROUTE_PRECEDENCE), len(set(ROUTE_PRECEDENCE)))
        self.assertNotIn(WorkflowRoute.EXIT_DRAIN, ROUTE_PRECEDENCE)

    def test_output_contract_preserves_unicode_and_is_strict(self) -> None:
        self.assertEqual(set(HOOK_OUTPUT_CONTRACTS), set(HookKind))
        self.assertFalse(HOOK_OUTPUT_CONTRACTS[HookKind.ADD].ensure_ascii)
        self.assertEqual(HOOK_OUTPUT_CONTRACTS[HookKind.EXIT].stdout, "empty")
        self.assertTrue(all(contract.diagnostics_stderr_only_when_enabled for contract in HOOK_OUTPUT_CONTRACTS.values()))

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

    def test_feedback_and_operational_result_are_deterministic(self) -> None:
        outcome = WorkflowOutcome(
            kind=WorkflowOutcomeKind.PASSTHROUGH,
            disposition=OutcomeDisposition.EMIT_TASK,
            route=WorkflowRoute.ORDINARY,
        )
        facts = FeedbackFacts(
            recurrence_kind="cp",
            natural_explanation="Every day",
            carry_changes=(("scheduled", "preserved"),),
            warnings=("warning",),
        )
        first = WorkflowOperationalResult(self._observation(), outcome, feedback=facts)
        second = WorkflowOperationalResult(self._observation(), outcome, feedback=facts)
        self.assertEqual(first, second)

    def test_lifecycle_effect_rejects_non_plan_values(self) -> None:
        with self.assertRaises(TypeError):
            LifecycleEffectRef(object())

    def test_operational_result_accepts_only_closed_effect_set(self) -> None:
        outcome = WorkflowOutcome(
            kind=WorkflowOutcomeKind.PASSTHROUGH,
            disposition=OutcomeDisposition.EMIT_TASK,
            route=WorkflowRoute.ORDINARY,
        )
        task = self._observation()
        patch = TaskPatch((TaskPatchOperation("value", PatchOperation.SET, 3),))
        result = WorkflowOperationalResult(
            task,
            outcome,
            effects=(TaskPatchEffect(patch), TerminalStateEffect(task, TaskLifecycleState.ACTIVE)),
        )
        self.assertEqual(len(result.effects), 2)
        with self.assertRaises(TypeError):
            WorkflowOperationalResult(task, outcome, effects=(object(),))

    def test_terminal_state_effect_normalizes_state_and_reason(self) -> None:
        effect = TerminalStateEffect(self._observation(), "terminal", "  completed  ")
        self.assertIs(effect.state, TaskLifecycleState.TERMINAL)
        self.assertEqual(effect.reason, "completed")

    def test_feedback_facts_deduplicate_repeated_messages(self) -> None:
        facts = FeedbackFacts(
            warnings=("same warning", "same warning", "different warning"),
            recovery_guidance=("retry", "retry"),
        )
        self.assertEqual(facts.warnings, ("same warning", "different warning"))
        self.assertEqual(facts.recovery_guidance, ("retry",))

    def test_feedback_facts_carry_actionable_identity(self) -> None:
        facts = FeedbackFacts(
            task_uuid=" task-1 ",
            chain_id=" chain-1 ",
            changed_fields=("due", "due", "scheduled"),
            next_action="run reconcile",
        )
        self.assertEqual(facts.task_uuid, "task-1")
        self.assertEqual(facts.chain_id, "chain-1")
        self.assertEqual(facts.changed_fields, ("due", "scheduled"))
        self.assertEqual(facts.next_action, "run reconcile")

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
