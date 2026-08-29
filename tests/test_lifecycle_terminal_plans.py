from __future__ import annotations

import unittest

from nautical_core.lifecycle_models import LifecycleEvent, TaskSnapshot
from nautical_core.lifecycle_recovery_models import RecoveryPlanResult, RecoveryRefusal, RecoveryStatus
from nautical_core.lifecycle_planner import LifecyclePlanner, RecurrenceCandidate, terminal_plan_for_snapshot
from nautical_core.chain_integrity_lifecycle import describe_recovery_result
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.task_models import NauticalTask, TaskDraft


def snapshot() -> TaskSnapshot:
    observation = DEFAULT_TASK_CODEC.decode_row(
        {
            "uuid": "11111111-1111-4111-8111-111111111111",
            "status": "pending",
            "chain": "on",
            "chainID": "abcd1234",
            "link": 4,
            "anchor": "w:mon",
            "due": "20260824T090000Z",
        },
        source_query="terminal-plan-test",
    )
    return TaskSnapshot.from_observation(observation)


class ExhaustedService:
    def next_candidate(self, *_args: object, **_kwargs: object) -> RecurrenceCandidate:
        return RecurrenceCandidate(child_due=None, terminal_reason="scheduler_exhausted")

    def build_child(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("an exhausted candidate must not build a child")


class LifecycleTerminalPlanTests(unittest.TestCase):
    def test_task_draft_drops_taskwarrior_native_urgency(self) -> None:
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "11111111-1111-4111-8111-111111111111",
                "status": "pending", "chain": "on", "chainID": "abcd1234", "link": 4,
                "description": "urgency carry regression", "anchor": "w:mon",
                "due": "20260824T090000Z", "urgency": 7.25,
            },
            source_query="task-draft-urgency-test",
        )
        draft = TaskDraft.from_task(NauticalTask.from_observation(observation))
        self.assertNotIn("urgency", draft.fields)
        self.assertNotIn("urgency", draft.to_mapping())

    def test_bound_events_preserve_terminal_kind(self) -> None:
        self.assertEqual(
            terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_MAX).terminal_kind,
            "chain_max",
        )
        self.assertEqual(
            terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_UNTIL).terminal_kind,
            "chain_until",
        )

    def test_scheduler_exhaustion_is_durable_terminal_provenance(self) -> None:
        plan = LifecyclePlanner(
            validated_configuration=object(),
            recurrence_service=ExhaustedService(),
        ).plan(snapshot(), LifecycleEvent.EXPIRE)
        self.assertEqual(plan.terminal_kind, "search_limit")

    def test_terminal_plan_replay_preserves_identity_and_provenance(self) -> None:
        plan = terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_UNTIL)
        restored = type(plan).from_dict(plan.to_dict())
        self.assertEqual(restored.identity.key, plan.identity.key)
        self.assertEqual(restored.identity.idempotency_key, plan.identity.idempotency_key)
        self.assertEqual(restored.terminal_kind, "chain_until")
        self.assertEqual(restored.action, plan.action)
        self.assertEqual(restored.expected_postconditions, plan.expected_postconditions)

    def test_shared_description_preserves_terminal_provenance(self) -> None:
        plan = terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_MAX)
        result = RecoveryPlanResult(
            snapshot().observation,
            plan,
            reason="reached chain maximum",
            terminal_kind=plan.terminal_kind,
        )
        evidence = describe_recovery_result(result)
        self.assertEqual(evidence["terminal_kind"], "chain_max")
        self.assertEqual(evidence["trigger"], "completion")

    def test_refusal_description_preserves_typed_status_and_evidence(self) -> None:
        refusal = RecoveryRefusal(
            snapshot().observation,
            RecoveryStatus.RETRYABLE,
            "scheduler evidence is unavailable",
            {"child": "deadbeef", "due": "2026-08-24T09:00:00Z"},
        )
        evidence = describe_recovery_result(refusal)
        self.assertEqual(evidence["status"], "retryable")
        self.assertEqual(evidence["reason"], "scheduler evidence is unavailable")
        self.assertEqual(evidence["child"], "deadbeef")
        self.assertEqual(evidence["due"], "2026-08-24T09:00:00Z")
        self.assertNotIn("action", evidence)
        self.assertNotIn("child_target", evidence)


if __name__ == "__main__":
    unittest.main()
