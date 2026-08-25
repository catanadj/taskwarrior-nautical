from __future__ import annotations

import unittest

from nautical_core.lifecycle_models import LifecycleEvent, TaskSnapshot
from nautical_core.lifecycle_planner import LifecyclePlanner, RecurrenceCandidate, terminal_plan_for_snapshot
from nautical_core.task_codec import DEFAULT_TASK_CODEC


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


if __name__ == "__main__":
    unittest.main()
