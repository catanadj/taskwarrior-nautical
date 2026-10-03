"""Completion-spawn lifecycle-plan callback contract."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from nautical_core.lifecycle.models import LifecyclePlan
from nautical_core.modify_completion_spawn import completion_build_and_spawn_child
from nautical_core.modify_models import CompletionSpawnServices
from nautical_core.task_models import NauticalTask, TaskDraft, TaskObservation, TaskPayload


class ModifyCompletionSpawnContractTests(unittest.TestCase):
    def test_spawn_callback_always_receives_the_optional_lifecycle_plan(self) -> None:
        draft = TaskDraft.from_task(
            NauticalTask.from_observation(
                TaskObservation.from_mapping(
                    {
                        "uuid": "00000000-0000-4000-8000-000000000111",
                        "status": "pending",
                        "chain": "on",
                        "chainID": "abcd1234",
                        "link": 2,
                        "description": "child",
                        "due": "2026-01-02T09:00:00Z",
                        "anchor": "w:mon",
                    },
                    source_query="completion spawn test",
                )
            )
        )
        received_plans: list[LifecyclePlan | None] = []

        def spawn_child_atomic(
            _child: TaskDraft | TaskPayload,
            _parent: TaskPayload,
            *,
            lifecycle_plan: LifecyclePlan | None,
        ) -> tuple[str, list[str], bool, bool, str | None, str | None]:
            received_plans.append(lifecycle_plan)
            return "child123", [], True, False, None, "intent-1"

        services = CompletionSpawnServices(
            build_child_draft=lambda *_args: draft,
            spawn_child_atomic=spawn_child_atomic,
            panel=lambda *_args, **_kwargs: None,
            print_task=lambda _task: None,
            diag=lambda _message: None,
        )
        new_task: TaskPayload = {
            "uuid": "00000000-0000-4000-8000-000000000010",
            "status": "completed",
        }

        result = completion_build_and_spawn_child(
            new_task,
            child_due=datetime(2026, 1, 2, 9, tzinfo=timezone.utc),
            next_no=2,
            parent_short="parent01",
            kind="anchor",
            cpmax=0,
            until_dt=None,
            services=services,
        )

        self.assertEqual(received_plans, [None])
        self.assertIsNotNone(result)
        self.assertEqual(result.outcome_state, "applied")
        self.assertEqual(new_task["nextLink"], "child123")


if __name__ == "__main__":
    unittest.main()
