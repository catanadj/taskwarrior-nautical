"""Completion-spawn lifecycle-plan callback contract."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

from nautical_core.lifecycle.models import LifecyclePlan
from nautical_core.modify_completion_spawn import completion_build_and_spawn_child
from nautical_core.modify_models import CompletionSpawnServices
from nautical_core.task_models import NauticalTask, TaskDraft, TaskObservation, TaskPayload


def _child_draft() -> TaskDraft:
    return TaskDraft.from_task(
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


class ModifyCompletionSpawnContractTests(unittest.TestCase):
    def test_spawn_callback_always_receives_the_optional_lifecycle_plan(self) -> None:
        draft = _child_draft()
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

    def test_child_build_failure_becomes_retryable_result_with_reason(self) -> None:
        diagnostics: list[str] = []
        spawn_calls: list[bool] = []

        def fail_build(*_args: object) -> TaskDraft:
            raise ValueError("invalid recurrence draft")

        def spawn_child_atomic(*_args: object, **_kwargs: object) -> tuple[str, list[str], bool, bool, None, str]:
            spawn_calls.append(True)
            return "child123", [], True, False, None, "intent-1"

        services = CompletionSpawnServices(
            build_child_draft=fail_build,
            spawn_child_atomic=spawn_child_atomic,
            panel=lambda *_args, **_kwargs: None,
            print_task=lambda _task: None,
            diag=diagnostics.append,
        )
        parent: TaskPayload = {"status": "completed"}

        result = completion_build_and_spawn_child(
            parent,
            child_due=datetime(2026, 1, 2, 9, tzinfo=timezone.utc),
            next_no=2,
            parent_short="parent01",
            kind="anchor",
            cpmax=0,
            until_dt=None,
            services=services,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.outcome_state, "retryable")
        self.assertEqual(result.reason, "invalid recurrence draft")
        self.assertEqual(diagnostics, ["build child failed: invalid recurrence draft"])
        self.assertEqual(spawn_calls, [])
        self.assertNotIn("nextLink", parent)

    def test_missing_due_is_rejected_before_child_builder_runs(self) -> None:
        builder_calls: list[bool] = []
        spawn_calls: list[bool] = []

        def build_child_draft(*_args: object) -> TaskDraft:
            builder_calls.append(True)
            return _child_draft()

        def spawn_child_atomic(*_args: object, **_kwargs: object) -> tuple[str, list[str], bool, bool, None, str]:
            spawn_calls.append(True)
            return "child123", [], True, False, None, "intent-1"

        services = CompletionSpawnServices(
            build_child_draft=build_child_draft,
            spawn_child_atomic=spawn_child_atomic,
            panel=lambda *_args, **_kwargs: None,
            print_task=lambda _task: None,
            diag=lambda _message: None,
        )
        result = completion_build_and_spawn_child(
            {"status": "completed"},
            child_due=None,
            next_no=2,
            parent_short="parent01",
            kind="anchor",
            cpmax=0,
            until_dt=None,
            services=services,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.outcome_state, "retryable")
        self.assertEqual(
            result.reason,
            "completion child due is required before building a child draft",
        )
        self.assertEqual(builder_calls, [])
        self.assertEqual(spawn_calls, [])

    def test_spawn_failure_becomes_retryable_result_without_parent_link(self) -> None:
        diagnostics: list[str] = []

        def fail_spawn(
            _child: TaskDraft | TaskPayload,
            _parent: TaskPayload,
            *,
            lifecycle_plan: LifecyclePlan | None,
        ) -> tuple[str, list[str], bool, bool, str | None, str | None]:
            self.assertIsNone(lifecycle_plan)
            raise OSError("outbox unavailable")

        services = CompletionSpawnServices(
            build_child_draft=lambda *_args: _child_draft(),
            spawn_child_atomic=fail_spawn,
            panel=lambda *_args, **_kwargs: None,
            print_task=lambda _task: None,
            diag=diagnostics.append,
        )
        parent: TaskPayload = {"status": "completed"}

        result = completion_build_and_spawn_child(
            parent,
            child_due=datetime(2026, 1, 2, 9, tzinfo=timezone.utc),
            next_no=2,
            parent_short="parent01",
            kind="anchor",
            cpmax=0,
            until_dt=None,
            services=services,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.outcome_state, "retryable")
        self.assertEqual(result.reason, "outbox unavailable")
        self.assertEqual(diagnostics, ["spawn child failed: outbox unavailable"])
        self.assertNotIn("nextLink", parent)


if __name__ == "__main__":
    unittest.main()
