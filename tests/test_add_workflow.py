from __future__ import annotations

import unittest

from datetime import datetime, timezone

from nautical_core.add_workflow import classify_add_route, plan_add, record_schedule
from nautical_core.hook_workflow_models import PatchOperation, WorkflowRoute
from nautical_core.task_models import TaskObservation, TaskTimestamp


def observation(values: dict[str, object]) -> TaskObservation:
    base: dict[str, object] = {
        "uuid": "11111111-1111-4111-8111-111111111111",
        "description": "workflow add",
        "status": "pending",
        "entry": "20260825T090000Z",
    }
    base.update(values)
    return TaskObservation.from_mapping(base, source_query="add-workflow-test")


class AddWorkflowTests(unittest.TestCase):
    def test_ordinary_add_is_a_noop_plan(self) -> None:
        task = observation({})
        plan = plan_add(task)
        self.assertTrue(plan.ordinary)
        self.assertEqual(plan.patch.operations, ())

    def test_cp_add_emits_root_defaults_without_mutation(self) -> None:
        task = observation({"cp": "P1D"})
        plan = plan_add(task)
        self.assertEqual(classify_add_route(task), WorkflowRoute.CP_ACTIVATION)
        self.assertEqual(plan.recurrence_kind, "cp")
        self.assertEqual(
            [(item.field, item.operation, item.value) for item in plan.patch.operations],
            [("chain", PatchOperation.SET, "on"), ("link", PatchOperation.SET, 1), ("chainID", PatchOperation.SET, "11111111")],
        )
        self.assertIsNone(task.get("chain"))

    def test_anchor_file_route_defaults_mode(self) -> None:
        plan = plan_add(observation({"anchor_file": "calendar.csv"}))
        self.assertEqual(plan.request.route, WorkflowRoute.ANCHOR_FILE_ACTIVATION)
        self.assertEqual(plan.patch.operations[-1].field, "anchor_mode")

    def test_combined_anchor_sources_keep_anchor_route(self) -> None:
        task = observation({"anchor": "w:mon", "anchor_file": "calendar.csv"})
        self.assertEqual(classify_add_route(task), WorkflowRoute.ANCHOR_ACTIVATION)

    def test_target_field_prefers_scheduled_when_due_is_implicit(self) -> None:
        plan = plan_add(observation({"cp": "P1D", "scheduled": "20260825T090000Z"}))
        self.assertEqual(plan.target_field, "scheduled")
        self.assertTrue(plan.target_explicit)

    def test_scheduler_result_is_typed_and_does_not_mutate_request(self) -> None:
        task = observation({"anchor": "w:mon"})
        plan = plan_add(task)
        timestamp = TaskTimestamp(datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        scheduled = record_schedule(plan, first_occurrence=timestamp)
        self.assertIsNone(plan.schedule)
        self.assertEqual(scheduled.schedule.target_field, "due")
        self.assertEqual(scheduled.schedule.first_occurrence, timestamp)
        self.assertEqual(scheduled.feedback.first_occurrence, timestamp)
        self.assertEqual(scheduled.feedback.recurrence_kind, "anchor")
        self.assertEqual(task.get("due"), None)

    def test_scheduler_unavailable_is_explicit(self) -> None:
        plan = plan_add(observation({"cp": "P1D"}))
        scheduled = record_schedule(plan, first_occurrence=None, status="unavailable")
        self.assertEqual(scheduled.schedule.status, "unavailable")
        self.assertTrue(scheduled.feedback.warnings)


if __name__ == "__main__":
    unittest.main()
