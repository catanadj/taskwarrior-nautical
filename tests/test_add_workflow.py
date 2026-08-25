from __future__ import annotations

import unittest

from datetime import datetime, timezone

from nautical_core.add_workflow import (
    AddScheduleLimits,
    AddScheduleFailure,
    classify_add_route,
    plan_add,
    record_limits,
    record_schedule,
    require_schedule,
    schedule_patch,
    preview_policy,
    record_preview,
    apply_task_patch,
    AddWorkflowApplication,
)
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
    def test_application_prepares_and_attaches_typed_plan(self) -> None:
        application = AddWorkflowApplication(
            record_schedule_fn=lambda plan, _task, _field: plan,
            record_limits_fn=lambda plan, _task, _context: plan,
            record_preview_fn=lambda plan: plan,
        )
        task = {"description": "test"}
        plan = application.prepare(task, observation({"anchor": "w:mon"}))
        self.assertEqual(plan.recurrence_kind, "anchor")
        self.assertEqual(task["chain"], "on")
    def test_apply_task_patch_owns_add_mutation_semantics(self) -> None:
        task = {"description": "test"}
        plan = plan_add(observation({"anchor": "w:mon"}))
        apply_task_patch(task, plan.patch)
        self.assertEqual(task["chain"], "on")
        self.assertEqual(task["link"], 1)

    def test_ordinary_add_is_a_noop_plan(self) -> None:
        task = observation({})
        plan = plan_add(task)
        self.assertTrue(plan.ordinary)
        self.assertEqual(plan.patch.operations, ())

    def test_plan_fingerprint_is_deterministic(self) -> None:
        first = plan_add(observation({"anchor": "w:mon"}))
        second = plan_add(observation({"anchor": "w:mon"}))
        self.assertEqual(first.deterministic_fingerprint, second.deterministic_fingerprint)

    def test_resolved_fingerprint_changes_with_scheduler_result(self) -> None:
        plan = plan_add(observation({"anchor": "w:mon"}))
        timestamp = TaskTimestamp(datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        resolved = record_schedule(plan, first_occurrence=timestamp)
        self.assertNotEqual(plan.resolved_fingerprint, resolved.resolved_fingerprint)

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
        with self.assertRaisesRegex(AddScheduleFailure, "unavailable"):
            require_schedule(scheduled)

    def test_schedule_is_required_before_consumption(self) -> None:
        with self.assertRaisesRegex(AddScheduleFailure, "unavailable"):
            require_schedule(plan_add(observation({"anchor": "w:mon"})))

    def test_explicit_target_is_preserved_after_scheduler_result(self) -> None:
        plan = plan_add(observation({"anchor": "w:mon", "due": "20260831T060000Z"}))
        timestamp = TaskTimestamp(datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        patch = schedule_patch(plan, first_occurrence=timestamp, encode_timestamp=str)
        self.assertEqual(patch.operations, ())

    def test_auto_target_receives_only_scheduler_timestamp(self) -> None:
        plan = plan_add(observation({"anchor": "w:mon"}))
        timestamp = TaskTimestamp(datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        with self.assertRaisesRegex(AddScheduleFailure, "unavailable"):
            schedule_patch(plan, first_occurrence=timestamp, encode_timestamp=str)
        resolved = record_schedule(plan, first_occurrence=timestamp)
        patch = schedule_patch(resolved, first_occurrence=timestamp, encode_timestamp=lambda value: value.value.strftime("%Y%m%dT%H%M%SZ"))
        self.assertEqual(patch.operations[0].field, "due")
        self.assertEqual(patch.operations[0].value, "20260831T060000Z")

    def test_auto_target_rejects_mismatched_scheduler_timestamp(self) -> None:
        plan = plan_add(observation({"anchor": "w:mon"}))
        selected = TaskTimestamp(datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        other = TaskTimestamp(datetime(2026, 9, 7, 6, tzinfo=timezone.utc))
        resolved = record_schedule(plan, first_occurrence=selected)
        with self.assertRaisesRegex(ValueError, "does not match"):
            schedule_patch(resolved, first_occurrence=other, encode_timestamp=str)

    def test_limits_are_typed_and_attached_without_mutation(self) -> None:
        plan = plan_add(observation({"anchor": "w:mon"}))
        limits = AddScheduleLimits(chain_max=3, expiration_hops=2)
        bounded = record_limits(plan, limits)
        self.assertIsNone(plan.limits)
        self.assertEqual(bounded.limits, limits)

    def test_limits_accept_temporal_carry_fields(self) -> None:
        timestamp = TaskTimestamp(datetime(2026, 8, 31, 6, tzinfo=timezone.utc))
        limits = AddScheduleLimits(wait=timestamp, scheduled=timestamp)
        self.assertEqual(limits.wait, timestamp)

    def test_compact_preview_is_bounded_to_one_occurrence(self) -> None:
        policy = preview_policy(panel_mode="quiet", requested_limit=6, hard_cap=32)
        self.assertTrue(policy.enabled)
        self.assertEqual(policy.occurrence_limit, 1)

    def test_preview_policy_honors_cap_for_rich_mode(self) -> None:
        policy = preview_policy(panel_mode="rich", requested_limit=20, hard_cap=4)
        self.assertEqual(policy.occurrence_limit, 4)

    def test_preview_policy_is_attached_separately(self) -> None:
        plan = plan_add(observation({"anchor": "w:mon"}))
        policy = preview_policy(panel_mode="minimal", requested_limit=5, hard_cap=32)
        rendered = record_preview(plan, policy)
        self.assertIsNone(plan.preview)
        self.assertEqual(rendered.preview, policy)


if __name__ == "__main__":
    unittest.main()
