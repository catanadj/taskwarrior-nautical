from __future__ import annotations

import unittest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from nautical_core.modify_carry_workflow import (
    NativeUntilDecision,
    TemporalCarryAdjustment,
    TemporalCarryDecision,
    apply_temporal_carry_patch,
    apply_native_until_patch,
    decision_from_cp_adjustments,
    verify_native_until_task,
    verify_temporal_carry_task,
)
from nautical_core.task_models import TaskTimestamp


class TemporalCarryWorkflowTests(unittest.TestCase):
    def test_adjustment_and_decision_are_immutable(self) -> None:
        old = TaskTimestamp(datetime(2026, 8, 25, 9, tzinfo=timezone.utc))
        new = TaskTimestamp(datetime(2026, 8, 26, 9, tzinfo=timezone.utc))
        adjustment = TemporalCarryAdjustment("scheduled", old, new, 86400)
        decision = TemporalCarryDecision("adjusted", (adjustment,))
        self.assertEqual(decision.adjustments[0].offset_seconds, 86400.0)

    def test_rejected_carry_requires_evidence(self) -> None:
        with self.assertRaises(ValueError):
            TemporalCarryDecision("rejected")

    def test_cp_result_is_normalized_to_typed_adjustments(self) -> None:
        old = datetime(2026, 8, 25, 9, tzinfo=timezone.utc)
        new = datetime(2026, 8, 26, 9, tzinfo=timezone.utc)
        result = decision_from_cp_adjustments((old, new, [("scheduled", old, new, 86400)]))
        self.assertEqual(result.status, "adjusted")
        self.assertEqual(result.adjustments[0].field, "scheduled")
        self.assertTrue(result)
        self.assertEqual(result.serialized_changes[0][0], "scheduled")

    def test_native_until_decision_requires_carried_value(self) -> None:
        with self.assertRaises(ValueError):
            NativeUntilDecision("carried")

    def test_native_until_decision_rejects_value_on_unchanged(self) -> None:
        stamp = TaskTimestamp(datetime(2026, 8, 25, 9, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, "cannot carry a value"):
            NativeUntilDecision("unchanged", value=stamp)

    def test_native_until_patch_uses_typed_task_patch(self) -> None:
        task = {"uuid": "11111111-1111-4111-8111-111111111111"}
        stamp = TaskTimestamp(datetime(2026, 8, 30, 20, tzinfo=timezone.utc))
        apply_native_until_patch(task, NativeUntilDecision("carried", value=stamp))
        self.assertEqual(task["until"], "2026-08-30T20:00:00Z")
        verify_native_until_task(task, NativeUntilDecision("carried", value=stamp))

    def test_temporal_carry_verification_rejects_lost_field(self) -> None:
        stamp = TaskTimestamp(datetime(2026, 8, 25, 9, tzinfo=timezone.utc))
        decision = TemporalCarryDecision(
            "adjusted",
            (TemporalCarryAdjustment("scheduled", stamp, stamp, 0),),
        )
        with self.assertRaisesRegex(ValueError, "verification failed"):
            verify_temporal_carry_task({}, decision)

    def test_dst_transition_keeps_typed_values_in_utc(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        before = TaskTimestamp(datetime(2026, 10, 25, 9, tzinfo=zone))
        after = TaskTimestamp(datetime(2026, 10, 26, 9, tzinfo=zone))
        decision = TemporalCarryDecision(
            "adjusted",
            (TemporalCarryAdjustment("scheduled", before, after, (after.value - before.value).total_seconds()),),
        )
        self.assertEqual(before.value.tzinfo, timezone.utc)
        self.assertEqual(after.value.tzinfo, timezone.utc)
        self.assertGreater(decision.adjustments[0].offset_seconds, 0)

    def test_malformed_cp_result_fails_closed(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            decision_from_cp_adjustments(("bad",))

    def test_adjusted_decision_applies_one_typed_patch(self) -> None:
        task = {"uuid": "11111111-1111-4111-8111-111111111111"}
        old = TaskTimestamp(datetime(2026, 8, 25, 8, tzinfo=timezone.utc))
        new = TaskTimestamp(datetime(2026, 8, 25, 9, tzinfo=timezone.utc))
        decision = TemporalCarryDecision(
            "adjusted",
            (TemporalCarryAdjustment("scheduled", old, new, 3600),),
        )
        apply_temporal_carry_patch(task, decision)
        self.assertEqual(task["scheduled"], "2026-08-25T09:00:00Z")

    def test_duplicate_adjustments_are_rejected_before_application(self) -> None:
        stamp = TaskTimestamp(datetime(2026, 8, 25, 9, tzinfo=timezone.utc))
        adjustment = TemporalCarryAdjustment("scheduled", stamp, stamp, 0)
        with self.assertRaisesRegex(ValueError, "more than once"):
            TemporalCarryDecision("adjusted", (adjustment, adjustment))

    def test_invalid_patch_target_does_not_partially_apply(self) -> None:
        task = {"uuid": "not-a-uuid", "scheduled": "original"}
        stamp = TaskTimestamp(datetime(2026, 8, 25, 9, tzinfo=timezone.utc))
        decision = TemporalCarryDecision(
            "adjusted",
            (TemporalCarryAdjustment("scheduled", stamp, stamp, 0),),
        )
        with self.assertRaises(ValueError):
            apply_temporal_carry_patch(task, decision)
        self.assertEqual(task["scheduled"], "original")


if __name__ == "__main__":
    unittest.main()
