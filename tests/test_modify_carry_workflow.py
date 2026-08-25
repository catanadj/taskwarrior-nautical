from __future__ import annotations

import unittest
from datetime import datetime, timezone

from nautical_core.modify_carry_workflow import TemporalCarryAdjustment, TemporalCarryDecision, decision_from_cp_adjustments
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


if __name__ == "__main__":
    unittest.main()
