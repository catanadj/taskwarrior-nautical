import unittest

import nautical_navigator


class NavigatorBudgetTests(unittest.TestCase):
    def test_anchor_presentation_exposes_bounded_budget_telemetry(self) -> None:
        result = nautical_navigator._anchor_presentation_result("w:mon", count=1)
        payload = result.to_dict()
        self.assertIsInstance(payload["budget"], dict)
        self.assertEqual(payload["budget"]["limits"]["occurrences"], 1)
        self.assertFalse(payload["budget"]["wall_time_exceeded"])


if __name__ == "__main__":
    unittest.main()
