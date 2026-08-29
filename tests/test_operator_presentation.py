import unittest

from nautical_core.operator_models import OperatorOperation, OperatorResult, OperatorStatus
from nautical_core.operator_presentation import ordered_records, render_result
from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits


class OperatorPresentationTests(unittest.TestCase):
    def test_ordered_records_is_deterministic_for_shuffled_canonical_rows(self) -> None:
        rows = [
            {"domain": "chains", "severity": "warning", "actionability": "repairable", "chain_id": "b", "link": 2, "code": "z"},
            {"domain": "config", "severity": "info", "actionability": "informational", "code": "a"},
            {"domain": "chains", "severity": "error", "actionability": "blocking", "chain_id": "a", "link": 1, "code": "x"},
        ]
        expected = tuple(ordered_records(rows))
        self.assertEqual(expected, tuple(ordered_records(list(reversed(rows)))))
        self.assertEqual(expected[0]["domain"], "chains")
        self.assertEqual(expected[0]["severity"], "error")

    def test_rich_renderer_failure_falls_back_to_text(self) -> None:
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK, data={"message": "stable"})

        def failing_renderer(_result: object) -> str:
            raise RuntimeError("injected renderer failure")

        rendered = render_result(result, "rich", rich_renderer=failing_renderer)
        self.assertIn("inspect: ok", rendered)
        self.assertIn("presentation unavailable", rendered)

    def test_render_budget_telemetry_is_an_extension_on_a_copy(self) -> None:
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK, data={"message": "stable"})
        budget = OperatorInvocationBudget(OperatorLimits(taskwarrior_calls=2))
        budget.consume("taskwarrior_calls")
        rendered = render_result(result, budget=budget)
        self.assertIn('"budget"', rendered)
        self.assertEqual(result.extensions, {})


if __name__ == "__main__":
    unittest.main()
