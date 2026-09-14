import json
import unittest

from nautical_core.operator_models import OperatorOperation, OperatorResult, OperatorStatus
from nautical_core.operator_presentation import ordered_records, render_result
from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits


class OperatorPresentationTests(unittest.TestCase):
    def test_operator_presentation_is_immutable_and_deterministic(self) -> None:
        from nautical_core.lifecycle_models import LifecycleDrainProgress, LifecycleDrainStage
        from nautical_core.operator_models import OperatorCursor, OperatorPage
        from nautical_core.operator_presentation import (
            ProgressView,
            bounded_text,
            ordered_findings,
            render_json,
            render_json_document,
            render_text,
        )

        result = OperatorResult(
            operation=OperatorOperation.INSPECT,
            status=OperatorStatus.OK,
            data={"items": [{"chain_id": "b", "link": 2}, {"chain_id": "a", "link": 1}]},
        )
        before = result.to_dict()
        encoded = render_json(result)
        self.assertIn('"schema": "nautical.operator.inspect"', encoded)
        self.assertEqual(render_text(result), "inspect: ok")
        paged = OperatorResult(
            operation=OperatorOperation.INSPECT,
            status=OperatorStatus.OK,
            page=OperatorPage(
                items=({"uuid": "one"},),
                cursor=OperatorCursor("snapshot", "config", "epoch", position=0, page_size=1),
                complete=False,
            ),
        )
        self.assertIn("more available", render_text(paged))
        exit_code = result.exit_code
        self.assertTrue(render_result(result, "json") and render_result(result, "text"))
        self.assertEqual(render_result(result, "disabled"), "")
        self.assertEqual(render_result(result, "rich", rich_renderer=lambda value: value.status.value), "ok")
        failed_render = render_result(
            result,
            "rich",
            rich_renderer=lambda _value: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        self.assertIn("presentation unavailable", failed_render)
        self.assertEqual(result.exit_code, exit_code)

        self.assertEqual(json.loads(render_json_document(result)), json.loads(encoded))
        self.assertEqual(result.to_dict(), before)
        findings = ordered_findings(
            [
                {"severity": "warning", "chain_id": "a", "link": 1},
                {"severity": "error", "chain_id": "b", "link": 2},
            ]
        )
        self.assertEqual(findings[0]["severity"], "error")
        records = ordered_records(
            [{"state": "retry", "intent_id": "b"}, {"state": "acknowledged", "intent_id": "a"}],
            keys=("state", "intent_id"),
        )
        self.assertEqual(records[0]["state"], "acknowledged")
        progress = ProgressView.from_event(
            LifecycleDrainProgress(LifecycleDrainStage.PROCESSING, completed=2, total=4, detail="spawn_child")
        )
        self.assertEqual(progress.fraction, 0.5)
        self.assertEqual(progress.label, "spawn child")
        self.assertEqual(bounded_text("line\nwith ⚓", width=20), "line with ⚓")
        self.assertEqual(bounded_text("0123456789", width=5), "0123…")

    def test_operator_presentation_modes_preserve_unicode_and_outcome(self) -> None:
        result = OperatorResult(
            operation=OperatorOperation.APPLY,
            status=OperatorStatus.MANUAL_REVIEW,
            data={"description": "Méditation ⚓", "plan": {"action": "review"}},
        )
        expected_code = result.exit_code
        outputs = {
            mode: render_result(result, mode, rich_renderer=lambda value: value.to_dict()["data"]["description"])
            for mode in ("json", "text", "rich", "disabled")
        }
        self.assertIn("Méditation", outputs["json"])
        self.assertIn("Méditation", outputs["rich"])
        self.assertEqual(outputs["disabled"], "")
        self.assertEqual(result.exit_code, expected_code)

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

    def test_static_renderer_exposes_budget_usage_and_overages(self) -> None:
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK)
        budget = OperatorInvocationBudget(OperatorLimits(taskwarrior_calls=1))
        budget.begin_effect()
        budget.consume("taskwarrior_calls", 2)
        rendered = render_result(result, "text", budget=budget)
        self.assertIn("budget calls=2", rendered)
        self.assertIn("exceeded=taskwarrior_calls", rendered)

    def test_expired_wall_time_is_visible_during_presentation(self) -> None:
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK)
        budget = OperatorInvocationBudget(OperatorLimits(wall_time_seconds=1))
        budget._started_monotonic -= 2
        rendered = render_result(result, budget=budget)
        self.assertIn('"wall_time_exceeded": true', rendered)


if __name__ == "__main__":
    unittest.main()
