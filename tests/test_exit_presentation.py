import unittest
from types import SimpleNamespace

from nautical_core.exit_presentation import ExitDrainProgress, render_drain_interrupted_panel


class ExitDrainProgressTests(unittest.TestCase):
    def test_internal_drain_stages_use_concise_user_facing_labels(self) -> None:
        expected = {
            "starting intent": "Preparing",
            "child mutation": "Created",
            "child verified": "Confirmed",
            "child mutation and verification": "Created",
            "parent mutation": "Linked",
            "parent verified": "Confirmed",
            "parent mutation and verification": "Linked",
            "intent verified": "Verified",
            "intent acknowledged": "Recorded",
            "intent finished": "Completed",
        }
        self.assertTrue(all(" " not in label for label in expected.values()))
        for internal, friendly in expected.items():
            with self.subTest(internal=internal):
                self.assertEqual(
                    ExitDrainProgress._description(internal),
                    friendly,
                )

    def test_unknown_detail_does_not_expose_internal_vocabulary(self) -> None:
        self.assertEqual(
            ExitDrainProgress._description("future_internal_phase"),
            "Processing",
        )

    def test_empty_detail_uses_compact_title(self) -> None:
        self.assertEqual(ExitDrainProgress._description(), "Processing")

    def test_interrupted_drain_panel_is_actionable_and_compact(self) -> None:
        rendered = []
        core = SimpleNamespace(
            PANEL_MODE="text",
            LIVE_PANEL_DURATION_MS=0,
            LIVE_PANEL_FOOTER="NAUTICAL",
            FAST_COLOR=False,
            panel_themes=lambda: {},
            render_panel=lambda title, rows, **kwargs: rendered.append((title, rows, kwargs)),
        )

        render_drain_interrupted_panel(core)

        self.assertEqual(len(rendered), 1)
        title, rows, options = rendered[0]
        self.assertEqual(title, "⚠ Nautical drain interrupted")
        values = dict(rows)
        self.assertIn("stopped", values["Status"])
        self.assertIn("preserved", values["Recovery"])
        self.assertEqual(values["Action"], "Run nautical reconcile --apply")
        self.assertEqual(options["kind"], "warning")


if __name__ == "__main__":
    unittest.main()
