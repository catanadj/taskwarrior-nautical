import unittest
from types import SimpleNamespace

from nautical_core.exit_presentation import ExitDrainProgress, render_drain_interrupted_panel


class ExitDrainProgressTests(unittest.TestCase):
    def test_all_internal_drain_stages_use_one_neutral_label(self) -> None:
        for internal in (
            "starting intent",
            "child mutation",
            "child verified",
            "child mutation and verification",
            "parent mutation",
            "parent verified",
            "parent mutation and verification",
            "intent verified",
            "intent acknowledged",
            "intent finished",
        ):
            with self.subTest(internal=internal):
                self.assertEqual(
                    ExitDrainProgress._description(internal),
                    "Draining",
                )

    def test_unknown_detail_does_not_expose_internal_vocabulary(self) -> None:
        self.assertEqual(
            ExitDrainProgress._description("future_internal_phase"),
            "Draining",
        )

    def test_empty_detail_uses_compact_title(self) -> None:
        self.assertEqual(ExitDrainProgress._description(), "Draining")

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
