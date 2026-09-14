"""Direct contracts for panel mode routing in the public renderer."""

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from unittest.mock import patch

import nautical_core as core
from nautical_core import ui


class PanelRendererContractTests(unittest.TestCase):
    def test_shared_rich_builder_preserves_static_layout_and_semantic_theme(self) -> None:
        from rich.console import Console

        panel = ui._build_rich_panel(
            "Nautical title",
            [("Pattern", "w:mon"), (None, "separator"), ("Warning", "check this")],
            kind="preview_anchor",
            themes={
                "preview_anchor": {
                    "border": "magenta",
                    "title": "bright_cyan",
                    "label": "green",
                }
            },
        )
        output = StringIO()
        Console(file=output, width=80, color_system=None).print(panel)
        rendered = output.getvalue()

        self.assertIn("Nautical title", rendered)
        self.assertIn("Pattern", rendered)
        self.assertIn("w:mon", rendered)
        self.assertIn("separator", rendered)
        self.assertIn("Warning", rendered)
        self.assertIn("check this", rendered)
        self.assertEqual(str(panel.border_style), "magenta")

        themes = ui.panel_themes()
        self.assertEqual(
            themes["preview_anchor"],
            {"border": "turquoise2", "title": "bright_cyan", "label": "sea_green2"},
        )
        self.assertEqual(themes["summary"]["border"], "magenta")
        self.assertEqual(themes["error"]["border"], "red")
        themes["info"]["border"] = "changed"
        self.assertEqual(ui.panel_themes()["info"]["border"], "blue")

    def test_static_rich_renderer_prints_the_shared_builder_result(self) -> None:
        from rich.text import Text

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        captured = {}
        stderr = TtyBuffer()

        def builder(title, rows, *, kind, themes):
            captured.update(title=title, rows=list(rows), kind=kind, themes=themes)
            return Text("shared builder output")

        with (
            patch.object(ui, "_build_rich_panel", side_effect=builder),
            redirect_stderr(stderr),
        ):
            rendered = ui._render_panel_rich(
                "Delegated",
                [("Key", "Value")],
                kind="info",
                themes={"info": {"border": "blue"}},
            )

        self.assertTrue(rendered)
        self.assertEqual(captured["title"], "Delegated")
        self.assertEqual(captured["rows"], [("Key", "Value")])
        self.assertEqual(captured["kind"], "info")
        self.assertIn("shared builder output", stderr.getvalue())

    def test_live_reveal_keeps_timeline_plain_and_unhighlighted(self) -> None:
        frames = ui._live_reveal_frames(
            [
                ("Summary", "ready"),
                ("Timeline", "old\ncurrent\nnext"),
                ("Chain", "on"),
            ]
        )
        timeline_frames = [rows[-1][1] for rows, active in frames if active == 1]

        self.assertEqual(
            timeline_frames,
            ["old", "old\ncurrent", "old\ncurrent\nnext"],
        )
        self.assertTrue(
            all("▸" not in value and "bright_cyan" not in value for value in timeline_frames)
        )

    def test_line_mode_promotes_force_rich_kinds_without_using_line_renderer(self) -> None:
        stderr = StringIO()
        with patch.object(core, "panel_line") as panel_line, redirect_stderr(stderr):
            core.render_panel(
                "Title",
                [("Key", "Value")],
                kind="preview_anchor",
                panel_mode="line",
                line_force_rich_kinds={"preview_anchor"},
            )

        panel_line.assert_not_called()
        self.assertIn("Title", stderr.getvalue())
        self.assertIn("Key", stderr.getvalue())

    def test_live_failure_preserves_generator_rows_for_static_fallback(self) -> None:
        stderr = StringIO()
        stdout = StringIO()

        def fail_after_consuming(_title, rows, **_kwargs):
            list(rows)
            return False

        with (
            patch.object(ui, "_render_panel_live", side_effect=fail_after_consuming),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            rows = ((key, value) for key, value in [("Key", "Value")])
            ui.render_panel("Fallback", rows, panel_mode="live")

        self.assertIn("Fallback", stderr.getvalue())
        self.assertIn("Key", stderr.getvalue())
        self.assertIn("Value", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_live_mode_uses_plain_fallback_when_stderr_is_not_a_tty(self) -> None:
        import rich.live

        stderr = StringIO()
        stdout = StringIO()
        with (
            patch.object(
                rich.live,
                "Live",
                side_effect=AssertionError("Live must not start for captured stderr"),
            ),
            patch.dict(os.environ, {"TERM": "xterm"}),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            ui.render_panel("Captured", [("Key", "Value")], panel_mode="live")

        output = stderr.getvalue()
        self.assertIn("Captured", output)
        self.assertIn("Key", output)
        self.assertIn("Value", output)
        self.assertNotIn("\x1b[", output)
        self.assertEqual(stdout.getvalue(), "")

    def test_live_renderer_rejects_dumb_terminal_even_when_stderr_is_a_tty(self) -> None:
        import rich.live

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        with (
            patch.dict(os.environ, {"TERM": "dumb"}),
            patch.object(
                rich.live,
                "Live",
                side_effect=AssertionError("Live must not start on TERM=dumb"),
            ),
            patch("sys.stderr", TtyBuffer()),
        ):
            rendered = ui._render_panel_live(
                "Dumb", [("Key", "Value")], kind="info", themes=None
            )

        self.assertFalse(rendered)

    def test_line_mode_routes_the_compact_message_through_panel_line(self) -> None:
        with patch.object(core, "panel_line") as panel_line:
            core.render_panel(
                "Title", [("Key", "Value")], kind="info", panel_mode="line"
            )

        panel_line.assert_called_once()
        self.assertEqual(panel_line.call_args.args[:2], ("Title", "Title — Key: Value"))
        self.assertEqual(panel_line.call_args.kwargs["kind"], "info")

    def test_successful_live_render_does_not_fall_through_to_static_rich(self) -> None:
        live_calls = []

        with (
            patch.object(
                ui, "_render_panel_live",
                side_effect=lambda title, rows, **kwargs: (
                    live_calls.append((title, list(rows), kwargs)) or True
                ),
            ),
            patch.object(
                ui,
                "_render_panel_rich",
                side_effect=AssertionError("successful live panel was duplicated"),
            ),
        ):
            ui.render_panel("Routed", [("Key", "Value")], kind="info", panel_mode="live")

        self.assertEqual(len(live_calls), 1)
        self.assertEqual(live_calls[0][0:2], ("Routed", [("Key", "Value")]))


if __name__ == "__main__":
    unittest.main()
