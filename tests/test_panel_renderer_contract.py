"""Direct contracts for panel mode routing in the public renderer."""

import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from unittest.mock import patch

import nautical_core as core
from nautical_core import ui


class PanelRendererContractTests(unittest.TestCase):
    def test_live_panel_branding_focus_and_footer_bounds_do_not_change_static_panels(self) -> None:
        from rich.console import Console

        static_panel = ui._build_rich_panel("Static", [("Key", "Value")], kind="info", themes=None)
        live_panel = ui._build_rich_panel(
            "Live", [("First", "one"), ("Second", "two")],
            kind="info", themes=None, live=True, active_row=1,
        )
        settled_panel = ui._build_rich_panel(
            "Live", [("First", "one"), ("Second", "two")], kind="info", themes=None, live=True,
        )
        active_output = StringIO()
        settled_output = StringIO()
        Console(file=active_output, width=80, color_system=None).print(live_panel)
        Console(file=settled_output, width=80, color_system=None).print(settled_panel)

        self.assertIsNone(static_panel.subtitle)
        self.assertEqual(str(live_panel.subtitle), "NAUTICAL")
        self.assertEqual(live_panel.subtitle_align, "right")
        self.assertIn("▸", active_output.getvalue())
        self.assertNotIn("▸", settled_output.getvalue())
        self.assertEqual(str(live_panel.border_style), "bold blue")
        self.assertEqual(str(settled_panel.border_style), "blue")
        self.assertNotIn("/", str(live_panel.subtitle))
        self.assertEqual(
            max(map(len, active_output.getvalue().splitlines())),
            max(map(len, settled_output.getvalue().splitlines())),
        )
        self.assertEqual(ui._live_reveal_delays(1, 160), [])

        narrow_output = StringIO()
        Console(file=narrow_output, width=40, color_system=None).print(
            ui._build_rich_panel("N", [("A", "1")], kind="info", themes=None, live=True)
        )
        self.assertIn("NAUTICAL", narrow_output.getvalue())

        custom = ui._build_rich_panel(
            "Custom", [("First", "one")], kind="info", themes=None,
            live=True, live_footer="STATUS",
        )
        bounded = ui._build_rich_panel(
            "Bounded", [("First", "one")], kind="info", themes=None,
            live=True, live_footer="A" * 80,
        )
        self.assertEqual(str(custom.subtitle), "STATUS")
        self.assertEqual(str(bounded.subtitle), "A" * 29 + "...")

        semantic = ui._build_rich_panel(
            "Semantic", [("Warning", "late")], kind="info", themes=None,
            live=True, active_row=0,
        )
        styles = [str(span.style) for span in semantic.renderable.columns[1]._cells[0].spans]
        self.assertTrue(any("yellow" in style for style in styles))
        self.assertIn("bold", styles)
        self.assertFalse(any("cyan" in style for style in styles))

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

    def test_live_reveals_cumulative_rows_and_multiline_values(self) -> None:
        import rich.live

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        built = []
        delays = []
        frames = []
        refreshes = []

        def builder(_title, rows, *, active_row=None, **_kwargs):
            snapshot = list(rows)
            built.append((snapshot, active_row))
            return tuple(snapshot)

        class FakeLive:
            def __init__(self, renderable, **kwargs):
                self.frames = frames
                frames.append(renderable)
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def update(self, renderable, *, refresh=False):
                refreshes.append(refresh)
                frames.append(renderable)

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        with (
            patch.dict(os.environ, {"TERM": "xterm"}),
            patch("sys.stderr", TtyBuffer()),
            patch.object(ui, "_build_rich_panel", side_effect=builder),
            patch.object(rich.live, "Live", FakeLive),
            patch.object(ui.time, "sleep", delays.append),
        ):
            ui._reset_live_animation_state()
            try:
                self.assertTrue(
                    ui._render_panel_live(
                        "Rows",
                        [("One", "1"), ("Two", "2")],
                        kind="info",
                        themes=None,
                    )
                )
                self.assertEqual(
                    built,
                    [
                        ([("One", "1"), ("Two", "2")], None),
                        ([("One", "[dim]1[/]")], 0),
                        ([("One", "1")], 0),
                        ([("One", "1"), ("Two", "[dim]2[/]")], 1),
                        ([("One", "1"), ("Two", "2")], 1),
                    ],
                )
                self.assertEqual(len(delays), 4)
                self.assertAlmostEqual(sum(delays), 0.16)
                self.assertEqual(refreshes, [True] * 4)

                built.clear()
                delays.clear()
                frames.clear()
                refreshes.clear()
                ui._reset_live_animation_state()
                self.assertTrue(
                    ui._render_panel_live(
                        "Multiline",
                        [("Upcoming", "one\ntwo\nthree"), ("Chain", "enabled")],
                        kind="info",
                        themes=None,
                        duration_ms=160,
                    )
                )
                self.assertEqual(
                    built,
                    [
                        ([("Upcoming", "one\ntwo\nthree"), ("Chain", "enabled")], None),
                        ([("Upcoming", "one")], 0),
                        ([("Upcoming", "one\ntwo")], 0),
                        ([("Upcoming", "one\ntwo\nthree")], 0),
                        ([("Upcoming", "one\ntwo\nthree"), ("Chain", "[dim]enabled[/]")], 1),
                        ([("Upcoming", "one\ntwo\nthree"), ("Chain", "enabled")], 1),
                    ],
                )
                self.assertEqual(len(delays), 5)
                self.assertAlmostEqual(sum(delays), 0.16)
                self.assertEqual(refreshes, [True] * 5)
                self.assertEqual(frames[-1], (("Upcoming", "one\ntwo\nthree"), ("Chain", "enabled")))
            finally:
                ui._reset_live_animation_state()

    def test_live_animation_policy_limits_motion_and_prioritizes_urgent_panels(self) -> None:
        import rich.live

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        live_starts = []
        delays = []

        class FakeLive:
            def __init__(self, renderable, **_kwargs):
                live_starts.append(renderable)

            def __enter__(self):
                return self

            def update(self, _renderable, *, refresh=False):
                pass

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        with (
            patch.dict(os.environ, {"TERM": "xterm"}),
            patch("sys.stderr", TtyBuffer()),
            patch.object(ui, "_build_rich_panel", side_effect=lambda _title, rows, **_kw: tuple(rows)),
            patch.object(rich.live, "Live", FakeLive),
            patch.object(ui.time, "sleep", delays.append),
        ):
            ui._reset_live_animation_state()
            try:
                self.assertTrue(ui._render_panel_live("Warning", [("A", "1"), ("B", "2"), ("C", "3")], kind="warning", themes=None, duration_ms=200))
                self.assertEqual(len(live_starts), 1)
                self.assertGreater(sum(delays), 0)
                self.assertLessEqual(sum(delays), 0.101)
                sleep_count = len(delays)
                self.assertTrue(ui._render_panel_live("Later", [("A", "1"), ("B", "2")], kind="info", themes=None, duration_ms=200))
                self.assertEqual(len(live_starts), 1)
                self.assertEqual(len(delays), sleep_count)

                ui._reset_live_animation_state()
                starts_before_error = len(live_starts)
                self.assertTrue(ui._render_panel_live("Error", [("Error", "bad")], kind="error", themes=None, duration_ms=200))
                self.assertEqual(len(live_starts), starts_before_error)
                self.assertTrue(ui._render_panel_live("After error", [("A", "1"), ("B", "2")], kind="info", themes=None, duration_ms=200))
                self.assertEqual(len(live_starts), starts_before_error + 1)

                ui._reset_live_animation_state()
                starts_before_zero = len(live_starts)
                self.assertTrue(ui._render_panel_live("No motion", [("A", "1"), ("B", "2")], kind="info", themes=None, duration_ms=0))
                self.assertEqual(len(live_starts), starts_before_zero)
            finally:
                ui._reset_live_animation_state()

    def test_live_frame_failure_settles_without_duplicate_static_panel(self) -> None:
        import rich.live
        from rich.text import Text

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        frames = []
        static_calls = []

        def flaky_builder(_title, rows, *, active_row=None, **_kwargs):
            if active_row == 1:
                raise RuntimeError("frame failed")
            return Text(f"rows={len(list(rows))} active={active_row}")

        class FakeLive:
            def __init__(self, renderable, **_kwargs):
                frames.append(str(renderable))

            def __enter__(self):
                return self

            def update(self, renderable, *, refresh=False):
                self.assert_refresh = refresh
                frames.append(str(renderable))

            def __exit__(self, _exc_type, _exc, _tb):
                return False

        with (
            patch.dict(os.environ, {"TERM": "xterm"}),
            patch("sys.stderr", TtyBuffer()),
            patch.object(ui, "_build_rich_panel", side_effect=flaky_builder),
            patch.object(rich.live, "Live", FakeLive),
            patch.object(ui, "_render_panel_rich", side_effect=lambda *_args, **_kw: static_calls.append(True) or False),
            patch.object(ui.time, "sleep", lambda _delay: None),
        ):
            ui._reset_live_animation_state()
            try:
                ui.render_panel("Recover", [("One", "1"), ("Two", "2"), ("Three", "3")], panel_mode="live", live_duration_ms=160)
            finally:
                ui._reset_live_animation_state()

        self.assertEqual(static_calls, [])
        self.assertEqual(frames[-1], "rows=3 active=None")

    def test_oversized_live_panel_settles_without_starting_animation(self) -> None:
        import rich.console
        import rich.live
        from rich.text import Text

        class TtyBuffer(StringIO):
            def isatty(self):
                return True

        starts = []
        settled = []

        class ShortConsole:
            height = 8

            def __init__(self, **_kwargs):
                pass

            def render_lines(self, _renderable, *, pad=True):
                self_pad.append(pad)
                return [[] for _ in range(6)]

            def print(self, renderable):
                settled.append(str(renderable))

        class ForbiddenLive:
            def __init__(self, *_args, **_kwargs):
                starts.append(True)

        self_pad = []
        with (
            patch.dict(os.environ, {"TERM": "xterm"}),
            patch("sys.stderr", TtyBuffer()),
            patch.object(ui, "_build_rich_panel", side_effect=lambda title, rows, **_kw: Text(f"{title}:{len(list(rows))}")),
            patch.object(rich.console, "Console", ShortConsole),
            patch.object(rich.live, "Live", ForbiddenLive),
        ):
            ui._reset_live_animation_state()
            try:
                rendered = ui._render_panel_live("Tall", [("A", "1"), ("B", "2"), ("C", "3")], kind="info", themes=None, duration_ms=160)
            finally:
                ui._reset_live_animation_state()

        self.assertTrue(rendered)
        self.assertEqual(self_pad, [False])
        self.assertEqual(settled, ["Tall:3"])
        self.assertEqual(starts, [])


if __name__ == "__main__":
    unittest.main()
