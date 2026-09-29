from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

import nautical_core
import nautical_core.modify_feedback as modify_feedback
from nautical_core.modify_models import (
    AnchorCompletionFeedbackModel,
    CpCompletionFeedbackModel,
    CompletionLifecycleResult,
    TaskView,
)


def _render_cp_completion_feedback(
    cp: str, *, mode: str = "panel"
) -> tuple[list[tuple[str, object]], str | None]:
    now = datetime(2026, 9, 29, 9, tzinfo=timezone.utc)
    panels = []
    core = SimpleNamespace(
        PANEL_MODE=mode,
        SHOW_ANALYTICS=False,
        strip_rich_markup=nautical_core.strip_rich_markup,
        humanize_delta=lambda *_args, **_kwargs: "in 15 days",
        fmt_dt_local=lambda value: value.isoformat(),
        coerce_int=lambda value, default: int(value) if value else default,
        parse_cp_duration=nautical_core.parse_cp_duration,
        parse_cp_sequence=nautical_core.parse_cp_sequence,
        parse_cp_sequence_tokens=nautical_core.parse_cp_sequence_tokens,
        cp_sequence_interval_for_token=nautical_core.cp_sequence_interval_for_token,
        cp_sequence_interval_for_link=nautical_core.cp_sequence_interval_for_link,
    )
    text_lines = []
    services = SimpleNamespace(
        core=core,
        diag_enabled=False,
        format_root_and_age=lambda *_args: "abcd1234",
        append_next_wait_sched_rows=lambda *_args, **_kwargs: None,
        timeline_lines=lambda *_args, **_kwargs: [],
        show_timeline_gaps=False,
        format_next_cp_rows=lambda rows: rows,
        format_line_preview=lambda *_args, **_kwargs: "00000000 ✓ next ⛓ · #2 · (due in 15 days)",
        panel_line=lambda *_args, **_kwargs: None,
        text_line=lambda line, **_kwargs: text_lines.append(line),
        panel=lambda _title, rows, **_kwargs: panels.append(list(rows)),
        chain_color_per_chain=False,
        chain_colour_for_task=lambda *_args: None,
        human_delta=lambda *_args, **_kwargs: "in 15 days",
    )
    feedback = CpCompletionFeedbackModel(
        new=TaskView.from_mapping(
            {
                "cp": cp,
                "link": 2,
                "uuid": "00000000-0000-4000-8000-000000000111",
                "chainID": "abcd1234",
            }
        ),
        child=TaskView.from_mapping({"uuid": "00000000-0000-4000-8000-000000000222"}),
        child_due=now,
        child_short="beeswax",
        next_no=3,
        parent_short="00000000",
        cap_no=None,
        finals=[],
        now_utc=now,
        until_dt=None,
        until_cap_no=None,
        meta={"cp_sequence_step": 1, "cp_sequence_len": 1},
        deferred_spawn=False,
        spawn_intent_id=None,
        lifecycle_result=CompletionLifecycleResult("applied"),
        chain_by_short=None,
        analytics_advice=None,
        integrity_warnings=None,
        base_no=2,
    )

    modify_feedback.render_cp_completion_feedback(feedback=feedback, services=services)
    return (panels[0] if panels else [], text_lines[0] if text_lines else None)


class ModifyFeedbackContractTests(unittest.TestCase):
    def test_cp_jitter_feedback_shows_the_selected_interval(self) -> None:
        rows, _text = _render_cp_completion_feedback("15d~0d")
        self.assertIn(("Step", "1/1 (15d)"), rows)

    def test_cp_random_feedback_shows_the_chain_scoped_selected_interval(self) -> None:
        cp = "rand(11d..14d)"
        rows, _text = _render_cp_completion_feedback(cp)
        selected = nautical_core.cp_sequence_interval_for_link(cp, 2, "abcd1234")
        selected_days = int(selected.total_seconds() // 86400)

        self.assertIn(("Step", f"1/1 ({selected_days}d)"), rows)

    def test_cp_text_feedback_uses_stacked_ascii_output(self) -> None:
        rows, text = _render_cp_completion_feedback("P1D", mode="text")

        self.assertEqual(rows, [])
        self.assertIsNotNone(text)
        output = text or ""
        self.assertGreaterEqual(output.count("\n"), 2)
        self.assertIn("[bold yellow]Next[/]", output)
        self.assertIn("[bold yellow]Period:[/] [white]P1D[/]", output)
        self.assertIn("[bold cyan]Result:[/] [white]Applied now[/]", output)

    def test_anchor_feedback_expands_presets_and_keeps_lifecycle_result_without_analytics(self) -> None:
        now = datetime(2026, 9, 29, 9, tzinfo=timezone.utc)
        panels = []

        def capture_panel(title, rows, **kwargs):
            panels.append((title, list(rows), kwargs))

        omit_api = SimpleNamespace(
            resolve_omit_presets=lambda value: "w:wed" if value == "@wed" else value
        )
        omit_module = SimpleNamespace(normalize_omit_expr=lambda value: value)
        core = SimpleNamespace(
            PANEL_MODE="panel",
            SHOW_ANALYTICS=False,
            anchor_preset_display=lambda value: (
                ("Preset", "@payday → m:15,-1bd") if value == "@payday" else None
            ),
            omit_preset_display=lambda value: (
                ("Omit", "@wed → w:wed") if value == "@wed" else None
            ),
            _parser_api=omit_api,
            _import_sibling=lambda name: omit_module if name == "anchor_omit" else None,
            describe_anchor_expr=lambda _value: "Wednesdays",
            describe_anchor_dnf=lambda _dnf, _task: "Every Monday",
            lint_anchor_expr=lambda _value: (None, []),
            humanize_delta=lambda *_args, **_kwargs: "in 7 days",
            expr_has_m_or_y=lambda _dnf: False,
            coerce_int=lambda value, default: int(value) if value else default,
            fmt_dt_local=lambda _value: "Mon 2026-10-05 09:00",
        )
        services = SimpleNamespace(
            core=core,
            debug_wait_sched=False,
            last_wait_sched_debug=None,
            diag_enabled=False,
            format_root_and_age=lambda *_args: "abcd1234",
            append_next_wait_sched_rows=lambda *_args, **_kwargs: None,
            timeline_lines=lambda *_args, **_kwargs: [],
            show_timeline_gaps=False,
            root_uuid_from=lambda _task: "00000000-0000-4000-8000-000000000111",
            short=lambda value: value[:8],
            format_next_anchor_rows=lambda rows: rows,
            format_line_preview=lambda *_args, **_kwargs: "unused line preview",
            panel_line=lambda *_args, **_kwargs: self.fail("panel mode must use the panel renderer"),
            text_line=lambda *_args, **_kwargs: self.fail("panel mode must not use text output"),
            panel=capture_panel,
            chain_color_per_chain=False,
            chain_colour_for_task=lambda *_args: None,
            strip_quotes=lambda value: value,
            human_delta=lambda *_args, **_kwargs: "in 7 days",
        )
        feedback = AnchorCompletionFeedbackModel(
            new=TaskView.from_mapping(
                {
                    "anchor": "@payday",
                    "omit": "@wed",
                    "anchor_mode": "skip",
                    "uuid": "00000000-0000-4000-8000-000000000111",
                    "chainID": "abcd1234",
                }
            ),
            child=TaskView.from_mapping(
                {"uuid": "00000000-0000-4000-8000-000000000222"}
            ),
            child_due=now,
            child_short="beeswax",
            next_no=2,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=now,
            until_dt=None,
            until_cap_no=None,
            dnf=[[{"typ": "w", "spec": "mon", "mods": {}}]],
            meta={"mode": "skip", "target_field": "due"},
            stripped_attrs=[],
            deferred_spawn=False,
            spawn_intent_id=None,
            lifecycle_result=CompletionLifecycleResult("applied"),
            chain_by_short=None,
            analytics_advice="healthy but intentionally hidden",
            integrity_warnings=None,
            base_no=1,
        )

        modify_feedback.render_anchor_completion_feedback(
            feedback=feedback,
            services=services,
        )

        self.assertEqual(len(panels), 1)
        title, rows, _kwargs = panels[0]
        self.assertIn("Next anchor", title)
        self.assertIn(("Omit", "@wed → w:wed"), rows)
        self.assertTrue(
            any(label == "Preset" and "@payday → m:15,-1bd" in value for label, value in rows)
        )
        self.assertTrue(any(label == "Natural" and "skip Wednesdays" in value for label, value in rows))
        self.assertTrue(any(label == "Result" and "Applied now" in value for label, value in rows))
        self.assertFalse(any(label == "Analytics" for label, _value in rows))


if __name__ == "__main__":
    unittest.main()
