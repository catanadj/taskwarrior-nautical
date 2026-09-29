from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest

import nautical_core.modify_feedback as modify_feedback
from nautical_core.modify_models import (
    AnchorCompletionFeedbackModel,
    CompletionLifecycleResult,
    TaskView,
)


class ModifyFeedbackContractTests(unittest.TestCase):
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
