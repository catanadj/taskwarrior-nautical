from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import importlib
from types import SimpleNamespace
from typing import get_type_hints
import unittest
from unittest.mock import patch

import nautical_core
import nautical_core.modify_feedback as modify_feedback
import nautical_core.modify_runtime as modify_runtime
from nautical_core.modify_models import (
    AnchorFeedbackServices,
    AnchorCompletionFeedbackModel,
    CompletionComputeResult,
    CpFeedbackServices,
    CpCompletionFeedbackModel,
    CompletionFinals,
    CompletionLifecycleResult,
    WaitScheduleDebug,
    NativeCarryDescription,
    PanelCallback,
)
from nautical_core.parsing.parser_models import AnchorDNF
from nautical_core.task_models import TaskPayload
from nautical_core.parsing.parser_models import ParseError


def _render_cp_completion_feedback(
    cp: str,
    *,
    mode: str = "panel",
    link_no: int = 2,
    next_no: int = 3,
    base_no: int = 2,
    sequence_step: int = 1,
    sequence_len: int = 1,
    task_values: dict[str, object] | None = None,
    child_values: dict[str, object] | None = None,
    child_due: datetime | None = None,
    now_utc: datetime | None = None,
    cap_no: int | None = None,
    finals: list[tuple[str, datetime]] | None = None,
    until_dt: datetime | None = None,
    until_cap_no: int | None = None,
) -> tuple[str | None, list[tuple[str, object]], str | None]:
    now = now_utc or datetime(2026, 9, 29, 9, tzinfo=timezone.utc)
    due = child_due or now
    add_validation = importlib.import_module("nautical_core.add_validation")
    panels = []
    core = SimpleNamespace(
        PANEL_MODE=mode,
        SHOW_ANALYTICS=False,
        strip_rich_markup=nautical_core.strip_rich_markup,
        _import_sibling=lambda name: add_validation if name == "add_validation" else None,
        to_local=nautical_core.to_local,
        humanize_delta=lambda *_args, **_kwargs: "in 15 days",
        fmt_dt_local=nautical_core.fmt_dt_local,
        coerce_int=lambda value, default: int(value) if value else default,
        parse_cp_duration=nautical_core.parse_cp_duration,
        parse_cp_sequence=nautical_core.parse_cp_sequence,
        parse_cp_sequence_tokens=nautical_core.parse_cp_sequence_tokens,
        cp_sequence_interval_for_token=nautical_core.cp_sequence_interval_for_token,
        cp_sequence_interval_for_link=nautical_core.cp_sequence_interval_for_link,
    )
    text_lines = []
    services = CpFeedbackServices(
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
        panel=lambda title, rows, **_kwargs: panels.append((title, list(rows))),
        chain_color_per_chain=False,
        chain_colour_for_task=lambda *_args: None,
        human_delta=lambda *_args, **_kwargs: "in 15 days",
    )
    new_task = {
        "cp": cp,
        "link": link_no,
        "uuid": "00000000-0000-4000-8000-000000000111",
        "chainID": "abcd1234",
    }
    new_task.update(task_values or {})
    child_task = {"uuid": "00000000-0000-4000-8000-000000000222"}
    child_task.update(child_values or {})
    feedback = CpCompletionFeedbackModel(
        new=new_task,
        child=child_task,
        child_due=due,
        child_short="beeswax",
        next_no=next_no,
        parent_short="00000000",
        cap_no=cap_no,
        finals=list(finals or []),
        now_utc=now,
        until_dt=until_dt,
        until_cap_no=until_cap_no,
        meta={"cp_sequence_step": sequence_step, "cp_sequence_len": sequence_len},
        deferred_spawn=False,
        spawn_intent_id=None,
        lifecycle_result=CompletionLifecycleResult("applied"),
        chain_by_short=None,
        analytics_advice=None,
        integrity_warnings=None,
        base_no=base_no,
    )

    modify_feedback.render_cp_completion_feedback(feedback=feedback, services=services)
    title, rows = panels[0] if panels else (None, [])
    return title, rows, text_lines[0] if text_lines else None


def _omit_summary_core(
    *, resolve, describe, lint
) -> SimpleNamespace:
    anchor_omit = SimpleNamespace(normalize_omit_expr=lambda expression: expression)
    return SimpleNamespace(
        _import_sibling=lambda _name: anchor_omit,
        _parser_api=SimpleNamespace(resolve_omit_presets=resolve),
        describe_anchor_expr=describe,
        lint_anchor_expr=lint,
    )


class ModifyFeedbackContractTests(unittest.TestCase):
    def test_feedback_renderers_use_the_shared_panel_callback_contract(self) -> None:
        renderers = (
            modify_feedback.render_cp_schedule_adjusted_panel,
            modify_feedback.render_explicit_timing_order_warning,
            modify_feedback.render_recurrence_updated_panel,
        )
        for renderer in renderers:
            with self.subTest(renderer=renderer.__name__):
                self.assertIs(get_type_hints(renderer)["panel"], PanelCallback)

    def test_recurrence_enabled_feedback_surfaces_unexpected_cp_parser_errors(self) -> None:
        def broken_parser(_value: str) -> None:
            raise RuntimeError("CP feedback parser invariant failed")

        with self.assertRaisesRegex(RuntimeError, "CP feedback parser invariant failed"):
            modify_feedback.recurrence_enabled_rows(
                {"cp": "1d"},
                "cp",
                describe_anchor=lambda value: value,
                parse_cp_sequence_tokens=broken_parser,
                first_recurrence_target=lambda _task, _source: None,
                format_local=lambda value: str(value),
            )

    def test_recurrence_enabled_feedback_surfaces_anchor_description_errors(self) -> None:
        def broken_description(_value: str) -> str:
            raise RuntimeError("anchor feedback description failed")

        with self.assertRaisesRegex(RuntimeError, "anchor feedback description failed"):
            modify_feedback.recurrence_enabled_rows(
                {"anchor": "w:mon", "anchor_mode": "skip"},
                "anchor",
                describe_anchor=broken_description,
                parse_cp_sequence_tokens=lambda _value: None,
                first_recurrence_target=lambda _task, _source: None,
                format_local=lambda value: str(value),
            )

    def test_native_until_feedback_uses_its_named_carry_contract(self) -> None:
        self.assertIs(
            get_type_hints(modify_feedback.render_recurrence_updated_panel)[
                "describe_native_until_carry"
            ],
            NativeCarryDescription,
        )

    def test_recurrence_feedback_local_conversion_uses_datetime_contract(self) -> None:
        expected = Callable[[datetime], datetime]
        self.assertEqual(
            get_type_hints(modify_feedback.CompletionFeedbackCore.to_local),
            {"value": datetime, "return": datetime},
        )
        self.assertEqual(
            get_type_hints(modify_feedback.render_recurrence_updated_panel)["to_local"],
            expected,
        )

    def test_feedback_timestamp_helper_returns_typed_task_timestamp(self) -> None:
        from nautical_core.task_models import TaskTimestamp

        self.assertEqual(
            get_type_hints(modify_feedback._timestamp),
            {
                "task": TaskPayload,
                "field": str,
                "return": TaskTimestamp | None,
            },
        )

    def test_final_occurrence_feedback_uses_datetime_contracts(self) -> None:
        hints = get_type_hints(modify_feedback._append_final_rows)
        self.assertEqual(
            hints,
            {
                "fb": list[tuple[str, object]],
                "finals": modify_feedback.CompletionFinals,
                "now_utc": datetime,
                "fmt_dt_local": Callable[[datetime], str],
                "human_delta": Callable[[datetime, datetime, bool], str],
                "return": type(None),
            },
        )

    def test_feedback_local_formatters_accept_datetimes(self) -> None:
        expected = Callable[[datetime], str]
        renderers = (
            modify_feedback.append_next_wait_sched_rows,
            modify_feedback.render_cp_schedule_adjusted_panel,
            modify_feedback._recurrence_display_value,
            modify_feedback._recurrence_change_row,
            modify_feedback.render_recurrence_updated_panel,
            modify_feedback.recurrence_enabled_rows,
        )
        for renderer in renderers:
            with self.subTest(renderer=renderer.__name__):
                self.assertEqual(
                    get_type_hints(renderer)["format_local"], expected
                )

    def test_feedback_recurrence_parser_callbacks_return_optional_datetime(self) -> None:
        from nautical_core.modify_models import DatetimeParserCallback

        helpers = (
            modify_feedback._recurrence_display_value,
            modify_feedback._recurrence_change_row,
            modify_feedback.render_recurrence_updated_panel,
        )
        for helper in helpers:
            with self.subTest(helper=helper.__name__):
                self.assertIs(
                    get_type_hints(helper)["parse_datetime"],
                    DatetimeParserCallback,
                )

    def test_recurrence_feedback_uses_shared_integer_coercion_contract(self) -> None:
        from nautical_core.modify_models import CoerceIntCallback

        self.assertIs(
            get_type_hints(modify_feedback.render_recurrence_updated_panel)[
                "coerce_int"
            ],
            CoerceIntCallback,
        )

    def test_completion_preview_uses_datetime_callback_contracts(self) -> None:
        from nautical_core.modify_models import (
            CompletionPreviewFormatter,
            MarkupStripper,
            PreviewLineFormatter,
        )

        datetime_or_none = datetime | None
        expected_arguments = {
            "child_due": datetime_or_none,
            "now_utc": datetime,
            "until_dt": datetime_or_none,
            "child_until_dt": datetime_or_none,
        }
        for formatter in (CompletionPreviewFormatter, PreviewLineFormatter):
            hints = get_type_hints(formatter.__call__)
            with self.subTest(formatter=formatter.__name__):
                for field, annotation in expected_arguments.items():
                    self.assertEqual(hints[field], annotation)
        expected_callbacks = {
            "format_local": Callable[[datetime], str],
            "on_time_delta": Callable[[datetime_or_none, datetime_or_none], str],
            "human_delta": Callable[[datetime, datetime_or_none, bool], str],
        }
        preview_hints = get_type_hints(PreviewLineFormatter.__call__)
        for field, annotation in expected_callbacks.items():
            with self.subTest(formatter=PreviewLineFormatter.__name__, field=field):
                self.assertEqual(preview_hints[field], annotation)
        formatter_hints = get_type_hints(modify_feedback.format_line_preview)
        self.assertIs(formatter_hints["core"], MarkupStripper)
        helper_fields = dict(expected_arguments, **expected_callbacks)
        helper_fields["child_due_utc"] = helper_fields.pop("child_due")
        for field, annotation in helper_fields.items():
            with self.subTest(formatter="format_line_preview", field=field):
                self.assertEqual(formatter_hints[field], annotation)

    def test_completion_feedback_renderers_use_owner_models(self) -> None:
        anchor_annotations = get_type_hints(modify_feedback.render_anchor_completion_feedback)
        cp_annotations = get_type_hints(modify_feedback.render_cp_completion_feedback)
        self.assertIs(anchor_annotations["feedback"], AnchorCompletionFeedbackModel)
        self.assertIs(anchor_annotations["services"], AnchorFeedbackServices)
        self.assertIs(cp_annotations["feedback"], CpCompletionFeedbackModel)
        self.assertIs(cp_annotations["services"], CpFeedbackServices)
        self.assertIs(get_type_hints(AnchorCompletionFeedbackModel)["new"], TaskPayload)
        self.assertIs(get_type_hints(CpCompletionFeedbackModel)["child"], TaskPayload)
        self.assertEqual(
            get_type_hints(AnchorFeedbackServices)["last_wait_sched_debug"],
            WaitScheduleDebug | None,
        )
        self.assertEqual(
            get_type_hints(CpCompletionFeedbackModel)["finals"],
            CompletionFinals,
        )
        self.assertEqual(get_type_hints(CompletionComputeResult)["dnf"], AnchorDNF | None)
        self.assertIs(
            get_type_hints(
                modify_feedback.orchestrate_anchor_completion_feedback,
                localns={"ModifyRuntimeServices": modify_runtime.ModifyRuntimeServices},
            )["request"],
            AnchorCompletionFeedbackModel,
        )
        self.assertIs(
            get_type_hints(
                modify_feedback.orchestrate_cp_completion_feedback,
                localns={"ModifyRuntimeServices": modify_runtime.ModifyRuntimeServices},
            )["request"],
            CpCompletionFeedbackModel,
        )
        self.assertIs(
            get_type_hints(
                modify_feedback.orchestrate_cp_completion_feedback,
                localns={"ModifyRuntimeServices": modify_runtime.ModifyRuntimeServices},
            )["core"],
            modify_feedback.CompletionFeedbackCore,
        )

    def test_cp_feedback_orchestration_uses_owner_models_without_module_bag(self) -> None:
        now = datetime(2026, 10, 4, tzinfo=timezone.utc)
        request = CpCompletionFeedbackModel(
            new={"cp": "P1D"},
            child={"uuid": "00000000-0000-4000-8000-000000000222"},
            child_due=now,
            child_short="beeswax",
            next_no=2,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=now,
            until_dt=None,
            until_cap_no=None,
            meta={},
            deferred_spawn=False,
            spawn_intent_id=None,
            lifecycle_result=CompletionLifecycleResult("applied"),
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=1,
        )
        services = object()
        runtime = SimpleNamespace(build_cp_feedback_services=lambda _runtime: services)
        diagnostics = SimpleNamespace(panel_warnings=lambda *_args, **_kwargs: [])

        with patch.object(modify_feedback, "render_cp_completion_feedback") as render:
            modify_feedback.orchestrate_cp_completion_feedback(
                request=request,
                core=object(),
                panel_warnings=diagnostics.panel_warnings,
                build_feedback_services=runtime.build_cp_feedback_services,
                build_runtime_services=lambda: object(),
            )

        rendered = render.call_args.kwargs["feedback"]
        self.assertIsInstance(rendered, CpCompletionFeedbackModel)
        self.assertEqual(rendered.new["cp"], "P1D")
        self.assertIs(render.call_args.kwargs["services"], services)

    def test_recurrence_update_panel_does_not_hide_expression_helper_failures(self) -> None:
        def broken_helper(_expression):
            raise RuntimeError("expression feedback implementation failed")

        cases = (
            (
                [("anchor", "", "w:mon")],
                {"anchor": "w:mon"},
                broken_helper,
                lambda value: value,
            ),
            (
                [("omit", "", "w:mon")],
                {"omit": "w:mon"},
                lambda _value: "natural",
                broken_helper,
            ),
        )
        for changes, new, describe_anchor, resolve_omit_presets in cases:
            with self.subTest(field=changes[0][0]):
                with self.assertRaisesRegex(RuntimeError, "expression feedback implementation failed"):
                    modify_feedback.render_recurrence_updated_panel(
                        changes,
                        new,
                        parse_datetime=lambda _value: None,
                        format_local=str,
                        describe_native_until_carry=lambda *_args, **_kwargs: None,
                        to_local=lambda value: value,
                        coerce_int=lambda _value, default: default,
                        describe_anchor=describe_anchor,
                        resolve_omit_presets=resolve_omit_presets,
                        first_recurrence_target=lambda *_args: None,
                        panel_mode="panel",
                        strip_markup=lambda value: value,
                panel=lambda *_args, **_kwargs: None,
            )

    def test_recurrence_updated_feedback_surfaces_carry_adapter_errors(self) -> None:
        def broken_carry(*_args, **_kwargs):
            raise RuntimeError("carry feedback adapter failed")

        with self.assertRaisesRegex(RuntimeError, "carry feedback adapter failed"):
            modify_feedback.render_recurrence_updated_panel(
                [("until", "old", "new")],
                {
                    "due": "20261004T070000Z",
                    "until": "20261004T080000Z",
                },
                parse_datetime=lambda _value: datetime(
                    2026, 10, 4, 7, tzinfo=timezone.utc
                ),
                format_local=lambda value: value.isoformat(),
                describe_native_until_carry=broken_carry,
                to_local=lambda value: value,
                coerce_int=lambda _value, default: default,
                describe_anchor=lambda value: value,
                resolve_omit_presets=lambda value: value,
                first_recurrence_target=lambda _task, _source: None,
                panel_mode="panel",
                strip_markup=lambda value: value,
                panel=lambda *_args, **_kwargs: None,
            )

    def test_omit_summary_falls_back_to_raw_expression_on_parse_error(self) -> None:
        def invalid_preset(_expression):
            raise ParseError("unknown omit preset")

        core = _omit_summary_core(
            resolve=invalid_preset,
            describe=lambda expression: f"description for {expression}",
            lint=lambda _expression: (None, ["lint warning"]),
        )

        self.assertEqual(
            modify_feedback._anchor_omit_summary(core, {"omit": "@unknown"}),
            ("@unknown", "description for @unknown", ["lint warning"], None),
        )

    def test_omit_summary_does_not_hide_unexpected_helper_failures(self) -> None:
        def resolver_failure(_expression):
            raise RuntimeError("preset resolver implementation failed")

        core = _omit_summary_core(
            resolve=resolver_failure,
            describe=lambda _expression: "description",
            lint=lambda _expression: (None, []),
        )
        with self.assertRaisesRegex(RuntimeError, "preset resolver implementation failed"):
            modify_feedback._anchor_omit_summary(core, {"omit": "@preset"})

        def describe_failure(_expression):
            raise RuntimeError("description implementation failed")

        core = _omit_summary_core(
            resolve=lambda expression: expression,
            describe=describe_failure,
            lint=lambda _expression: (None, []),
        )
        with self.assertRaisesRegex(RuntimeError, "description implementation failed"):
            modify_feedback._anchor_omit_summary(core, {"omit": "w:mon"})

        def lint_failure(_expression):
            raise RuntimeError("lint implementation failed")

        core = _omit_summary_core(
            resolve=lambda expression: expression,
            describe=lambda _expression: "description",
            lint=lint_failure,
        )
        with self.assertRaisesRegex(RuntimeError, "lint implementation failed"):
            modify_feedback._anchor_omit_summary(core, {"omit": "w:mon"})

    def test_pattern_rows_do_not_hide_unexpected_preset_lookup_failures(self) -> None:
        def broken_lookup(_expression):
            raise RuntimeError("preset lookup implementation failed")

        cases = (
            (modify_feedback._anchor_pattern_row, "anchor_preset_display"),
            (modify_feedback._omit_pattern_row, "omit_preset_display"),
        )
        for row_builder, lookup_name in cases:
            with self.subTest(lookup=lookup_name):
                core = SimpleNamespace(**{lookup_name: broken_lookup})
                with self.assertRaisesRegex(RuntimeError, "preset lookup implementation failed"):
                    row_builder(core, "invalid-pattern")

    def test_pattern_rows_fall_back_to_raw_text_for_invalid_preset_values(self) -> None:
        def invalid_lookup(_expression):
            return None

        cases = (
            (modify_feedback._anchor_pattern_row, "anchor_preset_display", "Pattern"),
            (modify_feedback._omit_pattern_row, "omit_preset_display", "Omit"),
        )
        for row_builder, lookup_name, label in cases:
            with self.subTest(lookup=lookup_name):
                core = SimpleNamespace(**{lookup_name: invalid_lookup})
                self.assertEqual(row_builder(core, "invalid-pattern"), (label, "invalid-pattern"))

    def test_anchor_file_feedback_renders_without_an_anchor_dnf(self) -> None:
        now = datetime(2026, 9, 29, 9, tzinfo=timezone.utc)
        panels = []
        core = SimpleNamespace(
            PANEL_MODE="panel",
            SHOW_ANALYTICS=False,
            anchor_preset_display=lambda _value: None,
            expr_has_m_or_y=lambda _dnf: False,
            humanize_delta=lambda *_args, **_kwargs: "in 1 day",
            fmt_dt_local=lambda value: value.isoformat(),
            coerce_int=lambda value, default: int(value) if value else default,
        )
        services = AnchorFeedbackServices(
            core=core,
            debug_wait_sched=False,
            last_wait_sched_debug=None,
            diag_enabled=False,
            format_root_and_age=lambda *_args: "abcd1234",
            append_next_wait_sched_rows=lambda *_args, **_kwargs: None,
            timeline_lines=lambda *_args, **_kwargs: [],
            show_timeline_gaps=False,
            root_uuid_from=lambda _task: "00000000-0000-4000-8000-000000000333",
            short=lambda value: value[:8],
            format_next_anchor_rows=lambda rows: rows,
            format_line_preview=lambda *_args, **_kwargs: "unused line preview",
            panel_line=lambda *_args, **_kwargs: self.fail("panel mode must use the panel renderer"),
            text_line=lambda *_args, **_kwargs: self.fail("panel mode must not use text output"),
            panel=lambda title, rows, **_kwargs: panels.append((title, list(rows))),
            chain_color_per_chain=False,
            chain_colour_for_task=lambda *_args: None,
            strip_quotes=lambda value: value,
            human_delta=lambda *_args, **_kwargs: "in 1 day",
        )
        feedback = AnchorCompletionFeedbackModel(
            new={
                "anchor_file": "calendar.csv@t=12:00",
                "anchor_mode": "skip",
                "uuid": "00000000-0000-4000-8000-000000000333",
                "chainID": "abcd1234",
            },
            child={"uuid": "00000000-0000-4000-8000-000000000444"},
            child_due=now,
            child_short="beeswax",
            next_no=2,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=now,
            until_dt=None,
            until_cap_no=None,
            dnf=None,
            meta={"mode": "skip"},
            stripped_attrs=[],
            deferred_spawn=False,
            spawn_intent_id=None,
            lifecycle_result=CompletionLifecycleResult("applied"),
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=1,
        )

        modify_feedback.render_anchor_completion_feedback(feedback=feedback, services=services)

        self.assertEqual(len(panels), 1)
        title, rows = panels[0]
        self.assertIn("Next anchor", title)
        self.assertTrue(
            any(label == "Anchor file" and value.startswith("calendar.csv@t=12:00") for label, value in rows)
        )
        self.assertIn(("Natural", "Dates from calendar.csv"), rows)

    def test_cp_jitter_feedback_shows_the_selected_interval(self) -> None:
        _title, rows, _text = _render_cp_completion_feedback("15d~0d")
        self.assertIn(("Step", "1/1 (15d)"), rows)

    def test_cp_random_step_does_not_hide_parser_or_interval_failures(self) -> None:
        for function_name in ("parse_cp_sequence_tokens", "cp_sequence_interval_for_token"):
            with self.subTest(function=function_name):
                with patch.object(
                    nautical_core,
                    function_name,
                    side_effect=RuntimeError("CP interval implementation failed"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "CP interval implementation failed"):
                        _render_cp_completion_feedback("rand(11d..14d)", sequence_len=1)

    def test_cp_random_feedback_shows_the_chain_scoped_selected_interval(self) -> None:
        cp = "rand(11d..14d)"
        _title, rows, _text = _render_cp_completion_feedback(cp)
        selected = nautical_core.cp_sequence_interval_for_link(cp, 2, "abcd1234")
        selected_days = int(selected.total_seconds() // 86400)

        self.assertIn(("Step", f"1/1 ({selected_days}d)"), rows)

    def test_cp_feedback_renders_the_current_sequence_step(self) -> None:
        title, rows, _text = _render_cp_completion_feedback(
            "3d,20d,7d", link_no=1, next_no=2, base_no=1, sequence_step=3, sequence_len=3
        )

        self.assertEqual(title, "⛓ Next link  #2  00000000 → beeswax")
        self.assertIn(("Step", "3/3 (7d)"), rows)
        self.assertTrue(any(label == "Result" and "Applied now" in str(value) for label, value in rows))

    def test_cp_feedback_separates_expiration_and_effective_chain_boundary(self) -> None:
        child_due = datetime(2026, 8, 10, 10, tzinfo=timezone.utc)
        child_expires = child_due + timedelta(hours=8)
        chain_end = child_due + timedelta(days=35)
        last_by_max = child_due + timedelta(days=60)
        last_by_end = child_due + timedelta(days=28)
        _title, rows, _text = _render_cp_completion_feedback(
            "7d",
            task_values={"chainMax": 10},
            child_values={"until": nautical_core.fmt_isoz(child_expires)},
            child_due=child_due,
            now_utc=datetime(2026, 7, 20, 9, tzinfo=timezone.utc),
            cap_no=6,
            finals=[("max", last_by_max), ("until", last_by_end)],
            until_dt=chain_end,
            until_cap_no=6,
        )
        add_validation = importlib.import_module("nautical_core.add_validation")
        expected_policy = add_validation.describe_native_until_carry(
            child_expires,
            child_due,
            to_local=nautical_core.to_local,
        )
        last_rows = [(label, value) for label, value in rows if label == "Last occurrence"]

        self.assertIn(("Expiration", expected_policy), rows)
        self.assertTrue(any(label == "Next expires" for label, _value in rows))
        self.assertIn(("Chain cap", "#10"), rows)
        self.assertTrue(any(label == "Chain end point" and "2026-09-14" in value for label, value in rows))
        self.assertEqual(len(last_rows), 1)
        self.assertIn("2026-09-07", last_rows[0][1])
        self.assertFalse(any(str(label).startswith("Final (") for label, _value in rows))

    def test_cp_text_feedback_uses_stacked_ascii_output(self) -> None:
        _title, rows, text = _render_cp_completion_feedback("P1D", mode="text")

        self.assertEqual(rows, [])
        self.assertIsNotNone(text)
        output = text or ""
        self.assertGreaterEqual(output.count("\n"), 2)
        self.assertIn("[bold yellow]Next[/]", output)
        self.assertIn("[bold yellow]Period:[/] [white]P1D[/]", output)
        self.assertIn("[bold cyan]Result:[/] [white]Applied now[/]", output)

    def test_expiration_summary_failure_preserves_primary_expiration_row(self) -> None:
        rows: list[tuple[str, object]] = []
        core = SimpleNamespace(
            _import_sibling=lambda _name: (_ for _ in ()).throw(
                RuntimeError("optional carry presentation unavailable")
            ),
            humanize_delta=lambda *_args, **_kwargs: "in 2 days",
            fmt_dt_local=lambda value: value.isoformat(),
        )
        child_due = datetime(2026, 10, 3, 9, tzinfo=timezone.utc)
        child = {
            "until": "20261005T090000Z",
        }

        modify_feedback._append_next_expiration_row(
            rows,
            child,
            child_due,
            core=core,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][0], "Next expires")
        self.assertIn("2026-10-05", str(rows[0][1]))

    def test_expiration_carry_caption_failure_keeps_next_expiration_feedback(self) -> None:
        with patch(
            "nautical_core.add_validation.describe_native_until_carry",
            side_effect=RuntimeError("optional carry caption unavailable"),
        ):
            _title, rows, text = _render_cp_completion_feedback(
                "P1D",
                child_values={"until": "20261005T090000Z"},
            )
            _text_title, _text_rows, text = _render_cp_completion_feedback(
                "P1D",
                mode="text",
                child_values={"until": "20261005T090000Z"},
            )

        self.assertTrue(any(label == "Next expires" for label, _value in rows))
        self.assertFalse(any(label == "Expiration" for label, _value in rows))
        self.assertIsNotNone(text)
        self.assertIn("Next expires", text or "")

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
            new={
                "anchor": "@payday",
                "omit": "@wed",
                "anchor_mode": "skip",
                "uuid": "00000000-0000-4000-8000-000000000111",
                "chainID": "abcd1234",
            },
            child={"uuid": "00000000-0000-4000-8000-000000000222"},
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
