"""Direct contracts for timeline warning behavior at failure boundaries."""

from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Callable, get_type_hints
import unittest

import nautical_core as core
from nautical_core.modify_timeline import (
    TimelineFormattingServices,
    TimelineProjectionServices,
    _timeline_base_line,
    _timeline_future_anchor_items,
    _timeline_omit_label,
)
from nautical_core.anchor_omit import OmitState
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.recurrence_evaluator import RecurrenceEvaluator
from nautical_core.scheduler_service import SchedulerService
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.task_models import TaskObservation, TaskPayload
from nautical_core.timeutil import parse_dt_any


class ModifyTimelineContractTests(unittest.TestCase):
    def test_timeline_gap_formatter_uses_datetime_values(self) -> None:
        from nautical_core.modify_timeline import format_gap

        self.assertEqual(
            get_type_hints(format_gap),
            {
                "prev_dt": datetime | None,
                "next_dt": datetime | None,
                "kind": str,
                "round_hours": bool,
                "return": str,
            },
        )

    def test_timeline_item_carries_nullable_datetime(self) -> None:
        from nautical_core.modify_timeline import TimelineItem

        self.assertEqual(
            TimelineItem,
            tuple[object, datetime | None, TaskPayload, str],
        )

    def test_timeline_recurrence_inputs_use_parser_dnf_model(self) -> None:
        from nautical_core.parsing.parser_models import AnchorDNF
        from nautical_core.modify_timeline import (
            _timeline_future_anchor_items,
            _timeline_omitted_before_next_anchor_items,
            timeline_lines,
            timeline_lines_for_task,
        )

        for function in (
            _timeline_future_anchor_items,
            _timeline_omitted_before_next_anchor_items,
            timeline_lines,
            timeline_lines_for_task,
        ):
            with self.subTest(function=function.__name__):
                self.assertEqual(
                    get_type_hints(function)["dnf"],
                    AnchorDNF | None,
                )

    def test_timeline_integer_coercion_parameters_have_explicit_call_shapes(self) -> None:
        from nautical_core.modify_timeline import (
            _timeline_initial_items,
            anchor_file_timeline_lines,
        )

        from nautical_core.modify_models import CoerceIntCallback

        expected = CoerceIntCallback
        for function in (_timeline_initial_items, anchor_file_timeline_lines):
            with self.subTest(function=function.__name__):
                self.assertEqual(get_type_hints(function)["coerce_int"], expected)

    def test_timeline_projection_services_expose_exact_callback_shapes(self) -> None:
        annotations = get_type_hints(TimelineProjectionServices)
        expected = {
            "collect_prev_two": Callable[[TaskPayload], list[TaskObservation]],
            "dtparse": Callable[[object], datetime | None],
            "to_local_cached": Callable[[datetime], datetime],
            "safe_parse_datetime": Callable[
                [object], tuple[datetime | None, str | None]
            ],
            "omit_dnf_from_parent": Callable[
                [TaskPayload], tuple[str, OmitState | None]
            ],
            "omit_description_for_date": Callable[
                [OmitState | None, date], str | None
            ]
            | None,
            "recurrence_evaluator_for_task": Callable[
                [TaskPayload], RecurrenceEvaluator
            ],
            "scheduler_service_for_task": Callable[
                [TaskPayload], SchedulerService
            ],
        }
        for name, annotation in expected.items():
            with self.subTest(callback=name):
                self.assertEqual(annotations[name], annotation)

    def test_missing_projection_dependencies_render_unavailable_warnings(self) -> None:
        from nautical_core.modify_timeline import timeline_lines

        formatting = SimpleNamespace(
            coerce_int=lambda value, default=None: int(value) if value is not None else default,
            future_style_for_chain=lambda _task, _kind: "yellow",
            fmt_dt_local=lambda _value: "local time",
            fmtlocal=lambda _value: "local time",
            fmt_on_time_delta=lambda _start, _end: "",
            short=lambda value: str(value or "–"),
            format_gap=lambda *_args: "",
        )
        projection = SimpleNamespace(
            max_iterations=4,
            collect_prev_two=lambda _task: [],
            dtparse=lambda _value: None,
        )

        cases = (
            ("cp", "recurrence evaluator"),
            ("anchor", "scheduler service"),
        )
        for kind, missing_dependency in cases:
            with self.subTest(kind=kind):
                lines = timeline_lines(
                    kind,
                    {"uuid": "parent-uuid", "chainID": "timeline-test", "link": 1, "cp": "1d"},
                    datetime(2026, 10, 6, 9, tzinfo=timezone.utc),
                    "child-uuid",
                    None,
                    next_count=1,
                    projection=projection,
                    formatting=formatting,
                    scheduler_service=None,
                    omit_dnf=None,
                    evaluator=None,
                )

                self.assertEqual(len(lines), 3)
                self.assertIn("Projection unavailable", lines[-1])
                self.assertIn(missing_dependency, lines[-1])

    def test_timeline_formatting_services_expose_exact_callback_shapes(self) -> None:
        from nautical_core.modify_models import CoerceIntCallback, ShortUuidCallback

        annotations = get_type_hints(TimelineFormattingServices)
        expected = {
            "future_style_for_chain": Callable[[TaskPayload, str], str],
            "coerce_int": CoerceIntCallback,
            "fmt_on_time_delta": Callable[
                [datetime | None, datetime | None], str
            ],
            "fmtlocal": Callable[[datetime], str],
            "fmt_dt_local": Callable[[datetime], str],
            "short": ShortUuidCallback,
            "format_gap": Callable[
                [datetime | None, datetime | None, str, bool], str
            ],
        }
        for name, annotation in expected.items():
            with self.subTest(callback=name):
                self.assertEqual(annotations[name], annotation)

    def test_omit_label_does_not_hide_unexpected_formatter_failure(self) -> None:
        def broken_formatter(_omit_dnf, _omit_date):
            raise RuntimeError("omit description implementation failed")

        with self.assertRaisesRegex(RuntimeError, "omit description implementation failed"):
            _timeline_omit_label(
                [[{"kind": "w", "value": "mon", "mods": {}}]],
                date(2026, 8, 3),
                omit_description_for_date=broken_formatter,
            )

    def test_positional_anchor_timeline_projects_future_selected_dates(self) -> None:
        expression = "(w:tue | w:thu)@in-month=last"
        task = {
            "uuid": "00000000-0000-4000-8000-000000000778",
            "description": "positional timeline",
            "status": "pending",
            "anchor": expression,
            "anchor_mode": "skip",
            "link": 1,
            "due": "20260730T090000Z",
            "end": "20260730T100000Z",
            "chainID": "abcd1234",
        }
        observation = TaskObservation.from_mapping(
            task, source_query="modify timeline positional contract"
        )
        scheduler = SchedulerService.from_observation(
            observation,
            context=RecurrenceContext(chain_id="abcd1234", timezone=timezone.utc),
        )
        due = datetime(2026, 8, 27, 9, tzinfo=timezone.utc)
        items = _timeline_future_anchor_items(
            task,
            core.validate_anchor_expr_strict(expression),
            due,
            start_no=2,
            allowed_future=5,
            cap_no=None,
            to_local_cached=lambda value: value.astimezone(timezone.utc),
            safe_parse_datetime=lambda value: (parse_dt_any(value, ()), None),
            scheduler_service=scheduler,
            omit_dnf=None,
            omit_description_for_date=None,
            max_iterations=64,
        )

        future_dates = [item[1].date() for item in items if item[3] == "future"]
        self.assertEqual(future_dates[:2], [date(2026, 9, 29), date(2026, 10, 29)])

    def test_post_selection_modifier_timeline_projects_transformed_dates(self) -> None:
        expression = "(w:tue | w:thu)@in-month=last@+2d@t=09:00"
        task = {
            "uuid": "00000000-0000-4000-8000-000000000779",
            "description": "post-selection timeline",
            "status": "pending",
            "anchor": expression,
            "anchor_mode": "skip",
            "link": 2,
            "due": "20260801T060000Z",
            "end": "20260801T070000Z",
            "chainID": "abcd1234",
        }
        observation = TaskObservation.from_mapping(
            task, source_query="modify timeline post-selection contract"
        )
        scheduler = SchedulerService.from_observation(
            observation,
            context=RecurrenceContext(chain_id="abcd1234", timezone=timezone.utc),
        )
        due = datetime(2026, 8, 29, 9, tzinfo=timezone.utc)
        items = _timeline_future_anchor_items(
            task,
            core.validate_anchor_expr_strict(expression),
            due,
            start_no=3,
            allowed_future=4,
            cap_no=None,
            to_local_cached=lambda value: value.astimezone(timezone.utc),
            safe_parse_datetime=lambda value: (parse_dt_any(value, ()), None),
            scheduler_service=scheduler,
            omit_dnf=None,
            omit_description_for_date=None,
            max_iterations=64,
        )

        future_dates = [item[1].date() for item in items if item[3] == "future"]
        self.assertEqual(future_dates[0], date(2026, 10, 1))

    def test_yearly_positional_timeline_projects_next_shifted_occurrence(self) -> None:
        expression = "(w:mon)@in-year=last@+7d@t=09:00"
        task = {
            "uuid": "00000000-0000-4000-8000-000000000780",
            "description": "yearly positional timeline",
            "status": "pending",
            "anchor": expression,
            "anchor_mode": "skip",
            "link": 2,
            "due": "20270104T060000Z",
            "end": "20270104T070000Z",
            "chainID": "abcd1234",
        }
        observation = TaskObservation.from_mapping(
            task, source_query="modify timeline yearly positional contract"
        )
        scheduler = SchedulerService.from_observation(
            observation,
            context=RecurrenceContext(chain_id="abcd1234", timezone=timezone.utc),
        )
        due = datetime(2028, 1, 3, 9, tzinfo=timezone.utc)
        items = _timeline_future_anchor_items(
            task,
            core.validate_anchor_expr_strict(expression),
            due,
            start_no=3,
            allowed_future=4,
            cap_no=None,
            to_local_cached=lambda value: value.astimezone(timezone.utc),
            safe_parse_datetime=lambda value: (parse_dt_any(value, ()), None),
            scheduler_service=scheduler,
            omit_dnf=None,
            omit_description_for_date=None,
            max_iterations=64,
        )

        future_dates = [item[1].date() for item in items if item[3] == "future"]
        self.assertEqual(future_dates[0], date(2029, 1, 1))

    def test_scheduler_failure_becomes_a_renderable_warning_row(self) -> None:
        child_due = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)

        class BrokenScheduler:
            session = SimpleNamespace(
                evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc))
            )

            def collect(self, *_args, **_kwargs):
                raise ValueError("provider contract broken")

        items = _timeline_future_anchor_items(
            {"chainID": "timeline-warning-test", "due": "20260803T090000Z"},
            [[{"kind": "w", "value": "mon", "mods": {}}]],
            child_due,
            start_no=2,
            allowed_future=1,
            cap_no=None,
            to_local_cached=lambda value: value,
            safe_parse_datetime=lambda _value: (child_due, None),
            scheduler_service=BrokenScheduler(),
            omit_dnf=None,
            omit_description_for_date=None,
            max_iterations=4,
        )

        self.assertTrue(items)
        self.assertEqual(items[-1][3], "warning")
        self.assertIn("provider contract broken", items[-1][2]["message"])
        line = _timeline_base_line(
            items[-1][0], items[-1][1], items[-1][2], items[-1][3],
            task={}, cap_no=None, prev_style="", cur_style="", next_style="",
            future_style="", fmt_dt_local=str, dtparse=lambda value: value,
            fmt_on_time_delta=lambda *_args: "", fmtlocal=lambda _value: "",
            short=lambda _value: "",
        )
        self.assertIn("provider contract broken", line)

    def test_date_limited_projection_keeps_typed_terminal_evidence(self) -> None:
        child_due = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)
        terminal = OccurrenceSearchExhausted(
            "timeline projection", reference=date(9999, 12, 31), limit=1
        )

        class DateLimitedScheduler:
            session = SimpleNamespace(
                evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc))
            )

            def collect(self, *_args, **_kwargs):
                return SimpleNamespace(occurrences=(), terminal=terminal, failure=None)

        items = _timeline_future_anchor_items(
            {"chainID": "timeline-terminal-test", "due": "20260803T090000Z"},
            [[{"kind": "w", "value": "mon", "mods": {}}]],
            child_due,
            start_no=2,
            allowed_future=1,
            cap_no=None,
            to_local_cached=lambda value: value,
            safe_parse_datetime=lambda _value: (child_due, None),
            scheduler_service=DateLimitedScheduler(),
            omit_dnf=None,
            omit_description_for_date=None,
            max_iterations=4,
        )

        self.assertTrue(items)
        self.assertEqual(items[-1][3], "warning")
        self.assertIn("Projection ended", items[-1][2]["message"])
        self.assertIn("9999-12-31", items[-1][2]["message"])

    def test_completed_timeline_rows_place_uuid_before_timing_delta(self) -> None:
        common = dict(
            dt=None,
            cap_no=None,
            prev_style="prev",
            cur_style="current",
            next_style="next",
            future_style="future",
            fmt_dt_local=lambda _value: "DATE",
            dtparse=lambda value: value,
            fmt_on_time_delta=lambda _due, _end: "(DELTA)",
            fmtlocal=lambda _value: "DATE",
            short=lambda value: str(value).replace("-", "")[:8],
        )
        previous = _timeline_base_line(
            1,
            obj={"due": "due", "end": "end", "uuid": "beeswax"},
            item_type="prev",
            task={},
            **common,
        )
        current = _timeline_base_line(
            2,
            obj={},
            item_type="current",
            task={"due": "due", "end": "end", "uuid": "cafebabe-0000"},
            **common,
        )
        self.assertIn("DATE beeswax (DELTA)", previous)
        self.assertIn("DATE cafebabe (DELTA)", current)

    def test_omit_evaluation_failure_is_warning_not_normal_future_slot(self) -> None:
        child_due = datetime(2026, 8, 3, 9, 0, tzinfo=timezone.utc)

        class BrokenOmitScheduler:
            session = SimpleNamespace(
                evaluator=SimpleNamespace(
                    context=SimpleNamespace(timezone=timezone.utc)
                )
            )

            def collect(self, *_args, **_kwargs):
                raise RuntimeError("omit backend unavailable")

        items = _timeline_future_anchor_items(
            {"chainID": "timeline-omit-warning-test", "due": "20260803T090000Z"},
            [[{"kind": "w", "value": "mon", "mods": {}}]],
            child_due,
            start_no=2,
            allowed_future=1,
            cap_no=None,
            to_local_cached=lambda value: value,
            safe_parse_datetime=lambda _value: (child_due, None),
            scheduler_service=BrokenOmitScheduler(),
            omit_dnf=[[{"kind": "w", "value": "mon", "mods": {}}]],
            omit_description_for_date=None,
            max_iterations=4,
        )

        self.assertTrue(items)
        self.assertEqual(items[-1][3], "warning")
        self.assertIn("omit backend unavailable", items[-1][2]["message"])


if __name__ == "__main__":
    unittest.main()
