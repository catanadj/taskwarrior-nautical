from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from nautical_core.add_anchor_preview import (
    AnchorExpressionPreviewServices,
    AnchorFilePreviewServices,
    _preview_occurrence_lines,
    handle_anchor_file_preview_on_add,
    handle_anchor_preview_on_add,
)
from nautical_core.modify_timeline import (
    TimelineFormattingServices,
    TimelineProjectionServices,
    timeline_lines_for_task,
)
from nautical_core.occurrence_provider import Occurrence
from nautical_core.occurrence_outcomes import FoundOccurrence


def _formatting() -> TimelineFormattingServices:
    return TimelineFormattingServices(
        future_style_for_chain=lambda _task, _kind: "cyan",
        coerce_int=lambda value, default=0: int(value or default),
        fmt_on_time_delta=lambda _due, _end: "",
        fmtlocal=lambda value: str(value),
        fmt_dt_local=lambda value: value.strftime("%Y-%m-%d %H:%M"),
        short=lambda value: str(value or ""),
        format_gap=lambda *_args: "",
    )


def _projection(*, evaluator_for_task, scheduler_for_task) -> TimelineProjectionServices:
    return TimelineProjectionServices(
        max_iterations=8,
        collect_prev_two=lambda _task: [],
        dtparse=lambda value: value,
        to_local_cached=lambda value: value,
        safe_parse_datetime=lambda value: (value, None),
        omit_dnf_from_parent=lambda _task: ("", None),
        omit_description_for_date=None,
        recurrence_evaluator_for_task=evaluator_for_task,
        scheduler_service_for_task=scheduler_for_task,
    )


class PresentationContextContractTests(unittest.TestCase):
    def test_preview_renderer_preserves_unicode_omit_description(self) -> None:
        event_date = datetime(2026, 1, 2, 9, tzinfo=timezone.utc)
        event = Occurrence(
            day=event_date.date(), hour=event_date.hour, minute=event_date.minute,
            local_datetime=event_date, omitted=True,
        )

        lines = _preview_occurrence_lines(
            [event],
            first_due_local_dt=event_date - timedelta(days=1),
            preview_limit=3,
            fmt_dt_local=lambda value: value.isoformat(),
            omit_description_for_task_date=lambda _task, _day: "Répéter 🌊",
            task={"omit_file": "calendar.csv"},
        )

        self.assertIn("Répéter 🌊", lines[0])

    def test_anchor_expression_preview_uses_its_context_without_file_only_services(self) -> None:
        first = datetime(2026, 1, 2, 9, tzinfo=timezone.utc)
        occurrence = Occurrence(
            day=first.date(), hour=first.hour, minute=first.minute, local_datetime=first
        )

        class Scheduler:
            session = SimpleNamespace(evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc)))

            @staticmethod
            def next(*_args, **_kwargs):
                return FoundOccurrence(
                    occurrence=occurrence, source="anchor", local_datetime=first, utc_datetime=first
                )

            @staticmethod
            def collect(*_args, **_kwargs):
                return SimpleNamespace(occurrences=(occurrence,), terminal=None, failure=None)

        panels = []
        services = AnchorExpressionPreviewServices(
            panel_mode="quiet",
            panel_warnings=lambda _task: ["Timezone data unavailable; using UTC fallback. Run nautical doctor."],
            prepare_anchor_dnf=lambda *_args: self.fail("Empty anchor should not parse an expression"),
            describe_anchor_natural=lambda *_args: self.fail("No natural-text renderer is needed"),
            prepare_omit_dnf=lambda _task, _rows: None,
            scheduler_service_for_task=lambda _task: Scheduler(),
            to_local=lambda value: value,
            fmt_dt_local=lambda value: value.isoformat(),
            coerce_int=lambda value, default=0: int(value or default),
            expr_has_m_or_y=lambda _dnf: False,
            append_dst_adjustment=lambda *_args: None,
            render_business_calendar_displacement=lambda *_args, **_kwargs: None,
            lint_and_validate=lambda *_args, **_kwargs: None,
            omit_description_for_task_date=lambda *_args: self.fail("No omit file was configured"),
            root_uuid_from=lambda _task: "root",
            short=lambda value: str(value or ""),
            validate_anchor_mode=lambda _mode: ("skip", None),
            validate_chain_duration_reasonable=lambda *_args: (True, None),
            append_wait_sched_rows=lambda *_args, **_kwargs: None,
            anchor_until_summary=lambda *_args, **_kwargs: (None, None),
            to_local_cached=lambda value: value,
            fmt_local_for_task=lambda value: value.isoformat(),
            format_anchor_rows=lambda rows: rows,
            panel=lambda *args, **kwargs: panels.append((args, kwargs)),
            human_delta=lambda *_args: "1d",
            error_and_exit=lambda rows: self.fail(f"Unexpected preview error: {rows!r}"),
            validate_native_until_after_target=lambda *_args: None,
            validate_native_until_anchor_slots=lambda *_args: None,
            append_first_expiration_row=lambda *_args: None,
        )

        handle_anchor_preview_on_add(
            task={"chainID": "expression-preview"},
            anchor_str="",
            ch="on",
            now_utc=first - timedelta(days=1),
            now_local=first - timedelta(days=1),
            user_provided_due=False,
            recurrence_field="due",
            due_dt=first - timedelta(days=1),
            due_day=first.date() - timedelta(days=1),
            due_hhmm=(9, 0),
            until_dt=None,
            past_due_warning=None,
            prof=SimpleNamespace(add_ms=lambda *_args: None),
            anchor_warn=False,
            upcoming_preview=3,
            preview_hard_cap=8,
            max_summary_links=8,
            services=services,
        )

        self.assertEqual(len(panels), 1)
        rendered_rows = panels[0][0][1]
        self.assertIn(
            ("Warning", "[yellow]Timezone data unavailable; using UTC fallback. Run nautical doctor.[/]"),
            rendered_rows,
        )
        self.assertNotIn("Upcoming", [label for label, _value in rendered_rows])

    def test_cp_timeline_uses_evaluator_without_constructing_anchor_scheduler(self) -> None:
        start = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)

        class Evaluator:
            cp_tokens = [{"kind": "interval"}]

            @staticmethod
            def cp_interval_for_link(_link):
                return timedelta(days=1)

            @staticmethod
            def project_cp(value, _link):
                return value + timedelta(days=1)

        evaluator = Evaluator()
        projection = _projection(
            evaluator_for_task=lambda _task: evaluator,
            scheduler_for_task=lambda _task: self.fail("CP projection constructed an anchor scheduler"),
        )
        task = {"uuid": "cp-task", "cp": "1d", "link": 1, "due": start, "end": start}

        lines = timeline_lines_for_task(
            "cp", task, start + timedelta(days=1), "child", None,
            projection=projection, formatting=_formatting(), next_count=1,
        )

        self.assertTrue(any("2026-01-03" in line for line in lines))

    def test_anchor_timeline_uses_scheduler_without_constructing_cp_evaluator(self) -> None:
        start = datetime(2026, 1, 1, 9, tzinfo=timezone.utc)
        next_local = start + timedelta(days=2)

        class Scheduler:
            session = SimpleNamespace(evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc)))

            @staticmethod
            def collect(*_args, **_kwargs):
                return SimpleNamespace(
                    occurrences=(Occurrence(
                        day=next_local.date(), hour=next_local.hour, minute=next_local.minute,
                        local_datetime=next_local,
                    ),), terminal=None, failure=None
                )

        projection = _projection(
            evaluator_for_task=lambda _task: self.fail("Anchor projection constructed a CP evaluator"),
            scheduler_for_task=lambda _task: Scheduler(),
        )
        task = {"uuid": "anchor-task", "anchor": "w:thu", "link": 1, "due": start, "end": start}

        lines = timeline_lines_for_task(
            "anchor", task, start + timedelta(days=1), "child", None,
            projection=projection, formatting=_formatting(), next_count=1,
        )

        self.assertTrue(any("2026-01-03" in line for line in lines))

    def test_anchor_file_preview_uses_only_its_focused_context(self) -> None:
        first = datetime(2026, 1, 2, 9, tzinfo=timezone.utc)

        class Scheduler:
            session = SimpleNamespace(evaluator=SimpleNamespace(context=SimpleNamespace(timezone=timezone.utc)))

            @staticmethod
            def collect(*_args, **_kwargs):
                return SimpleNamespace(
                    occurrences=(Occurrence(
                        day=first.date(), hour=first.hour, minute=first.minute, local_datetime=first
                    ),), terminal=None, failure=None
                )

        panels = []
        services = AnchorFilePreviewServices(
            panel_mode="quiet",
            timezone_fallback_warning=lambda _expr: False,
            prepare_omit_dnf=lambda _task, _rows: None,
            scheduler_service_for_task=lambda _task: Scheduler(),
            to_local=lambda value: value,
            fmt_dt_local=lambda value: value.isoformat(),
            coerce_int=lambda value, default=0: int(value or default),
            render_business_calendar_displacement=lambda *_args, **_kwargs: None,
            omit_description_for_task_date=lambda _task, _day: self.fail("No omit file was configured"),
            append_wait_sched_rows=lambda *_args, **_kwargs: None,
            validate_chain_duration_reasonable=lambda *_args: (True, None),
            format_anchor_rows=lambda rows: rows,
            panel=lambda *args, **kwargs: panels.append((args, kwargs)),
            fmt_local_for_task=lambda value: value.isoformat(),
            human_delta=lambda *_args: "1d",
            error_and_exit=lambda rows: self.fail(f"Unexpected preview error: {rows!r}"),
        )

        handle_anchor_file_preview_on_add(
            task={"anchor_file": "calendar.csv"},
            anchor_file_str="calendar.csv",
            ch="on",
            now_utc=first - timedelta(days=1),
            now_local=first - timedelta(days=1),
            user_provided_due=False,
            recurrence_field="due",
            due_dt=first - timedelta(days=1),
            due_hhmm=(9, 0),
            until_dt=None,
            past_due_warning=None,
            prof=SimpleNamespace(add_ms=lambda *_args: None),
            anchor_warn=False,
            upcoming_preview=3,
            preview_hard_cap=8,
            services=services,
        )

        self.assertEqual(len(panels), 1)
        self.assertIn("First due", [label for label, _value in panels[0][0][1]])


if __name__ == "__main__":
    unittest.main()
