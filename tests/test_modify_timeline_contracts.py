"""Direct contracts for timeline warning behavior at failure boundaries."""

from datetime import date, datetime, timezone
from types import SimpleNamespace
import unittest

from nautical_core.modify_timeline import _timeline_base_line, _timeline_future_anchor_items
from nautical_core.scheduler_models import OccurrenceSearchExhausted


class ModifyTimelineContractTests(unittest.TestCase):
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
