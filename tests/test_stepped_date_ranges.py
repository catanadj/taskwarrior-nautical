from __future__ import annotations

from datetime import date, datetime, timezone
import unittest

import nautical_core as core
from nautical_core.parsing.parser_models import ParseError
from nautical_core.occurrence_outcomes import FoundOccurrence
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from nautical_core.scheduler_service import SchedulerService
from nautical_core.task_models import TaskObservation


def _scheduler(anchor: str) -> SchedulerService:
    task = {
        "uuid": "00000000-0000-4000-8000-000000000321",
        "description": "stepped range fixture",
        "status": "pending",
        "link": 1,
        "anchor": anchor,
        "chainID": "stepped-range-tests",
    }
    return SchedulerService.from_observation(
        TaskObservation.from_mapping(task, source_query="stepped range test"),
        context=RecurrenceContext(chain_id="stepped-range-tests", timezone=timezone.utc),
    )


class SteppedDateRangeTests(unittest.TestCase):
    def test_monthly_range_accepts_day_step_and_expands_from_range_start(self) -> None:
        dnf = core.validate_anchor_expr_strict("m:1..31/3d")

        self.assertEqual(dnf[0][0]["spec"], "1..31/3d")
        self.assertEqual(
            core.expand_monthly_cached("1..31/3d", 2026, 2),
            [1, 4, 7, 10, 13, 16, 19, 22, 25, 28],
        )
        self.assertEqual(core.expand_monthly_cached("30..31/3d", 2026, 2), [])

    def test_yearly_range_accepts_day_step_and_crosses_month_boundaries(self) -> None:
        dnf = core.validate_anchor_expr_strict("y:06-01..08-15/2d")

        self.assertEqual(dnf[0][0]["spec"], "06-01..08-15/2d")
        dates = core.expand_yearly_cached("06-01..08-15/2d", 2026)

        self.assertEqual(dates[0], date(2026, 6, 1))
        self.assertEqual(dates[1], date(2026, 6, 3))
        self.assertEqual(dates[-1], date(2026, 8, 14))
        self.assertEqual(len(dates), 38)

    def test_descriptions_explain_the_shared_step(self) -> None:
        self.assertIn("3-day steps", core.describe_anchor_expr("m:1..31/3d"))
        self.assertIn("2-day steps", core.describe_anchor_expr("y:06-01..08-15/2d"))

    def test_yearly_step_skips_february_29_when_the_year_is_not_leap(self) -> None:
        non_leap = core.expand_yearly_cached("02-29..03-03/2d", 2025)
        leap = core.expand_yearly_cached("02-29..03-03/2d", 2024)

        self.assertEqual(non_leap, [date(2025, 3, 1), date(2025, 3, 3)])
        self.assertEqual(leap, [date(2024, 2, 29), date(2024, 3, 2)])

    def test_step_suffix_requires_positive_calendar_days(self) -> None:
        for expression in (
            "m:1..31/0d",
            "y:06-01..08-15/2h",
            "m:15/2d",
            "y:d100..d110/2d",
            "m:1..31/367d",
        ):
            with self.subTest(expression=expression):
                with self.assertRaises(ParseError):
                    core.validate_anchor_expr_strict(expression)

    def test_monthly_intersections_do_not_clamp_invalid_stepped_days(self) -> None:
        service = _scheduler("m:30..31/3d + w:sat@t=09:00")
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc
        )

        first = service.next(cursor)

        self.assertIsInstance(first, FoundOccurrence)
        self.assertEqual(first.occurrence.local_datetime.date(), date(2026, 5, 30))

    def test_cross_paths_agree_for_stepped_monthly_and_yearly_ranges(self) -> None:
        cases = (
            ("m:1..31/3d@t=09:00", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            ("y:06-01..08-15/2d@t=09:00", datetime(2026, 1, 1, tzinfo=timezone.utc)),
        )
        for anchor, start in cases:
            with self.subTest(anchor=anchor):
                service = _scheduler(anchor)
                cursor = OccurrenceCursor.strict_after(start, timezone=timezone.utc)
                first = service.next(cursor)
                collected = service.collect(cursor, limit=5)
                preview = service.preview(start, limit=1, timezone=timezone.utc)
                ranged = service.collect_request(OccurrenceRangeRequest(cursor, limit=5))

                self.assertIsInstance(first, FoundOccurrence)
                self.assertTrue(collected.occurrences)
                self.assertTrue(preview.occurrences)
                self.assertTrue(ranged.occurrences)
                self.assertEqual(first.occurrence.local_datetime, collected.occurrences[0].local_datetime)
                self.assertEqual(first.occurrence.local_datetime, preview.occurrences[0].local_datetime)
                self.assertEqual(first.occurrence.local_datetime, ranged.occurrences[0].local_datetime)
                self.assertEqual(
                    [item.local_datetime for item in collected.occurrences],
                    [item.local_datetime for item in ranged.occurrences],
                )

    def test_month_interval_composes_with_stepped_range(self) -> None:
        service = _scheduler("m/3:1..31/3d@t=09:00")
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc
        )

        occurrences = service.collect(cursor, limit=12).occurrences

        self.assertEqual(occurrences[0].local_datetime.date(), date(2026, 1, 1))
        self.assertEqual(occurrences[10].local_datetime.date(), date(2026, 1, 31))
        self.assertEqual(occurrences[11].local_datetime.date(), date(2026, 4, 1))

    def test_year_interval_composes_with_stepped_range(self) -> None:
        service = _scheduler("y/2:06-01..08-15/2d@t=09:00")
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc
        )

        occurrences = service.collect(cursor, limit=39).occurrences

        self.assertEqual(occurrences[0].local_datetime.date(), date(2026, 6, 1))
        self.assertEqual(occurrences[37].local_datetime.date(), date(2026, 8, 14))
        self.assertEqual(occurrences[38].local_datetime.date(), date(2028, 6, 1))


if __name__ == "__main__":
    unittest.main()
