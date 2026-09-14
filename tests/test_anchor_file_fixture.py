from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import nautical_core.anchor_files as anchor_files
from nautical_core import business_calendar
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.scheduler_cursor import OccurrenceCursor
from nautical_core.scheduler_service import SchedulerService
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.add_anchor_preview import (
    _collect_events_with_provider,
    _collect_included_with_provider,
)
from tests.support.anchor_file import canonical_anchor_file_fixture


class CanonicalAnchorFileFixtureTests(unittest.TestCase):
    def test_scheduler_and_preview_collectors_preserve_file_description(self) -> None:
        from nautical_core.occurrence_provider import Occurrence

        with TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date,description\n2026-08-03,Water the plants\n",
                encoding="utf-8",
            )
            observation = DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "33333333-0000-4000-8000-000000000003",
                    "chainID": "event-metadata-test",
                    "status": "pending",
                    "link": 1,
                    "anchor_file": "calendar.csv@t=09:00",
                },
                source_query="anchor-file-metadata-contract",
            )
            context = RecurrenceContext(
                chain_id="event-metadata-test",
                timezone=timezone.utc,
                anchor_file_dir=directory,
            )
            scheduler = SchedulerService.from_observation(observation, context=context)
            cursor = OccurrenceCursor(
                datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
                inclusive=False,
                timezone=timezone.utc,
            )

            events = _collect_events_with_provider(
                after_local_dt=cursor.local_datetime,
                inclusive=cursor.inclusive,
                limit_included=1,
                fallback_hhmm=(9, 0),
                default_seed_date=date(2026, 8, 2),
                return_occurrences=True,
                scheduler_service=scheduler,
            )
            included = _collect_included_with_provider(
                after_local_dt=cursor.local_datetime,
                inclusive=cursor.inclusive,
                limit=1,
                fallback_hhmm=(9, 0),
                default_seed_date=date(2026, 8, 2),
                return_occurrences=True,
                scheduler_service=scheduler,
            )

        for collection in (events, included):
            self.assertEqual(len(collection), 1)
            self.assertIsInstance(collection[0], Occurrence)
            self.assertEqual(collection[0].source, "anchor_file")
            self.assertEqual(collection[0].description, "Water the plants")

    def test_merged_scheduler_carries_context_and_loads_anchor_specs_once(self) -> None:
        original = anchor_files.load_anchor_file_occurrence_specs
        contexts = []

        def counted(*args, **kwargs):
            contexts.append(kwargs.get("context"))
            return original(*args, **kwargs)

        with TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date\n2026-08-03\n2026-08-04\n", encoding="utf-8"
            )
            row = {
                "uuid": "44444444-0000-4000-8000-000000000004",
                "chainID": "merged-provider-test",
                "status": "pending",
                "link": 1,
                "anchor_file": "calendar.csv@t=rand(06..18)",
            }
            observation = DEFAULT_TASK_CODEC.decode_row(
                row, source_query="merged-anchor-file-context"
            )
            calendar = business_calendar.DEFAULT_BUSINESS_CALENDAR
            context = RecurrenceContext(
                chain_id="merged-provider-test",
                timezone=timezone.utc,
                business_calendar=calendar,
                anchor_file_dir=directory,
            )
            scheduler = SchedulerService.from_observation(observation, context=context)
            with patch.object(anchor_files, "load_anchor_file_occurrence_specs", counted):
                result = _collect_events_with_provider(
                    after_local_dt=datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc),
                    inclusive=False,
                    limit_included=2,
                    fallback_hhmm=(9, 0),
                    default_seed_date=date(2026, 8, 2),
                    return_occurrences=True,
                    scheduler_service=scheduler,
                )

        self.assertEqual(len(result), 2)
        self.assertEqual(len(contexts), 1)
        self.assertIsNotNone(contexts[0])
        self.assertEqual(contexts[0].chain_id, "merged-provider-test")
        self.assertIs(contexts[0].business_calendar, calendar)

    def test_fixture_supports_long_omission_scan(self) -> None:
        with canonical_anchor_file_fixture(count=514, omitted=lambda item, first: (item.date() - first).days < 513) as (core, root, _first):
            observation = DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "11111111-0000-4000-8000-000000000001",
                    "chainID": "long-omission-test",
                    "link": 1,
                    "chain": "on",
                    "status": "pending",
                    "anchor_file": "calendar.csv",
                },
                source_query="long-anchor-omission-contract",
            )
            context = RecurrenceContext(
                chain_id="long-omission-test",
                timezone=timezone.utc,
                anchor_file_dir=str(root),
            )
            scheduler = SchedulerService.from_observation(
                observation, context=context
            )
            result = _collect_included_with_provider(
                after_local_dt=datetime(2025, 12, 31, 9, 0, tzinfo=timezone.utc),
                inclusive=False,
                limit=1,
                max_iterations=600,
                fallback_hhmm=(9, 0),
                default_seed_date=None,
                scheduler_service=scheduler,
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].date(), date(2027, 5, 29))
        self.assertEqual((result[0].hour, result[0].minute), (9, 0))

    def test_all_omitted_file_slots_end_as_an_empty_finite_stream(self) -> None:
        with canonical_anchor_file_fixture(count=3, omitted=lambda _item, _first: True) as (_core, root, _first):
            observation = DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "22222222-0000-4000-8000-000000000002",
                    "chainID": "all-omitted-test",
                    "link": 1,
                    "chain": "on",
                    "status": "pending",
                    "anchor_file": "calendar.csv",
                },
                source_query="all-omitted-anchor-contract",
            )
            context = RecurrenceContext(
                chain_id="all-omitted-test",
                timezone=timezone.utc,
                anchor_file_dir=str(root),
            )
            scheduler = SchedulerService.from_observation(
                observation, context=context
            )
            result = _collect_included_with_provider(
                after_local_dt=datetime(2025, 12, 31, 9, 0, tzinfo=timezone.utc),
                inclusive=False,
                limit=1,
                max_iterations=4,
                fallback_hhmm=(9, 0),
                default_seed_date=None,
                scheduler_service=scheduler,
            )

        self.assertEqual(result, [])
        self.assertIsNone(result.terminal)


if __name__ == "__main__":
    unittest.main()
