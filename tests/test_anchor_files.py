import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from nautical_core import anchor_files


class AnchorFileOccurrenceCacheTests(unittest.TestCase):
    def _provider(self, directory: str) -> anchor_files.AnchorFileOccurrenceProvider:
        return anchor_files.AnchorFileOccurrenceProvider("calendar.csv", directory, (9, 0))

    @staticmethod
    def _build(day, hhmm):
        return datetime(day.year, day.month, day.day, *hhmm)

    def test_static_records_are_reused_across_cold_providers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text("date,description\n2026-08-24,review\n", encoding="utf-8")
            original = anchor_files._load_anchor_source_data
            with patch.object(anchor_files, "_load_anchor_source_data", wraps=original) as loader:
                first = self._provider(directory).occurrences()
                second = self._provider(directory).occurrences()
            self.assertEqual(first, second)
            self.assertEqual(loader.call_count, 1)

    def test_file_metadata_change_invalidates_static_record_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "calendar.csv")
            path.write_text("date,description\n2026-08-24,old\n", encoding="utf-8")
            self.assertEqual(self._provider(directory).occurrences()[0].description, "old")
            path.write_text("date,description\n2026-08-25,new\n", encoding="utf-8")
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            values = self._provider(directory).occurrences()
            self.assertEqual([(item.day, item.description) for item in values], [(values[0].day, "new")])

    def test_cached_lookup_cursor_advances_and_resets_for_earlier_requests(self) -> None:
        provider = anchor_files.AnchorFileOccurrenceProvider(None, None, (9, 0))
        provider._spec_cache = [
            (date(2026, 1, 1) + timedelta(days=index), (9, 0))
            for index in range(1000)
        ]
        repeat_provider = anchor_files.AnchorFileOccurrenceProvider(None, None, (9, 0))
        repeat_provider._spec_cache = provider._spec_cache
        built = []

        def build(day, hhmm):
            built.append(day)
            return datetime(day.year, day.month, day.day, *hhmm)

        identity = lambda value: value
        first = repeat_provider.next_after(
            datetime(2025, 12, 31, 9),
            build_local_datetime=build,
            to_local=identity,
        )
        repeated = repeat_provider.next_after(
            datetime(2025, 12, 31, 9),
            build_local_datetime=build,
            to_local=identity,
        )
        self.assertEqual(first.local_datetime, repeated.local_datetime)
        built.clear()
        cursor = datetime(2025, 12, 31, 9)
        for _ in range(100):
            occurrence = provider.next_after(
                cursor,
                build_local_datetime=build,
                to_local=identity,
            )
            self.assertIsNotNone(occurrence)
            cursor = occurrence.local_datetime

        self.assertEqual(len(built), 1000)
        reset = provider.next_after(
            datetime(2026, 1, 1, 9),
            build_local_datetime=build,
            to_local=identity,
        )
        self.assertEqual(reset.day, date(2026, 1, 2))

    def test_nonmonotonic_cursor_uses_cached_binary_search(self) -> None:
        provider = anchor_files.AnchorFileOccurrenceProvider(None, None, (9, 0))
        provider._record_cache = [
            (date(2026, 1, 1) + timedelta(days=index), (9, 0), f"slot-{index}")
            for index in range(10_000)
        ]
        identity = lambda value: value
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        first = provider.next_after(
            datetime(2026, 1, 1, 9),
            build_local_datetime=build,
            to_local=identity,
        )
        self.assertEqual(first.description, "slot-1")

        with patch.object(
            anchor_files,
            "compare_datetimes",
            side_effect=AssertionError("backward lookup rescanned cached candidates"),
        ):
            backward = provider.next_after(
                datetime(2026, 1, 1, 9),
                build_local_datetime=build,
                to_local=identity,
            )

        self.assertEqual(backward.description, "slot-1")

    def test_successor_order_uses_dst_normalized_local_time(self) -> None:
        from nautical_core.timeutil import build_local_datetime

        zone = ZoneInfo("Europe/Bucharest")
        provider = anchor_files.AnchorFileOccurrenceProvider(None, None, (9, 0))
        provider._spec_cache = [
            (date(2026, 3, 29), (3, 30)),
            (date(2026, 3, 29), (4, 0)),
        ]

        first = provider.next_after(
            datetime(2026, 3, 29, 2, 0, tzinfo=zone),
            build_local_datetime=lambda day, hhmm: build_local_datetime(day, hhmm, zone),
            to_local=lambda value: value.astimezone(zone),
        )
        self.assertEqual(first.local_datetime.strftime("%H:%M"), "04:00")
        second = provider.next_after(
            first.local_datetime,
            build_local_datetime=lambda day, hhmm: build_local_datetime(day, hhmm, zone),
            to_local=lambda value: value.astimezone(zone),
        )
        self.assertEqual(second.local_datetime.strftime("%H:%M"), "04:30")

    def test_failed_provider_load_is_retried_without_caching_empty_result(self) -> None:
        provider = anchor_files.AnchorFileOccurrenceProvider("calendar.csv", ".", (9, 0))
        original = anchor_files.load_anchor_file_occurrence_specs
        calls = []

        def flaky(*_args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise ValueError("transient calendar read failure")
            kwargs["_records_sink"].append(
                (date(2026, 8, 8), (9, 0), "recovered")
            )
            return [(date(2026, 8, 8), (9, 0))]

        with patch.object(anchor_files, "load_anchor_file_occurrence_specs", side_effect=flaky):
            with self.assertRaisesRegex(ValueError, "transient calendar read failure"):
                provider.occurrences()
            occurrences = provider.occurrences()

        self.assertEqual(len(calls), 2)
        self.assertEqual(occurrences[0].description, "recovered")

    def test_overnight_slots_keep_source_date_description(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date,description\n2026-08-03,Overnight maintenance\n",
                encoding="utf-8",
            )
            occurrences = anchor_files.AnchorFileOccurrenceProvider(
                "calendar.csv@t=22:30..06:30/2", directory, (9, 0)
            ).occurrences()

        self.assertEqual([item.day for item in occurrences], [date(2026, 8, 3), date(2026, 8, 4)])
        self.assertEqual(
            [item.description for item in occurrences],
            ["Overnight maintenance", "Overnight maintenance"],
        )

    def test_duplicate_source_can_fill_missing_occurrence_description(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "first.csv").write_text(
                "date,description\n2026-08-08,\n", encoding="utf-8"
            )
            Path(directory, "second.csv").write_text(
                "date,description\n2026-08-08,Backup description\n",
                encoding="utf-8",
            )
            occurrences = anchor_files.AnchorFileOccurrenceProvider(
                "first.csv@t=09:00 | second.csv@t=09:00", directory, (9, 0)
            ).occurrences()

        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0].description, "Backup description")

    def test_dst_fold_description_tracks_the_selected_instant(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        provider = anchor_files.AnchorFileOccurrenceProvider(None, None, (9, 0))
        provider._record_cache = [
            (date(2026, 10, 25), (3, 30), "first fold"),
            (date(2026, 10, 25), (3, 45), "second fold"),
        ]

        def build(day, hhmm):
            fold = 1 if hhmm == (3, 45) else 0
            return datetime(
                day.year, day.month, day.day, *hhmm, tzinfo=zone, fold=fold
            )

        occurrence = provider.next_after(
            datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1),
            build_local_datetime=build,
            to_local=lambda value: value,
        )

        self.assertEqual(occurrence.description, "second fold")

    def test_dst_fold_successors_are_ordered_by_instant(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        provider = anchor_files.AnchorFileOccurrenceProvider(None, None, (9, 0))
        provider._spec_cache = [
            (date(2026, 10, 25), (3, 30)),
            (date(2026, 10, 25), (3, 45)),
        ]

        def build(day, hhmm):
            fold = 1 if hhmm == (3, 45) else 0
            return datetime(
                day.year, day.month, day.day, *hhmm, tzinfo=zone, fold=fold
            )

        occurrence = provider.next_after(
            datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1),
            build_local_datetime=build,
            to_local=lambda value: value,
        )

        self.assertEqual(occurrence.local_datetime.fold, 1)
        self.assertEqual(occurrence.hhmm, (3, 45))

    def test_provider_rejects_incomparable_naive_and_aware_datetimes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date\n2026-08-04\n", encoding="utf-8"
            )
            provider = self._provider(directory)
            with self.assertRaisesRegex(ValueError, "incomparable"):
                provider.next_after(
                    datetime(2026, 8, 3, 9),
                    build_local_datetime=lambda day, hhmm: datetime(
                        day.year, day.month, day.day, *hhmm, tzinfo=timezone.utc
                    ),
                    to_local=lambda value: value,
                )


if __name__ == "__main__":
    unittest.main()
