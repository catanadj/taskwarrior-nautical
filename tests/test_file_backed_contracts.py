"""Direct contracts for anchor-file and omit-file input semantics."""

import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import nautical_core as core
from nautical_core import anchor_files, omit_files


class FileBackedRecurrenceContractTests(unittest.TestCase):
    def test_csv_without_date_column_reports_available_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "name,description\nHoliday,No date column\n", encoding="utf-8"
            )
            for label, loader in (
                ("anchor_file", anchor_files.load_anchor_file_dates),
                ("omit_file", omit_files.load_omit_file_dates),
            ):
                with self.subTest(source=label), self.assertRaisesRegex(
                    ValueError, "CSV must contain a 'date' column"
                ) as raised:
                    loader("calendar.csv", directory)
                self.assertIn("Found columns: name, description", str(raised.exception))

    def test_empty_and_unusable_date_files_report_actionable_errors(self) -> None:
        for label, loader in (
            ("anchor_file", anchor_files.load_anchor_file_dates),
            ("omit_file", omit_files.load_omit_file_dates),
        ):
            with self.subTest(source=label), tempfile.TemporaryDirectory() as directory:
                Path(directory, "empty.csv").write_text(
                    "# no dates here\n\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "empty or has no date rows"):
                    loader("empty.csv", directory)

                Path(directory, "blank_dates.csv").write_text(
                    "date,description\n,Missing date\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "did not contain any usable dates") as raised:
                    loader("blank_dates.csv", directory)
                self.assertIn("1 data row(s), 0 non-empty date value(s)", str(raised.exception))

    def test_same_size_rewrite_does_not_return_stale_cached_dates(self) -> None:
        from nautical_core import file_backed_dates

        for label, loader in (
            ("anchor_file", anchor_files.load_anchor_file_dates),
            ("omit_file", omit_files.load_omit_file_dates),
        ):
            file_backed_dates._CACHE_BY_PATH.clear()
            with self.subTest(source=label), tempfile.TemporaryDirectory() as directory:
                sample = Path(directory, "calendar.csv")
                sample.write_text("date\n2026-01-01\n", encoding="utf-8")
                first_stat = sample.stat()
                self.assertEqual(loader("calendar.csv", directory), frozenset({date(2026, 1, 1)}))

                sample.write_text("date\n2026-01-02\n", encoding="utf-8")
                os.utime(sample, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))
                second_stat = sample.stat()
                if (
                    second_stat.st_dev == first_stat.st_dev
                    and second_stat.st_ino == first_stat.st_ino
                    and second_stat.st_size == first_stat.st_size
                    and second_stat.st_mtime_ns == first_stat.st_mtime_ns
                    and second_stat.st_ctime_ns == first_stat.st_ctime_ns
                ):
                    cached = file_backed_dates._CACHE_BY_PATH[str(sample)]
                    file_backed_dates._CACHE_BY_PATH[str(sample)] = file_backed_dates._FileCacheEntry(
                        (*cached.metadata[:4], cached.metadata[4] + 1),
                        cached.digest,
                        cached.dates,
                        cached.descriptions,
                    )
                self.assertEqual(loader("calendar.csv", directory), frozenset({date(2026, 1, 2)}))
        file_backed_dates._CACHE_BY_PATH.clear()

    def test_file_date_cache_uses_metadata_and_enforces_lru_bound(self) -> None:
        from nautical_core import file_backed_dates

        saved_limit = file_backed_dates._FILE_CACHE_MAX_ENTRIES
        file_backed_dates._CACHE_BY_PATH.clear()
        file_backed_dates._FILE_CACHE_MAX_ENTRIES = 2
        try:
            with tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory, name) for name in ("first.txt", "second.txt", "third.txt")]
                paths[0].write_text("2026-01-01\n" * 10_000, encoding="utf-8")
                paths[1].write_text("2026-01-02\n", encoding="utf-8")
                paths[2].write_text("2026-01-03\n", encoding="utf-8")
                first_value = file_backed_dates.load_file_date_data(
                    str(paths[0]), label="anchor_file first.txt"
                )
                with patch.object(Path, "read_bytes", side_effect=AssertionError("cache hit read file")):
                    self.assertEqual(
                        file_backed_dates.load_file_date_data(
                            str(paths[0]), label="anchor_file first.txt"
                        ),
                        first_value,
                    )

                file_backed_dates.load_file_date_data(str(paths[1]), label="anchor_file second.txt")
                file_backed_dates.load_file_date_data(str(paths[0]), label="anchor_file first.txt")
                file_backed_dates.load_file_date_data(str(paths[2]), label="anchor_file third.txt")
                self.assertEqual(len(file_backed_dates._CACHE_BY_PATH), 2)
                self.assertIn(str(paths[0]), file_backed_dates._CACHE_BY_PATH)
                self.assertNotIn(str(paths[1]), file_backed_dates._CACHE_BY_PATH)
        finally:
            file_backed_dates._FILE_CACHE_MAX_ENTRIES = saved_limit
            file_backed_dates._CACHE_BY_PATH.clear()

    def test_file_date_cache_reuses_identical_digest_after_metadata_change(self) -> None:
        from nautical_core import file_backed_dates

        file_backed_dates._CACHE_BY_PATH.clear()
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory, "calendar.txt")
            sample.write_text("2026-01-01\n", encoding="utf-8")
            parse = file_backed_dates._parse_text_dates
            parse_calls = []

            def counted(*args, **kwargs):
                parse_calls.append(1)
                return parse(*args, **kwargs)

            try:
                with patch.object(file_backed_dates, "_parse_text_dates", counted):
                    first = file_backed_dates.load_file_date_data(
                        str(sample), label="anchor_file calendar.txt"
                    )
                    os.utime(sample, ns=(sample.stat().st_atime_ns, sample.stat().st_mtime_ns + 1))
                    second = file_backed_dates.load_file_date_data(
                        str(sample), label="anchor_file calendar.txt"
                    )
            finally:
                file_backed_dates._CACHE_BY_PATH.clear()

        self.assertEqual(first, second)
        self.assertEqual(first, (frozenset({date(2026, 1, 1)}), {}))
        self.assertEqual(len(parse_calls), 1)

    def test_file_date_resources_bound_bytes_lines_ranges_and_dates(self) -> None:
        from nautical_core import file_backed_dates, file_resource_limits

        limits = file_resource_limits
        original = (
            limits.MAX_FILE_BYTES,
            limits.MAX_FILE_LINES,
            limits.MAX_DATE_RANGE_DAYS,
            limits.MAX_RESOLVED_DATES,
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                cases = (
                    ("oversized.txt", "2026-01-01\n2026-01-02\n", "MAX_FILE_BYTES", 16, "too large"),
                    ("lines.txt", "2026-01-01\n2026-01-02\n2026-01-03\n", "MAX_FILE_LINES", 2, "contains 3 lines"),
                    ("range.txt", "2026-01-01..2026-01-03\n", "MAX_DATE_RANGE_DAYS", 2, "range spans 3 days"),
                    ("dates.txt", "2026-01-01\n2026-01-02\n2026-01-03\n", "MAX_RESOLVED_DATES", 2, "more than 2 unique dates"),
                )
                for filename, content, limit_name, limit, message in cases:
                    with self.subTest(limit=limit_name):
                        (
                            limits.MAX_FILE_BYTES,
                            limits.MAX_FILE_LINES,
                            limits.MAX_DATE_RANGE_DAYS,
                            limits.MAX_RESOLVED_DATES,
                        ) = original
                        setattr(limits, limit_name, limit)
                        path = base / filename
                        path.write_text(content, encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, message):
                            file_backed_dates.load_file_date_data(str(path), label=filename)
                        file_backed_dates._CACHE_BY_PATH.clear()

                limits.MAX_DATE_RANGE_DAYS = original[2]
                max_date = base / "max-date.txt"
                max_date.write_text("9999-12-31..9999-12-31\n", encoding="utf-8")
                dates, _ = file_backed_dates.load_file_date_data(str(max_date), label="max date")
                self.assertEqual(dates, frozenset({date.max}))
        finally:
            (
                limits.MAX_FILE_BYTES,
                limits.MAX_FILE_LINES,
                limits.MAX_DATE_RANGE_DAYS,
                limits.MAX_RESOLVED_DATES,
            ) = original
            file_backed_dates._CACHE_BY_PATH.clear()

    def test_wildcard_resolution_bounds_directory_and_file_fanout(self) -> None:
        from nautical_core import file_resource_limits

        limits = file_resource_limits
        original = (limits.MAX_DIRECTORY_ENTRIES, limits.MAX_RESOLVED_FILES)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for idx in range(3):
                    Path(directory, f"{idx}.csv").write_text(
                        f"date\n2026-01-0{idx + 1}\n", encoding="utf-8"
                    )
                limits.MAX_DIRECTORY_ENTRIES = 2
                with self.assertRaisesRegex(ValueError, "contains more than 2 entries"):
                    anchor_files.load_anchor_file_dates("*.csv", directory)

                limits.MAX_DIRECTORY_ENTRIES = 10
                limits.MAX_RESOLVED_FILES = 2
                with self.assertRaisesRegex(ValueError, "more than 2 files"):
                    anchor_files.load_anchor_file_dates("*.csv", directory)
        finally:
            limits.MAX_DIRECTORY_ENTRIES, limits.MAX_RESOLVED_FILES = original

    def test_aggregate_file_limits_cover_anchor_omit_occurrences_and_calendars(self) -> None:
        from nautical_core import file_resource_limits

        original = file_resource_limits.MAX_RESOLVED_DATES
        try:
            file_resource_limits.MAX_RESOLVED_DATES = 1
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                (base / "one.csv").write_text("date\n2026-01-01\n", encoding="utf-8")
                (base / "two.csv").write_text("date\n2026-01-02\n", encoding="utf-8")
                for label, loader in (
                    ("anchor_file", anchor_files.load_anchor_file_dates),
                    ("omit_file", omit_files.load_omit_file_dates),
                ):
                    with self.subTest(source=label), self.assertRaisesRegex(
                        ValueError, f"{label} resolves to more than 1 unique dates"
                    ):
                        loader("one.csv | two.csv", directory)
                with self.assertRaisesRegex(ValueError, "more than 1 occurrences"):
                    anchor_files.load_anchor_file_occurrence_specs(
                        "one.csv@t=09:00,17:00", directory, (8, 0)
                    )
                with self.assertRaisesRegex(
                    ValueError,
                    "business_calendar.work.anchor_file resolves to more than 1 unique dates",
                ):
                    core.resolve_business_calendar_config(
                        {"work": {"anchor_file": ["one.csv", "two.csv"]}},
                        anchor_file_dir=directory,
                    )
        finally:
            file_resource_limits.MAX_RESOLVED_DATES = original

    def test_anchor_file_sources_merge_modifiers_and_descriptions_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            anchor_dir = Path(directory)
            (anchor_dir / "first.csv").write_text(
                "date,description\n2026-04-20,First description\n", encoding="utf-8"
            )
            (anchor_dir / "second.csv").write_text(
                "date,description\n2026-04-21,Second description\n2026-04-22,Later date\n",
                encoding="utf-8",
            )

            expression = "(first.csv@+1d | second.csv)@+1d"
            self.assertEqual(
                anchor_files.load_anchor_file_dates(expression, directory),
                frozenset({date(2026, 4, 22), date(2026, 4, 23)}),
            )
            self.assertEqual(
                anchor_files.load_anchor_file_descriptions(expression, directory),
                {
                    date(2026, 4, 22): "First description",
                    date(2026, 4, 23): "Later date",
                },
            )

    def test_file_wildcards_are_ordered_bounded_and_single_scan(self) -> None:
        from nautical_core import file_source_expr

        with tempfile.TemporaryDirectory() as directory:
            anchor_dir = Path(directory)
            (anchor_dir / "b.csv").write_text("date\n2026-04-22\n", encoding="utf-8")
            (anchor_dir / "a").write_text("2026-04-21\n", encoding="utf-8")
            (anchor_dir / ".hidden.csv").write_text("date\n2026-04-23\n", encoding="utf-8")
            (anchor_dir / "folder.csv").mkdir()

            self.assertEqual(
                anchor_files.load_anchor_file_dates("*.*", directory),
                frozenset({date(2026, 4, 21), date(2026, 4, 22)}),
            )
            self.assertEqual(
                anchor_files.load_anchor_file_dates("missing-*.csv", directory),
                frozenset(),
            )
            self.assertEqual(
                anchor_files.unmatched_anchor_file_patterns(
                    "missing-*.csv | b.csv", directory
                ),
                ("missing-*.csv",),
            )
            scan = file_source_expr.os.scandir
            calls = []

            def counted(path):
                calls.append(path)
                return scan(path)

            with patch.object(file_source_expr.os, "scandir", counted):
                file_source_expr.resolve_file_source_expression(
                    "* | *.csv | missing-?.txt", directory, label="anchor_file"
                )
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, "Unknown anchor_file modifier"):
                anchor_files.load_anchor_file_dates("missing-*.csv@unknown", directory)

    def test_anchor_file_source_times_survive_merging_and_dedupe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name in ("early.csv", "late.csv"):
                (Path(directory) / name).write_text("date\n2026-04-25\n", encoding="utf-8")

            self.assertEqual(
                anchor_files.load_anchor_file_occurrence_specs(
                    "late.csv@t=15:00 | early.csv@t=09:00 | early*.csv@t=09:00",
                    directory,
                    (12, 0),
                ),
                [(date(2026, 4, 25), (9, 0)), (date(2026, 4, 25), (15, 0))],
            )
            self.assertEqual(
                anchor_files.load_anchor_file_occurrence_specs(
                    "(early.csv | late.csv)@t=11:00", directory, (12, 0)
                ),
                [(date(2026, 4, 25), (11, 0))],
            )
            with self.assertRaisesRegex(ValueError, "more than one @t modifier"):
                anchor_files.load_anchor_file_dates(
                    "(early.csv@t=09:00 | late.csv)@t=15:00", directory
                )
            occurrence = anchor_files.next_anchor_file_occurrence_after(
                "late.csv@t=15:00 | early.csv@t=09:00",
                directory,
                datetime(2026, 4, 25, 10, 0),
                (12, 0),
                build_local_datetime=lambda day, hhmm: datetime(
                    day.year, day.month, day.day, *hhmm
                ),
                to_local=lambda value: value,
            )
            self.assertEqual((occurrence.hour, occurrence.minute), (15, 0))

    def test_omit_file_sources_are_atomic_and_reject_time_modifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            omit_dir = Path(directory)
            (omit_dir / "public.csv").write_text("date\n2026-04-20\n", encoding="utf-8")
            (omit_dir / "local.txt").write_text("2026-04-21\n", encoding="utf-8")
            (omit_dir / "bad.csv").write_text("date\nnot-a-date\n", encoding="utf-8")

            self.assertEqual(
                omit_files.load_omit_file_dates(
                    "(public.csv | local.txt)@+1d", directory
                ),
                frozenset({date(2026, 4, 21), date(2026, 4, 22)}),
            )
            for expression in (
                "(public.csv | local.txt)@t=09:00",
                "missing-*.csv@t=09:00",
            ):
                with self.subTest(expression=expression), self.assertRaisesRegex(
                    ValueError, "omit_file does not support time modifiers"
                ):
                    omit_files.load_omit_file_dates(expression, directory)
            with self.assertRaisesRegex(ValueError, "omit_file 'bad.csv'"):
                omit_files.load_omit_file_dates("public.csv | bad.csv", directory)

    def test_file_symlinks_cannot_escape_the_configured_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            anchor_dir = base / "anchors"
            anchor_dir.mkdir()
            outside = base / "outside.csv"
            outside.write_text("date\n2026-04-20\n", encoding="utf-8")
            (anchor_dir / "linked.csv").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "resolves outside its configured directory"):
                anchor_files.load_anchor_file_dates("linked.csv", str(anchor_dir))

    def test_anchor_file_slots_shifted_to_same_dst_instant_are_deduplicated(self) -> None:
        from nautical_core.occurrence_provider import Occurrence
        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.scheduler_cursor import OccurrenceCursor
        from nautical_core.scheduler_service import SchedulerService
        from nautical_core.task_codec import DEFAULT_TASK_CODEC

        zone = ZoneInfo("Europe/Bucharest")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-03-29\n", encoding="utf-8"
            )
            observation = DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "11111111-0000-4000-8000-000000000041",
                    "status": "pending",
                    "chainID": "dst-add-test",
                    "link": 1,
                    "anchor_file": "calendar.csv@t=03:30,04:30",
                },
                source_query="anchor-file-dst-gap",
            )
            context = RecurrenceContext(
                chain_id="dst-add-test",
                timezone=zone,
                anchor_file_dir=directory,
            )
            scheduler = SchedulerService.from_observation(observation, context=context)
            result = scheduler.collect(
                OccurrenceCursor(
                    datetime(2026, 3, 28, 0, 0, tzinfo=zone),
                    inclusive=False,
                    timezone=zone,
                ),
                limit=4,
                fallback_hhmm=(9, 0),
            )

        self.assertEqual(len(result.occurrences), 1)
        occurrence = result.occurrences[0]
        self.assertIsInstance(occurrence, Occurrence)
        self.assertEqual((occurrence.hour, occurrence.minute), (4, 30))

    def test_next_anchor_file_occurrence_uses_shared_dst_ordering(self) -> None:
        from nautical_core.timeutil import build_local_datetime

        zone = ZoneInfo("Europe/Bucharest")
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-03-29\n2026-03-29\n", encoding="utf-8"
            )
            with patch.object(
                anchor_files,
                "load_anchor_file_occurrence_specs",
                return_value=[
                    (date(2026, 3, 29), (3, 30)),
                    (date(2026, 3, 29), (4, 0)),
                ],
            ):
                occurrence = anchor_files.next_anchor_file_occurrence_after(
                    "calendar.csv",
                    directory,
                    datetime(2026, 3, 29, 2, 0, tzinfo=zone),
                    (9, 0),
                    build_local_datetime=lambda day, hhmm: build_local_datetime(
                        day, hhmm, zone
                    ),
                    to_local=lambda value: value.astimezone(zone),
                )

        self.assertIsNotNone(occurrence)
        self.assertEqual((occurrence.hour, occurrence.minute), (4, 0))
    def test_file_source_expression_flattens_group_modifiers_and_rejects_unsafe_syntax(self) -> None:
        from nautical_core.file_source_expr import parse_file_source_expression

        sources = parse_file_source_expression(
            "(one.csv@-1d | team-?.csv)@+1d | final.txt",
            label="anchor_file",
        )
        self.assertEqual(
            [(item.pattern, item.modifier_layers) for item in sources],
            [
                ("one.csv", ("@-1d", "@+1d")),
                ("team-?.csv", ("@+1d",)),
                ("final.txt", ()),
            ],
        )

        invalid = (
            ("one.csv || two.csv", "empty branch"),
            ("../one.csv", "not a path"),
            ("**", "recursive '**'"),
            ("team-[ab].csv", "only '*' and '?'"),
        )
        for expression, message in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError) as raised:
                    parse_file_source_expression(expression, label="anchor_file")
                self.assertIn(message, str(raised.exception))

    def test_anchor_and_omit_file_names_reject_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchor_file must be a file name, not a path"):
            anchor_files.validate_anchor_file_name("../calendar.csv")
        with self.assertRaisesRegex(ValueError, "omit_file must be a file name, not a path"):
            omit_files.validate_omit_file_name("../holidays.csv")

    def test_omit_file_csv_uses_headers_deduplicates_dates_and_preserves_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holidays.csv"
            path.write_text(
                "description,region,date\n"
                "New Year,AU,2025-01-01\n"
                "New Year duplicate,NSW,2025-01-01\n"
                "Holiday,AU,2025-01-26\n",
                encoding="utf-8",
            )

            self.assertEqual(
                omit_files.load_omit_file_dates(path.name, directory),
                frozenset({date(2025, 1, 1), date(2025, 1, 26)}),
            )
            descriptions_path = Path(directory) / "holiday_descriptions.csv"
            descriptions_path.write_text(
                "region,description,date\n"
                "AU,New Year,2025-01-01\n"
                "NSW,Anniversary Day,2025-01-26\n",
                encoding="utf-8",
            )
            self.assertEqual(
                omit_files.load_omit_file_descriptions(descriptions_path.name, directory),
                {date(2025, 1, 1): "New Year", date(2025, 1, 26): "Anniversary Day"},
            )

    def test_omit_file_roll_moves_dates_and_descriptions_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "holidays.csv").write_text(
                "date,description\n2026-04-25,Weekend holiday\n",
                encoding="utf-8",
            )

            dates = omit_files.load_omit_file_dates(
                "holidays.csv@nbd", directory
            )
            descriptions = omit_files.load_omit_file_descriptions(
                "holidays.csv@nbd", directory
            )

        self.assertEqual(dates, frozenset({date(2026, 4, 27)}))
        self.assertEqual(
            descriptions, {date(2026, 4, 27): "Weekend holiday"}
        )

    def test_omit_file_supports_negative_calendar_and_business_day_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "holidays.csv").write_text(
                "date\n2026-04-27\n", encoding="utf-8"
            )

            calendar_days = omit_files.load_omit_file_dates(
                "holidays.csv@-2d", directory
            )
            business_days = omit_files.load_omit_file_dates(
                "holidays.csv@-2bd", directory
            )

        self.assertEqual(calendar_days, frozenset({date(2026, 4, 25)}))
        self.assertEqual(business_days, frozenset({date(2026, 4, 23)}))

    def test_omit_file_modifiers_apply_after_base_file_data_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "holidays.csv").write_text(
                "date\n2026-04-25\n", encoding="utf-8"
            )

            plain = omit_files.load_omit_file_dates("holidays.csv", directory)
            shifted = omit_files.load_omit_file_dates(
                "holidays.csv@nbd", directory
            )

        self.assertEqual(plain, frozenset({date(2026, 4, 25)}))
        self.assertEqual(shifted, frozenset({date(2026, 4, 27)}))

    def test_omit_file_rejects_time_modifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "holidays.csv").write_text(
                "date\n2026-04-25\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(
                ValueError, r"omit_file does not support time modifiers \(@t\)"
            ):
                omit_files.load_omit_file_dates(
                    "holidays.csv@t=09:00", directory
                )

    def test_anchor_file_spec_parses_times_offsets_and_bounded_windows(self) -> None:
        name, modifiers = anchor_files.parse_anchor_file_spec(
            "calendar.csv@t=09:00,17:00@-2d@+1bd"
        )
        self.assertEqual(name, "calendar.csv")
        self.assertEqual(modifiers.get("t"), [(9, 0), (17, 0)])
        self.assertEqual(modifiers.get("day_offset"), -2)
        self.assertEqual(modifiers.get("business_day_offset"), 1)

        name, modifiers = anchor_files.parse_anchor_file_spec("calendar.csv@t=06..17/3h")
        self.assertEqual(name, "calendar.csv")
        self.assertEqual(modifiers.get("time_window"), "06:00..17:00/3h")
        self.assertEqual(modifiers.get("t"), [(6, 0), (9, 0), (12, 0), (15, 0)])
        with self.assertRaisesRegex(ValueError, "end time must differ"):
            anchor_files.parse_anchor_file_spec("calendar.csv@t=18..18/2h")

    def test_anchor_file_random_time_spec_normalizes_to_canonical_window(self) -> None:
        name, modifiers = anchor_files.parse_anchor_file_spec(
            "events.csv@t=rand(06..18)"
        )

        self.assertEqual(name, "events.csv")
        self.assertEqual(modifiers.get("time_random"), "rand(06:00..18:00)")

    def test_anchor_file_time_schedules_reject_unpadded_or_empty_members(self) -> None:
        with self.assertRaisesRegex(ValueError, "leading zero.*03:00"):
            anchor_files.parse_anchor_file_spec("calendar.csv@t=3:00")
        with self.assertRaisesRegex(ValueError, "(?i)empty"):
            anchor_files.parse_anchor_file_spec("calendar.csv@t=06..12/2h,,18")

    def test_anchor_file_time_window_expands_only_generated_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            occurrences = anchor_files.load_anchor_file_occurrence_specs(
                "calendar.csv@t=06..17/3h", directory, (8, 0)
            )

        self.assertEqual(
            occurrences,
            [
                (date(2026, 8, 3), (6, 0)),
                (date(2026, 8, 3), (9, 0)),
                (date(2026, 8, 3), (12, 0)),
                (date(2026, 8, 3), (15, 0)),
            ],
        )

    def test_anchor_file_random_window_uses_supplied_recurrence_context(self) -> None:
        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.time_windows import parse_random_time_window_spec

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            occurrences = anchor_files.load_anchor_file_occurrence_specs(
                "calendar.csv@t=rand(06..18/3)",
                directory,
                (8, 0),
                context=RecurrenceContext(chain_id="file-chain"),
            )

        slots = parse_random_time_window_spec("rand(06..18/3)").slots_with_offsets(
            "file-chain/2026-08-03"
        )
        expected = [(date(2026, 8, 3), (slot[1], slot[2])) for slot in slots]
        self.assertEqual(occurrences, expected)

    def test_anchor_file_provider_returns_typed_occurrences(self) -> None:
        from nautical_core.occurrence_provider import Occurrence

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            values = anchor_files.AnchorFileOccurrenceProvider(
                "calendar.csv@t=06,12:30", directory, (8, 0)
            ).occurrences()

        self.assertTrue(all(isinstance(value, Occurrence) for value in values))
        self.assertEqual(
            [(value.day, value.hhmm, value.source) for value in values],
            [
                (date(2026, 8, 3), (6, 0), "anchor_file"),
                (date(2026, 8, 3), (12, 30), "anchor_file"),
            ],
        )

    def test_anchor_file_occurrence_retains_csv_description(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date,description\n2026-08-03,Water the plants\n", encoding="utf-8"
            )
            occurrences = anchor_files.AnchorFileOccurrenceProvider(
                "calendar.csv@t=09:00", directory, (9, 0)
            ).occurrences()

        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0].source, "anchor_file")
        self.assertEqual(occurrences[0].description, "Water the plants")

    def test_anchor_file_provider_projects_one_lazy_successor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            provider = anchor_files.AnchorFileOccurrenceProvider(
                "calendar.csv@t=06,12:30", directory, (8, 0)
            )
            occurrence = provider.next_after(
                datetime(2026, 8, 3, 6, 0),
                build_local_datetime=lambda day, hhmm: datetime(
                    day.year, day.month, day.day, *hhmm
                ),
                to_local=lambda value: value,
            )

        self.assertIsNotNone(occurrence)
        self.assertEqual(
            (occurrence.day, occurrence.hhmm), (date(2026, 8, 3), (12, 30))
        )

    def test_anchor_file_provider_expands_specs_once_for_repeated_lookups(self) -> None:
        original = anchor_files.load_anchor_file_occurrence_specs
        calls = []

        def counted(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n2026-08-04\n", encoding="utf-8"
            )
            anchor_files.load_anchor_file_occurrence_specs = counted
            try:
                provider = anchor_files.AnchorFileOccurrenceProvider(
                    "calendar.csv@t=09:00", directory, (8, 0)
                )
                build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
                provider.next_after(
                    datetime(2026, 8, 2, 9, 0),
                    build_local_datetime=build,
                    to_local=lambda value: value,
                )
                provider.next_after(
                    datetime(2026, 8, 3, 9, 0),
                    build_local_datetime=build,
                    to_local=lambda value: value,
                )
            finally:
                anchor_files.load_anchor_file_occurrence_specs = original

        self.assertEqual(len(calls), 1)

    def test_anchor_file_overnight_window_keeps_source_date_for_each_slot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            occurrences = anchor_files.load_anchor_file_occurrence_specs(
                "calendar.csv@t=22:30..06:30/7",
                directory,
                (9, 0),
            )

        self.assertEqual(
            occurrences,
            [
                (date(2026, 8, 3), (22, 30)),
                (date(2026, 8, 3), (23, 50)),
                (date(2026, 8, 4), (1, 10)),
                (date(2026, 8, 4), (2, 30)),
                (date(2026, 8, 4), (3, 50)),
                (date(2026, 8, 4), (5, 10)),
                (date(2026, 8, 4), (6, 30)),
            ],
        )

    def test_anchor_file_window_and_explicit_times_compose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            occurrences = anchor_files.load_anchor_file_occurrence_specs(
                "calendar.csv@t=06..12/3h,18", directory, (8, 0)
            )

        self.assertEqual(
            [hhmm for _day, hhmm in occurrences],
            [(6, 0), (9, 0), (12, 0), (18, 0)],
        )

    def test_anchor_file_date_modifiers_transform_dates_and_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date,description\n2026-04-25,Weekend anchor\n",
                encoding="utf-8",
            )
            dates = anchor_files.load_anchor_file_dates("calendar.csv@nbd", directory)
            descriptions = anchor_files.load_anchor_file_descriptions(
                "calendar.csv@nbd", directory
            )
            shifted = anchor_files.load_anchor_file_dates(
                "calendar.csv@nbd@-1bd", directory
            )

        self.assertEqual(dates, frozenset({date(2026, 4, 27)}))
        self.assertEqual(descriptions, {date(2026, 4, 27): "Weekend anchor"})
        self.assertEqual(shifted, frozenset({date(2026, 4, 24)}))

    def test_next_anchor_file_occurrence_prefers_task_times_to_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "calendar.csv").write_text(
                "date\n2026-04-25\n", encoding="utf-8"
            )
            occurrence = anchor_files.next_anchor_file_occurrence_after(
                "calendar.csv@t=12:00,17:00",
                directory,
                datetime(2026, 4, 24, 10, 0),
                (9, 0),
                build_local_datetime=lambda day, hhmm: datetime(
                    day.year, day.month, day.day, *hhmm
                ),
                to_local=lambda value: value,
            )

        self.assertEqual(occurrence, datetime(2026, 4, 25, 12, 0))


if __name__ == "__main__":
    unittest.main()
