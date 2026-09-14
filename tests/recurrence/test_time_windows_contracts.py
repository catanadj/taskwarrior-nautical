"""Direct contracts for deterministic time-window and schedule parsing."""

import unittest
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from nautical_core import file_resource_limits
from nautical_core.time_windows import (
    parse_random_time_window_spec,
    parse_time_schedule_spec,
    parse_time_window_spec,
)
from nautical_core.timeutil import build_local_datetime, to_local


class TimeWindowContractTests(unittest.TestCase):
    def test_window_expands_exact_and_non_divisible_boundaries(self) -> None:
        exact = parse_time_window_spec("09:00..17:00/2h")
        bounded = parse_time_window_spec("09:00..18:00/2h")

        self.assertEqual(exact.slots, ((9, 0), (11, 0), (13, 0), (15, 0), (17, 0)))
        self.assertEqual(bounded.slots, ((9, 0), (11, 0), (13, 0), (15, 0), (17, 0)))
        self.assertEqual(bounded.canonical, "09:00..18:00/2h")

    def test_window_rejects_unsafe_or_ambiguous_ranges(self) -> None:
        self.assertIsNone(parse_time_window_spec("09:00"))
        for value in (
            "09:00..09:00/2h",
            "09:00..17:00/0m",
            "09:00..17:00/2d",
            "09:00..17:00/9h",
            "09:00..17:00/1m",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time_window_spec(value)

    def test_window_honors_shared_slot_resource_limit(self) -> None:
        original = file_resource_limits.MAX_TIME_WINDOW_SLOTS
        try:
            file_resource_limits.MAX_TIME_WINDOW_SLOTS = 3
            with self.assertRaisesRegex(ValueError, "below 3 slots"):
                parse_time_window_spec("06:00..18:00/3h")
        finally:
            file_resource_limits.MAX_TIME_WINDOW_SLOTS = original

    def test_compound_minute_intervals_normalize_and_expand(self) -> None:
        compact = parse_time_window_spec("08:00..12:00/1h30m")
        long_form = parse_time_window_spec("04:30..19:30/3h30min")
        decimal = parse_time_window_spec("04:30..19:30/3.5h")

        self.assertEqual(compact.interval_minutes, 90)
        self.assertEqual(compact.slots, ((8, 0), (9, 30), (11, 0)))
        self.assertEqual(long_form.interval_minutes, 210)
        self.assertEqual(long_form.slots, ((4, 30), (8, 0), (11, 30), (15, 0), (18, 30)))
        self.assertEqual(decimal.interval_minutes, 210)
        self.assertEqual(decimal.canonical, "04:30..19:30/3h30m")

    def test_hour_only_and_mixed_precision_endpoints_normalize(self) -> None:
        shorthand = parse_time_window_spec("06..18/3h")
        mixed = parse_time_window_spec("06:30..18/2h")

        self.assertEqual(shorthand.slots, ((6, 0), (9, 0), (12, 0), (15, 0), (18, 0)))
        self.assertEqual(shorthand.canonical, "06:00..18:00/3h")
        self.assertEqual(mixed.slots, ((6, 30), (8, 30), (10, 30), (12, 30), (14, 30), (16, 30)))
        self.assertEqual(mixed.canonical, "06:30..18:00/2h")

    def test_partition_count_generates_evenly_spaced_slots(self) -> None:
        window = parse_time_window_spec("04:30..19:30/3")
        four = parse_time_window_spec("06..18/4")

        self.assertEqual(window.partition_count, 3)
        self.assertIsNone(window.interval_minutes)
        self.assertEqual(window.slots, ((4, 30), (12, 0), (19, 30)))
        self.assertEqual(window.canonical, "04:30..19:30/3")
        self.assertEqual(four.slots, ((6, 0), (10, 0), (14, 0), (18, 0)))
        overnight = parse_time_window_spec("22:30..06:30/7")
        self.assertTrue(overnight.crosses_midnight)
        self.assertEqual(
            overnight.slots_with_offsets,
            ((0, 22, 30), (0, 23, 50), (1, 1, 10), (1, 2, 30), (1, 3, 50), (1, 5, 10), (1, 6, 30)),
        )
        overnight_interval = parse_time_window_spec("22:30..06:30/2h")
        self.assertEqual(overnight_interval.interval_minutes, 120)
        self.assertEqual(
            overnight_interval.slots_with_offsets,
            ((0, 22, 30), (1, 0, 30), (1, 2, 30), (1, 4, 30), (1, 6, 30)),
        )
        for value in ("06..18/1", "06..18/800"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time_window_spec(value)

    def test_random_windows_are_seeded_bounded_and_ordered(self) -> None:
        single = parse_random_time_window_spec("rand(06..18)")
        first = single.slots_with_offsets("chain-a/2026-08-04")
        self.assertEqual(first, single.slots_with_offsets("chain-a/2026-08-04"))
        self.assertEqual(len(first), 1)
        self.assertTrue((6, 0) <= first[0][1:] <= (18, 0))

        grouped = parse_random_time_window_spec("rand(06..18/3)")
        slots = grouped.slots_with_offsets("chain-a/2026-08-04")
        minutes = [offset * 1440 + hour * 60 + minute for offset, hour, minute in slots]
        self.assertEqual(len(set(minutes)), 3)
        self.assertEqual(minutes, sorted(minutes))
        self.assertTrue(all(6 * 60 <= minute <= 18 * 60 for minute in minutes))

        overnight = parse_random_time_window_spec("rand(22:30..02:30/3)")
        self.assertTrue(overnight.crosses_midnight)
        self.assertTrue(all(slot[0] in (0, 1) for slot in overnight.slots_with_offsets("chain-a/2026-08-04")))
        with self.assertRaisesRegex(ValueError, "stable chain seed"):
            single.slots_with_offsets("")
        for value in ("rand(06..18/0)", "rand(06..18/1441)"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_random_time_window_spec(value)

    def test_random_window_slots_project_uniquely_across_dst_transitions(self) -> None:
        window = parse_random_time_window_spec("rand(01:00..04:00/3)")
        zone = ZoneInfo("America/New_York")

        for day in (date(2025, 3, 9), date(2025, 11, 2)):
            with self.subTest(day=day):
                slots = window.slots_with_offsets(f"dst/{day.isoformat()}")
                instants = [
                    build_local_datetime(
                        day + timedelta(days=day_offset), (hour, minute), zone
                    )
                    for day_offset, hour, minute in slots
                ]

                self.assertEqual(len(instants), len(set(instants)))
                self.assertTrue(
                    all(to_local(value, zone).tzinfo is not None for value in instants)
                )

    def test_time_schedule_deduplicates_overlap_and_has_stable_canonical_form(self) -> None:
        boundary = parse_time_schedule_spec("00..04/2h,04..08/2h,08,23")
        overlap = parse_time_schedule_spec("06..12/2h,08..14/3h,08")
        reordered = parse_time_schedule_spec("22,06..18/3h,06..18/3h,02")

        self.assertEqual(boundary.slots, ((0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (23, 0)))
        self.assertEqual(overlap.slots, ((6, 0), (8, 0), (10, 0), (11, 0), (12, 0), (14, 0)))
        self.assertEqual(reordered.canonical, "02:00,06:00..18:00/3h,22:00")

    def test_partition_rounding_preserves_endpoints_and_monotonicity(self) -> None:
        window = parse_time_window_spec("06:00..18:01/4")
        minutes = [hour * 60 + minute for hour, minute in window.slots]

        self.assertEqual(window.slots, ((6, 0), (10, 0), (14, 1), (18, 1)))
        self.assertEqual(minutes[0], 6 * 60)
        self.assertEqual(minutes[-1], 18 * 60 + 1)
        self.assertTrue(all(left < right for left, right in zip(minutes, minutes[1:])))

    def test_composed_schedule_honors_aggregate_slot_limit(self) -> None:
        original = file_resource_limits.MAX_TIME_WINDOW_SLOTS
        try:
            file_resource_limits.MAX_TIME_WINDOW_SLOTS = 4
            with self.assertRaisesRegex(ValueError, "too many slots"):
                parse_time_schedule_spec("06..12/2h,16..20/2h")
        finally:
            file_resource_limits.MAX_TIME_WINDOW_SLOTS = original

    def test_window_endpoints_require_padded_valid_clock_values(self) -> None:
        for value in ("6..18/3h", "06:0..18/3h", "06..24/3h"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time_window_spec(value)


if __name__ == "__main__":
    unittest.main()
