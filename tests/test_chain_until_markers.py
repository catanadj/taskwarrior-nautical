from __future__ import annotations

import unittest
import importlib
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from nautical_core.chain_until_markers import resolve_chain_until


season_support = importlib.import_module("nautical_core.season_support")


class ChainUntilMarkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._season_mode = season_support.active_mode()
        self._hemisphere = season_support.active_hemisphere()
        self._timezone = season_support._ACTIVE_TIMEZONE

    def tearDown(self) -> None:
        season_support.configure_mode(self._season_mode)
        season_support.configure_hemisphere(self._hemisphere)
        season_support.configure_timezone(self._timezone)

    def test_concrete_timestamp_is_preserved(self) -> None:
        value = "2026-12-25T18:30:00Z"

        self.assertEqual(
            resolve_chain_until(value, reference=datetime(2026, 11, 8, tzinfo=timezone.utc)),
            value,
        )

    def test_annual_holiday_resolves_to_local_end_of_day(self) -> None:
        self.assertEqual(
            resolve_chain_until(
                "christmas",
                reference=datetime(2026, 11, 8, tzinfo=timezone.utc),
            ),
            "2026-12-25T23:59:59Z",
        )

    def test_easter_family_markers_resolve_from_reference_year(self) -> None:
        reference = datetime(2027, 1, 10, tzinfo=timezone.utc)

        self.assertEqual(resolve_chain_until("easter", reference=reference), "2027-03-28T23:59:59Z")
        self.assertEqual(resolve_chain_until("good-friday", reference=reference), "2027-03-26T23:59:59Z")
        self.assertEqual(resolve_chain_until("easter-monday", reference=reference), "2027-03-29T23:59:59Z")

        after_easter = datetime(2027, 4, 10, tzinfo=timezone.utc)
        self.assertEqual(resolve_chain_until("easter", reference=after_easter), "2028-04-16T23:59:59Z")

    def test_fixed_season_markers_use_configured_calendar_boundaries(self) -> None:
        season_support.configure_mode("fixed")
        season_support.configure_hemisphere("north")

        reference = datetime(2026, 1, 10, tzinfo=timezone.utc)

        self.assertEqual(resolve_chain_until("summer-start", reference=reference), "2026-06-01T23:59:59Z")
        self.assertEqual(resolve_chain_until("summer-end", reference=reference), "2026-08-31T23:59:59Z")

    def test_astronomical_season_marker_uses_transition_local_date(self) -> None:
        season_support.configure_mode("astronomical")
        season_support.configure_hemisphere("north")
        season_support.configure_timezone("UTC")

        value = resolve_chain_until(
            "summer-start",
            reference=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(value.startswith("2026-06-"))
        self.assertTrue(value.endswith("T23:59:59Z"))

    def test_calendar_period_markers_resolve_to_inclusive_period_end(self) -> None:
        reference = datetime(2026, 5, 12, tzinfo=timezone.utc)

        self.assertEqual(resolve_chain_until("month-end", reference=reference), "2026-05-31T23:59:59Z")
        self.assertEqual(resolve_chain_until("quarter-end", reference=reference), "2026-06-30T23:59:59Z")
        self.assertEqual(resolve_chain_until("year-end", reference=reference), "2026-12-31T23:59:59Z")

    def test_explicit_season_event_resolves_to_its_local_calendar_date(self) -> None:
        season_support.configure_timezone("UTC")

        value = resolve_chain_until(
            "summer-solstice",
            reference=datetime(2026, 1, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(value.startswith("2026-06-"))
        self.assertTrue(value.endswith("T23:59:59Z"))

    def test_end_of_day_uses_configured_timezone_across_dst(self) -> None:
        value = resolve_chain_until(
            "christmas",
            reference=datetime(2026, 11, 8, tzinfo=timezone.utc),
            local_tz=ZoneInfo("America/New_York"),
        )

        self.assertEqual(value, "2026-12-26T04:59:59Z")

    def test_unknown_marker_is_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Unknown chainUntil marker 'not-a-holiday'.*holiday"):
            resolve_chain_until(
                "not-a-holiday",
                reference=datetime(2026, 11, 8, tzinfo=timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
