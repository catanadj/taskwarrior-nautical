"""Direct contracts for fixed seasons and seasonal selector syntax."""

from __future__ import annotations

import json
import unittest
from datetime import date, datetime, timedelta

import nautical_core as core
from nautical_core import (
    acf_support,
    astronomical_seasons,
    cache_payload,
    position_selection,
    season_support,
)


class SeasonCalendarContractTests(unittest.TestCase):
    def setUp(self) -> None:
        core._refresh_facade_config_exports()
        self.previous_mode = core.SEASON_MODE
        self.previous_hemisphere = season_support.active_hemisphere()
        core.SEASON_MODE = "fixed"
        season_support.configure_mode("fixed")
        season_support.configure_hemisphere("north")

    def tearDown(self) -> None:
        core.SEASON_MODE = self.previous_mode
        season_support.configure_mode(self.previous_mode)
        season_support.configure_hemisphere(self.previous_hemisphere)
        core._refresh_facade_config_exports()

    def test_astronomical_season_calculator_returns_ordered_immutable_results(self) -> None:
        events = astronomical_seasons.seasonal_events_utc(2026)
        self.assertEqual(tuple(events), astronomical_seasons.SEASON_EVENT_NAMES)
        values = tuple(events.values())
        self.assertTrue(all(value.tzinfo == astronomical_seasons.UTC for value in values))
        self.assertTrue(all(left < right for left, right in zip(values, values[1:])))
        self.assertEqual(
            tuple(value.date().isoformat() for value in values),
            ("2026-03-20", "2026-06-21", "2026-09-23", "2026-12-21"),
        )
        self.assertEqual(
            astronomical_seasons.season_boundary_utc(2026, "spring"),
            events["spring_equinox"],
        )

        events["spring_equinox"] = datetime.min.replace(tzinfo=astronomical_seasons.UTC)
        self.assertEqual(
            astronomical_seasons.seasonal_event_utc(2026, "spring-equinox").date().isoformat(),
            "2026-03-20",
        )
        for invalid_year in (True, 2026.0, 0, 10000):
            with self.subTest(year=invalid_year), self.assertRaises((TypeError, ValueError)):
                astronomical_seasons.seasonal_events_utc(invalid_year)
        with self.assertRaisesRegex(astronomical_seasons.AstronomicalSeasonError, "Expected one of"):
            astronomical_seasons.seasonal_event_utc(2026, "equinox")
        with self.assertRaises(TypeError):
            astronomical_seasons.solar_longitude(datetime(2026, 1, 1))

    def test_seasonal_selection_scheduler_traverses_windows_and_rollover(self) -> None:
        seed = date(2026, 1, 1)
        self.assertEqual(
            position_selection.next_period_start("spring", date(2026, 4, 1)),
            date(2027, 3, 1),
        )
        self.assertEqual(
            position_selection.next_period_start("winter", date(2026, 1, 1)),
            date(2026, 12, 1),
        )
        cases = (
            ("(w:mon)@in-spring=first", date(2026, 1, 1), date(2026, 3, 2)),
            ("(w:mon)@in-spring=last", date(2026, 1, 1), date(2026, 5, 25)),
            ("(w:mon)@in-summer=first", date(2026, 1, 1), date(2026, 6, 1)),
            ("(w:fri)@in-autumn=last", date(2026, 1, 1), date(2026, 11, 27)),
            ("(w:mon)@in-winter=first", date(2026, 7, 1), date(2026, 12, 7)),
            ("(w:mon)@in-winter=last", date(2026, 7, 1), date(2027, 2, 22)),
            ("(w:mon)@in-winter=last", date(2027, 2, 22), date(2028, 2, 28)),
            ("(y:02-29)@in-winter=first", date(2026, 3, 1), date(2028, 2, 29)),
        )
        for expression, after_date, expected in cases:
            with self.subTest(expression=expression, after=after_date):
                dnf = core.validate_anchor_expr_strict(expression)
                actual, _meta = core.next_after_expr(dnf, after_date, default_seed=seed)
                self.assertEqual(actual, expected)
                self.assertTrue(core.factor_matches_on(dnf[0][0], expected, seed))

    def test_seasonal_business_calendar_shift_and_candidate_cache_identity(self) -> None:
        class ClosingCalendar:
            name = "season-closing"
            fingerprint = "season-closing-v1"

            @staticmethod
            def is_business_day(value):
                return value.weekday() < 5 and value != date(2027, 3, 1)

        expression = "(w:fri)@in-winter=last@+1bd"
        dnf = core.validate_anchor_expr_strict(expression)
        shifted, _meta = core.next_after_expr(
            dnf,
            date(2026, 7, 1),
            default_seed=date(2026, 1, 1),
            business_calendar=ClosingCalendar(),
        )
        self.assertEqual(shifted, date(2027, 3, 2))

        position_selection.clear_candidate_cache()
        seed = date(2026, 1, 1)
        spring = core.validate_anchor_expr_strict("(w:mon)@in-spring=first")[0][0]
        summer = core.validate_anchor_expr_strict("(w:mon)@in-summer=first")[0][0]
        kwargs = {
            "matches_on": core.atom_matches_on,
            "default_seed": seed,
            "seed_base": "season-cache-contract",
            "calendar_fingerprint": "calendar-a",
        }
        position_selection.selected_candidates_in_period(spring, seed, **kwargs)
        position_selection.selected_candidates_in_period(spring, seed, **kwargs)
        position_selection.selected_candidates_in_period(summer, seed, **kwargs)
        position_selection.selected_candidates_in_period(
            spring, seed, **{**kwargs, "calendar_fingerprint": "calendar-b"}
        )
        info = position_selection.candidate_cache_info()
        self.assertGreaterEqual(info.hits, 1)
        self.assertEqual(info.misses, 3)

    def test_fixed_season_windows_are_inclusive_and_start_year_keyed(self) -> None:
        previous = season_support.active_hemisphere()
        season_support.configure_hemisphere("north")
        try:
            expected = {
                "spring": (date(2026, 3, 1), date(2026, 5, 31)),
                "summer": (date(2026, 6, 1), date(2026, 8, 31)),
                "autumn": (date(2026, 9, 1), date(2026, 11, 30)),
                "winter": (date(2026, 12, 1), date(2027, 2, 28)),
            }
            self.assertEqual(season_support.SEASON_NAMES, tuple(expected))
            for name, bounds in expected.items():
                with self.subTest(season=name):
                    self.assertEqual(season_support.season_bounds(name, 2026), bounds)
                    self.assertEqual(
                        season_support.season_bounds(f" {name.upper()} ", 2026), bounds
                    )
            self.assertEqual(
                season_support.season_bounds("winter", 2027),
                (date(2027, 12, 1), date(2028, 2, 29)),
            )
        finally:
            season_support.configure_hemisphere(previous)

    def test_season_lookup_returns_active_or_next_named_window(self) -> None:
        previous = season_support.active_hemisphere()
        season_support.configure_hemisphere("north")
        try:
            cases = (
                ("spring", date(2026, 1, 15), (date(2026, 3, 1), date(2026, 5, 31))),
                ("spring", date(2026, 4, 15), (date(2026, 3, 1), date(2026, 5, 31))),
                ("spring", date(2026, 5, 31), (date(2026, 3, 1), date(2026, 5, 31))),
                ("spring", date(2026, 6, 1), (date(2027, 3, 1), date(2027, 5, 31))),
                ("winter", date(2026, 1, 15), (date(2025, 12, 1), date(2026, 2, 28))),
                ("winter", date(2026, 2, 28), (date(2025, 12, 1), date(2026, 2, 28))),
                ("winter", date(2026, 7, 1), (date(2026, 12, 1), date(2027, 2, 28))),
                ("winter", date(2026, 12, 1), (date(2026, 12, 1), date(2027, 2, 28))),
            )
            for season, reference, expected in cases:
                with self.subTest(season=season, reference=reference):
                    self.assertEqual(
                        season_support.season_window_on_or_after(season, reference), expected
                    )
        finally:
            season_support.configure_hemisphere(previous)

    def test_season_calendar_rejects_invalid_names_years_and_reference_types(self) -> None:
        previous = season_support.active_hemisphere()
        season_support.configure_hemisphere("north")
        try:
            for value in ("", "rainy", "monsoon"):
                with self.subTest(season=value), self.assertRaisesRegex(
                    ValueError, "Expected one of"
                ):
                    season_support.season_bounds(value, 2026)

            for year in (True, 2026.0):
                with self.subTest(year=year), self.assertRaises(TypeError):
                    season_support.season_bounds("spring", year)

            with self.assertRaisesRegex(ValueError, "supported date range"):
                season_support.season_bounds("winter", 9999)
            with self.assertRaises(TypeError):
                season_support.season_window_on_or_after("spring", "2026-03-01")
        finally:
            season_support.configure_hemisphere(previous)

    def test_southern_profile_remaps_fixed_seasons_and_selector_windows(self) -> None:
        previous = season_support.active_hemisphere()
        season_support.configure_hemisphere("south")
        try:
            self.assertEqual(
                season_support.season_bounds("spring", 2026),
                (date(2026, 9, 1), date(2026, 11, 30)),
            )
            self.assertEqual(
                season_support.season_bounds("summer", 2026),
                (date(2026, 12, 1), date(2027, 2, 28)),
            )
            self.assertEqual(
                season_support.fixed_season_boundary_description("winter"),
                "June 1 through August 31",
            )
            self.assertEqual(
                position_selection.period_bounds("summer", date(2027, 1, 15)),
                (date(2026, 12, 1), date(2027, 2, 28)),
            )
            dnf = core.validate_anchor_expr_strict("(y:12-01)@in-summer=first")
            actual, _meta = core.next_after_expr(
                dnf, date(2026, 7, 1), default_seed=date(2026, 1, 1)
            )
            self.assertEqual(actual, date(2026, 12, 1))
            with self.assertRaisesRegex(ValueError, "north, south"):
                season_support.configure_hemisphere("equatorial")
        finally:
            season_support.configure_hemisphere(previous)

    def test_astronomical_season_mode_and_hemisphere_boundaries(self) -> None:
        previous_mode = season_support.active_mode()
        previous_hemisphere = season_support.active_hemisphere()
        try:
            season_support.configure_mode("astronomical")
            season_support.configure_timezone("UTC")
            season_support.configure_hemisphere("north")
            self.assertEqual(
                season_support.season_bounds("spring", 2026),
                (date(2026, 3, 20), date(2026, 6, 20)),
            )
            self.assertEqual(
                season_support.season_bounds("winter", 2026),
                (date(2026, 12, 21), date(2027, 3, 19)),
            )
            season_support.configure_hemisphere("south")
            self.assertEqual(
                season_support.season_bounds("summer", 2026),
                (date(2026, 12, 21), date(2027, 3, 19)),
            )
            with self.assertRaisesRegex(ValueError, "fixed, astronomical"):
                season_support.configure_mode("sidereal")
            with self.assertRaisesRegex(ValueError, "invalid or unavailable"):
                season_support.configure_timezone("Not/A_Timezone")
        finally:
            season_support.configure_mode(previous_mode)
            season_support.configure_hemisphere(previous_hemisphere)
            season_support.configure_timezone(core.LOCAL_TZ_NAME)

    def test_astronomical_windows_use_local_dates_and_reject_range_overflow(self) -> None:
        previous_mode = season_support.active_mode()
        previous_hemisphere = season_support.active_hemisphere()
        try:
            season_support.configure_mode("astronomical")
            season_support.configure_hemisphere("north")
            season_support.configure_timezone("Pacific/Kiritimati")
            self.assertEqual(
                season_support.season_bounds("spring", 2026),
                (date(2026, 3, 21), date(2026, 6, 20)),
            )
            season_support.configure_timezone("America/Adak")
            self.assertEqual(
                season_support.season_bounds("summer", 2026),
                (date(2026, 6, 20), date(2026, 9, 21)),
            )
            for timezone_name in ("Pacific/Kiritimati", "America/Adak"):
                with self.subTest(timezone=timezone_name):
                    season_support.configure_timezone(timezone_name)
                    windows = [
                        season_support.season_bounds(name, 2026)
                        for name in season_support.SEASON_NAMES
                    ]
                    self.assertTrue(
                        all(
                            left[1] + timedelta(days=1) == right[0]
                            for left, right in zip(windows, windows[1:])
                        ),
                        f"astronomical windows have a gap or overlap: {windows}",
                    )
            season_support.configure_timezone("UTC")
            self.assertEqual(season_support.season_bounds("spring", 9999)[0].year, 9999)
            with self.assertRaisesRegex(ValueError, "supported date range"):
                season_support.season_bounds("winter", 9999)
        finally:
            season_support.configure_mode(previous_mode)
            season_support.configure_hemisphere(previous_hemisphere)
            season_support.configure_timezone(core.LOCAL_TZ_NAME)

    def test_seasonal_selector_parser_normalizes_scopes_and_enforces_bounds(self) -> None:
        limits = {"spring": 92, "summer": 92, "autumn": 91, "winter": 91}
        for scope, limit in limits.items():
            with self.subTest(scope=scope):
                final_position = position_selection.format_position(limit)
                parsed = position_selection.parse_group_selection_modifier(
                    f"@in-{scope}=first,{final_position},last@+1d"
                )
                self.assertEqual(parsed, (scope, (1, limit, -1), "@+1d"))
                node = position_selection.normalize_selection_node(
                    {
                        "kind": "select",
                        "scope": scope,
                        "positions": parsed[1],
                        "expr": [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {}}]],
                        "mods": {"day_offset": 1},
                    }
                )
                self.assertEqual(node["scope"], scope)
                with self.assertRaisesRegex(ValueError, f"{scope} limit of {limit}"):
                    position_selection.parse_positions(str(limit + 1), scope)

        self.assertEqual(
            position_selection.parse_group_selection_modifier("@in-season=first,92nd,last"),
            ("season", (1, 92, -1), ""),
        )
        with self.assertRaisesRegex(ValueError, "season limit of 92"):
            position_selection.parse_positions("93rd", "season")

        impossible = (
            ("(w:mon)@in-autumn=15th", "at most 14 matching dates per autumn"),
            ("(m:1)@in-winter=4th", "at most 3 matching dates per winter"),
        )
        for expression, message in impossible:
            with self.subTest(expression=expression), self.assertRaises(core.ParseError) as raised:
                core.validate_anchor_expr_strict(expression)
            self.assertIn(message, str(raised.exception))

    def test_seasonal_selection_round_trips_through_acf_and_cache_shape(self) -> None:
        node = {
            "kind": "select",
            "scope": "spring",
            "positions": (1, -1),
            "expr": [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {}}]],
            "mods": {},
        }
        terms = acf_support._build_acf_terms(
            [[node]],
            coerce_int=lambda value, default: int(value or default),
            normalize_spec_for_acf=lambda _typ, spec: spec,
            mods_to_acf=lambda _mods: {},
            atom_sort_key=lambda atom: acf_support.atom_sort_key(atom, json_mod=json),
            json_mod=json,
        )
        self.assertEqual(terms[0][0].get("c"), "spring")
        rendered = acf_support.acf_to_original_format(
            "valid:payload",
            is_valid_acf=lambda _value: True,
            acf_unpack=lambda _payload: {"terms": terms},
            acf_spec_to_string=lambda _typ, spec: spec,
            acf_mods_to_string=lambda _mods: "",
            format_selection_positions=position_selection.format_positions,
        )
        self.assertEqual(rendered, "(w:mon)@in-spring=first,last")

        expression = "(w:mon)@in-spring=last@+7d"
        dnf = core.validate_anchor_expr_strict(expression)
        self.assertEqual(dnf[0][0].get("scope"), "spring")
        self.assertTrue(cache_payload.is_selection_like(dnf[0][0]))
        acf = core.build_acf(expression)
        self.assertNotIn(acf, ("", "!PARSE_ERROR"))
        self.assertEqual(core.acf_to_original_format(acf), expression)

    def test_generic_season_scope_schedules_each_season_and_round_trips(self) -> None:
        expression = "(w:mon)@in-season=1st"
        dnf = core.validate_anchor_expr_strict(expression)
        self.assertEqual(dnf[0][0].get("scope"), "season")
        first, _meta = core.next_after_expr(
            dnf, date(2026, 1, 1), default_seed=date(2026, 1, 1)
        )
        self.assertEqual(first, date(2026, 3, 2))
        for reference, expected in (
            (date(2026, 3, 2), date(2026, 6, 1)),
            (date(2026, 6, 1), date(2026, 9, 7)),
            (date(2026, 9, 7), date(2026, 12, 7)),
            (date(2026, 12, 7), date(2027, 3, 1)),
        ):
            with self.subTest(reference=reference):
                actual, _meta = core.next_after_expr(
                    dnf, reference, default_seed=date(2026, 1, 1)
                )
                self.assertEqual(actual, expected)
        self.assertEqual(
            core.describe_anchor_expr(expression), "the first Monday of each season"
        )
        self.assertEqual(
            core.acf_to_original_format(core.build_acf(expression)),
            "(w:mon)@in-season=first",
        )
        season_support.configure_hemisphere("south")
        southern, _meta = core.next_after_expr(
            dnf, date(2026, 8, 1), default_seed=date(2026, 1, 1)
        )
        self.assertEqual(southern, date(2026, 9, 7))

    def test_season_modifiers_shift_selected_dates_across_window_boundaries(self) -> None:
        seed = date(2026, 1, 1)
        cases = (
            ("(w:mon)@in-spring=last@+7d", date(2026, 1, 1), date(2026, 6, 1)),
            ("(w:mon)@in-spring=first@-7d", date(2026, 1, 1), date(2026, 2, 23)),
            ("(w:fri)@in-winter=last@+1bd", date(2026, 7, 1), date(2027, 3, 1)),
        )
        for expression, after_date, expected in cases:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                actual, _meta = core.next_after_expr(
                    dnf, after_date, default_seed=seed
                )
                self.assertEqual(actual, expected)
                self.assertTrue(core.factor_matches_on(dnf[0][0], expected, seed))

        next_year, _meta = core.next_after_expr(
            core.validate_anchor_expr_strict("(w:mon)@in-spring=last@+7d"),
            date(2026, 6, 1),
            default_seed=seed,
        )
        self.assertEqual(next_year, date(2027, 6, 7))

    def test_season_selector_edges_modifiers_and_date_overflow(self) -> None:
        seed = date(2026, 1, 1)
        edge_cases = (
            ("(y:03-01)@in-spring=first", date(2026, 3, 1)),
            ("(y:05-31)@in-spring=first", date(2026, 5, 31)),
            ("(y:06-01)@in-summer=first", date(2026, 6, 1)),
            ("(y:08-31)@in-summer=first", date(2026, 8, 31)),
            ("(y:09-01)@in-autumn=first", date(2026, 9, 1)),
            ("(y:11-30)@in-autumn=first", date(2026, 11, 30)),
            ("(y:12-01)@in-winter=first", date(2026, 12, 1)),
            ("(y:02-28)@in-winter=first", date(2026, 2, 28)),
        )
        for expression, expected in edge_cases:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                actual, _meta = core.next_after_expr(
                    dnf, date(2026, 1, 1), default_seed=seed
                )
                self.assertEqual(actual, expected)

        crossing_cases = (
            ("(y:05-31)@in-spring=first@+1d", date(2026, 6, 1)),
            ("(y:03-01)@in-spring=first@-1d", date(2026, 2, 28)),
            ("(y:12-01)@in-winter=first@-1d", date(2026, 11, 30)),
        )
        for expression, expected in crossing_cases:
            with self.subTest(expression=expression):
                actual, _meta = core.next_after_expr(
                    core.validate_anchor_expr_strict(expression),
                    date(2026, 1, 1),
                    default_seed=seed,
                )
                self.assertEqual(actual, expected)

        self.assertEqual(
            season_support.season_bounds("spring", 9999),
            (date(9999, 3, 1), date(9999, 5, 31)),
        )
        with self.assertRaises(OverflowError):
            season_support.season_window_on_or_after("spring", date(9999, 6, 1))
        with self.assertRaisesRegex(ValueError, "between 1 and 9999"):
            position_selection.next_period_start("spring", date(9999, 4, 1))

    def test_semantic_guard_rejects_seasonally_impossible_anchor_candidates(self) -> None:
        invalid = (
            ("(y:jan)@in-spring=first", "fixed March 1 through May 31 window"),
            ("(y:q4)@in-summer=first", "fixed June 1 through August 31 window"),
            (
                "(y:jan + w:mon)@in-spring=first",
                "fixed March 1 through May 31 window",
            ),
        )
        for expression, message in invalid:
            with self.subTest(expression=expression):
                with self.assertRaises(core.ParseError) as raised:
                    core.validate_anchor_expr_strict(expression)
                self.assertIn(message, str(raised.exception))

        valid = (
            "(y:02-29)@in-winter=first",
            "(y:jan | w:mon)@in-spring=first",
            "(w/100:mon)@in-spring=first",
        )
        for expression in valid:
            with self.subTest(expression=expression):
                self.assertTrue(core.validate_anchor_expr_strict(expression))


if __name__ == "__main__":
    unittest.main()
