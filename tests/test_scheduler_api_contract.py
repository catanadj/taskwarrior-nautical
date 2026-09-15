import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
from nautical_core import scheduler_atom, scheduler_api
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.parsing.parser_models import ParseError
from nautical_core.schedule_utils import roll_apply


class SchedulerAtomContractTests(unittest.TestCase):
    def test_roll_apply_guard_fails_when_weekday_never_converges(self):
        class NonConvergingDate(date):
            def weekday(self):
                return 9

        with self.assertRaisesRegex(ParseError, "failed to reach business day"):
            roll_apply(
                NonConvergingDate(2025, 1, 1),
                {"roll": "pbd"},
                parse_error_cls=ParseError,
            )

    def test_next_for_and_rejects_a_term_without_forward_progress(self):
        with patch.object(core, "next_after_atom_with_mods", side_effect=lambda _atom, day, _seed, **_kwargs: day):
            with self.assertRaises(core.ParseError):
                core._next_for_and(
                    [{"typ": "w", "spec": "mon"}],
                    date(2025, 1, 1),
                    date(2025, 1, 1),
                )

    def test_next_for_and_recovers_after_a_transient_stall(self):
        calls = 0

        def next_after(_atom, day, _seed, **_kwargs):
            nonlocal calls
            calls += 1
            return day if calls == 1 else day + timedelta(days=1)

        with (
            patch.object(core, "next_after_atom_with_mods", side_effect=next_after),
            patch.object(core, "atom_matches_on", return_value=True),
        ):
            result = core._next_for_and(
                [{"typ": "w", "spec": "mon"}],
                date(2025, 1, 1),
                date(2025, 1, 1),
            )

        self.assertGreater(result, date(2025, 1, 1))

    def test_base_next_after_atom_weekly_monthly_yearly_and_moon(self):
        ref = date(2026, 1, 5)  # Monday
        weekly = scheduler_atom.base_next_after_atom(
            {"typ": "w", "spec": "mon"}, ref,
            expand_weekly_cached_mods=lambda spec, bd: {0},
            split_csv_tokens=lambda value: value.split(","),
            expand_monthly_cached=lambda *args: [],
            expand_yearly_cached=lambda *args: [],
            weekly_rand_pick=lambda *args, **kwargs: None,
            week_monday=lambda d: d - timedelta(days=d.weekday()),
            date_cls=date,
        )
        self.assertEqual(weekly, date(2026, 1, 12))

        monthly = scheduler_atom.base_next_after_atom(
            {"typ": "m", "spec": "last"}, date(2026, 1, 31),
            expand_weekly_cached_mods=lambda *args: set(),
            split_csv_tokens=lambda value: value.split(","),
            expand_monthly_cached=lambda token, year, month: [31] if month in (1, 3) else [],
            expand_yearly_cached=lambda *args: [],
            weekly_rand_pick=lambda *args, **kwargs: None,
            week_monday=lambda d: d - timedelta(days=d.weekday()),
            date_cls=date,
        )
        self.assertEqual(monthly, date(2026, 3, 31))

        yearly = scheduler_atom.base_next_after_atom(
            {"typ": "y", "spec": "01-01"}, date(2026, 3, 1),
            expand_weekly_cached_mods=lambda *args: set(),
            split_csv_tokens=lambda value: value.split(","),
            expand_monthly_cached=lambda *args: [],
            expand_yearly_cached=lambda spec, year: [date(year, 1, 1)],
            weekly_rand_pick=lambda *args, **kwargs: None,
            week_monday=lambda d: d - timedelta(days=d.weekday()),
            date_cls=date,
        )
        self.assertEqual(yearly, date(2027, 1, 1))

        moon = scheduler_atom.base_next_after_atom(
            {"typ": "moon", "spec": "full"}, ref,
            expand_weekly_cached_mods=lambda *args: set(),
            split_csv_tokens=lambda value: value.split(","),
            expand_monthly_cached=lambda *args: [],
            expand_yearly_cached=lambda *args: [],
            weekly_rand_pick=lambda *args, **kwargs: None,
            week_monday=lambda d: d - timedelta(days=d.weekday()),
            date_cls=date,
            resolve_moon_phase_date=lambda phase, day: date(2026, 1, 20),
        )
        self.assertEqual(moon, date(2026, 1, 20))

    def test_random_weekly_uses_injected_pick_and_identity(self):
        calls = []
        result = scheduler_atom.base_next_after_atom(
            {"typ": "w", "spec": "rand", "ival": 1, "mods": {"x": 1}},
            date(2026, 1, 5), seed_base="seed",
            expand_weekly_cached_mods=lambda *args: set(),
            split_csv_tokens=lambda value: value.split(","),
            expand_monthly_cached=lambda *args: [],
            expand_yearly_cached=lambda *args: [],
            weekly_rand_pick=lambda year, week, mods, **kwargs: calls.append(
                (year, week, kwargs["atom_identity"])
            ) or 2,
            week_monday=lambda d: d - timedelta(days=d.weekday()),
            date_cls=date,
        )
        self.assertEqual(result, date(2026, 1, 7))
        self.assertEqual(calls[0][2], '{"ival":1,"mods":{"x":1},"spec":"rand","typ":"w"}')

    def test_interval_acceptance_and_probe_advancement(self):
        weeks = lambda a, b: (b - a).days // 7
        years = lambda d: d.year
        seed = date(2024, 1, 1)
        self.assertTrue(scheduler_atom.interval_allowed_for_atom("w", 2, seed, date(2024, 1, 15), weeks_between=weeks, year_index=years))
        self.assertFalse(scheduler_atom.interval_allowed_for_atom("w", 2, seed, date(2024, 1, 8), weeks_between=weeks, year_index=years))
        # The ISO year changes at the end of December; this must use the
        # candidate's ISO year rather than its Gregorian calendar year.
        self.assertFalse(scheduler_atom.interval_allowed_for_atom("y", 2, date(2024, 12, 30), date(2026, 1, 1), weeks_between=weeks, year_index=years, spec="w1"))
        self.assertTrue(scheduler_atom.interval_allowed_for_atom("y", 2, date(2024, 12, 30), date(2028, 1, 1), weeks_between=weeks, year_index=years, spec="w1"))
        iso_probe = scheduler_atom.advance_probe_for_interval_bucket(
            "y", 2, date(2024, 12, 30), date(2026, 1, 1),
            weeks_between=weeks, year_index=years, date_cls=date, spec="w1"
        )
        self.assertEqual(iso_probe, date(2027, 1, 3))
        advanced = scheduler_atom.advance_probe_for_interval_bucket("w", 2, seed, date(2024, 1, 8), weeks_between=weeks, year_index=years, date_cls=date)
        self.assertEqual(advanced, date(2024, 1, 14))
        with self.assertRaises(OccurrenceSearchExhausted) as ctx:
            scheduler_atom.advance_probe_for_interval_bucket("y", 2, date(9999, 1, 1), date(9999, 2, 1), weeks_between=weeks, year_index=years, date_cls=date)
        self.assertEqual(ctx.exception.kind, OccurrenceSearchExhausted.DATE_LIMIT)

    def test_roll_acceptance_and_lookback(self):
        self.assertTrue(scheduler_atom.accept_roll_candidate(date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 5), "nbd"))
        self.assertFalse(scheduler_atom.accept_roll_candidate(date(2026, 1, 5), date(2026, 1, 5), date(2026, 1, 5), "nbd"))
        self.assertEqual(scheduler_atom._atom_match_lookback_days({"mods": {"day_offset": 2, "business_day_offset": 1, "roll": "next-wd"}}), 16)

    def test_atom_matches_on_uses_lookback_and_moon_filter(self):
        target = date(2026, 1, 10)
        calls = []
        matches = scheduler_atom.atom_matches_on(
            {"typ": "w", "spec": "sat"}, target, target,
            next_after_atom_with_mods=lambda atom, ref, seed, **kwargs: calls.append(ref) or (
                target if ref == target - timedelta(days=1) else target - timedelta(days=1)
            ),
        )
        self.assertTrue(matches)
        self.assertEqual(calls, [target - timedelta(days=1)])
        self.assertFalse(scheduler_atom.atom_matches_on(
            {"typ": "w", "spec": "sat", "mods": {"moon": "full"}}, target, target,
            next_after_atom_with_mods=lambda *args, **kwargs: target,
            moon_phase_matches_date=lambda phase, day: False,
        ))

    def test_atom_matches_on_rejects_nonmatching_concrete_candidate(self):
        target = date(2026, 1, 10)
        self.assertFalse(scheduler_atom.atom_matches_on(
            {"typ": "w", "spec": "sat"}, target, target,
            next_after_atom_with_mods=lambda atom, ref, seed, **kwargs: target - timedelta(days=1),
        ))


class SchedulerApiDelegationTests(unittest.TestCase):
    def test_base_wrapper_delegates_callbacks_and_calendar(self):
        seen = {}
        calendars = []
        atom_module = SimpleNamespace(base_next_after_atom=lambda atom, ref, **kwargs: seen.update(kwargs) or date(2026, 2, 1))
        module = SimpleNamespace(
            _scheduler_atom=atom_module,
            expand_weekly_cached_mods="weekly",
            _split_csv_tokens="split",
            _with_business_calendar=lambda fn, cal: (calendars.append(cal) or fn),
            expand_monthly_cached="monthly",
            expand_yearly_cached="yearly",
            _weekly_rand_pick="random",
            _week_monday="monday",
            _resolve_moon_phase_date="moon",
        )
        calendar = object()
        self.assertEqual(scheduler_api._base_next_after_atom_impl(module, {"typ": "w"}, date(2026, 1, 1), business_calendar=calendar), date(2026, 2, 1))
        self.assertEqual(seen["expand_weekly_cached_mods"], "weekly")
        self.assertEqual(seen["date_cls"].__name__, "date")
        self.assertEqual(calendars, [calendar, calendar])

    def test_modified_atom_wrapper_delegates_all_contract_callbacks(self):
        seen = {}
        atom_module = SimpleNamespace(
            next_after_atom_with_mods=lambda *args, **kwargs: seen.update(kwargs) or date(2026, 2, 2),
        )
        calendar = SimpleNamespace(is_business_day=lambda day: True)
        module = SimpleNamespace(
            _business_calendar=SimpleNamespace(effective_business_calendar=lambda value: calendar),
            _with_business_calendar=lambda fn, cal: fn,
            base_next_after_atom="base",
            _monthly_align_base_for_interval="monthly",
            roll_apply="roll",
            apply_day_offset="offset",
            _scheduler_atom=atom_module,
            _active_mod_keys="active",
            MAX_ANCHOR_ITER=12,
            _warn_once_per_day="warn",
            os="os",
            _resolve_moon_phase_date="moon",
            _moon_phase_matches_date="phase",
        )
        self.assertEqual(
            scheduler_api._next_after_atom_with_mods_impl(
                module, {"typ": "w"}, date(2026, 1, 1), date(2026, 1, 1), business_calendar="custom"
            ),
            date(2026, 2, 2),
        )
        self.assertIs(seen["is_business_day"], calendar.is_business_day)
        self.assertEqual(seen["max_anchor_iter"], 12)

    def test_interval_wrapper_preserves_typed_exhaustion(self):
        failure = OccurrenceSearchExhausted("test", reference=date(2026, 1, 1), limit=2, kind=OccurrenceSearchExhausted.DATE_LIMIT)
        module = SimpleNamespace(
            _scheduler_atom=SimpleNamespace(advance_probe_for_interval_bucket=lambda *args, **kwargs: (_ for _ in ()).throw(failure)),
            _weeks_between=lambda a, b: (b - a).days // 7,
            _year_index=lambda d: d.year,
        )
        with self.assertRaises(OccurrenceSearchExhausted) as ctx:
            scheduler_api._advance_probe_for_interval_bucket(module, "y", 2, date(2026, 1, 1), date(2026, 2, 1))
        self.assertIs(ctx.exception, failure)


class SchedulerExpressionContractTests(unittest.TestCase):
    def test_complex_weekday_union_projects_the_next_day(self) -> None:
        expression = " | ".join(
            f"w:{day}" for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
        )
        occurrence, _meta = core.next_after_expr(
            core.validate_anchor_expr_strict(expression), date(2024, 1, 1)
        )
        self.assertEqual(occurrence, date(2024, 1, 2))

    def test_modified_atom_interval_roll_and_offset_characterization(self) -> None:
        cases = (
            ("m/2:31", date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 31)),
            ("m/2:31", date(2024, 2, 1), date(2024, 1, 1), date(2024, 5, 31)),
            ("m/2:31", date(2024, 3, 31), date(2024, 1, 1), date(2024, 5, 31)),
            ("w/2:mon", date(2024, 12, 9), date(2024, 12, 9), date(2024, 12, 23)),
            ("y/2:06-15", date(2024, 1, 1), date(2024, 1, 1), date(2024, 6, 15)),
            ("m:3@bd", date(2026, 1, 1), date(2026, 1, 1), date(2026, 2, 3)),
            ("m:1@nbd@+1d", date(2024, 8, 31), date(2024, 8, 31), date(2024, 9, 3)),
            ("m:-1@pbd@-2bd", date(2026, 1, 1), date(2026, 1, 1), date(2026, 1, 28)),
            ("y:04-24@+1bd", date(2026, 1, 1), date(2026, 1, 1), date(2026, 4, 27)),
            ("y:04-27@-1bd", date(2026, 1, 1), date(2026, 1, 1), date(2026, 4, 24)),
            ("y:04-25@+1d@+1bd", date(2026, 1, 1), date(2026, 1, 1), date(2026, 4, 27)),
            ("m:15@nw", date(2024, 6, 14), date(2024, 6, 14), date(2024, 6, 14)),
        )
        for expression, reference, seed, expected in cases:
            with self.subTest(expression=expression, reference=reference):
                atom = core.validate_anchor_expr_strict(expression)[0][0]
                self.assertEqual(
                    core.next_after_atom_with_mods(atom, reference, seed), expected
                )

    def test_bucket_signature_requires_one_compatible_monthly_range(self) -> None:
        term = [
            {"typ": "m", "spec": "1..7", "mods": {"t": (9, 30)}},
            {"typ": "m", "spec": "rand", "mods": {"bd": True}},
        ]
        self.assertEqual(
            core._rand_bucket_signature(term), (1, "09:30", True, "1–7")
        )
        self.assertIsNone(
            core._rand_bucket_signature(term + [{"typ": "w", "spec": "mon"}])
        )
        bad_range = [
            {"typ": "m", "spec": "1..7"},
            {"typ": "m", "spec": "8..14"},
            {"typ": "m", "spec": "rand"},
        ]
        self.assertIsNone(core._rand_bucket_signature(bad_range))

    def test_time_resolver_keeps_time_after_positive_day_offset(self) -> None:
        dnf = core.validate_anchor_expr_strict("y:04-25@+10d@t=12:00")
        self.assertEqual(
            core.pick_hhmm_from_dnf_for_date(
                dnf, date(2026, 5, 5), date(2026, 4, 12)
            ),
            (12, 0),
        )

    def test_atom_matcher_recognizes_positive_day_offset_result(self) -> None:
        atom = core.validate_anchor_expr_strict("y:04-25@+10d")[0][0]
        self.assertTrue(
            core.atom_matches_on(atom, date(2026, 5, 5), date(2026, 4, 12))
        )

    def test_month_name_window_filters_monthly_weekday_occurrences(self) -> None:
        dnf = core.validate_anchor_expr_strict("m:1st-mon + y:mar..sep")
        seed = date(2026, 1, 1)
        cursor = seed
        occurrences = []
        for _ in range(6):
            cursor, _meta = core.next_after_expr(
                dnf, cursor, default_seed=seed, seed_base="month-name-window-contract"
            )
            self.assertIsNotNone(cursor)
            occurrences.append(cursor)
        self.assertTrue(all(3 <= occurrence.month <= 9 for occurrence in occurrences))

    def test_grouped_weekly_day_ranges_respect_two_week_interval(self) -> None:
        dnf = core.validate_anchor_expr_strict("w/2:mon..tue,thu..sat")
        seed = date(2026, 1, 1)
        cursor = seed
        occurrences = []
        for _ in range(8):
            cursor, _meta = core.next_after_expr(
                dnf, cursor, default_seed=seed, seed_base="weekly-range-interval-contract"
            )
            self.assertIsNotNone(cursor)
            occurrences.append(cursor)
        self.assertTrue(all(item.weekday() in {0, 1, 3, 4, 5} for item in occurrences))
        weeks = []
        for item in occurrences:
            iso = item.isocalendar()
            weeks.append(date.fromisocalendar(iso.year, iso.week, 1))
        for previous, current in zip(weeks, weeks[1:]):
            self.assertIn((current - previous).days // 7, (0, 2))

    def test_large_random_intervals_scale_and_fail_with_typed_exhaustion(self) -> None:
        seed = date(2026, 1, 1)
        cases = (
            ("m/2:rand + y:01-01..12-31", "month", 2),
            ("m/25:rand", "month", 25),
            ("y/11:rand", "year", 11),
        )
        for expression, unit, interval in cases:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                first, _meta = core.next_after_expr(
                    dnf, seed, default_seed=seed, seed_base="large-interval"
                )
                second, _meta = core.next_after_expr(
                    dnf, first, default_seed=seed, seed_base="large-interval"
                )
                if unit == "month":
                    offset = (second.year - seed.year) * 12 + second.month - seed.month
                else:
                    offset = second.year - seed.year
                self.assertEqual(offset % interval, 0)

        for expression in ("m/25:rand", "y/11:rand"):
            with self.subTest(exhaustion=expression), self.assertRaises(
                core.OccurrenceSearchExhausted
            ):
                core.next_after_expr(
                    core.validate_anchor_expr_strict(expression),
                    date(9999, 12, 31),
                    default_seed=seed,
                    seed_base="large-interval-exhaustion",
                )

    def test_large_weekly_random_interval_scales_and_exhausts_safely(self) -> None:
        seed = date(2026, 1, 1)
        dnf = core.validate_anchor_expr_strict("w/100:2rand")
        first, _meta = core.next_after_expr(
            dnf, seed, default_seed=seed, seed_base="weekly-large-interval"
        )
        second, _meta = core.next_after_expr(
            dnf, first, default_seed=seed, seed_base="weekly-large-interval"
        )
        seed_monday = seed - timedelta(days=seed.weekday())
        second_monday = second - timedelta(days=second.weekday())
        week_offset = (second_monday - seed_monday).days // 7
        self.assertEqual(week_offset % 100, 0)

        with self.assertRaises(core.OccurrenceSearchExhausted):
            core.next_after_expr(
                dnf,
                date(9999, 12, 31),
                default_seed=seed,
                seed_base="weekly-large-interval-exhaustion",
            )
    def test_or_and_expression_matrix_schedules_expected_dates(self) -> None:
        cases = (
            ("w:mon | w:fri", date(2024, 12, 11), date(2024, 12, 13)),
            ("m:1 | m:15", date(2024, 12, 11), date(2024, 12, 15)),
            ("w:mon + m:1", date(2024, 12, 1), date(2025, 9, 1)),
            ("w:fri + m:13", date(2024, 12, 1), date(2024, 12, 13)),
            (
                "(w:mon + y:01-01..01-31) | (w:fri + y:02-01..02-28)",
                date(2024, 1, 1),
                date(2024, 1, 8),
            ),
        )
        for expression, reference, expected in cases:
            with self.subTest(expression=expression):
                occurrence, _meta = core.next_after_expr(
                    core.validate_anchor_expr_strict(expression), reference
                )
                self.assertEqual(occurrence, expected)

    def test_weekly_monthly_and_yearly_intervals_produce_expected_dates(self) -> None:
        cases = (
            ("w/2:mon", date(2024, 12, 9), (date(2024, 12, 23), date(2025, 1, 6), date(2025, 1, 20))),
            ("m/2:15", date(2024, 1, 1), (date(2024, 1, 15), date(2024, 3, 15), date(2024, 5, 15))),
            ("m/3:-1", date(2024, 1, 1), (date(2024, 1, 31), date(2024, 4, 30), date(2024, 7, 31))),
            ("y/2:06-15", date(2024, 1, 1), (date(2024, 6, 15), date(2026, 6, 15), date(2028, 6, 15))),
        )
        for expression, seed, expected_dates in cases:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                cursor = seed
                actual = []
                for _ in expected_dates:
                    occurrence, _meta = core.next_after_expr(
                        dnf, cursor, default_seed=seed
                    )
                    self.assertIsNotNone(occurrence)
                    actual.append(occurrence)
                    cursor = occurrence + timedelta(days=1)
                self.assertEqual(tuple(actual), expected_dates)

    def test_next_after_expression_matches_anchor_date_boundaries(self) -> None:
        cases = (
            ("w:mon", date(2024, 12, 11), date(2024, 12, 16)),
            ("w:mon,fri", date(2024, 12, 11), date(2024, 12, 13)),
            ("m:15", date(2024, 12, 11), date(2024, 12, 15)),
            ("m:15", date(2024, 12, 20), date(2025, 1, 15)),
            ("m:-1", date(2024, 12, 11), date(2024, 12, 31)),
            ("m:1", date(2024, 12, 31), date(2025, 1, 1)),
            ("y:12-25", date(2024, 12, 11), date(2024, 12, 25)),
            ("y:12-25", date(2024, 12, 26), date(2025, 12, 25)),
        )
        for expression, reference, expected in cases:
            with self.subTest(expression=expression, reference=reference):
                occurrence, _meta = core.next_after_expr(
                    core.validate_anchor_expr_strict(expression), reference
                )
                self.assertEqual(occurrence, expected)

    def test_next_after_expression_handles_leap_month_and_iso_week_edges(self) -> None:
        cases = (
            ("y:02-29", date(2023, 1, 1), date(2024, 2, 29)),
            ("y:02-29", date(2024, 2, 29), date(2028, 2, 29)),
            ("m:31", date(2024, 2, 1), date(2024, 3, 31)),
            ("m:-31", date(2024, 2, 1), date(2024, 3, 1)),
            ("y:12-31", date(2024, 12, 31), date(2025, 12, 31)),
            ("y:01-01", date(2024, 12, 31), date(2025, 1, 1)),
            ("w/2:mon", date(2024, 12, 30), date(2025, 1, 13)),
        )
        for expression, reference, expected in cases:
            with self.subTest(expression=expression, reference=reference):
                occurrence, _meta = core.next_after_expr(
                    core.validate_anchor_expr_strict(expression), reference
                )
                self.assertEqual(occurrence, expected)

    def test_random_anchor_sequences_are_repeatable_for_same_seed(self) -> None:
        expressions = (
            "w:rand",
            "m:rand",
            "m:rand@bd",
            "m:1..10 + m:rand",
            "y:07-rand",
            "y:rand-07",
            "y:rand",
        )
        start = date(2024, 1, 1)
        for expression in expressions:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                sequences = []
                for _ in range(2):
                    cursor = start
                    sequence = []
                    for _ in range(3):
                        occurrence, _meta = core.next_after_expr(
                            dnf, cursor, seed_base="test_seed"
                        )
                        sequence.append(occurrence)
                        cursor = occurrence + timedelta(days=1)
                    sequences.append(sequence)
                self.assertEqual(sequences[0], sequences[1])

    def test_weekly_and_monthly_random_draws_are_chain_scoped(self) -> None:
        weekly = core.parse_anchor_expr_to_dnf_cached("w:rand")
        weekly_start = date(2026, 6, 7)

        def weekly_pick(seed: str, reference: date = weekly_start) -> date:
            occurrence, _meta = core.next_after_expr(
                weekly,
                reference,
                default_seed=weekly_start,
                seed_base=seed,
            )
            self.assertIsNotNone(occurrence)
            return occurrence

        first_week_picks = [weekly_pick(f"weekly-chain-{index}") for index in range(64)]
        self.assertGreaterEqual(len({day.weekday() for day in first_week_picks}), 5)
        self.assertEqual(weekly_pick("weekly-replay"), weekly_pick("weekly-replay"))

        cursor = weekly_start
        period_weekdays = []
        for _ in range(24):
            cursor = weekly_pick("weekly-periods", cursor)
            period_weekdays.append(cursor.weekday())
        self.assertGreaterEqual(len(set(period_weekdays)), 5)

        monthly = core.parse_anchor_expr_to_dnf_cached("m:rand + y:apr")
        monthly_start = date(2026, 3, 31)

        def monthly_pick(seed: str) -> date:
            occurrence, _meta = core.next_after_expr(
                monthly,
                monthly_start,
                default_seed=monthly_start,
                seed_base=seed,
            )
            self.assertIsNotNone(occurrence)
            self.assertEqual(occurrence.month, 4)
            return occurrence

        monthly_picks = [monthly_pick(f"monthly-intersection-{index}") for index in range(48)]
        self.assertGreaterEqual(len(set(monthly_picks)), 16)
        self.assertEqual(monthly_pick("monthly-replay"), monthly_pick("monthly-replay"))

    def test_random_anchor_forms_replay_and_vary_across_chain_identities(self) -> None:
        cases = (
            (
                "w:rand",
                date(2026, 6, 7),
                lambda day: day.weekday(),
                lambda _day: True,
                5,
            ),
            (
                "m:rand",
                date(2026, 5, 31),
                lambda day: day.day,
                lambda day: (day.year, day.month) == (2026, 6),
                16,
            ),
            (
                "y:rand",
                date(2025, 12, 31),
                lambda day: (day.month, day.day),
                lambda day: day.year == 2026,
                36,
            ),
            (
                "y:rand + y:apr,jul,oct",
                date(2025, 12, 31),
                lambda day: (day.month, day.day),
                lambda day: day.year == 2026 and day.month in {4, 7, 10},
                32,
            ),
            (
                "m:rand@bd",
                date(2026, 5, 31),
                lambda day: day.day,
                lambda day: (day.year, day.month) == (2026, 6) and day.weekday() < 5,
                14,
            ),
        )
        for expression, start, key, valid, min_unique in cases:
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                picks = []
                for index in range(64):
                    seed_base = f"cross-chain-{expression}-{index}"
                    first, _meta = core.next_after_expr(
                        dnf,
                        start,
                        default_seed=start,
                        seed_base=seed_base,
                    )
                    replay, _meta = core.next_after_expr(
                        dnf,
                        start,
                        default_seed=start,
                        seed_base=seed_base,
                    )
                    self.assertEqual(first, replay)
                    self.assertTrue(valid(first), f"{expression} selected {first}")
                    picks.append(key(first))
                self.assertGreaterEqual(len(set(picks)), min_unique)

    def test_random_weekday_filters_group_as_one_candidate_pool(self) -> None:
        start = date(2025, 12, 31)
        for expression in ("m:rand + w:mon,sat", "w:mon,sat + m:rand"):
            with self.subTest(expression=expression):
                dnf = core.validate_anchor_expr_strict(expression)
                self.assertEqual(len(dnf), 1)
                self.assertEqual(
                    core.describe_anchor_expr(expression),
                    "one random Monday or Saturday each month",
                )
                cursor = start
                picks = []
                for _ in range(12):
                    cursor, _meta = core.next_after_expr(
                        dnf,
                        cursor,
                        default_seed=start,
                        seed_base=f"grouped-weekday-{expression}",
                    )
                    self.assertIn(cursor.weekday(), {0, 5})
                    picks.append(cursor)
                self.assertEqual(len({(pick.year, pick.month) for pick in picks}), len(picks))

        yearly = core.validate_anchor_expr_strict("y:rand + w:mon,sat")
        self.assertEqual(len(yearly), 1)
        self.assertEqual(
            core.describe_anchor_expr("y:rand + w:mon,sat"),
            "one random Monday or Saturday each year",
        )

    def test_explicit_random_weekday_or_keeps_one_draw_per_branch(self) -> None:
        expression = "m:rand + (w:mon | w:sat)"
        dnf = core.validate_anchor_expr_strict(expression)
        self.assertEqual(len(dnf), 2)
        start = date(2025, 12, 31)
        current = start
        selected = []
        for _ in range(6):
            current, _metadata = core.next_after_expr(
                dnf,
                current,
                default_seed=start,
                seed_base="explicit-random-weekday-or",
            )
            selected.append(current)

        by_month = {}
        for day in selected:
            by_month.setdefault((day.year, day.month), []).append(day)
        self.assertEqual(set(by_month), {(2026, 1), (2026, 2), (2026, 3)})
        self.assertTrue(all(len(days) == 2 for days in by_month.values()))
        self.assertTrue(all({day.weekday() for day in days} == {0, 5} for days in by_month.values()))

    def test_business_day_modifiers_produce_expected_calendar_dates(self) -> None:
        cases = (
            ("m:15@bd", date(2024, 1, 14), date(2024, 1, 15)),
            ("m:15@bd", date(2024, 6, 14), date(2024, 7, 15)),
            ("m:-1@pbd", date(2024, 3, 28), date(2024, 3, 29)),
            ("m:1@nbd", date(2024, 6, 28), date(2024, 7, 1)),
            ("m:1@nbd", date(2024, 8, 31), date(2024, 9, 2)),
            ("m:15@nw", date(2024, 6, 14), date(2024, 7, 15)),
            ("m:15@nw", date(2024, 9, 14), date(2024, 9, 16)),
        )
        for expression, reference, expected in cases:
            with self.subTest(expression=expression, reference=reference):
                dnf = core.validate_anchor_expr_strict(expression)
                occurrence, _meta = core.next_after_expr(dnf, reference)
                self.assertEqual(occurrence, expected)

        skip_dnf = core.validate_anchor_expr_strict("m:3@bd")
        self.assertEqual(core.next_after_expr(skip_dnf, date(2026, 1, 1))[0], date(2026, 2, 3))

    def test_yearly_month_aliases_and_ranges_stay_within_their_months(self) -> None:
        seed = date(2026, 1, 1)
        for expression in ("y:apr", "y:04", "y:jul", "y:07"):
            with self.subTest(expression=expression):
                occurrence, _meta = core.next_after_expr(
                    core.validate_anchor_expr_strict(expression),
                    seed,
                    default_seed=seed,
                )
                self.assertEqual(occurrence.month, 4 if expression.endswith("apr") or expression.endswith("04") else 7)

        expression = "y:jan..jun + m:rand"
        dnf = core.validate_anchor_expr_strict(expression)
        cursor = seed
        occurrences = []
        for _ in range(8):
            cursor, _meta = core.next_after_expr(
                dnf,
                cursor,
                default_seed=seed,
                seed_base="month-window-contract",
            )
            self.assertIsNotNone(cursor)
            occurrences.append(cursor)
        self.assertTrue(all(1 <= occurrence.month <= 6 for occurrence in occurrences))

    def test_monthly_interval_with_fifth_weekday_only_returns_valid_month_hits(self) -> None:
        dnf = core.validate_anchor_expr_strict("m/2:5th-mon")
        seed = date(2026, 1, 1)
        cursor = seed
        occurrences = []
        for _ in range(8):
            occurrence, _meta = core.next_after_expr(
                dnf,
                cursor,
                default_seed=seed,
                seed_base="monthly-fifth-weekday-contract",
            )
            self.assertIsNotNone(occurrence)
            occurrences.append(occurrence)
            cursor = occurrence

        for occurrence in occurrences:
            mondays = [
                day
                for day in range(1, occurrence.day + 1)
                if date(occurrence.year, occurrence.month, day).weekday() == 0
            ]
            self.assertEqual(occurrence.weekday(), 0)
            self.assertEqual(len(mondays), 5)

    def test_weekday_rolls_skip_same_day_and_cross_year_correctly(self) -> None:
        cases = (
            ("y:12-31@next-thu", date(2026, 12, 1), date(2027, 1, 7)),
            ("y:12-31@prev-thu", date(2026, 12, 1), date(2026, 12, 24)),
            ("y:12-31@next-thu", date(2026, 12, 31), date(2027, 1, 7)),
        )
        for expression, reference, expected in cases:
            with self.subTest(expression=expression, reference=reference):
                dnf = core.validate_anchor_expr_strict(expression)
                occurrence, _meta = core.next_after_expr(
                    dnf,
                    reference,
                    default_seed=reference,
                    seed_base="weekday-roll-contract",
                )
                self.assertEqual(occurrence, expected)

    def test_weekly_multi_day_interval_respects_iso_week_buckets(self) -> None:
        dnf = core.validate_anchor_expr_strict("w/2:mon,thu")
        seed = date(2026, 1, 1)
        cursor = seed
        occurrences = []
        for _ in range(8):
            occurrence, _meta = core.next_after_expr(
                dnf,
                cursor,
                default_seed=seed,
                seed_base="weekly-two-week-contract",
            )
            self.assertIsNotNone(occurrence)
            occurrences.append(occurrence)
            cursor = occurrence

        self.assertTrue(all(day.weekday() in (0, 3) for day in occurrences))
        weeks = [date.fromisocalendar(*day.isocalendar()[:2], 1) for day in occurrences]
        for previous, current in zip(weeks, weeks[1:]):
            week_gap = (current - previous).days // 7
            self.assertIn(week_gap, (0, 2))

    def test_random_weekly_interval_keeps_a_stable_iso_week_bucket(self) -> None:
        dnf = core.validate_anchor_expr_strict("w/4:rand")
        seed = date(2026, 1, 1)
        cursor = seed
        occurrences = []
        for _ in range(8):
            occurrence, _meta = core.next_after_expr(
                dnf,
                cursor,
                default_seed=seed,
                seed_base="random-weekly-interval-contract",
            )
            self.assertIsNotNone(occurrence)
            occurrences.append(occurrence)
            cursor = occurrence

        buckets = [
            (day.isocalendar().year * 53 + day.isocalendar().week) % 4
            for day in occurrences
        ]
        self.assertTrue(all(bucket == buckets[0] for bucket in buckets))

    def test_random_monthly_candidates_stay_inside_yearly_window(self) -> None:
        dnf = core.validate_anchor_expr_strict("y:04-20..05-15 + m:rand")
        seed = date(2026, 1, 1)
        cursor = seed
        occurrences = []
        for _ in range(8):
            occurrence, _meta = core.next_after_expr(
                dnf,
                cursor,
                default_seed=seed,
                seed_base="random-monthly-window-contract",
            )
            self.assertIsNotNone(occurrence)
            occurrences.append(occurrence)
            cursor = occurrence

        for occurrence in occurrences:
            self.assertLessEqual(date(occurrence.year, 4, 20), occurrence)
            self.assertLessEqual(occurrence, date(occurrence.year, 5, 15))


if __name__ == "__main__":
    unittest.main()
