import os
import unittest
from datetime import date, timedelta
from types import SimpleNamespace

from nautical_core import scheduler_atom, scheduler_api
from nautical_core.scheduler_models import OccurrenceSearchExhausted


class SchedulerAtomContractTests(unittest.TestCase):
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
        self.assertTrue(scheduler_atom.interval_allowed_for_atom("y", 2, date(2024, 12, 30), date(2026, 1, 1), weeks_between=weeks, year_index=years, spec="w:1"))
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
            next_after_atom_with_mods=lambda atom, ref, seed, **kwargs: calls.append(ref) or target,
        )
        self.assertTrue(matches)
        self.assertEqual(calls, [target - timedelta(days=1)])
        self.assertFalse(scheduler_atom.atom_matches_on(
            {"typ": "w", "spec": "sat", "mods": {"moon": "full"}}, target, target,
            next_after_atom_with_mods=lambda *args, **kwargs: target,
            moon_phase_matches_date=lambda phase, day: False,
        ))


class SchedulerApiDelegationTests(unittest.TestCase):
    def test_base_wrapper_delegates_callbacks_and_calendar(self):
        seen = {}
        atom_module = SimpleNamespace(base_next_after_atom=lambda atom, ref, **kwargs: seen.update(kwargs) or date(2026, 2, 1))
        module = SimpleNamespace(
            _scheduler_atom=atom_module,
            expand_weekly_cached_mods="weekly",
            _split_csv_tokens="split",
            _with_business_calendar=lambda fn, cal: fn,
            expand_monthly_cached="monthly",
            expand_yearly_cached="yearly",
            _weekly_rand_pick="random",
            _week_monday="monday",
            _resolve_moon_phase_date="moon",
        )
        self.assertEqual(scheduler_api._base_next_after_atom_impl(module, {"typ": "w"}, date(2026, 1, 1)), date(2026, 2, 1))
        self.assertEqual(seen["expand_weekly_cached_mods"], "weekly")
        self.assertEqual(seen["date_cls"].__name__, "date")

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


if __name__ == "__main__":
    unittest.main()
