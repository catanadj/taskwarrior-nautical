"""Direct contracts for astronomy configuration and moon-phase semantics."""

import os
import unittest
from datetime import date, timedelta
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
from nautical_core import add_anchor_compute
from nautical_core import astronomy
from nautical_core import scheduler_expr
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.task_codec import DEFAULT_TASK_CODEC

try:
    from astral import Observer, moon, sun  # noqa: F401
except ImportError:
    ASTRAL_AVAILABLE = False
else:
    ASTRAL_AVAILABLE = True


class AstronomyContractTests(unittest.TestCase):
    def test_unavailable_astronomy_candidate_advances_to_next_candidate_date(self) -> None:
        first = date(2027, 7, 1)

        class FakeCore:
            MAX_ANCHOR_ITER = 4
            _scheduler_api = SimpleNamespace()
            factor_matches_on = staticmethod(lambda *_args, **_kwargs: True)
            dnf_has_counted_random = staticmethod(lambda _dnf: False)
            _import_sibling = staticmethod(core._import_sibling)
            build_local_datetime = staticmethod(
                lambda day, hhmm: datetime(
                    day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=timezone.utc
                )
            )
            to_local = staticmethod(lambda value: value)

        FakeCore._scheduler_api.factor_matches_on = FakeCore.factor_matches_on
        FakeCore._scheduler_api.dnf_has_counted_random = FakeCore.dnf_has_counted_random

        def resolve_slots(_mods, day):
            if day == first:
                raise astronomy.AstronomyEventUnavailableError(
                    "astronomical event 'moonrise' is unavailable on 2027-07-01 at home"
                )
            return [(6, 10)]

        with patch.object(
            add_anchor_compute,
            "anchor_step_once_with_omit",
            return_value=date(2028, 7, 1),
        ):
            actual = add_anchor_compute.anchor_next_occurrence_after_local_dt(
                [[{"mods": {"t": "moonrise"}}]],
                datetime(2027, 7, 1, 12, 0, tzinfo=timezone.utc),
                (9, 0),
                first,
                first,
                core=FakeCore(),
                norm_t_mod=lambda _value: [],
                resolve_time_slots=resolve_slots,
            )

        self.assertEqual(actual, datetime(2027, 7, 2, 6, 10, tzinfo=timezone.utc))

    def test_unavailable_moonrise_advances_within_phase_month_intersection(self) -> None:
        first = date(2027, 7, 1)

        class FakeCore:
            MAX_ANCHOR_ITER = 4
            _scheduler_api = SimpleNamespace()
            dnf_has_counted_random = staticmethod(lambda _dnf: False)
            _import_sibling = staticmethod(core._import_sibling)
            build_local_datetime = staticmethod(
                lambda day, hhmm: datetime(
                    day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=timezone.utc
                )
            )
            to_local = staticmethod(lambda value: value)

            @staticmethod
            def factor_matches_on(atom, day, *_args, **_kwargs):
                typ = atom.get("typ")
                if typ == "y":
                    return day.month == 7
                if typ == "moon":
                    return day.month == 7 and day.day <= 7
                return True

        FakeCore._scheduler_api.factor_matches_on = FakeCore.factor_matches_on
        FakeCore._scheduler_api.dnf_has_counted_random = FakeCore.dnf_has_counted_random

        def resolve_slots(_mods, day):
            if day == first:
                raise astronomy.AstronomyEventUnavailableError(
                    "astronomical event 'moonrise' is unavailable on 2027-07-01 at home"
                )
            return [(3, 4)]

        dnf = core.validate_anchor_expr_strict("(moon:last-quarter + y:jul)@t=moonrise")
        with patch.object(add_anchor_compute, "anchor_step_once_with_omit", return_value=None):
            actual = add_anchor_compute.anchor_next_occurrence_after_local_dt(
                dnf,
                datetime(2027, 7, 1, 0, 0, tzinfo=timezone.utc),
                (9, 0),
                first,
                first,
                core=FakeCore(),
                norm_t_mod=lambda _value: [],
                resolve_time_slots=resolve_slots,
            )

        self.assertEqual(actual, datetime(2027, 7, 2, 3, 4, tzinfo=timezone.utc))

    def test_unavailable_previous_astronomy_date_does_not_block_next_occurrence(self) -> None:
        first = date(2027, 7, 1)

        class FakeCore:
            MAX_ANCHOR_ITER = 4
            _scheduler_api = SimpleNamespace()
            dnf_has_counted_random = staticmethod(lambda _dnf: False)
            _import_sibling = staticmethod(core._import_sibling)
            build_local_datetime = staticmethod(
                lambda day, hhmm: datetime(
                    day.year, day.month, day.day, hhmm[0], hhmm[1], tzinfo=timezone.utc
                )
            )
            to_local = staticmethod(lambda value: value)

            @staticmethod
            def factor_matches_on(atom, day, *_args, **_kwargs):
                typ = atom.get("typ")
                if typ == "y":
                    return day.month == 7
                if typ == "moon":
                    return day.month == 7 and day.day <= 3
                return True

        FakeCore._scheduler_api.factor_matches_on = FakeCore.factor_matches_on
        FakeCore._scheduler_api.dnf_has_counted_random = FakeCore.dnf_has_counted_random

        def resolve_slots(_mods, day):
            if day == first:
                raise astronomy.AstronomyEventUnavailableError(
                    "astronomical event 'moonrise' is unavailable on 2027-07-01 at home"
                )
            return [(3, 4)]

        dnf = core.validate_anchor_expr_strict("(moon:last-quarter + y:jul)@t=moonrise")
        with patch.object(
            add_anchor_compute,
            "anchor_step_once_with_omit",
            return_value=date(2028, 7, 1),
        ):
            actual = add_anchor_compute.anchor_next_occurrence_after_local_dt(
                dnf,
                datetime(2027, 7, 2, 4, 0, tzinfo=timezone.utc),
                (9, 0),
                first,
                first,
                core=FakeCore(),
                norm_t_mod=lambda _value: [],
                resolve_time_slots=resolve_slots,
            )

        self.assertEqual(actual, datetime(2028, 7, 1, 3, 4, tzinfo=timezone.utc))

    def test_provider_failure_is_actionable_and_reconcile_fails_closed(self) -> None:
        message = astronomy.scheduling_error_message(
            astronomy.AstronomyUnavailableError(
                "moon phase anchors require astral"
            )
        )
        self.assertIn("Install astral", message)
        self.assertIn("moon-based recurrence", message)

        foreign_error = type(
            "AstronomyUnavailableError", (RuntimeError,), {}
        )("provider missing")
        self.assertTrue(astronomy.is_astronomy_error(foreign_error))
        self.assertIn("Install astral", astronomy.scheduling_error_message(foreign_error))

        from nautical_core.chain_integrity_lifecycle import plan_recovery_decision

        class FailingGeneration:
            core = core

            @staticmethod
            def safe_parse_datetime(value):
                return core.parse_dt_any(value), None

            @staticmethod
            def compute_anchor_child_due(_parent):
                raise astronomy.AstronomyUnavailableError(
                    "moon phase anchors require astral"
                )

        parent = {
            "uuid": "11111111-0000-4000-8000-000000000001",
            "status": "completed",
            "chain": "on",
            "chainID": "11111111",
            "link": 1,
            "anchor": "moon:full",
        }
        observation = DEFAULT_TASK_CODEC.decode_row(
            parent, source_query="astronomy-failure-contract"
        )
        result = plan_recovery_decision(
            observation,
            existing_children=[],
            hook=SimpleNamespace(core=core),
            generation=FailingGeneration(),
        )

        self.assertEqual(result.status.value, "error")
        self.assertIn("Astronomy provider unavailable", result.reason)

    def test_preflight_distinguishes_disabled_invalid_and_healthy_profiles(self) -> None:
        self.assertEqual(astronomy.preflight({}).get("status"), "not_configured")

        invalid = astronomy.preflight(
            {
                "default_location": "home",
                "locations": {
                    "home": {
                        "latitude": "bad",
                        "longitude": 0,
                        "timezone": "UTC",
                    }
                },
            }
        )
        self.assertEqual(invalid.get("status"), "error")

        with (
            patch.object(astronomy, "_observer", return_value=("home", object(), "UTC")),
            patch.object(
                astronomy,
                "resolve_event",
                return_value=datetime(2026, 7, 31, 6, 0, tzinfo=timezone.utc),
            ),
        ):
            healthy = astronomy.preflight(
                {
                    "default_location": "home",
                    "locations": {
                        "home": {
                            "latitude": 1,
                            "longitude": 2,
                            "timezone": "UTC",
                        }
                    },
                },
                reference_day=date(2026, 7, 31),
            )

        self.assertEqual(healthy.get("status"), "ok")
        self.assertEqual(healthy.get("event"), "sunrise")

    def test_none_from_real_event_cache_becomes_actionable_unavailable_error(self) -> None:
        config = {
            "default_location": "home",
            "locations": {
                "home": {
                    "latitude": 45,
                    "longitude": 27,
                    "timezone": "UTC",
                }
            },
        }
        with patch.object(astronomy, "_resolve_event_cached", return_value=None):
            with self.assertRaises(astronomy.AstronomyEventUnavailableError) as raised:
                astronomy.resolve_event("moonrise", date(2027, 7, 1), config=config)

        self.assertIn("moonrise", str(raised.exception))

    def test_moon_intersection_exhaustion_is_typed_not_a_synthetic_date(self) -> None:
        reference = date(2026, 7, 1)
        term = [
            {"typ": "moon", "spec": "full"},
            {"typ": "w", "spec": "fri"},
        ]

        with self.assertRaises(OccurrenceSearchExhausted) as raised:
            scheduler_expr.next_after_term(
                term,
                reference,
                reference,
                next_after_atom_with_mods=lambda _atom, current, _seed, **_kwargs: current + timedelta(days=1),
                atom_matches_on=lambda *_args, **_kwargs: False,
                intersection_guard_steps=2,
            )

        self.assertEqual(raised.exception.scope, "AND-term scheduling")

    def test_moon_source_and_filter_preserve_weekday_intersection(self) -> None:
        windows = (
            (date(2026, 8, 7), date(2026, 8, 13)),
            (date(2026, 9, 4), date(2026, 9, 10)),
        )

        def resolve(_phase, reference_day, **_kwargs):
            return next(start for start, _end in windows if start > reference_day)

        def matches(_phase, day, **_kwargs):
            return any(start <= day <= end for start, end in windows)

        with (
            patch.object(astronomy, "resolve_phase_date", side_effect=resolve),
            patch.object(astronomy, "phase_matches_date", side_effect=matches),
        ):
            source = core.validate_anchor_expr_strict("moon:full + w:fri")
            source_next, _meta = core.next_after_expr(
                source, date(2026, 7, 1), default_seed=date(2026, 7, 1)
            )
            filtered = core.validate_anchor_expr_strict("w:fri@moon=full")
            filtered_next, _meta = core.next_after_expr(
                filtered, date(2026, 7, 1), default_seed=date(2026, 7, 1)
            )

        self.assertEqual(source_next, date(2026, 8, 7))
        self.assertEqual(filtered_next, date(2026, 8, 7))

    def test_moon_source_emits_once_per_multi_day_phase_window(self) -> None:
        windows = (
            (date(2026, 7, 30), date(2026, 7, 31)),
            (date(2027, 7, 19), date(2027, 7, 22)),
            (date(2027, 8, 17), date(2027, 8, 23)),
        )

        def resolve(_phase, reference_day, **_kwargs):
            return next(start for start, _end in windows if start > reference_day)

        def matches(_phase, day, **_kwargs):
            return any(start <= day <= end for start, end in windows)

        with (
            patch.object(astronomy, "resolve_phase_date", side_effect=resolve),
            patch.object(astronomy, "phase_matches_date", side_effect=matches),
        ):
            dnf = core.validate_anchor_expr_strict("moon:full + y:jul")
            first, _meta = core.next_after_expr(
                dnf, date(2026, 7, 1), default_seed=date(2026, 7, 1)
            )
            second, _meta = core.next_after_expr(
                dnf, first, default_seed=date(2026, 7, 1)
            )

        self.assertEqual(first, date(2026, 7, 30))
        self.assertEqual(second, date(2027, 7, 19))

    def _require_astral(self) -> None:
        if ASTRAL_AVAILABLE:
            return
        if os.environ.get("NAUTICAL_REQUIRE_ASTRAL") == "1":
            self.fail("Astral is required for this test job; install requirements.txt")
        self.skipTest("Astral is an optional astronomy provider")

    def test_moon_phase_anchor_names_are_canonicalized(self) -> None:
        cases = {
            "moon:new": "new",
            "moon:first_quarter": "first-quarter",
            "moon:full-moon": "full",
            "moon:third-quarter": "last-quarter",
        }
        for expression, expected in cases.items():
            with self.subTest(expression=expression):
                atom = core.validate_anchor_expr_strict(expression)[0][0]
                self.assertEqual(atom["typ"], "moon")
                self.assertEqual(atom["spec"], expected)

        with self.assertRaisesRegex(Exception, "(?i)moon phase"):
            core.validate_anchor_expr_strict("moon:blue")

    def test_astronomy_observer_requires_explicit_timezone(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit timezone"):
            astronomy._observer(
                {"locations": {"home": {"latitude": 40.0, "longitude": -74.0}}},
                "home",
            )

    def test_phase_distance_wraps_at_new_moon(self) -> None:
        self.assertAlmostEqual(astronomy._phase_distance(27.8, 0.0), 0.2, places=9)
        self.assertEqual(astronomy._phase_distance(14.0, 14.0), 0.0)

    def test_phase_matches_documented_seven_day_bands(self) -> None:
        self.assertTrue(astronomy._phase_matches(14.0, "full"))
        self.assertTrue(astronomy._phase_matches(20.99, "full"))
        self.assertFalse(astronomy._phase_matches(13.99, "full"))

    def test_real_phase_resolver_returns_the_start_of_its_phase_band(self) -> None:
        self._require_astral()
        config = {
            "default_location": "test",
            "locations": {
                "test": {
                    "latitude": 40.0,
                    "longitude": -74.0,
                    "timezone": "UTC",
                }
            },
        }

        boundary = astronomy.resolve_phase_date("full", date(2026, 1, 1), config=config)

        self.assertTrue(astronomy.phase_matches_date("full", boundary, config=config))
        self.assertFalse(
            astronomy.phase_matches_date(
                "full", boundary - timedelta(days=1), config=config
            )
        )

    def test_real_event_times_preserve_timezone_across_dst(self) -> None:
        self._require_astral()
        config = {
            "default_location": "new-york",
            "locations": {
                "new-york": {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "timezone": "America/New_York",
                }
            },
        }

        before = astronomy.resolve_event("sunrise", date(2026, 3, 7), config=config)
        after = astronomy.resolve_event("sunrise", date(2026, 3, 8), config=config)

        self.assertIsNotNone(before.tzinfo)
        self.assertIsNotNone(after.tzinfo)
        self.assertEqual(before.tzinfo.key, "America/New_York")
        self.assertEqual(after.tzinfo.key, "America/New_York")
        self.assertNotEqual(before.utcoffset(), after.utcoffset())

    def test_unavailable_polar_summer_moonrise_fails_with_event_name(self) -> None:
        self._require_astral()
        config = {
            "default_location": "tromso",
            "locations": {
                "tromso": {
                    "latitude": 69.6492,
                    "longitude": 18.9553,
                    "timezone": "Europe/Oslo",
                }
            },
        }
        unavailable = False
        for day_offset in range(7):
            with self.subTest(day_offset=day_offset):
                try:
                    astronomy.resolve_event(
                        "moonrise",
                        date(2026, 6, 21) + timedelta(days=day_offset),
                        config=config,
                    )
                except LookupError as exc:
                    self.assertIn("moonrise", str(exc))
                    unavailable = True
                    break

        self.assertTrue(unavailable, "expected a missing Tromso moonrise in polar summer")


if __name__ == "__main__":
    unittest.main()
