import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
from nautical_core import business_calendar
from nautical_core import business_calendar_api
from nautical_core import business_calendar_config


class BusinessCalendarHelperContractTests(unittest.TestCase):
    def test_task_calendar_policy_is_scoped_and_restored_after_scheduling(self):
        calendars = core.resolve_business_calendar_config(
            {"weekend": {"anchor": "w:sat,sun"}},
        )
        with patch.object(core, "configured_business_calendars", return_value=calendars):
            task = {"bc": "WEEKEND"}
            policy = core.normalize_task_business_calendar_in_place(task)
            self.assertEqual(task["bc"], "weekend")

            dnf = core.validate_anchor_expr_strict("m:1bd")
            with core.use_business_calendar(policy):
                selected, _meta = core.next_after_expr(dnf, date(2026, 6, 30))
                self.assertEqual(selected, date(2026, 7, 4))
                self.assertEqual(core.business_calendar_fingerprint(), policy.fingerprint)

            restored, _meta = core.next_after_expr(dnf, date(2026, 6, 30))
            self.assertEqual(restored, date(2026, 7, 1))
    def test_weekday_and_configured_calendar_membership(self):
        self.assertTrue(business_calendar.DEFAULT_BUSINESS_CALENDAR.is_business_day(date(2026, 9, 10)))
        self.assertFalse(business_calendar.DEFAULT_BUSINESS_CALENDAR.is_business_day(date(2026, 9, 12)))
        configured = business_calendar.ConfiguredBusinessCalendar(
            name="work",
            fingerprint="work-v1",
            anchor_dates=frozenset({date(2026, 9, 12)}),
            omit_dates=frozenset({date(2026, 9, 14)}),
            _anchor_matches=lambda value: value == date(2026, 9, 13),
            _omit_matches=lambda value: value == date(2026, 9, 15),
        )
        self.assertTrue(configured.is_business_day(date(2026, 9, 12)))
        self.assertTrue(configured.is_business_day(date(2026, 9, 13)))
        self.assertFalse(configured.is_business_day(date(2026, 9, 14)))
        self.assertFalse(configured.is_business_day(date(2026, 9, 15)))

    def test_search_shift_and_month_helpers_obey_limits(self):
        saturday = date(2026, 9, 12)
        self.assertEqual(business_calendar.find_business_day(saturday, 1), date(2026, 9, 14))
        self.assertEqual(business_calendar.nearest_business_day(saturday), date(2026, 9, 11))
        self.assertEqual(business_calendar.shift_business_days(date(2026, 9, 11), 1), date(2026, 9, 14))
        self.assertEqual(business_calendar.shift_business_days(date(2026, 9, 14), -1), date(2026, 9, 11))
        self.assertEqual(business_calendar.business_days_in_month(2026, 9)[0], date(2026, 9, 1))
        self.assertEqual(business_calendar.nth_business_day_of_month(2026, 9, 1), date(2026, 9, 1))
        self.assertEqual(business_calendar.nth_business_day_of_month(2026, 9, -1), date(2026, 9, 30))
        self.assertEqual(business_calendar.nth_business_day_of_month(2026, 9, 0), None)
        self.assertEqual(business_calendar.business_day_offsets_for_iso_week(2026, 37), [0, 1, 2, 3, 4])
        with self.assertRaises(business_calendar.BusinessCalendarSearchError):
            business_calendar.find_business_day(saturday, 1, max_scan_days=1)
        never_business = business_calendar.ConfiguredBusinessCalendar(
            name="closed",
            fingerprint="closed-v1",
            anchor_dates=frozenset(),
            omit_dates=frozenset(),
            _anchor_matches=lambda value: False,
            _omit_matches=lambda value: False,
        )
        with self.assertRaises(business_calendar.BusinessCalendarSearchError):
            business_calendar.shift_business_days(
                date(2026, 9, 11), 1, never_business, max_scan_days_per_step=2
            )
        with self.assertRaises(ValueError):
            business_calendar.find_business_day(saturday, 0)

        default = business_calendar.DEFAULT_BUSINESS_CALENDAR
        self.assertTrue(default.is_business_day(date(2026, 4, 24)))
        self.assertFalse(default.is_business_day(date(2026, 4, 25)))
        self.assertEqual(
            business_calendar.find_business_day(date(2026, 4, 25), -1, default),
            date(2026, 4, 24),
        )
        self.assertEqual(
            business_calendar.find_business_day(date(2026, 4, 25), 1, default),
            date(2026, 4, 27),
        )
        self.assertEqual(
            business_calendar.nearest_business_day(date(2026, 4, 25), default),
            date(2026, 4, 24),
        )
        self.assertEqual(
            business_calendar.shift_business_days(date(2026, 4, 24), 1, default),
            date(2026, 4, 27),
        )
        self.assertEqual(
            business_calendar.shift_business_days(date(2026, 4, 27), -1, default),
            date(2026, 4, 24),
        )
        self.assertEqual(
            business_calendar.nth_business_day_of_month(2026, 1, 1, default),
            date(2026, 1, 1),
        )
        self.assertEqual(
            business_calendar.nth_business_day_of_month(2026, 1, -1, default),
            date(2026, 1, 30),
        )

    def test_displacements_chain_and_filtering(self):
        calendar = business_calendar.WeekdayBusinessCalendar()
        with business_calendar.capture_business_calendar_displacements() as events:
            business_calendar.record_business_calendar_displacement(
                date(2026, 9, 12), date(2026, 9, 13), calendar, operation="pbd"
            )
            business_calendar.record_business_calendar_displacement(
                date(2026, 9, 13), date(2026, 9, 14), calendar, operation="nbd"
            )
            business_calendar.record_business_calendar_displacement(
                date(2026, 9, 14), date(2026, 9, 14), calendar, operation="noop"
            )
            self.assertEqual(len(events), 2)
            self.assertEqual(
                business_calendar.business_calendar_displacement_for_date(date(2026, 9, 14)),
                business_calendar.CalendarDisplacement("weekday", date(2026, 9, 12), date(2026, 9, 14), "pbd+nbd"),
            )
            self.assertIsNone(
                business_calendar.business_calendar_displacement_for_date(date(2026, 9, 14), calendar_name="other")
            )

    def test_context_managers_restore_previous_values(self):
        custom = business_calendar.WeekdayBusinessCalendar(name="custom", fingerprint="custom-v1")
        self.assertIs(business_calendar.active_business_calendar(), business_calendar.DEFAULT_BUSINESS_CALENDAR)
        with business_calendar.use_business_calendar(custom):
            self.assertIs(business_calendar.active_business_calendar(), custom)
        self.assertIs(business_calendar.active_business_calendar(), business_calendar.DEFAULT_BUSINESS_CALENDAR)
        with business_calendar.capture_business_calendar_displacements() as events:
            self.assertIsInstance(events, list)
        self.assertIsNone(business_calendar._ACTIVE_DISPLACEMENTS.get())

    def test_calendar_helpers_follow_active_context_when_not_explicitly_overridden(self):
        class ContextCalendar:
            name = "context"

            def is_business_day(self, value):
                return value == date(2026, 9, 12)

        calendar = ContextCalendar()
        with business_calendar.use_business_calendar(calendar):
            self.assertTrue(business_calendar.is_business_day(date(2026, 9, 12)))
            self.assertEqual(
                business_calendar.find_business_day(date(2026, 9, 11), 1),
                date(2026, 9, 12),
            )
            self.assertEqual(
                business_calendar.business_days_in_month(2026, 9),
                [date(2026, 9, 12)],
            )


class BusinessCalendarConfigContractTests(unittest.TestCase):
    def test_definition_normalization_and_invalid_shapes(self):
        definitions = business_calendar_config.parse_business_calendar_definitions(
            {
                " Work-Days ": {
                    "anchor": [" w:mon..fri ", "w:mon..fri", " y:04-25 "],
                    "anchor_file": "open-*.csv",
                    "omit": " y:04-20 ",
                    "omit_file": ["closed-*.csv"],
                }
            }
        )
        definition = definitions["work-days"]
        self.assertEqual(definition.anchor, ("w:mon..fri", "y:04-25"))
        self.assertEqual(definition.anchor_file, ("open-*.csv",))
        self.assertEqual(definition.omit, ("y:04-20",))
        self.assertEqual(definition.omit_file, ("closed-*.csv",))
        with self.assertRaises(TypeError):
            definitions["other"] = definition
        for raw, message in (([], "TOML table"), ({"x": {}}, "must define anchor"), ({"x": {"other": "v"}}, "Unknown")):
            with self.subTest(raw=raw), self.assertRaisesRegex(business_calendar_config.BusinessCalendarConfigError, message):
                business_calendar_config.parse_business_calendar_definitions(raw)

    def test_definition_rejects_unstable_rules_and_unmatched_file_sources(self):
        parse_cases = (
            ({"bad.name": {"anchor": "w:mon"}}, "Invalid business_calendar name"),
            ({"work": {"anchor_files": "*.csv"}}, "Unknown business_calendar.work field"),
            ({"work": {"omit": "y:04-20"}}, "must define anchor or anchor_file"),
        )
        for raw, expected in parse_cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(
                business_calendar_config.BusinessCalendarConfigError, expected
            ):
                business_calendar_config.parse_business_calendar_definitions(raw)

        resolve_cases = (
            ({"work": {"anchor": "w/2:mon"}}, "interval recurrences"),
            ({"work": {"anchor": "w:rand"}}, "random selectors"),
            ({"work": {"anchor": "w:mon@t=09:00"}}, "time modifiers"),
            ({"work": {"anchor": "w:mon@bd"}}, "business-day modifiers"),
            ({"work": {"anchor": "m:lbd"}}, "business-day ordinals"),
        )
        for raw, expected in resolve_cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(
                business_calendar_config.BusinessCalendarConfigError, expected
            ):
                core.resolve_business_calendar_config(raw)

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "days.csv").write_text("date\n2026-04-20\n", encoding="utf-8")
            with self.assertRaisesRegex(
                business_calendar_config.BusinessCalendarConfigError, "matched no files"
            ):
                core.resolve_business_calendar_config(
                    {"work": {"anchor_file": "missing-*.csv"}},
                    anchor_file_dir=directory,
                )
            with self.assertRaisesRegex(
                business_calendar_config.BusinessCalendarConfigError,
                "business-day modifiers",
            ):
                core.resolve_business_calendar_config(
                    {"work": {"anchor_file": "days.csv@nbd"}},
                    anchor_file_dir=directory,
                )


class BusinessCalendarApiContractTests(unittest.TestCase):
    def setUp(self):
        self.calendar = business_calendar.WeekdayBusinessCalendar(name="work", fingerprint="work-v1")
        self.namespace = {
            "_business_calendar_config": business_calendar_config,
            "_business_calendar": business_calendar,
            "BUSINESS_CALENDAR_CONFIG": {},
            "_unwrap_quotes": lambda value: value.strip().strip("'\"") if value else "",
            "configured_business_calendars": lambda: {"work": self.calendar},
        }
        self.api = business_calendar_api.for_core(namespace=self.namespace)

    def test_task_selection_normalization_fingerprint_and_context(self):
        task = {"bc": ' "work" '}
        self.assertIs(self.api.business_calendar_for_task(task), self.calendar)
        self.assertIs(self.api.normalize_task_business_calendar_in_place(task), self.calendar)
        self.assertEqual(task["bc"], "work")
        self.assertEqual(self.api.business_calendar_fingerprint(), "weekday-v1")
        self.assertEqual(self.api.business_calendar_fingerprint(self.calendar), "work-v1")
        with self.api.use_task_business_calendar({"bc": "work"}):
            self.assertIs(business_calendar.active_business_calendar(), self.calendar)
        self.assertIs(business_calendar.active_business_calendar(), business_calendar.DEFAULT_BUSINESS_CALENDAR)

    def test_empty_and_object_tasks_and_unknown_names(self):
        self.assertIs(self.api.business_calendar_for_task({}), business_calendar.DEFAULT_BUSINESS_CALENDAR)
        state = SimpleNamespace(value="work")
        task = SimpleNamespace(field=lambda key: state if key == "bc" else None)
        self.assertIs(self.api.business_calendar_for_task(task), self.calendar)
        with self.assertRaisesRegex(business_calendar_config.BusinessCalendarConfigError, "Unknown business calendar"):
            self.api.business_calendar_for_task({"bc": "missing"})


class BusinessCalendarResolutionContractTests(unittest.TestCase):
    def test_custom_calendar_drives_scheduler_filters_rolls_offsets_ordinals_and_random_pools(self):
        class SetCalendar:
            name = "test-calendar"

            def __init__(self, open_dates):
                self.open_dates = frozenset(open_dates)

            def is_business_day(self, value):
                return value in self.open_dates

        policy = SetCalendar(
            {
                date(2026, 1, 2),
                date(2026, 1, 7),
                date(2026, 1, 21),
                date(2026, 4, 25),
                date(2026, 4, 28),
            }
        )
        cases = (
            ("y:04-25@bd", date(2026, 4, 12), date(2026, 4, 25)),
            ("y:04-26@pbd", date(2026, 4, 12), date(2026, 4, 25)),
            ("y:04-26@nbd", date(2026, 4, 12), date(2026, 4, 28)),
            ("y:04-26@nw", date(2026, 4, 12), date(2026, 4, 25)),
            ("y:04-24@+1bd", date(2026, 4, 12), date(2026, 4, 25)),
            ("m:1bd", date(2026, 1, 1), date(2026, 1, 2)),
            ("m:-1bd", date(2026, 1, 1), date(2026, 1, 21)),
            ("w:rand@bd", date(2026, 1, 4), date(2026, 1, 7)),
            ("w:rand@bd", date(2026, 1, 7), date(2026, 1, 21)),
        )
        for expression, start, expected in cases:
            with self.subTest(expression=expression, start=start):
                dnf = core.validate_anchor_expr_strict(expression)
                actual, _meta = core.next_after_expr(
                    dnf,
                    start,
                    default_seed=start,
                    seed_base="business-calendar-policy",
                    business_calendar=policy,
                )
                self.assertEqual(actual, expected)

        random_dnf = core.validate_anchor_expr_strict("m:rand@bd")
        random_date, _meta = core.next_after_expr(
            random_dnf,
            date(2026, 1, 1),
            default_seed=date(2026, 1, 1),
            seed_base="business-calendar-policy",
            business_calendar=policy,
        )
        self.assertIn(random_date, {date(2026, 1, 2), date(2026, 1, 7), date(2026, 1, 21)})

    def test_calendar_resolution_unions_rules_files_then_applies_omissions(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            anchor_dir = base / "anchors"
            omit_dir = base / "omits"
            anchor_dir.mkdir()
            omit_dir.mkdir()
            (anchor_dir / "open-local.csv").write_text(
                "date\n2026-04-25\n", encoding="utf-8"
            )
            (omit_dir / "closed-local.csv").write_text(
                "date\n2026-04-20\n", encoding="utf-8"
            )
            calendars = core.resolve_business_calendar_config(
                {
                    "work": {
                        "anchor": ["w:mon..fri", "y:04-25"],
                        "anchor_file": "open-*.csv@+1d",
                        "omit": "y:04-22",
                        "omit_file": "closed-*.csv@+1d",
                    }
                },
                anchor_file_dir=str(anchor_dir),
                omit_file_dir=str(omit_dir),
            )

        policy = calendars["work"]
        for value, expected in (
            (date(2026, 4, 17), True),
            (date(2026, 4, 18), False),
            (date(2026, 4, 21), False),
            (date(2026, 4, 22), False),
            (date(2026, 4, 25), True),
            (date(2026, 4, 26), True),
        ):
            with self.subTest(value=value):
                self.assertIs(policy.is_business_day(value), expected)
        self.assertIsInstance(hash(policy), int)
        with self.assertRaises(TypeError):
            calendars["other"] = policy

    def test_injected_calendar_controls_anchor_and_omit_file_modifiers(self):
        import nautical_core.anchor_files as anchor_files
        import nautical_core.omit_files as omit_files

        class SetCalendar:
            name = "file-calendar"

            def __init__(self, open_dates):
                self.open_dates = frozenset(open_dates)

            def is_business_day(self, value):
                return value in self.open_dates

        policy = SetCalendar({date(2026, 4, 25), date(2026, 4, 27)})
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory)
            (source_dir / "saturday.csv").write_text(
                "date\n2026-04-25\n", encoding="utf-8"
            )
            (source_dir / "friday.csv").write_text(
                "date\n2026-04-24\n", encoding="utf-8"
            )
            self.assertEqual(
                anchor_files.load_anchor_file_dates("saturday.csv@bd", str(source_dir)),
                frozenset(),
            )
            self.assertEqual(
                anchor_files.load_anchor_file_dates(
                    "saturday.csv@bd", str(source_dir), business_calendar=policy
                ),
                frozenset({date(2026, 4, 25)}),
            )
            self.assertEqual(
                anchor_files.load_anchor_file_dates(
                    "friday.csv@+1bd", str(source_dir), business_calendar=policy
                ),
                frozenset({date(2026, 4, 25)}),
            )
            self.assertEqual(
                omit_files.load_omit_file_dates(
                    "saturday.csv@bd", str(source_dir), business_calendar=policy
                ),
                frozenset({date(2026, 4, 25)}),
            )

    def test_displacement_capture_records_only_actual_rule_and_file_rolls(self):
        import nautical_core.anchor_files as anchor_files

        policy = core.resolve_business_calendar_config(
            {"work": {"anchor": "w:mon..fri", "omit": "y:04-24"}}
        )["work"]
        shifted_dnf = core.validate_anchor_expr_strict("y:04-24@nbd")
        chained_dnf = core.validate_anchor_expr_strict("y:04-24@nbd@+1bd")
        unchanged_dnf = core.validate_anchor_expr_strict("y:04-23@nbd")

        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "dates.csv").write_text(
                "date\n2026-04-24\n", encoding="utf-8"
            )
            with core.use_business_calendar(policy), core.capture_business_calendar_displacements():
                shifted, _meta = core.next_after_expr(
                    shifted_dnf, date(2026, 4, 20), date(2026, 4, 20)
                )
                displacement = core.business_calendar_displacement_for_date(
                    shifted, calendar_name="work"
                )
                chained, _meta = core.next_after_expr(
                    chained_dnf, date(2026, 4, 20), date(2026, 4, 20)
                )
                chained_displacement = core.business_calendar_displacement_for_date(
                    chained, calendar_name="work"
                )
                unchanged, _meta = core.next_after_expr(
                    unchanged_dnf, date(2026, 4, 20), date(2026, 4, 20)
                )
                no_displacement = core.business_calendar_displacement_for_date(
                    unchanged, calendar_name="work"
                )
            with core.capture_business_calendar_displacements():
                file_dates = anchor_files.load_anchor_file_dates(
                    "dates.csv@nbd", directory, business_calendar=policy
                )
                file_displacement = core.business_calendar_displacement_for_date(
                    date(2026, 4, 27), calendar_name="work"
                )

        self.assertEqual(shifted, date(2026, 4, 27))
        self.assertEqual(
            displacement,
            business_calendar.CalendarDisplacement(
                "work", date(2026, 4, 24), date(2026, 4, 27), "nbd"
            ),
        )
        self.assertEqual(chained, date(2026, 4, 28))
        self.assertEqual(chained_displacement.original, date(2026, 4, 24))
        self.assertEqual(chained_displacement.adjusted, date(2026, 4, 28))
        self.assertEqual(file_dates, frozenset({date(2026, 4, 27)}))
        self.assertEqual(file_displacement.original, date(2026, 4, 24))
        self.assertEqual(unchanged, date(2026, 4, 23))
        self.assertIsNone(no_displacement)

    def test_calendar_fingerprint_separates_rule_file_and_hint_cache_identity(self):
        monday = core.resolve_business_calendar_config(
            {"work": {"anchor": "w:mon"}}
        )["work"]
        tuesday = core.resolve_business_calendar_config(
            {"work": {"anchor": "w:tue"}}
        )["work"]
        self.assertNotEqual(monday.fingerprint, tuesday.fingerprint)
        self.assertEqual(
            core.cache_key_for_task("m:1bd", "skip"),
            core.cache_key_for_task(
                "m:1bd", "skip", core.DEFAULT_BUSINESS_CALENDAR.fingerprint
            ),
        )
        self.assertNotEqual(
            core.cache_key_for_task("m:1bd", "skip", monday.fingerprint),
            core.cache_key_for_task("m:1bd", "skip", tuesday.fingerprint),
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "open.csv"
            source.write_text("date\n2026-04-20\n", encoding="utf-8")
            config = {"work": {"anchor_file": "open.csv"}}
            first = core.resolve_business_calendar_config(
                config, anchor_file_dir=directory
            )["work"]
            source.write_text("date\n2026-04-21\n", encoding="utf-8")
            second = core.resolve_business_calendar_config(
                config, anchor_file_dir=directory
            )["work"]
            self.assertNotEqual(first.fingerprint, second.fingerprint)

        hints = core.build_and_cache_hints(
            "m:1bd", "skip", business_calendar=monday
        )
        self.assertEqual(
            hints.get("meta", {}).get("cfg", {}).get("bc"), monday.fingerprint
        )


if __name__ == "__main__":
    unittest.main()
