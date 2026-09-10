import unittest
from datetime import date
from types import SimpleNamespace

from nautical_core import business_calendar
from nautical_core import business_calendar_api
from nautical_core import business_calendar_config


class BusinessCalendarHelperContractTests(unittest.TestCase):
    def test_weekday_and_configured_calendar_membership(self):
        self.assertTrue(business_calendar.DEFAULT_BUSINESS_CALENDAR.is_business_day(date(2026, 9, 10)))
        self.assertFalse(business_calendar.DEFAULT_BUSINESS_CALENDAR.is_business_day(date(2026, 9, 12)))
        configured = business_calendar.ConfiguredBusinessCalendar(
            name="work",
            fingerprint="work-v1",
            anchor_dates=frozenset({date(2026, 9, 12)}),
            omit_dates=frozenset({date(2026, 9, 12)}),
            _anchor_matches=lambda value: value == date(2026, 9, 13),
            _omit_matches=lambda value: value == date(2026, 9, 13),
        )
        self.assertFalse(configured.is_business_day(date(2026, 9, 12)))
        self.assertFalse(configured.is_business_day(date(2026, 9, 13)))

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
        with self.assertRaises(ValueError):
            business_calendar.find_business_day(saturday, 0)

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


class BusinessCalendarConfigContractTests(unittest.TestCase):
    def test_definition_normalization_and_invalid_shapes(self):
        definitions = business_calendar_config.parse_business_calendar_definitions(
            {" Work ": {"anchor": [" y:01 ", "y:01"], "omit": " y:02 "}}
        )
        self.assertEqual(definitions["work"].anchor, ("y:01",))
        self.assertEqual(definitions["work"].omit, ("y:02",))
        for raw, message in (([], "TOML table"), ({"x": {}}, "must define anchor"), ({"x": {"other": "v"}}, "Unknown")):
            with self.subTest(raw=raw), self.assertRaisesRegex(business_calendar_config.BusinessCalendarConfigError, message):
                business_calendar_config.parse_business_calendar_definitions(raw)


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


if __name__ == "__main__":
    unittest.main()
