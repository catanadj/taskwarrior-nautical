import unittest
from datetime import date, datetime, timezone
from types import MappingProxyType
from zoneinfo import ZoneInfo

from nautical_core import business_calendar_config
from nautical_core import timeutil


class BusinessCalendarConfigContractTests(unittest.TestCase):
    def test_empty_config_is_immutable_and_definitions_are_normalized(self):
        empty = business_calendar_config.parse_business_calendar_definitions(None)
        self.assertIsInstance(empty, MappingProxyType)
        self.assertEqual(dict(empty), {})
        with self.assertRaises(TypeError):
            empty["new"] = object()

        definitions = business_calendar_config.parse_business_calendar_definitions(
            {
                " Work-Day ": {
                    "anchor": [" y:01 ", "y:01", ""],
                    "anchor_file": " holidays.txt ",
                    "omit": [" y:02 ", "y:02"],
                }
            }
        )
        definition = definitions["work-day"]
        self.assertEqual(definition.anchor, ("y:01",))
        self.assertEqual(definition.anchor_file, ("holidays.txt",))
        self.assertEqual(definition.omit, ("y:02",))
        self.assertEqual(definition.omit_file, ())

    def test_invalid_shapes_and_rule_modifiers_have_stable_errors(self):
        invalid = (
            ([], "TOML table"),
            ({"bad name": {"anchor": "y:01"}}, "Invalid business_calendar name"),
            ({"work": {}}, "must define anchor"),
            ({"work": {"unknown": "x", "anchor": "y:01"}}, "Unknown"),
            ({"work": {"anchor": [1]}}, "only strings"),
            ({"work": {"anchor": "y:01"}, " WORK ": {"anchor": "y:02"}}, "Duplicate"),
        )
        for raw, message in invalid:
            with self.subTest(raw=raw), self.assertRaisesRegex(
                business_calendar_config.BusinessCalendarConfigError, message
            ):
                business_calendar_config.parse_business_calendar_definitions(raw)

        for mods, message in (
            ({"t": True}, "time modifiers"),
            ({"bd": True}, "business-day modifiers"),
            ({"wd": True}, "business-day modifiers"),
            ({"business_day_offset": 1}, "business-day modifiers"),
            ({"roll": "pbd"}, "business-day modifiers"),
        ):
            with self.subTest(mods=mods), self.assertRaisesRegex(
                business_calendar_config.BusinessCalendarConfigError, message
            ):
                business_calendar_config.validate_calendar_rule_modifiers(mods, label="calendar.work")

    def test_resolve_loads_rules_and_file_dates_with_deterministic_fingerprint(self):
        calls = []

        def validate_rule(expression):
            calls.append(("rule", expression))
            return (({"typ": "m", "spec": expression, "ival": 1, "mods": {}},),)

        def matches(rule, value, calendar_name):
            return rule[0][0]["spec"] == "y:01" and value == date(2026, 1, 1) and calendar_name == "work"

        def validate_file(expression):
            calls.append(("file", expression))

        def unmatched(expression, directory):
            return ()

        def load(expression, directory):
            calls.append(("load", expression, directory))
            return frozenset({date(2026, 1, 2) if "anchor" in directory else date(2026, 1, 3)})

        raw_config = {
            " Work ": {
                "anchor": "y:01",
                "anchor_file": "anchor.txt",
                "omit": "y:02",
                "omit_file": "omit.txt",
            }
        }
        resolve_kwargs = dict(
            anchor_file_dir="anchor-dir",
            omit_file_dir="omit-dir",
            validate_anchor_expr=validate_rule,
            validate_omit_expr=validate_rule,
            expression_matches_date=matches,
            validate_anchor_file_expr=validate_file,
            validate_omit_file_expr=validate_file,
            unmatched_anchor_file_patterns=unmatched,
            unmatched_omit_file_patterns=unmatched,
            load_anchor_file_dates=load,
            load_omit_file_dates=load,
        )
        result = business_calendar_config.resolve_business_calendars(raw_config, **resolve_kwargs)
        self.assertIsInstance(result, MappingProxyType)
        calendar = result["work"]
        self.assertEqual(calendar.anchor_dates, frozenset({date(2026, 1, 2)}))
        self.assertEqual(calendar.omit_dates, frozenset({date(2026, 1, 3)}))
        self.assertTrue(calendar.is_business_day(date(2026, 1, 1)))
        self.assertFalse(calendar.is_business_day(date(2026, 1, 3)))
        self.assertEqual(len(calendar.fingerprint), 16)
        self.assertTrue(any(item[0] == "load" for item in calls))

        equivalent = business_calendar_config.resolve_business_calendars(raw_config, **resolve_kwargs)
        self.assertEqual(equivalent["work"].fingerprint, calendar.fingerprint)

        changed_rule = {"work": {**raw_config[" Work "], "anchor": "y:03"}}
        changed_rule_result = business_calendar_config.resolve_business_calendars(
            changed_rule, **resolve_kwargs
        )
        self.assertNotEqual(changed_rule_result["work"].fingerprint, calendar.fingerprint)

        def changed_load(expression, directory):
            if "anchor" in directory:
                return frozenset({date(2026, 1, 4)})
            return frozenset({date(2026, 1, 3)})

        changed_files = business_calendar_config.resolve_business_calendars(
            raw_config,
            **{**resolve_kwargs, "load_anchor_file_dates": changed_load},
        )
        self.assertNotEqual(changed_files["work"].fingerprint, calendar.fingerprint)

    def test_resolve_reports_expression_file_and_loader_failures(self):
        base = {"work": {"anchor": "y:01"}}
        kwargs = dict(
            anchor_file_dir="a",
            omit_file_dir="o",
            validate_anchor_expr=lambda _value: (_ for _ in ()).throw(RuntimeError("bad expression")),
            validate_omit_expr=lambda _value: (),
            expression_matches_date=lambda *_args: False,
            validate_anchor_file_expr=lambda _value: None,
            validate_omit_file_expr=lambda _value: None,
            unmatched_anchor_file_patterns=lambda *_args: (),
            unmatched_omit_file_patterns=lambda *_args: (),
            load_anchor_file_dates=lambda *_args: frozenset(),
            load_omit_file_dates=lambda *_args: frozenset(),
        )
        with self.assertRaisesRegex(business_calendar_config.BusinessCalendarConfigError, "Invalid business_calendar.work.anchor"):
            business_calendar_config.resolve_business_calendars(base, **kwargs)

        file_config = {"work": {"anchor_file": "missing.txt"}}
        kwargs["unmatched_anchor_file_patterns"] = lambda *_args: ("missing.txt",)
        with self.assertRaisesRegex(business_calendar_config.BusinessCalendarConfigError, "matched no files"):
            business_calendar_config.resolve_business_calendars(file_config, **kwargs)

        file_config = {"work": {"anchor_file": "broken.txt"}}
        kwargs["unmatched_anchor_file_patterns"] = lambda *_args: ()
        kwargs["load_anchor_file_dates"] = lambda *_args: (_ for _ in ()).throw(OSError("unreadable"))
        with self.assertRaisesRegex(business_calendar_config.BusinessCalendarConfigError, "Invalid business_calendar.work.anchor_file"):
            business_calendar_config.resolve_business_calendars(file_config, **kwargs)


class TimeUtilContractTests(unittest.TestCase):
    def test_comparison_and_utc_normalization(self):
        naive = datetime(2026, 1, 1, 12)
        aware = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        self.assertEqual(timeutil.compare_datetimes(naive, naive), 0)
        self.assertEqual(
            timeutil.compare_datetimes(datetime(2026, 1, 1, 12, tzinfo=timezone.utc), datetime(2026, 1, 1, 7, tzinfo=ZoneInfo("America/New_York"))),
            0,
        )
        with self.assertRaises(ValueError):
            timeutil.compare_datetimes(naive, aware)
        with self.assertRaises(TypeError):
            timeutil.compare_datetimes("bad", aware)
        self.assertEqual(timeutil.ensure_utc(naive), aware)
        self.assertEqual(timeutil.ensure_utc(datetime(2026, 1, 1, 7, tzinfo=ZoneInfo("America/New_York"))), aware)

    def test_local_conversion_formatting_and_dst_boundaries(self):
        zone = ZoneInfo("America/New_York")
        instant = datetime(2026, 7, 1, 16, 5, tzinfo=timezone.utc)
        self.assertEqual(timeutil.to_local(instant, zone).strftime("%Y-%m-%d %H:%M"), "2026-07-01 12:05")
        self.assertEqual(timeutil.utc_to_local_naive(instant, zone), datetime(2026, 7, 1, 12, 5))
        self.assertEqual(timeutil.fmt_dt_local(instant, zone), "Wed 2026-07-01 12:05 EDT")
        self.assertEqual(timeutil.fmt_isoz(instant), "2026-07-01T16:05:00Z")
        self.assertEqual(timeutil.local_naive_to_utc(datetime(2026, 11, 1, 1, 30), zone), datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc))
        self.assertEqual(timeutil.local_naive_to_utc(datetime(2026, 3, 8, 2, 30), zone), datetime(2026, 3, 8, 7, 30, tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            timeutil.local_naive_to_utc(instant, zone)
        with self.assertRaises(TypeError):
            timeutil.utc_to_local_naive("bad", zone)
        self.assertEqual(timeutil.build_local_datetime(date(2026, 7, 1), (12, 5), zone), instant)

    def test_parse_dt_any_accepts_iso_formats_and_rejects_invalid_values(self):
        formats = ("%d/%m/%Y %H:%M",)
        self.assertEqual(timeutil.parse_dt_any("2026-01-02T03:04:05Z", formats), datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
        self.assertEqual(timeutil.parse_dt_any("2026-01-02T03:04:05+02:00", formats), datetime(2026, 1, 2, 1, 4, 5, tzinfo=timezone.utc))
        self.assertEqual(timeutil.parse_dt_any("02/01/2026 03:04", formats), datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        self.assertEqual(timeutil.parse_dt_any("2026-01-02 trailing", formats), datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.assertIsNone(timeutil.parse_dt_any("", formats))
        self.assertIsNone(timeutil.parse_dt_any("not-a-date", formats))


if __name__ == "__main__":
    unittest.main()
