from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from tests.support.hook_process import HookSubprocessFixture


class OnAddHookRouteTests(HookSubprocessFixture):
    """Executable on-add routes keep the Taskwarrior JSON protocol intact."""

    def setUp(self) -> None:
        super().setUp()
        self._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self._temporary_directory.name)
        self.taskdata = root / "taskdata"
        self.taskdata.mkdir()
        self.anchor_files = root / "anchors"
        self.anchor_files.mkdir()
        self.omit_files = root / "omit"
        self.omit_files.mkdir()
        self.config = root / "config-nautical.toml"
        self.config.write_text(
            f'tz = "UTC"\nanchor_file_dir = "{self.anchor_files}"\nomit_file_dir = "{self.omit_files}"\npanel_mode = "rich"\n',
            encoding="utf-8",
        )
        (self.anchor_files / "dates.csv").write_text(
            "date,description\n2099-01-05,Întâlnire café\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()
        super().tearDown()

    def _task(self, **fields: object) -> dict[str, object]:
        task: dict[str, object] = {
            "uuid": "11111111-1111-4111-8111-111111111111",
            "description": "route task",
            "status": "pending",
            "entry": "20990101T000000Z",
            "modified": "20990101T000000Z",
        }
        task.update(fields)
        return task

    def _run(self, task: dict[str, object], *, diagnostics: bool = False):
        return self.run_hook(
            "on-add.nautical",
            json.dumps(task, ensure_ascii=False),
            diagnostics=diagnostics,
            extra_environment={"NAUTICAL_CONFIG": str(self.config), "NO_COLOR": "1"},
        )

    def _assert_valid(self, task: dict[str, object]) -> dict[str, object]:
        quiet = self._run(task)
        self.assertEqual(quiet.returncode, 0, quiet.stderr)
        self.assertNotIn("Preview", quiet.stdout)
        self.assertNotIn("[nautical]", quiet.stderr)
        result = json.loads(quiet.stdout)

        diagnostic = self._run(task, diagnostics=True)
        self.assertEqual(diagnostic.returncode, 0, diagnostic.stderr)
        self.assertEqual(json.loads(diagnostic.stdout), result)
        self.assertNotIn("[nautical]", diagnostic.stdout)
        self.assertIn("[nautical]", diagnostic.stderr)
        return result

    def _assert_invalid(self, task: dict[str, object], expected: tuple[str, ...]) -> None:
        quiet = self._run(task)
        self.assertNotEqual(quiet.returncode, 0)
        self.assertEqual(quiet.stdout, "")
        quiet_stderr = " ".join(quiet.stderr.replace("│", " ").split())
        for text in expected:
            self.assertIn(text, quiet_stderr)
        self.assertNotIn("Traceback", quiet.stderr)
        self.assertNotIn("[nautical]", quiet.stderr)

        diagnostic = self._run(task, diagnostics=True)
        self.assertNotEqual(diagnostic.returncode, 0)
        self.assertEqual(diagnostic.stdout, "")
        diagnostic_stderr = " ".join(diagnostic.stderr.replace("│", " ").split())
        for text in expected:
            self.assertIn(text, diagnostic_stderr)
        self.assertNotIn("Traceback", diagnostic.stderr)
        self.assertNotIn("[nautical]", diagnostic.stdout)
        self.assertIn("[nautical]", diagnostic.stderr)

    def test_position_selection_advice_preserves_hook_json(self) -> None:
        expression = "(y:w-1)@in-year=8th"
        result = self._run(
            self._task(anchor=expression, anchor_mode="skip"),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["anchor"], expression)
        self.assertIn("Advice", result.stderr)
        self.assertIn("ISO-week candidates by calendar year", result.stderr)

    def test_positional_anchor_preview_preserves_expression_and_explains_rule(self) -> None:
        expression = "(w:tue | w:thu)@in-month=last"
        result = self._run(
            self._task(
                entry="20260701T090000Z",
                due="20260730T090000Z",
                anchor=expression,
                anchor_mode="skip",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["anchor"], expression)
        self.assertEqual(payload["chain"], "on")
        self.assertIn("last matching date", result.stderr)

    def test_post_selection_modifiers_are_shown_in_anchor_preview(self) -> None:
        expression = "(w:tue | w:thu)@in-month=last@+2d@t=09:00"
        result = self._run(
            self._task(
                entry="20260703T090000Z",
                due="20260801T060000Z",
                anchor=expression,
                anchor_mode="skip",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["anchor"], expression)
        self.assertIn("2 days later at 09:00", result.stderr)

    def test_multitime_preview_emits_all_explicit_slots(self) -> None:
        expression = "w:wed@t=06:00,12:00,22:00"
        task = self._task(
            description="multitime preview",
            entry="20251217T000000Z",
            anchor=expression,
            anchor_mode="skip",
            due="20251217T060000Z",
        )
        result = self._run(task)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["due"], task["due"])
        self.assertIn("12:00", result.stderr)
        self.assertIn("22:00", result.stderr)

    def test_time_window_preview_emits_only_bounded_slots(self) -> None:
        task = self._task(
            description="time-window preview",
            entry="20251217T000000Z",
            anchor="w:wed@t=06..17/3h",
            anchor_mode="skip",
            due="20251217T060000Z",
        )
        result = self._run(task)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["uuid"], task["uuid"])
        self.assertIn("09:00", result.stderr)
        self.assertIn("15:00", result.stderr)
        self.assertNotIn("17:00 UTC", result.stderr)

    def test_overnight_window_preview_keeps_json_and_next_day_slots(self) -> None:
        task = self._task(
            description="overnight window preview",
            entry="20260804T000000Z",
            anchor="w:mon@t=22:30..06:30/7",
            anchor_mode="skip",
            due="20260810T193000Z",
        )
        result = self._run(task)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["uuid"], task["uuid"])
        self.assertIn("23:50", result.stderr)
        self.assertIn("01:10", result.stderr)

    def test_random_window_preview_keeps_anchor_and_upcoming_panel(self) -> None:
        expression = "w:mon@t=rand(06..18/3)"
        task = self._task(
            description="random window preview",
            entry="20260803T000000Z",
            anchor=expression,
            anchor_mode="skip",
            chainID="randomhook1",
            due="20260803T060000Z",
        )
        result = self._run(task)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["anchor"], expression)
        self.assertIn("Upcoming", result.stderr)

    def test_anchor_preset_preview_preserves_reference_and_shows_expansion(self) -> None:
        config = self.taskdata.parent / "preset-config.toml"
        config.write_text('[anchor_presets]\npayday = "m:15"\n', encoding="utf-8")
        task = self._task(
            description="anchor preset preview",
            entry="20260101T000000Z",
            anchor="@payday",
            anchor_mode="skip",
            due="20260101T090000Z",
        )
        result = self.run_hook(
            "on-add.nautical",
            json.dumps(task),
            extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["anchor"], "@payday")
        self.assertNotIn("Invalid anchor", result.stderr)
        self.assertIn("@payday → m:15", result.stderr)
        self.assertIn("2026-01-15", result.stderr)

    def test_dst_gap_shifted_window_slot_is_not_duplicated(self) -> None:
        self._assert_dst_window_slot_occurs_once(
            "w:sun@t=01..04/1h",
            "20250301T000000Z",
            "20250309T060000Z",
            "Sun 2025-03-09 03:00 EDT",
        )

    def test_dst_gap_shifted_partition_slot_is_not_duplicated(self) -> None:
        self._assert_dst_window_slot_occurs_once(
            "w:sun@t=01..05/5",
            "20250301T000000Z",
            "20250309T060000Z",
            "Sun 2025-03-09 03:00 EDT",
        )

    def test_dst_fallback_overnight_slot_is_not_duplicated(self) -> None:
        self._assert_dst_window_slot_occurs_once(
            "w:sat@t=22:30..02:30/5",
            "20261020T000000Z",
            "20261101T023000Z",
            "Sun 2026-11-01 01:30",
        )

    def test_live_panel_mode_falls_back_when_hook_stderr_is_captured(self) -> None:
        config = self.taskdata.parent / "live-config.toml"
        config.write_text('tz = "UTC"\npanel_mode = "live"\n', encoding="utf-8")
        task = self._task(
            description="live panel protocol",
            entry="20260101T000000Z",
            cp="1d",
            due="20260102T090000Z",
        )
        result = self.run_hook(
            "on-add.nautical",
            json.dumps(task),
            extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual(len(output_lines), 1)
        self.assertEqual(json.loads(output_lines[0])["uuid"], task["uuid"])
        self.assertIn("Recurring Chain Preview", result.stderr)
        self.assertIn("Period", result.stderr)
        self.assertNotIn("\x1b[", result.stderr)

    def test_counted_random_preview_uses_group_time(self) -> None:
        expression = "(m:2rand + w:mon..fri)@t=09:00"
        task = self._task(
            description="counted random group time",
            entry="20260101T000000Z",
            anchor=expression,
            anchor_mode="skip",
        )
        result = self._run(task)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["anchor"], expression)
        due = datetime.fromisoformat(payload["due"])
        self.assertEqual((due.hour, due.minute), (9, 0))
        self.assertLess(due.weekday(), 5)
        self.assertIn("2 random days each", result.stderr)
        self.assertIn("month at 09:00", result.stderr)

    def test_grouped_date_modifiers_are_accepted_and_keep_shared_time(self) -> None:
        expression = "(y:04-24 | y:04-30)@pbd@-1bd@t=09:00"
        task = self._task(
            description="grouped date modifiers",
            entry="20260101T000000Z",
            anchor=expression,
            anchor_mode="skip",
        )
        result = self._run(task)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["anchor"], expression)
        due = datetime.fromisoformat(payload["due"])
        self.assertEqual((due.hour, due.minute), (9, 0))
        self.assertLess(due.weekday(), 5)
        self.assertNotIn("Invalid anchor", result.stderr)

    def test_unknown_anchor_preset_fails_with_actionable_error(self) -> None:
        config = self.taskdata.parent / "unknown-preset-config.toml"
        config.write_text("[anchor_presets]\n", encoding="utf-8")
        task = self._task(
            description="unknown anchor preset",
            entry="20260101T000000Z",
            anchor="@missing",
            anchor_mode="skip",
            due="20260101T090000Z",
        )
        result = self.run_hook(
            "on-add.nautical",
            json.dumps(task),
            extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid anchor", result.stderr)
        self.assertIn("Unknown anchor preset '@missing'", result.stderr)

    def test_composed_anchor_preset_resolves_before_preview(self) -> None:
        task = self._task(
            description="composed anchor preset",
            entry="20260401T000000Z",
            anchor="@workout + y:apr",
            anchor_mode="skip",
            due="20260401T090000Z",
        )
        result = self._run_with_config(
            task, '[anchor_presets]\nworkout = "w:mon,wed,fri"\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["anchor"], task["anchor"])
        self.assertNotIn("Invalid anchor", result.stderr)
        self.assertIn("Natural", result.stderr)
        self.assertIn("Apr", result.stderr)

    def test_recursive_anchor_preset_fails_with_guidance(self) -> None:
        task = self._task(
            description="recursive anchor preset",
            entry="20260101T000000Z",
            anchor="@a",
            anchor_mode="skip",
            due="20260101T090000Z",
        )
        result = self._run_with_config(
            task, '[anchor_presets]\na = "@b"\nb = "@a"\n'
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid Nautical configuration", result.stderr)
        self.assertIn("Recursive", result.stderr)
        self.assertIn("anchor preset reference detected", result.stderr)

    def test_omit_preset_resolves_and_preserves_reference(self) -> None:
        task = self._task(
            description="omit preset",
            entry="20260301T000000Z",
            anchor="w:mon",
            omit="@april",
            anchor_mode="skip",
            due="20260302T090000Z",
        )
        result = self._run_with_config(task, '[omit_presets]\napril = "y:apr"\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["omit"], "@april")
        self.assertNotIn("Invalid omit", result.stderr)
        self.assertIn("@april → y:apr", result.stderr)
        self.assertIn("Except", result.stderr)
        self.assertIn("Apr", result.stderr)

    def test_unknown_omit_preset_fails_with_guidance(self) -> None:
        task = self._task(
            description="unknown omit preset",
            entry="20260301T000000Z",
            anchor="w:mon",
            omit="@missing",
            anchor_mode="skip",
            due="20260302T090000Z",
        )
        result = self._run_with_config(task, "[omit_presets]\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid omit", result.stderr)
        self.assertIn("Unknown omit preset '@missing'", result.stderr)

    def test_recursive_omit_preset_fails_with_guidance(self) -> None:
        task = self._task(
            description="recursive omit preset",
            entry="20260301T000000Z",
            anchor="w:mon",
            omit="@a",
            anchor_mode="skip",
            due="20260302T090000Z",
        )
        result = self._run_with_config(task, '[omit_presets]\na = "@b"\nb = "@a"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("Invalid Nautical configuration", result.stderr)
        self.assertIn("Recursive omit", result.stderr)
        self.assertIn("preset reference detected", result.stderr)

    def _assert_dst_window_slot_occurs_once(
        self, anchor: str, entry: str, due: str, expected_slot: str
    ) -> None:
        config = self.taskdata.parent / "dst-config.toml"
        config.write_text('tz = "America/New_York"\n', encoding="utf-8")
        task = self._task(
            description="DST time-window preview",
            entry=entry,
            anchor=anchor,
            anchor_mode="skip",
            due=due,
        )
        result = self.run_hook(
            "on-add.nautical",
            json.dumps(task),
            extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["uuid"], task["uuid"])
        self.assertEqual(result.stderr.count(expected_slot), 1, result.stderr)

    def _run_with_config(self, task: dict[str, object], config_text: str):
        config = self.taskdata.parent / "route-config.toml"
        config.write_text(config_text, encoding="utf-8")
        return self.run_hook(
            "on-add.nautical",
            json.dumps(task),
            extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
        )

    def test_yearly_positional_anchor_preview_supports_post_selection_offset(self) -> None:
        expression = "(w:mon)@in-year=last@+7d@t=09:00"
        result = self._run(
            self._task(
                entry="20260701T090000Z",
                due="20270104T090000Z",
                anchor=expression,
                anchor_mode="skip",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["anchor"], expression)
        self.assertIn("in each year", result.stderr)

    def test_seasonal_anchor_preview_explains_rule_and_fixed_boundary(self) -> None:
        expression = "(w:mon)@in-spring=first"
        result = self._run(
            self._task(
                entry="20260723T090000Z",
                anchor=expression,
                anchor_mode="skip",
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["anchor"], expression)
        self.assertIn("the first Monday of each spring", result.stderr)
        self.assertIn("Advice", result.stderr)
        self.assertIn("fixed March 1 through May 31", result.stderr)

    def test_named_business_calendar_normalizes_and_selects_its_weekend(self) -> None:
        self.config.write_text(
            'tz = "UTC"\n'
            f'anchor_file_dir = "{self.anchor_files}"\n'
            f'omit_file_dir = "{self.omit_files}"\n'
            'panel_mode = "rich"\n'
            '[business_calendar.weekend]\n'
            'anchor = "w:sat,sun"\n',
            encoding="utf-8",
        )
        result = self._run(
            self._task(
                entry="20260714T000000Z",
                anchor="m:1bd@t=09:00",
                anchor_mode="skip",
                bc="WEEKEND",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        due = datetime.fromisoformat(payload["due"])
        self.assertIn(due.weekday(), {5, 6})
        self.assertEqual(payload["bc"], "weekend")

    def test_business_calendar_displacement_is_reported_only_for_shifted_anchor(self) -> None:
        self.config.write_text(
            'tz = "UTC"\n'
            f'anchor_file_dir = "{self.anchor_files}"\n'
            f'omit_file_dir = "{self.omit_files}"\n'
            'panel_mode = "rich"\n'
            '[business_calendar.work]\n'
            'anchor = "w:mon..fri"\n'
            'omit = "y:04-24"\n',
            encoding="utf-8",
        )
        base = self._task(
            due="20260420T060000Z",
            anchor_mode="skip",
            bc="work",
        )

        shifted = self._run({**base, "anchor": "y:04-24@nbd@t=09:00"})
        unchanged = self._run(
            {
                **base,
                "uuid": "22222222-2222-4222-8222-222222222222",
                "anchor": "y:04-23@nbd@t=09:00",
            }
        )

        self.assertEqual(shifted.returncode, 0, shifted.stderr)
        self.assertEqual(unchanged.returncode, 0, unchanged.stderr)
        self.assertIn("Business calendar adjusted", shifted.stderr)
        self.assertIn("Calendar work", shifted.stderr)
        self.assertIn("Fri 2026-04-24", shifted.stderr)
        self.assertIn("Mon 2026-04-27", shifted.stderr)
        self.assertNotIn("Business calendar adjusted", unchanged.stderr)

    def test_unknown_business_calendar_fails_with_configured_choices(self) -> None:
        self.config.write_text(
            'tz = "UTC"\n'
            f'anchor_file_dir = "{self.anchor_files}"\n'
            f'omit_file_dir = "{self.omit_files}"\n'
            'panel_mode = "rich"\n'
            '[business_calendar.work]\n'
            'anchor = "w:mon..fri"\n',
            encoding="utf-8",
        )
        self._assert_invalid(
            self._task(anchor="w:mon", bc="missing"),
            ("Invalid business calendar", "configured calendars:", "work."),
        )

    def test_invalid_timezone_blocks_scheduled_on_add_task(self) -> None:
        self.config.write_text('tz = "Invalid/Timezone"\n', encoding="utf-8")
        self._assert_invalid(
            self._task(anchor="w:mon"),
            ("Invalid Nautical configuration", "timezone"),
        )

    def test_valid_routes_preserve_or_mutate_the_expected_task_fields(self) -> None:
        explicit_due = "20990102T090000Z"
        routes = (
            (
                "ordinary",
                self._task(description="café ăîșț"),
                {"description": "café ăîșț"},
            ),
            (
                "cp implicit due",
                self._task(cp="1d"),
                {"cp": "1d", "chain": "on", "chainID": "11111111", "link": 1, "due": "2099-01-02T00:00:00+00:00"},
            ),
            (
                "cp explicit due",
                self._task(cp="1d", due=explicit_due),
                {"cp": "1d", "chain": "on", "chainID": "11111111", "link": 1, "due": explicit_due},
            ),
            (
                "anchor",
                self._task(anchor="w:mon", due=explicit_due),
                {"anchor": "w:mon", "chain": "on", "chainID": "11111111", "link": 1, "due": explicit_due},
            ),
            (
                "anchor file",
                self._task(anchor_file="dates.csv@t=12:00", due=explicit_due),
                {"anchor_file": "dates.csv@t=12:00", "chain": "on", "chainID": "11111111", "link": 1, "due": explicit_due},
            ),
            (
                "cp scheduled only",
                self._task(cp="1d", scheduled=explicit_due),
                {"cp": "1d", "chain": "on", "chainID": "11111111", "link": 1, "scheduled": explicit_due, "due": None},
            ),
            (
                "anchor scheduled only",
                self._task(anchor="w:mon", scheduled=explicit_due),
                {"anchor": "w:mon", "chain": "on", "chainID": "11111111", "link": 1, "scheduled": explicit_due, "due": None},
            ),
            (
                "chain limits",
                self._task(cp="1d", due=explicit_due, chainMax="3", chainUntil="20990110T090000Z"),
                {"cp": "1d", "chain": "on", "chainID": "11111111", "link": 1, "chainMax": 3, "chainUntil": "20990110T090000Z", "due": explicit_due},
            ),
        )

        for label, task, expected in routes:
            with self.subTest(route=label):
                result = self._assert_valid(task)
                for field, value in expected.items():
                    if value is None:
                        self.assertNotIn(field, result)
                    else:
                        self.assertEqual(result.get(field), value)
                if label == "ordinary":
                    self.assertFalse(
                        {"anchor", "anchor_file", "cp", "chain", "chainID", "link", "due", "scheduled"}.intersection(result),
                        result,
                    )

    def test_due_mirroring_entry_is_auto_assigned_to_first_anchor_occurrence(self) -> None:
        task = self._task(
            entry="20260412T111500Z",
            due="20260412T111500Z",
            anchor="w:mon,wed,fri",
            anchor_mode="skip",
        )
        started = datetime.now(timezone.utc)

        result = self._assert_valid(task)
        finished = datetime.now(timezone.utc)

        def first_anchor_after(reference: datetime) -> datetime:
            for offset in range(8):
                day = reference.date() + timedelta(days=offset)
                candidate = datetime.combine(day, time(9), timezone.utc)
                if day.weekday() in {0, 2, 4} and candidate >= reference:
                    return candidate
            self.fail("no anchor occurrence found in an eight-day horizon")

        due = datetime.fromisoformat(result["due"])
        self.assertIn(due, {first_anchor_after(started), first_anchor_after(finished)})

    def test_anchor_file_unicode_values_keep_json_stdout_for_implicit_due(self) -> None:
        import nautical_core.anchor_files as anchor_files

        records: list[tuple[date, tuple[int, int], str]] = []
        specs = anchor_files.load_anchor_file_occurrence_specs(
            "dates.csv@t=12:00",
            str(self.anchor_files),
            (9, 0),
            _records_sink=records,
        )
        self.assertEqual(specs, [(date(2099, 1, 5), (12, 0))])
        self.assertEqual(records, [(date(2099, 1, 5), (12, 0), "Întâlnire café")])

        task = self._task(description="întâlnire café", anchor_file="dates.csv@t=12:00")
        raw_process = self._run(task)
        self.assertEqual(raw_process.returncode, 0, raw_process.stderr)
        self.assertIn("întâlnire café", raw_process.stdout)
        self.assertNotIn("\\u00", raw_process.stdout)
        self.assertEqual(json.loads(raw_process.stdout)["description"], "întâlnire café")
        result = self._assert_valid(task)
        self.assertEqual(result["description"], "întâlnire café")
        self.assertEqual(result["anchor_file"], "dates.csv@t=12:00")
        self.assertEqual(result["due"], "2099-01-05T12:00:00+00:00")

    def test_anchor_preview_explains_explicit_omit_rules_without_contaminating_json(self) -> None:
        (self.anchor_files / "2026.csv").write_text(
            "date\n2026-05-01\n2026-05-06\n", encoding="utf-8"
        )
        (self.omit_files / "2026.csv").write_text("date\n2026-05-05\n", encoding="utf-8")
        task = self._task(
            entry="20260413T000000Z",
            anchor="w:tue,fri | y:05-05",
            anchor_file="2026.csv@-1d@t=12:00,18:00",
            omit="w:sun",
            omit_file="2026.csv",
            anchor_mode="skip",
        )

        process = self._run(task)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["anchor_file"], "2026.csv@-1d@t=12:00,18:00")
        self.assertIn("either Tuesdays, Fridays, or May 5 each year; omit", process.stderr)
        self.assertIn("Sundays and Dates from 2026.csv", process.stderr)
        self.assertNotIn("skip missed anchors and Dates from 2026.csv", process.stderr)
        self.assertNotIn("Preview", process.stdout)

    def test_hook_on_add_anchor_preview_skips_omit_date(self) -> None:
        task = self._task(
            project="testing",
            entry="20250108T000000Z",
            anchor="w:mon,wed,fri@t=09:00",
            omit="w:wed",
            anchor_mode="skip",
            due="20250108T090000Z",
        )

        process = self._run(task)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["due"], task["due"])
        self.assertIn("Omit", process.stderr)
        self.assertIn("Except", process.stderr)
        self.assertTrue(
            "Wednesdays" in process.stderr or "Wednesday" in process.stderr,
            process.stderr,
        )
        self.assertIn("2025-01-10", process.stderr)

        diagnostic = self._run(task, diagnostics=True)
        self.assertEqual(diagnostic.returncode, 0, diagnostic.stderr)
        self.assertEqual(json.loads(diagnostic.stdout), result)
        self.assertIn("[nautical]", diagnostic.stderr)

    def test_hook_on_add_anchor_preview_skips_omit_file_date(self) -> None:
        (self.omit_files / "holidays.csv").write_text(
            "date,description\n2025-01-10,Company holiday blackout\n", encoding="utf-8"
        )
        task = self._task(
            project="testing",
            entry="20250108T000000Z",
            anchor="w:mon,wed,fri@t=09:00",
            omit_file="holidays.csv",
            anchor_mode="skip",
            due="20250108T090000Z",
        )

        process = self._run(task)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["due"], task["due"])
        self.assertIn("Omit file", process.stderr)
        self.assertIn("holidays.csv", process.stderr)
        self.assertIn("2025-01-13", process.stderr)
        self.assertNotIn("Company holida...", process.stderr)
        self.assertNotIn("Fri 2025-01-10", process.stderr)
        self.assertNotIn("2025-01-10 09:00", process.stderr)

        diagnostic = self._run(task, diagnostics=True)
        self.assertEqual(diagnostic.returncode, 0, diagnostic.stderr)
        self.assertEqual(json.loads(diagnostic.stdout), result)
        self.assertNotIn("Company holida...", diagnostic.stderr)
        self.assertNotIn("2025-01-10 09:00", diagnostic.stderr)
        self.assertIn("[nautical]", diagnostic.stderr)

    def test_hook_on_add_anchor_preview_marks_omitted_future_slots(self) -> None:
        task = self._task(
            project="testing",
            entry="20250108T000000Z",
            anchor="w:mon,wed,fri@t=09:00",
            omit="w:wed",
            anchor_mode="skip",
            due="20250108T090000Z",
        )

        process = self._run(task)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["due"], task["due"])
        self.assertNotIn("(omitted)", process.stderr)
        self.assertNotIn("2025-01-15", process.stderr)

        diagnostic = self._run(task, diagnostics=True)
        self.assertEqual(diagnostic.returncode, 0, diagnostic.stderr)
        self.assertEqual(json.loads(diagnostic.stdout), result)
        self.assertNotIn("2025-01-15", diagnostic.stderr)
        self.assertIn("[nautical]", diagnostic.stderr)

    def test_hook_on_add_anchor_preview_applies_omit_file_business_day_modifier(self) -> None:
        (self.omit_files / "holidays.csv").write_text(
            "date,description\n2026-04-25,Weekend holiday\n", encoding="utf-8"
        )
        task = self._task(
            project="testing",
            entry="20260412T000000Z",
            due="20260412T090000Z",
            anchor="y:04-25@nbd@t=09:00",
            omit_file="holidays.csv@nbd",
            anchor_mode="skip",
        )

        process = self._run(task)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["due"], task["due"])
        self.assertIn("Mon 2027-04-26 09:00", process.stderr)
        self.assertNotIn("Mon 2026-04-27 09:00", process.stderr)

        diagnostic = self._run(task, diagnostics=True)
        self.assertEqual(diagnostic.returncode, 0, diagnostic.stderr)
        self.assertEqual(json.loads(diagnostic.stdout), result)
        self.assertIn("Mon 2027-04-26 09:00", diagnostic.stderr)
        self.assertNotIn("Mon 2026-04-27 09:00", diagnostic.stderr)
        self.assertIn("[nautical]", diagnostic.stderr)

    def test_large_weekly_interval_hook_route_is_not_clamped(self) -> None:
        task = self._task(
            entry="20260809T090000Z",
            anchor="w/500:mon + y:01-01",
        )
        result = self._assert_valid(task)
        self.assertEqual(result["anchor"], "w/500:mon + y:01-01")
        self.assertEqual(result["chain"], "on")
        self.assertGreater(str(result["due"]), "2026-01-01")

    def test_large_interval_with_unreachable_chain_until_fails_actionably(self) -> None:
        task = self._task(
            entry="20260809T090000Z",
            anchor="w/500:mon + y:01-01",
            chain="on",
            chainID="00000901",
            chainUntil="20270101T000000Z",
        )
        self._assert_invalid(task, ("Chain end point is earlier",))

    def test_invalid_routes_fail_without_json_or_tracebacks(self) -> None:
        cases = (
            ("malformed cp reversed range", self._task(cp="rand(7d..3d)"), ("Invalid cp", "lower bound", "upper")),
            ("malformed cp separator", self._task(cp="rand(3d-7d)"), ("Invalid cp", "expected rand(<duration>..<duration>)")),
            ("malformed cp jitter bound", self._task(cp="14d~abc"), ("Invalid cp", "invalid", "bound")),
            ("malformed cp negative jitter", self._task(cp="2d~3d"), ("Invalid cp", "lower bound must be >= 0")),
            ("malformed cp empty sequence", self._task(cp="3d,,7d"), ("Invalid cp", "empty duration", "position 2")),
            ("malformed anchor", self._task(anchor="not-a-valid-expression"), ("Invalid anchor",)),
            (
                "astronomy without a resolvable event",
                self._task(anchor="(moon:last-quarter + y:jul)@t=moonrise"),
                ("No astronomical occurrence", "astral", "astronomy profile"),
            ),
            ("missing anchor file", self._task(anchor_file="missing.csv"), ("Invalid anchor_file",)),
            ("cp zero limit", self._task(cp="1d", chainMax=0), ("Invalid chainMax", "chainMax must be a positive integer")),
            ("cp negative limit", self._task(cp="1d", chainMax=-1), ("Invalid chainMax", "chainMax must be a positive integer")),
            ("cp fractional limit", self._task(cp="1d", chainMax=2.5), ("Invalid chainMax", "chainMax must be a positive integer")),
            ("anchor zero limit", self._task(anchor="w:mon", chainMax=0), ("Invalid chainMax", "chainMax must be a positive integer")),
            ("conflicting kinds", self._task(cp="1d", anchor="w:mon"), ("Invalid Nautical task", "cannot be combined")),
        )
        for label, task, expected in cases:
            with self.subTest(route=label):
                self._assert_invalid(task, expected)


if __name__ == "__main__":
    unittest.main()
