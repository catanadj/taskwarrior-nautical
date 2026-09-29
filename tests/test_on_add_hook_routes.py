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
