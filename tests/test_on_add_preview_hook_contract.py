"""Focused contracts for on-add preview orchestration and fallback output."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from datetime import date, timezone
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_add_hook():
    path = ROOT / "nautical_core" / "hooks" / "add_impl.py"
    name = f"_nautical_add_hook_contract_{id(path)}_{len(sys.modules)}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load on-add implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.core = __import__("nautical_core")
    module._CORE_READY = True
    return module


class OnAddPreviewHookContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self._run_isolated = os.environ.get("NAUTICAL_ADD_HOOK_CONTRACT_CHILD") != "1"
        if self._run_isolated:
            return
        self.hook = _load_add_hook()
        now_utc = self.hook.core.build_local_datetime(date(2026, 4, 12), (12, 0)).astimezone(timezone.utc)
        self.task = {
            "uuid": "00000000-0000-4000-8000-000000000143",
            "description": "on-add evaluator contract",
            "status": "pending",
            "entry": self.hook.core.fmt_isoz(now_utc),
            "anchor": "w:mon",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "00000000",
        }
        self.context = self.hook._module("add_composition").build_on_add_context(
            self.hook, self.task, now_utc, self.hook.core.to_local(now_utc)
        )

    def _run_in_child_process(self) -> None:
        environment = os.environ.copy()
        environment["NAUTICAL_ADD_HOOK_CONTRACT_CHILD"] = "1"
        test_id = self.id()
        if test_id.startswith("test_on_add_preview_hook_contract."):
            test_id = f"tests.{test_id}"
        result = subprocess.run(
            [sys.executable, "-m", "unittest", test_id, "-q"],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_on_add_preview_fails_closed_when_evaluator_initialization_fails(self) -> None:
        if self._run_isolated:
            self._run_in_child_process()
            return
        panels = []
        scheduler_service = __import__("nautical_core.scheduler_service", fromlist=["SchedulerService"])

        def fail_from_task(cls, *args, **kwargs):
            raise RuntimeError("astronomy profile is unavailable")

        with (
            patch.object(scheduler_service.SchedulerService, "from_task", classmethod(fail_from_task)),
            patch.object(self.hook, "_panel", side_effect=lambda title, rows, **kwargs: panels.append((title, list(rows)))),
            self.assertRaises(SystemExit) as raised,
        ):
            self.hook._module("add_composition").render_anchor_preview(
                self.hook, self.context, prof=self.hook._NoopProfiler()
            )

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(panels[-1][0], "❌ Invalid Chain")
        self.assertIn("Recurrence evaluator", [label for label, _ in panels[-1][1]])
        self.assertIn("Fix", [label for label, _ in panels[-1][1]])

    def test_on_add_preview_reports_scheduler_exhaustion_actionably(self) -> None:
        if self._run_isolated:
            self._run_in_child_process()
            return
        panels = []
        preview = self.hook._module("add_anchor_preview")
        exhaustion = self.hook.core.OccurrenceSearchExhausted(
            "test preview", reference=date(2026, 4, 12), limit=1
        )

        with (
            patch.object(preview, "handle_anchor_preview_on_add", side_effect=exhaustion),
            patch.object(self.hook, "_panel", side_effect=lambda title, rows, **kwargs: panels.append((title, list(rows)))),
            self.assertRaises(SystemExit) as raised,
        ):
            self.hook._module("add_composition").render_anchor_preview(
                self.hook, self.context, prof=self.hook._NoopProfiler()
            )

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(panels[-1][0], "❌ Invalid Chain")
        rows = dict(panels[-1][1])
        self.assertIn("test preview", rows["Scheduler"])
        self.assertIn("less sparse", rows["Fix"])

    def test_on_add_preview_uses_evaluator_for_first_due_and_upcoming_rows(self) -> None:
        if self._run_isolated:
            self._run_in_child_process()
            return
        captured = {}
        with patch.object(
            self.hook,
            "_panel",
            side_effect=lambda title, rows, **kwargs: captured.update(title=title, rows=list(rows)),
        ), patch.object(self.hook, "_fmt_local_for_task", self.hook.core.fmt_isoz):
            self.hook._module("add_composition").render_anchor_preview(
                self.hook, self.context, prof=self.hook._NoopProfiler()
            )

        self.assertTrue(self.task.get("due"), captured)
        self.assertEqual(captured["title"], "⚓︎ Anchor Preview")
        labels = [label for label, _ in captured["rows"]]
        self.assertIn("First due", labels)
        self.assertIn("Upcoming", labels)

    def test_on_add_fail_and_exit_emits_no_task_json(self) -> None:
        if self._run_isolated:
            self._run_in_child_process()
            return
        task = {"uuid": "00000000-0000-4000-8000-000000000abc", "description": "fail test"}
        self.hook._PARSED_TASK = dict(task)
        self.hook._RAW_INPUT_TEXT = json.dumps(task, ensure_ascii=False)
        stdout = io.StringIO()
        with (
            patch.object(self.hook, "_panel"),
            contextlib.redirect_stdout(stdout),
            self.assertRaises(SystemExit) as raised,
        ):
            self.hook._fail_and_exit("Invalid anchor", "anchor syntax error: bad")

        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(stdout.getvalue(), "")

    def test_on_add_panic_passthrough_emits_valid_json(self) -> None:
        if self._run_isolated:
            self._run_in_child_process()
            return
        self.hook._PARSED_TASK = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "description": "panic-add",
        }
        self.hook._RAW_INPUT_TEXT = "{not-json"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.hook._panic_passthrough()

        result = json.loads(stdout.getvalue())
        self.assertEqual(result["uuid"], "00000000-0000-4000-8000-000000000111")
        self.assertEqual(result["description"], "panic-add")

    def test_on_add_rejects_oversized_stdin_early(self) -> None:
        if self._run_isolated:
            self._run_in_child_process()
            return

        self.hook._MAX_JSON_BYTES = 32
        raw = json.dumps({"uuid": "u", "status": "pending", "description": "x" * 128})
        failure = RuntimeError("hook rejected oversized input")
        with (
            patch.object(self.hook, "_fail_and_exit", side_effect=failure) as fail,
            patch.object(sys, "stdin", io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")),
            self.assertRaisesRegex(RuntimeError, "rejected oversized input"),
        ):
            self.hook._read_on_add_task(self.hook._NoopProfiler())

        fail.assert_called_once()
        self.assertEqual(fail.call_args.args[0], "Invalid input")
        self.assertIn("exceeds 32 bytes", fail.call_args.args[1])
