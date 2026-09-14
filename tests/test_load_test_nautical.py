"""Deterministic contract tests for the load-test support functions."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from dev_tools import load_test_nautical as loadtest


class LoadTestSupportTests(unittest.TestCase):
    def test_percentile_is_deterministic_and_clamps_percentiles(self) -> None:
        self.assertEqual(loadtest._percentile([], 95), 0.0)
        self.assertEqual(loadtest._percentile([3.0, 1.0, 2.0], 50), 2.0)
        self.assertEqual(loadtest._percentile([3.0, 1.0, 2.0], -10), 1.0)
        self.assertEqual(loadtest._percentile([3.0, 1.0, 2.0], 110), 3.0)

    def test_parse_created_id_accepts_taskwarrior_output_only(self) -> None:
        self.assertEqual(loadtest._parse_created_id("Created task 42\n"), 42)
        self.assertEqual(loadtest._parse_created_id("Created task 7 (loadtest task)"), 7)
        self.assertIsNone(loadtest._parse_created_id("task 42 modified"))
        self.assertIsNone(loadtest._parse_created_id(""))

    def test_verify_link_rows_accepts_one_exact_child(self) -> None:
        parent = {"uuid": "parent-uuid", "chainID": "chain", "link": 3, "nextLink": "child-uu"}
        child = {"uuid": "child-uuid", "chainID": "chain", "link": 4, "prevLink": "parent-u"}
        result = loadtest._verify_link_rows([parent, child], ["parent-uuid"])
        self.assertEqual(result["expected"], 1)
        self.assertEqual(result["verified"], 1)
        self.assertEqual(result["failures"], [])

    def test_verify_link_rows_reports_missing_and_duplicate_children(self) -> None:
        parent = {"uuid": "parent-uuid", "chainID": "chain", "link": 3, "nextLink": "child-u"}
        missing = loadtest._verify_link_rows([parent], ["parent-uuid"])
        self.assertEqual(missing["verified"], 0)
        self.assertIn("expected one child", missing["failures"][0])

        child = {"uuid": "child-uuid", "chainID": "chain", "link": 4, "prevLink": "parent-u"}
        duplicate = loadtest._verify_link_rows([parent, child, dict(child, uuid="child-2")], ["parent-uuid"])
        self.assertEqual(duplicate["verified"], 0)
        self.assertIn("found 2", duplicate["failures"][0])

    def test_verify_link_rows_ignores_deleted_children(self) -> None:
        parent = {"uuid": "parent-uuid", "chainID": "chain", "link": 3, "nextLink": "child-u"}
        deleted = {
            "uuid": "child-uuid",
            "chainID": "chain",
            "link": 4,
            "prevLink": "parent-u",
            "status": "deleted",
        }
        result = loadtest._verify_link_rows([parent, deleted], ["parent-uuid"])
        self.assertEqual(result["verified"], 0)
        self.assertIn("found 0", result["failures"][0])

    def test_run_task_converts_timeout_to_safe_failure(self) -> None:
        with patch.object(
            loadtest.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["task"], timeout=0.01),
        ):
            ok, stdout, stderr, elapsed = loadtest._run_task(["task", "export"], {})
        self.assertFalse(ok)
        self.assertEqual((stdout, stderr), ("", "timeout"))
        self.assertGreaterEqual(elapsed, 0.0)

    def test_queue_metrics_fails_closed_for_missing_database(self) -> None:
        with TemporaryDirectory() as directory:
            result = loadtest._queue_metrics(Path(directory))
        self.assertEqual(result, {"items": 0, "bytes": 0})

    def test_main_reports_json_when_taskwarrior_is_unavailable(self) -> None:
        with patch.object(loadtest, "_which_task", return_value=None), patch.object(
            sys, "argv", ["load_test_nautical.py", "--json"]
        ), patch("sys.stdout") as stdout:
            # Use a real StringIO-like capture while keeping the test isolated.
            from io import StringIO

            capture = StringIO()
            stdout.write.side_effect = capture.write
            stdout.flush.side_effect = capture.flush
            result = loadtest.main()
        self.assertEqual(result, 2)
        payload = json.loads(capture.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["mode"], "unavailable")


if __name__ == "__main__":
    unittest.main()
