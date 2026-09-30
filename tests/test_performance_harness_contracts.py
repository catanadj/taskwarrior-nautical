"""Direct contracts for standalone performance and soak harnesses."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PerformanceHarnessContractTests(unittest.TestCase):
    def test_hook_replay_harness_reports_ok(self) -> None:
        path = ROOT / "dev_tools" / "nautical_hook_replay.py"
        corpus = ROOT / "dev_tools" / "nautical_hook_replay_corpus.jsonl"
        process = subprocess.run(
            [sys.executable, str(path), "--json", "--corpus", str(corpus)],
            text=True,
            capture_output=True,
            timeout=12.0,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads((process.stdout or "").strip() or "{}")
        self.assertEqual(payload.get("status"), "ok", payload)
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        self.assertTrue(results, "replay harness should report per-case results")
        self.assertTrue(
            all(bool(result.get("ok")) for result in results if isinstance(result, dict)),
            f"failing replay result: {results}",
        )

    def test_mixed_recurrence_loop_harness_reports_ok(self) -> None:
        path = ROOT / "dev_tools" / "nautical_mixed_recurrence_loop.py"
        process = subprocess.run(
            [sys.executable, str(path), "--cycles", "3", "--json"],
            text=True,
            capture_output=True,
            timeout=30.0,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads((process.stdout or "").strip() or "{}")
        self.assertIs(payload.get("ok"), True, payload)
        self.assertGreaterEqual(int(payload.get("cycles_completed") or 0), 1, payload)
        self.assertFalse(payload.get("violations"), payload)

    def test_soak_runner_reports_ok(self) -> None:
        path = ROOT / "dev_tools" / "nautical_soak_test.py"
        process = subprocess.run(
            [
                sys.executable,
                str(path),
                "--seconds",
                "2",
                "--batch-size",
                "4",
                "--anchor-rate",
                "0.5",
                "--cp-rate",
                "0.5",
                "--done-rate",
                "0.5",
                "--progress-every-seconds",
                "0",
                "--json",
                "--enforce",
            ],
            text=True,
            capture_output=True,
            timeout=240,
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        payload = json.loads((process.stdout or "").strip() or "{}")
        self.assertIs(payload.get("ok"), True, payload)
        self.assertFalse(payload.get("violations"), payload)


if __name__ == "__main__":
    unittest.main()
