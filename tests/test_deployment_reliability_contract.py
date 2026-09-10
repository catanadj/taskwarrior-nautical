from __future__ import annotations

import contextlib
import io
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dev_tools import nautical_deploy_sanity as deploy
from dev_tools import nautical_reliability_smoke as reliability


ROOT = Path(__file__).parents[1]


class DeploymentSanityContractTests(unittest.TestCase):
    def test_strict_json_object_accepts_unicode_and_rejects_non_object_envelopes(self) -> None:
        ok, message = deploy._strict_json_object('{"status":"ok","label":"café"}')
        self.assertTrue(ok, message)

        for payload in ("[]", '{"status":"ok"}\nnoise', ""):
            accepted, _ = deploy._strict_json_object(payload)
            self.assertFalse(accepted, payload)

    def test_required_inventory_reports_missing_runtime_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-deploy-contract-") as td:
            root = Path(td)
            (root / "nautical_core").mkdir()
            (root / "nautical_core" / "__init__.py").write_text("", encoding="utf-8")
            results = deploy._check_required_files(root, require_exec=False)
            missing = [item for item in results if not item["ok"]]
            self.assertTrue(missing)
            self.assertTrue(any(item["path"] == "bootstrap.sh" for item in missing))

    def test_broken_temporary_inventory_produces_json_failure_and_nonzero_status(self) -> None:
        script = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
        with tempfile.TemporaryDirectory(prefix="nautical-deploy-contract-") as td:
            candidate = Path(td) / "candidate"
            shutil.copytree(ROOT, candidate, ignore=shutil.ignore_patterns(".git", "__pycache__", ".nautical-cache"))
            (candidate / "nautical_core" / "tools" / "nautical_doctor.py").unlink()
            proc = subprocess.run(
                [sys.executable, str(script), "--root", str(candidate), "--no-require-exec", "--json"],
                text=True,
                capture_output=True,
                timeout=20.0,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertEqual(proc.stderr, "")
            payload = json.loads(proc.stdout)
            self.assertEqual(payload["status"], "fail")
            self.assertTrue(any(item.get("path") == "nautical_core/tools/nautical_doctor.py" and not item.get("ok")
                                for item in payload["results"]))

    def test_valid_json_report_is_stdout_only(self) -> None:
        script = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--json", "--no-require-exec"],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        self.assertEqual(proc.stderr, "")
        payload = json.loads(proc.stdout)
        # The current dirty checkout may intentionally contain an unrelated
        # deploy finding; the contract under test is the report envelope and
        # its stdout isolation, not a second assertion of the full audit.
        self.assertIn(payload["status"], {"ok", "fail"})
        self.assertIsInstance(payload["results"], list)
        self.assertTrue(payload["results"])

    def test_argument_validation_rejects_unknown_deploy_option(self) -> None:
        script = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
        proc = subprocess.run([sys.executable, str(script), "--not-an-option"], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("usage:", proc.stderr)


class ReliabilitySmokeContractTests(unittest.TestCase):
    def test_export_one_supports_array_json_jsonl_and_invalid_output(self) -> None:
        completed = lambda text: SimpleNamespace(stdout=text)
        with patch.object(reliability, "_task", return_value=completed('[{"uuid":"u1"}]')):
            self.assertEqual(reliability._export_one(["export"], {}), {"uuid": "u1"})
        with patch.object(reliability, "_task", return_value=completed('{"uuid":"u2"}\n')):
            self.assertEqual(reliability._export_one(["export"], {}), {"uuid": "u2"})
        with patch.object(reliability, "_task", return_value=completed("not-json")):
            self.assertIsNone(reliability._export_one(["export"], {}))

    def test_outbox_failure_and_child_uuid_contracts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-reliability-contract-") as td:
            state = Path(td) / ".nautical-state"
            state.mkdir()
            db = state / ".nautical_lifecycle_outbox.db"
            with sqlite3.connect(db) as connection:
                connection.execute("CREATE TABLE lifecycle_outbox (processing_state TEXT, failure_json TEXT, plan_json TEXT, updated_at TEXT, intent_id TEXT)")
                connection.execute("INSERT INTO lifecycle_outbox VALUES (?, ?, ?, ?, ?)",
                                   ("ready", "", json.dumps({"child_payload": {"uuid": "child-1"}}), "1", "1"))
                connection.execute("INSERT INTO lifecycle_outbox VALUES (?, ?, ?, ?, ?)",
                                   ("manual_review", '{"error":"broken"}', "{}", "2", "2"))
                connection.commit()
            self.assertEqual(reliability._read_queue_child_uuid(Path(td)), "child-1")
            failed, reason = reliability._check_failure_detail(Path(td))
            self.assertTrue(failed)
            self.assertIn("broken", reason or "")

    def test_reset_signal_files_is_safe_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-reliability-contract-") as td:
            state = Path(td) / ".nautical-state"
            state.mkdir()
            for name in (".nautical_lifecycle_outbox.db", ".nautical_lifecycle_outbox.db-shm", ".nautical_lifecycle_outbox.db-wal"):
                (state / name).write_text("x", encoding="utf-8")
            reliability._reset_signal_files(Path(td))
            reliability._reset_signal_files(Path(td))
            self.assertFalse(any(state.iterdir()))

    def test_default_smoke_profile_runs_without_live_taskdata_when_boundaries_are_stubbed(self) -> None:
        completed = SimpleNamespace(stdout="{}\n", stderr="", returncode=0)
        output = io.StringIO()
        with patch.object(reliability, "_add_task", side_effect=[1, 2, 3]), \
             patch.object(reliability, "_task", return_value=completed), \
             patch.object(reliability, "_check_failure_detail", return_value=(False, None)), \
             contextlib.redirect_stdout(output), \
             patch.object(sys, "argv", ["nautical_reliability_smoke.py"]):
            self.assertEqual(reliability.main(), 0)
        self.assertIn("all tests completed", output.getvalue())
        self.assertIn("happy path ok", output.getvalue())

    def test_argument_validation_rejects_non_numeric_load(self) -> None:
        script = ROOT / "dev_tools" / "nautical_reliability_smoke.py"
        proc = subprocess.run([sys.executable, str(script), "--load", "many"], text=True, capture_output=True)
        self.assertEqual(proc.returncode, 2)
        self.assertEqual(proc.stdout, "")
        self.assertIn("invalid int value", proc.stderr)


if __name__ == "__main__":
    unittest.main()
