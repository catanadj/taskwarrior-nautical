from __future__ import annotations

import contextlib
import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from dev_tools import nautical_stress_campaign as campaign
from dev_tools import nautical_mixed_recurrence_loop as mixed


class StressCampaignTests(unittest.TestCase):
    def test_profiles_have_explicit_cycle_and_timeout_budgets(self) -> None:
        self.assertEqual(campaign.PROFILE_BUDGETS["ci"], (8, 300.0))
        self.assertEqual(campaign.PROFILE_BUDGETS["nightly"], (24, 300.0))
        self.assertEqual(campaign.PROFILE_BUDGETS["stress"], (64, 900.0))

    def test_health_warning_is_retained_without_failing_campaign(self) -> None:
        payload = '{"status":"warn","checks":[]}'
        process = SimpleNamespace(returncode=1, stdout=payload, stderr="")
        with patch.object(campaign.subprocess, "run", return_value=process):
            result = mixed._health_snapshot({}, None)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "warn")

    def test_stage_selects_profile_budget_and_enforces_json_success(self) -> None:
        process = SimpleNamespace(
            returncode=0,
            stdout='{"ok":true,"cycles_completed":24,"violations":[]}',
            stderr="",
        )
        with patch.object(campaign.subprocess, "run", return_value=process) as run:
            result = campaign._run_stage("nightly")

        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--cycles") + 1], "24")
        self.assertIn("--enforce", command)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["cycles"], 24)
        self.assertEqual(result["violations"], [])

    def test_stage_rejects_malformed_result_and_preserves_stderr(self) -> None:
        process = SimpleNamespace(returncode=0, stdout="not-json", stderr="runner warning")
        with patch.object(campaign.subprocess, "run", return_value=process):
            result = campaign._run_stage("ci")

        self.assertEqual(result["status"], "failed")
        self.assertIn("invalid JSON", result["error"])
        self.assertEqual(result["stderr"], "runner warning")

    def test_stage_fails_when_campaign_report_is_unhealthy(self) -> None:
        process = SimpleNamespace(
            returncode=0,
            stdout='{"ok":false,"cycles_completed":7,"violations":["lost anchor"]}',
            stderr="",
        )
        with patch.object(campaign.subprocess, "run", return_value=process):
            result = campaign._run_stage("ci")

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["cycles"], 7)
        self.assertEqual(result["violations"], ["lost anchor"])

    def test_main_emits_strict_json_without_diagnostics(self) -> None:
        stage = {
            "stage": "mixed_recurrence",
            "status": "passed",
            "duration_s": 0.01,
            "cycles": 8,
            "violations": [],
            "error": "",
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(campaign, "_run_stage", return_value=stage), patch.object(
            sys, "argv", ["nautical_stress_campaign.py", "--profile", "ci", "--json"]
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = campaign.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr.getvalue(), "")
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["profile"], "ci")
        self.assertEqual(payload["failed_stages"], [])

    def test_main_enforce_returns_nonzero_for_failed_stage(self) -> None:
        stage = {"stage": "mixed_recurrence", "status": "failed", "error": "threshold"}
        stdout = io.StringIO()
        with patch.object(campaign, "_run_stage", return_value=stage), patch.object(
            sys,
            "argv",
            ["nautical_stress_campaign.py", "--profile", "stress", "--json", "--enforce"],
        ), contextlib.redirect_stdout(stdout):
            exit_code = campaign.main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failed_stages"], ["mixed_recurrence"])

    def test_main_rejects_unknown_profile_before_running_campaign(self) -> None:
        with patch.object(campaign, "_run_stage") as run, patch.object(
            sys, "argv", ["nautical_stress_campaign.py", "--profile", "unknown"]
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    campaign.main()

        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
