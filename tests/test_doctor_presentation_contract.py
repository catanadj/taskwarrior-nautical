"""Direct contracts for the concise text projection emitted by Doctor."""

import importlib
import io
import unittest


doctor = importlib.import_module("nautical_core.tools.nautical_doctor")


class DoctorPresentationContractTests(unittest.TestCase):
    def test_timezone_summary_is_prominent(self) -> None:
        payload = {
            "status": "warn",
            "taskdata": "/tmp/task",
            "operator_findings": [
                {
                    "code": "config.timezone.invalid",
                    "domain": "config",
                    "severity": "warning",
                    "actionability": "actionable",
                    "message": "Nautical timezone 'Europe/Bucharest' is not available; hooks will use UTC fallback.",
                    "observed": {},
                    "expected": {},
                    "evidence": {"tz": "Europe/Bucharest"},
                    "guidance": "Use an available timezone.",
                }
            ],
        }
        output = io.StringIO()
        doctor._render_text(payload, stream=output)
        self.assertIn(
            "Timezone: Europe/Bucharest unavailable; UTC fallback active",
            output.getvalue(),
        )

    def test_large_healthy_history_is_compact_and_actionable(self) -> None:
        findings = [
            {
                "code": f"healthy.{index}", "domain": "configuration",
                "severity": "info", "actionability": "informational",
                "message": f"healthy {index}", "observed": {}, "expected": {},
                "evidence": {}, "guidance": "",
            }
            for index in range(1000)
        ]
        findings.append(
            {
                "code": "chains.active_issue", "domain": "chains",
                "severity": "error", "actionability": "actionable",
                "message": "one active issue", "observed": {}, "expected": {},
                "evidence": {}, "guidance": "Run nautical query integrity --all.",
            }
        )
        output = io.StringIO()
        doctor._render_text(
            {"status": "error", "taskdata": "/tmp/task", "operator_findings": findings},
            stream=output,
        )
        rendered = output.getvalue()
        self.assertIn("one active issue", rendered)
        self.assertNotIn("healthy 0", rendered)
        self.assertNotIn("healthy 999", rendered)
        self.assertLess(len(rendered.splitlines()), 20)

    def test_historical_findings_are_grouped_across_chains(self) -> None:
        findings = [
            {
                "code": "chains.carry.child_relative_offset", "domain": "chains",
                "severity": "info", "actionability": "informational",
                "message": "historical carry difference",
                "observed": {"field": "scheduled"}, "expected": {},
                "evidence": {"historical": True, "chainID": f"chain-{index}"},
                "guidance": "No action is required; current pending-chain findings are reported separately.",
            }
            for index in range(100)
        ]
        output = io.StringIO()
        doctor._render_text(
            {"status": "ok", "taskdata": "/tmp/task", "operator_findings": findings},
            stream=output,
        )
        rendered = output.getvalue()
        self.assertIn("100 completed-link scheduled observation(s)", rendered)
        self.assertEqual(rendered.count("chains.historical_summary"), 1)


if __name__ == "__main__":
    unittest.main()
