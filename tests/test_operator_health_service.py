import unittest

from nautical_core.operator_health_service import OperatorHealthService
from nautical_core.operator_findings import FindingActionability, FindingSeverity, OperatorFinding
from nautical_core.operator_models import OperatorStatus


class OperatorHealthServiceTests(unittest.TestCase):
    def test_report_is_deterministic_and_deduplicated(self) -> None:
        finding = OperatorFinding(
            "config.invalid", "configuration", FindingSeverity.ERROR,
            FindingActionability.BLOCKING, "Configuration is invalid.", guidance="Fix configuration.",
        )
        report = OperatorHealthService.report([finding, finding])
        self.assertEqual(report.status, OperatorStatus.ERROR)
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.to_dict()["findings"][0]["code"], "config.invalid")

    def test_empty_report_is_healthy(self) -> None:
        report = OperatorHealthService.report(())
        self.assertEqual(report.status, OperatorStatus.OK)
        self.assertEqual(report.to_dict(), {"status": "ok", "findings": []})


if __name__ == "__main__":
    unittest.main()
