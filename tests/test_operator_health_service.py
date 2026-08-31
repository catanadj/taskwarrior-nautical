import unittest
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from nautical_core.operator_health_service import OperatorHealthService
from nautical_core.operator_findings import FindingActionability, FindingSeverity, OperatorFinding
from nautical_core.operator_models import OperatorStatus


class OperatorHealthServiceTests(unittest.TestCase):
    def test_storage_findings_report_capacity_without_writes(self) -> None:
        stats = SimpleNamespace(f_frsize=4096, f_bavail=10, f_blocks=20, f_favail=30, f_files=40)
        findings = OperatorHealthService.storage_findings(
            {"taskdata": "/tmp/taskdata"}, statvfs_factory=lambda _path: stats,
        )
        self.assertEqual(len(findings), 1)
        payload = findings[0].to_dict()
        self.assertEqual(payload["severity"], "info")
        self.assertEqual(payload["observed"]["free_bytes"], 40960)
        self.assertEqual(payload["observed"]["free_inodes"], 30)

    def test_storage_findings_report_unavailable_path(self) -> None:
        findings = OperatorHealthService.storage_findings(
            {"backup": "/tmp/missing"}, statvfs_factory=lambda _path: (_ for _ in ()).throw(OSError("denied")),
        )
        payload = findings[0].to_dict()
        self.assertEqual(payload["severity"], "error")
        self.assertIn("denied", payload["observed"]["error"])
    def test_astronomy_finding_normalizes_timezone_for_json(self) -> None:
        findings = OperatorHealthService.astronomy_findings(
            {}, effective_timezone=ZoneInfo("Europe/Bucharest"), source_hint="config",
            preflight=lambda _: {
                "status": "ok", "event": "sunrise", "provider_timezone": ZoneInfo("UTC")
            },
        )
        payload = findings[0].to_dict()
        self.assertEqual(payload["observed"]["effective_timezone"], "Europe/Bucharest")
        self.assertEqual(payload["observed"]["provider_timezone"], "UTC")

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
