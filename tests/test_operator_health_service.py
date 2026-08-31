import unittest
import json
import tempfile
import sys
from pathlib import Path
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

    def test_deep_identity_findings_are_injectable_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime_root = Path(td) / "runtime"
            (runtime_root / "releases" / "r-test").mkdir(parents=True)
            runtime = {
                "runtime_root": str(runtime_root),
                "active_release": "r-test",
                "manifest": {"content_sha256": "digest"},
            }
            calls: list[str] = []
            def probe(path: str) -> tuple[bool, str]:
                calls.append(path)
                return True, "version"
            findings = OperatorHealthService.deep_identity_findings(
                runtime, sys.executable, sys.executable,
                digest_factory=lambda path: "digest",
                version_probe=probe,
            )
            self.assertEqual([item.code for item in findings], ["install.release_digest", "taskwarrior.identity", "python.identity"])
            self.assertEqual(len(calls), 2)

    def test_deep_resource_findings_validate_timezone_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            resource = Path(td) / "calendar.json"
            resource.write_text("{}", encoding="utf-8")
            findings = OperatorHealthService.deep_resource_findings(
                "UTC", {"calendar": resource}, timezone_factory=lambda value: object(),
            )
            self.assertEqual([item.severity.value for item in findings], ["info", "info"])
            missing = OperatorHealthService.deep_resource_findings(
                "No/Such/Zone", {"calendar": resource}, timezone_factory=lambda value: (_ for _ in ()).throw(ValueError("bad zone")),
            )
            self.assertEqual(missing[0].severity.value, "error")
            self.assertEqual(missing[1].severity.value, "info")

    def test_deep_local_state_checks_are_injectable_and_select_newest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            backup = root / "backups"
            backup.mkdir()
            older = backup / "older"
            newest = backup / "newest"
            older.mkdir()
            newest.mkdir()
            manifest = {"metadata": {"restore_tool_schema": 1, "created_at": 90.0}}
            (older / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (newest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            import os
            os.utime(older, (1, 1))
            os.utime(newest, (2, 2))
            checked: list[Path] = []
            def verify(path: Path) -> bool:
                checked.append(path)
                return True
            findings = OperatorHealthService.deep_local_state_findings(
                root / "outbox.db", backup,
                quick_check=lambda path: "ok",
                backup_checker=verify,
                clock=lambda: 100.0,
            )
            self.assertEqual([item.severity.value for item in findings], ["info", "info"])
            self.assertEqual(checked, [newest])
            self.assertEqual(findings[1].to_dict()["observed"]["age_seconds"], 10.0)

    def test_deep_local_state_rejects_missing_backup_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            generation = Path(td) / "backups" / "generation"
            generation.mkdir(parents=True)
            (generation / "manifest.json").write_text(json.dumps({"metadata": {}}), encoding="utf-8")
            findings = OperatorHealthService.deep_local_state_findings(
                Path(td) / "outbox.db", generation.parent,
                quick_check=lambda path: "ok", backup_checker=lambda path: True,
            )
            self.assertEqual(findings[1].severity.value, "error")
            self.assertIn("restore-tool schema", findings[1].observed["error"])

    def test_deep_clock_reports_only_clock_before_evidence(self) -> None:
        runtime = {"manifest": {"created_at": 200.0}}
        findings = OperatorHealthService.deep_clock_findings(runtime, clock=lambda: 100.0)
        self.assertEqual([item.code for item in findings], ["time.before_release"])
        self.assertEqual(OperatorHealthService.deep_clock_findings(runtime, clock=lambda: 300.0), ())

    def test_deep_ownership_checks_permissions_and_release_containment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            release = root / "runtime" / "releases" / "r-test"
            release.mkdir(parents=True)
            for name in ("nautical", "nautical_navigator.py"):
                (release / name).write_text("x", encoding="utf-8")
            implementation = release / "nautical_core" / "hooks" / "add_impl.py"
            implementation.parent.mkdir(parents=True)
            implementation.write_text("x", encoding="utf-8")
            hooks = root / "hooks"
            hooks.mkdir()
            for name in ("on-add.nautical", "on-modify.nautical", "on-exit.nautical"):
                (hooks / name).write_text("x", encoding="utf-8")
            runtime = {"runtime_root": str(root / "runtime"), "active_release": "r-test"}
            mode = SimpleNamespace(st_mode=0o700)
            records = {"on-add": {"implementation": str(implementation)}}
            findings = OperatorHealthService.deep_ownership_findings(runtime, hooks, records, stat_factory=lambda path: mode)
            self.assertEqual(findings[0].severity.value, "info")
            broken = OperatorHealthService.deep_ownership_findings(runtime, hooks, records, stat_factory=lambda path: SimpleNamespace(st_mode=0o400))
            self.assertEqual(broken[0].severity.value, "error")
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
