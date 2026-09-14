"""Direct contracts for Doctor's configuration reporting boundary."""

import importlib
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo


doctor = importlib.import_module("nautical_core.tools.nautical_doctor")


class DoctorConfigurationContractTests(unittest.TestCase):
    def test_timezone_findings_distinguish_missing_and_unavailable_zones(self) -> None:
        service = doctor.OperatorHealthService
        missing = service.timezone_findings({}, ZoneInfo)[0].to_doctor_dict()
        self.assertEqual(missing.get("id"), "config.timezone.missing")
        self.assertEqual(
            ((missing.get("details") or {}).get("observed") or {}).get("tz"), "UTC"
        )

        def unavailable(_name: str) -> object:
            raise ValueError("timezone database unavailable")

        invalid = service.timezone_findings(
            {"tz": "Europe/Bucharest"}, unavailable
        )[0].to_doctor_dict()
        self.assertEqual(invalid.get("id"), "config.timezone.invalid")
        self.assertIn("pip install tzdata", invalid.get("fix", ""))

    def test_astronomy_findings_distinguish_unconfigured_and_healthy(self) -> None:
        service = doctor.OperatorHealthService
        unconfigured = service.astronomy_findings(
            {}, effective_timezone=ZoneInfo("UTC"), source_hint="defaults",
            preflight=lambda _config: {"status": "not_configured"},
        )[0]
        self.assertEqual(unconfigured.code, "astronomy.not_configured")
        self.assertEqual(unconfigured.severity.value, "info")

        healthy = service.astronomy_findings(
            {"astronomy": {"locations": {"home": {}}}},
            effective_timezone=ZoneInfo("UTC"), source_hint="user-config",
            preflight=lambda _config: {
                "status": "ok", "provider": "astral", "location": "home"
            },
        )[0].to_dict()
        self.assertEqual(healthy["code"], "astronomy.preflight")
        self.assertEqual(healthy["severity"], "info")
        self.assertEqual(healthy["observed"]["config_source"], "user-config")
        self.assertEqual(healthy["observed"]["effective_timezone"], "UTC")

    def test_season_findings_report_astronomical_events_and_reject_bad_mode(self) -> None:
        events = {
            "spring_equinox": datetime(2026, 3, 20, 14, tzinfo=timezone.utc),
            "summer_solstice": datetime(2026, 6, 21, 8, tzinfo=timezone.utc),
        }
        findings = doctor.OperatorHealthService.season_findings(
            {"season_mode": "astronomical", "season_hemisphere": "north", "tz": "UTC"},
            {}, ZoneInfo, lambda year: events if year == 2026 else {}, year=2026,
        )
        self.assertEqual(findings[0].code, "config.season_mode")
        self.assertEqual(
            findings[0].to_dict()["observed"]["events"]["spring_equinox"],
            "2026-03-20",
        )

        invalid = doctor.OperatorHealthService.season_findings(
            {"season_mode": "sidereal", "tz": "UTC"}, {}, ZoneInfo,
            lambda _year: {}, year=2026,
        )[0].to_doctor_dict()
        self.assertEqual(invalid.get("id"), "config.season_mode.invalid")
        self.assertIn("season_mode", invalid.get("fix", ""))

    def test_configuration_drift_is_filtered_by_source_and_reports_changes(self) -> None:
        source = str(Path(tempfile.gettempdir()) / "nautical-config-drift-test.toml")
        resolved = str(Path(source).resolve())
        current = {
            "changed": False, "source": resolved,
            "loaded_fingerprint": "abc", "current_fingerprint": "abc",
        }
        service = doctor.OperatorHealthService
        healthy = service.configuration_drift_findings(
            source, lambda: current.copy()
        )[0].to_doctor_dict()
        self.assertEqual(healthy.get("severity"), "info")

        current.update(
            changed=True, current_fingerprint="def", status="changed"
        )
        changed = service.configuration_drift_findings(
            source, lambda: current.copy()
        )[0].to_doctor_dict()
        self.assertEqual(changed.get("severity"), "warning")
        self.assertIn("Restart Navigator", changed.get("fix", ""))
        self.assertEqual(service.configuration_drift_findings(
            str(Path(source).with_name("other.toml")), lambda: current.copy()
        ), ())

    def test_navigator_dependency_findings_name_missing_packages(self) -> None:
        available = lambda name: name not in {"rich", "dateutil"}
        findings = [
            item.to_doctor_dict()
            for item in doctor.OperatorHealthService.navigator_dependency_findings(
                {}, available, python_executable="/usr/bin/python3"
            )
        ]
        item = next(item for item in findings if item.get("id") == "navigator.dependencies")
        observed = (item.get("details") or {}).get("observed") or {}
        self.assertEqual(item.get("severity"), "warning")
        self.assertEqual(set(observed.get("missing") or ()), {"rich", "dateutil"})

    def test_configuration_schema_findings_are_stable_and_actionable(self) -> None:
        findings = [
            item.to_doctor_dict()
            for item in doctor.OperatorHealthService.configuration_schema_findings(
                {
                    "verify_import": False,
                    "outbox_drain_max_items": 0,
                    "panel_mode": "sparkle",
                    "setting_typo": True,
                }
            )
        ]
        ids = {item.get("id") for item in findings}
        self.assertTrue(
            {
                "config.schema.deprecated",
                "config.schema.range",
                "config.schema.choice",
                "config.schema.unknown",
            }.issubset(ids)
        )
        self.assertTrue(all(item.get("fix") for item in findings))
        drain = next(
            item
            for item in findings
            if ((item.get("details") or {}).get("observed") or {}).get("key")
            == "outbox_drain_max_items"
        )
        self.assertEqual(
            ((drain.get("details") or {}).get("observed") or {}).get("effective"), 1
        )

    def test_live_panel_configuration_reports_fallbacks_and_dependency_health(self) -> None:
        service = doctor.OperatorHealthService
        available = lambda _name: object()
        findings = [
            item.to_doctor_dict()
            for item in service.panel_findings(
                {"panel_mode": "live", "live_panel_duration_ms": 275}, available
            )
        ]
        live = next(item for item in findings if item.get("id") == "config.panel.live")
        self.assertEqual(live.get("severity"), "info")
        observed = (live.get("details") or {}).get("observed") or {}
        self.assertEqual(observed.get("configured_duration_ms"), 275)
        self.assertEqual(observed.get("effective_duration_ms"), 275)
        self.assertEqual(observed.get("non_tty_fallback"), "static")
        self.assertIs(observed.get("rich_available"), True)
        self.assertIn("Rich is available", live.get("message", ""))

        invalid_findings = [
            item.to_doctor_dict()
            for item in service.panel_findings(
                {"panel_mode": "live", "live_panel_duration_ms": "slow"}, available
            )
        ]
        invalid = next(
            item for item in invalid_findings
            if item.get("id") == "config.panel.duration.invalid"
        )
        self.assertEqual(invalid.get("severity"), "warning")
        self.assertEqual(
            ((invalid.get("details") or {}).get("observed") or {}).get(
                "effective_duration_ms"
            ),
            160,
        )

        clamped_findings = [
            item.to_doctor_dict()
            for item in service.panel_findings(
                {"panel_mode": "live", "live_panel_duration_ms": 5000}, available
            )
        ]
        clamped = next(
            item for item in clamped_findings
            if item.get("id") == "config.panel.duration.clamped"
        )
        self.assertEqual(clamped.get("severity"), "warning")
        self.assertEqual(
            ((clamped.get("details") or {}).get("observed") or {}).get(
                "effective_duration_ms"
            ),
            1000,
        )

        missing_findings = [
            item.to_doctor_dict()
            for item in service.panel_findings(
                {"panel_mode": "live", "live_panel_duration_ms": 160},
                lambda _name: None,
            )
        ]
        missing = next(
            item for item in missing_findings
            if item.get("id") == "config.panel.rich_missing"
        )
        self.assertEqual(missing.get("severity"), "warning")
        self.assertIn("pip install rich", missing.get("fix", ""))

    def test_uda_alias_configuration_is_actionable_and_defaults_off(self) -> None:
        findings = [
            item.to_doctor_dict()
            for item in doctor.OperatorHealthService.uda_alias_findings(
                {"enable_uda_aliases": True}
            )
        ]
        enabled = next(item for item in findings if item.get("id") == "config.uda_aliases")
        self.assertEqual(enabled.get("severity"), "info")
        observed = (enabled.get("details") or {}).get("observed") or {}
        self.assertIs(observed.get("enabled"), True)
        self.assertEqual(observed.get("clear_syntax"), "alias:")
        self.assertIn("Description UDA aliases are enabled", enabled.get("message", ""))

        with tempfile.TemporaryDirectory() as td:
            env = {
                "HOME": td,
                "TASKRC": str(Path(td) / ".taskrc"),
                "NAUTICAL_CONFIG": str(Path(td) / "missing-nautical.toml"),
            }
            missing_findings: list[dict[str, object]] = []
            with patch.dict(os.environ, env):
                doctor._check_config(missing_findings, Path(td))

        defaulted = next(
            item for item in missing_findings if item.get("id") == "config.uda_aliases"
        )
        observed = (defaulted.get("details") or {}).get("observed") or {}
        self.assertIs(observed.get("enabled"), False)


if __name__ == "__main__":
    unittest.main()
