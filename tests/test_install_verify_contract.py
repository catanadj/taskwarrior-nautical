from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import get_type_hints
import unittest

from nautical_core.installation_report import InstallationVerificationReport
from nautical_core.tools import nautical_install_verify


class InstallationVerificationInputContractTests(unittest.TestCase):
    def test_report_builder_returns_validated_public_report(self) -> None:
        report = nautical_install_verify.build_report(
            {
                "operator_findings": [
                    {"code": code, "severity": "info"}
                    for code in (
                        "taskwarrior.available",
                        "taskdata.available",
                        "install.active",
                        "hook.available",
                        "config.timezone",
                    )
                ],
                "taskdata": "/tmp/taskdata",
            },
            platform="Linux",
            launcher=Path(sys.executable),
        )

        self.assertIsInstance(report, InstallationVerificationReport)
        self.assertEqual(report["schema"], "nautical.install.verification")
        self.assertEqual(report["status"], "passed")

    def test_finding_reader_ignores_malformed_non_list_findings(self) -> None:
        self.assertEqual(nautical_install_verify._findings({"operator_findings": 7}), [])

    def test_finding_reader_uses_object_mapping_boundary(self) -> None:
        hints = get_type_hints(nautical_install_verify._findings)

        self.assertEqual(hints["payload"], Mapping[str, object])
        self.assertEqual(hints["return"], list[Mapping[str, object]])

    def test_finding_filter_uses_object_mapping_boundary(self) -> None:
        hints = get_type_hints(nautical_install_verify._items)

        self.assertEqual(hints["payload"], Mapping[str, object])
        self.assertEqual(hints["return"], list[Mapping[str, object]])


if __name__ == "__main__":
    unittest.main()
