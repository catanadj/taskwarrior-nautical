import unittest
from datetime import datetime, timezone

from nautical_core.operator_application import DomainApplicationRegistry
from nautical_core.operator_control_plane import OperatorControlPlane
from nautical_core.operator_inspectors import inspect_operator_snapshot
from nautical_core.operator_models import CoverageKind, CoverageRequirement, OperatorCoverage, OperatorLimits, OperatorScope
from nautical_core.operator_snapshot import OperatorSnapshot


class OperatorConformanceTests(unittest.TestCase):
    def test_control_plane_and_direct_inspection_share_one_projection(self) -> None:
        snapshot = OperatorSnapshot(
            "conformance-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
        )
        requirement = CoverageRequirement(CoverageKind.COMPLETE)
        limits = OperatorLimits()
        expected = inspect_operator_snapshot(snapshot, requirement, limits, scope=OperatorScope.system())

        class Configuration:
            fingerprint = "config-1"
            scheduler_fingerprint = "schedule-1"

        control_plane = OperatorControlPlane.from_configuration(Configuration(), DomainApplicationRegistry())
        actual = control_plane.inspect(snapshot, requirement, limits, scope=OperatorScope.system())
        self.assertEqual(actual, expected)
        self.assertEqual(
            tuple(item.to_dict() for item in actual),
            tuple(item.to_dict() for item in control_plane.inspect(snapshot, requirement, limits, scope=OperatorScope.system())),
        )


if __name__ == "__main__":
    unittest.main()
