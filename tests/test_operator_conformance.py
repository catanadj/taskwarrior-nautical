import unittest
import json
from datetime import datetime, timezone

from nautical_core.operator_application import DomainApplicationRegistry
from nautical_core.operator_control_plane import OperatorControlPlane
from nautical_core.operator_inspectors import inspect_operator_snapshot
from nautical_core.operator_models import (CoverageKind, CoverageRequirement, OperatorCoverage, OperatorLimits,
    OperatorScope, OperatorV2Result, OperatorV2Status)
from nautical_core.operator_findings import FindingActionability, FindingSeverity, OperatorFinding
from nautical_core.operator_presentation import ordered_findings, render_contract_json
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

    def test_shuffled_findings_have_one_stable_order(self) -> None:
        findings = [
            OperatorFinding("b", "chain", FindingSeverity.WARNING, FindingActionability.INFORMATIONAL, "b", affected=("z",), guidance="inspect"),
            OperatorFinding("a", "chain", FindingSeverity.ERROR, FindingActionability.BLOCKING, "a", affected=("a",), guidance="inspect"),
        ]
        records = [item.to_dict() for item in findings]
        reversed_records = list(reversed(records))
        self.assertEqual(ordered_findings(records), ordered_findings(reversed_records))

    def test_versioned_result_json_round_trip_preserves_envelope(self) -> None:
        result = OperatorV2Result(
            "nautical.query.test", "query", OperatorV2Status.OK,
            payload={"message": "héllo", "items": [1, 2]},
        )
        encoded = render_contract_json(result)
        decoded = OperatorV2Result.from_mapping(json.loads(encoded))
        self.assertEqual(decoded, result)


if __name__ == "__main__":
    unittest.main()
