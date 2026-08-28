import json
import unittest

from nautical_core.operator_findings import (
    FindingActionability,
    FindingSeverity,
    OperatorFinding,
    deduplicate_findings,
    highest_severity,
    status_for_findings,
    sort_findings,
)
from nautical_core.operator_models import OperatorStatus
from nautical_core.operator_models import OperatorContractError, OperatorScope, OperatorScopeKind


class OperatorFindingTests(unittest.TestCase):
    def test_finding_round_trip_is_json_native(self) -> None:
        finding = OperatorFinding(
            "chain.missing_successor",
            "chain",
            FindingSeverity.ERROR,
            FindingActionability.REPAIRABLE,
            "A completed chain node has no successor.",
            scope=OperatorScope(OperatorScopeKind.CHAIN, ("abc",)),
            affected=("task-1", "task-1"),
            observed={"nextLink": ""},
            expected={"nextLink": "successor"},
            evidence={"snapshot_id": "snap-1"},
            command="nautical reconcile --apply",
        )
        restored = OperatorFinding.from_mapping(finding.to_dict())
        self.assertEqual(restored, finding)
        json.dumps(finding.to_dict(), ensure_ascii=False)
        self.assertEqual(finding.affected, ("task-1",))

    def test_doctor_mapping_normalizes_to_canonical_fields(self) -> None:
        finding = OperatorFinding.from_doctor_mapping({
            "id": "chains.repair_review",
            "severity": "warn",
            "message": "Review chain evidence.",
            "fix": "Run nautical query integrity --all.",
            "details": {"subjects": ["task-1"], "reason": "manual"},
        })
        self.assertEqual(finding.code, "chains.repair_review")
        self.assertEqual(finding.domain, "chains")
        self.assertEqual(finding.affected, ("task-1",))
        self.assertEqual(finding.to_dict()["actionability"], "actionable")

    def test_actionable_finding_requires_guidance(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorFinding("x", "domain", FindingSeverity.ERROR, FindingActionability.BLOCKING, "bad")
        with self.assertRaises(OperatorContractError):
            OperatorFinding("x", "domain", FindingSeverity.ERROR, "unknown", "bad")
        with self.assertRaises(OperatorContractError):
            OperatorFinding("x", "domain", FindingSeverity.WARNING, FindingActionability.RETRYABLE, "retry")

    def test_deduplication_merges_affected_identities_deterministically(self) -> None:
        def build(identity: str) -> OperatorFinding:
            return OperatorFinding(
                "x", "domain", FindingSeverity.WARNING,
                FindingActionability.INFORMATIONAL, "same", affected=(identity,),
            )
        merged = deduplicate_findings([build("b"), build("a"), build("b")])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].affected, ("a", "b"))
        self.assertEqual(highest_severity(merged), FindingSeverity.WARNING)
        self.assertIsNone(highest_severity([]))
        self.assertEqual(status_for_findings([]), OperatorStatus.OK)
        self.assertEqual(status_for_findings(merged), OperatorStatus.ATTENTION)
        deferred = OperatorFinding("d", "chain", FindingSeverity.INFO, FindingActionability.DEFERRED, "later", guidance="review")
        retryable = OperatorFinding("r", "chain", FindingSeverity.ERROR, FindingActionability.RETRYABLE, "retry", guidance="retry")
        self.assertEqual(status_for_findings((deferred,)), OperatorStatus.DEFERRED)
        self.assertEqual(status_for_findings((retryable,)), OperatorStatus.UNAVAILABLE)
        self.assertEqual([item.code for item in sort_findings((deferred, retryable))], ["r", "d"])


if __name__ == "__main__":
    unittest.main()
