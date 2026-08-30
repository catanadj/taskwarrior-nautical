import unittest
from datetime import datetime, timezone
from types import MappingProxyType

from nautical_core.operator_findings import FindingActionability
from nautical_core.operator_inspectors import (ChainIntegrityInspector, ConfigurationInspector, DependenciesInspector,
    InstallationInspector, LifecycleOutboxInspector, PerformanceInspector,
    STANDARD_COMPONENTS, ScheduleAvailabilityInspector, TaskDomainInspector, aggregate_historical,
    classify_historical, inspect_component_availability, inspect_component_validity, inspect_snapshot,
    inspect_snapshot_consistency, inspect_snapshot_coverage, inspect_snapshot_limits, inspect_standard_components,
    inspect_integrity_findings, inspect_lifecycle_outcomes, inspect_occurrence_collection,
    inspect_operator_snapshot, prioritize_findings, run_inspectors, standard_inspector_bundle)
from nautical_core.operator_findings import FindingSeverity, OperatorFinding
from nautical_core.operator_models import CoverageKind, CoverageRequirement, OperatorLimits, OperatorScope, OperatorScopeKind
from nautical_core.operator_snapshot import OperatorSnapshot, SnapshotComponent, SnapshotIndexes
from nautical_core.operator_models import OperatorCoverage


class OperatorInspectorTests(unittest.TestCase):
    def test_coverage_inspector_reports_blocking_insufficient_evidence(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.BOUNDED, "taskwarrior", omitted_count=2),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
        )
        findings = inspect_snapshot_coverage(
            snapshot,
            CoverageRequirement(CoverageKind.COMPLETE),
            scope=OperatorScope(OperatorScopeKind.SYSTEM),
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].actionability, FindingActionability.BLOCKING)
        self.assertEqual(findings[0].observed["coverage"], "bounded")

    def test_coverage_inspector_is_pure_for_sufficient_evidence(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
        )
        self.assertEqual(inspect_snapshot_coverage(snapshot, CoverageRequirement(CoverageKind.BOUNDED)), ())

    def test_inspector_runner_is_deterministic_and_validates_protocol(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
        )
        finding = OperatorFinding("same", "test", FindingSeverity.INFO, "informational", "message")

        class Inspector:
            def inspect(self, _snapshot):
                return (finding,)

        self.assertEqual(run_inspectors(snapshot, (Inspector(), Inspector())), (finding,))
        with self.assertRaises(TypeError):
            run_inspectors(snapshot, (object(),))

    def test_consistency_inspector_reports_mixed_component_epoch(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
            component_evidence=(
                SnapshotComponent("tasks", datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-2"),
            ),
        )
        findings = inspect_snapshot_consistency(snapshot)
        self.assertEqual(findings[0].code, "snapshot.inconsistent")
        self.assertEqual(findings[0].actionability, FindingActionability.BLOCKING)

    def test_limits_inspector_reports_each_exceeded_dimension(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
            indexes=SnapshotIndexes(task_uuids=("a", "b"), chain_ids=("c",), links=(1, 2)),
        )
        findings = inspect_snapshot_limits(snapshot, OperatorLimits(tasks=1, chains=1, history_links=1))
        self.assertEqual([item.observed for item in findings], [{"history_links": 2}, {"tasks": 2}])

    def test_component_inspector_distinguishes_absent_and_unavailable(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
            components={"configuration": {"available": False, "reason": "invalid TOML"}},
        )
        unavailable = inspect_component_availability(snapshot, "configuration")
        self.assertEqual(unavailable[0].domain, "configuration")
        self.assertEqual(inspect_component_availability(snapshot, "dependencies")[0].observed["available"], False)

    def test_component_validity_is_distinct_from_availability(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
            components={"task_domain": {"available": True, "valid": False, "reason": "bad UDA"}},
        )
        findings = inspect_component_validity(snapshot, "task_domain")
        self.assertEqual(findings[0].code, "component.invalid")
        self.assertEqual(findings[0].actionability, FindingActionability.ACTIONABLE)
        self.assertEqual(TaskDomainInspector().component, "task_domain")
        self.assertEqual(ScheduleAvailabilityInspector().component, "schedule")
        self.assertEqual(InstallationInspector().component, "installation")
        self.assertEqual(ConfigurationInspector().component, "configuration")
        self.assertEqual(DependenciesInspector().component, "dependencies")
        self.assertEqual(ChainIntegrityInspector().component, "chain_integrity")
        self.assertEqual(LifecycleOutboxInspector().component, "lifecycle")
        self.assertEqual(PerformanceInspector().component, "performance")

    def test_component_inspectors_accept_immutable_mapping_evidence(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
            components={"configuration": MappingProxyType({"available": True, "valid": True})},
        )
        self.assertEqual(inspect_component_availability(snapshot, "configuration"), ())
        self.assertEqual(inspect_component_validity(snapshot, "configuration"), ())

    def test_component_inspectors_fail_closed_for_malformed_evidence(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1",
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1",
            "config-1",
            components={"configuration": "not an evidence object"},
        )
        unavailable = inspect_component_availability(snapshot, "configuration")
        invalid = inspect_component_validity(snapshot, "configuration")
        self.assertEqual(unavailable[0].actionability, FindingActionability.BLOCKING)
        self.assertIn("must be an object", unavailable[0].message)
        self.assertEqual(invalid[0].actionability, FindingActionability.BLOCKING)
        self.assertIn("must be an object", invalid[0].message)

    def test_historical_classification_preserves_evidence_but_defers_action(self) -> None:
        finding = OperatorFinding("x", "chain", FindingSeverity.ERROR, FindingActionability.BLOCKING, "old", guidance="inspect")
        historical = classify_historical(finding, active=False)
        self.assertEqual(historical.severity, FindingSeverity.INFO)
        self.assertEqual(historical.actionability, FindingActionability.DEFERRED)
        self.assertEqual(historical.code, finding.code)

    def test_prioritization_puts_active_findings_first(self) -> None:
        old = OperatorFinding("old", "chain", FindingSeverity.ERROR, FindingActionability.DEFERRED, "old", affected=("task-old",), guidance="review")
        active = OperatorFinding("active", "chain", FindingSeverity.WARNING, FindingActionability.INFORMATIONAL, "active", affected=("task-new",))
        self.assertEqual(prioritize_findings((old, active), {"task-new"})[0], active)

    def test_standard_component_bundle_is_stable(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1", OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior"),
            datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-1", "config-1",
            components={"configuration": {"available": True}},
        )
        findings = inspect_standard_components(snapshot)
        self.assertEqual(len(findings), len(STANDARD_COMPONENTS) - 1)
        self.assertEqual(findings[0].domain, "dependencies")

    def test_standard_pipeline_has_one_inspector_per_domain(self) -> None:
        bundle = standard_inspector_bundle()
        self.assertEqual(
            tuple(item.component for item in bundle),
            ("installation", "configuration", "dependencies", "task_domain", "schedule", "chain_integrity", "lifecycle", "performance"),
        )

    def test_historical_aggregation_groups_related_findings(self) -> None:
        first = OperatorFinding("carry", "chain", FindingSeverity.INFO, "deferred", "old", affected=("a",), guidance="review")
        second = OperatorFinding("carry", "chain", FindingSeverity.INFO, "deferred", "old", affected=("b",), guidance="review")
        aggregated = aggregate_historical((second, first))
        self.assertEqual(len(aggregated), 1)
        self.assertEqual(aggregated[0].affected, ("a", "b"))
        self.assertEqual(aggregated[0].evidence["aggregated_count"], 2)

    def test_composite_snapshot_inspection_is_stable(self) -> None:
        snapshot = OperatorSnapshot(
            "snap-1", OperatorCoverage(CoverageKind.BOUNDED, "taskwarrior", omitted_count=1),
            datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-1", "config-1",
        )
        findings = inspect_snapshot(snapshot, CoverageRequirement(CoverageKind.COMPLETE), OperatorLimits())
        self.assertEqual([item.code for item in findings], ["snapshot.coverage_insufficient"])

    def test_typed_integrity_finding_preserves_reason_and_evidence(self) -> None:
        from nautical_core.chain_integrity_models import FindingSeverity as IntegritySeverity, FindingStatus, IntegrityFinding

        finding = IntegrityFinding(
            "continuity.child", FindingStatus.REPAIRABLE, IntegritySeverity.ERROR,
            "snap-1", chain_id="chain-a", subject_uuids=("task-a",),
            reason_code="missing_successor", message="successor is missing",
            observed=(('nextLink', ''),), expected=(('nextLink', 'successor'),),
            evidence=(('coverage', 'complete'),),
        )
        projected = inspect_integrity_findings((finding,))
        self.assertEqual(projected[0].code, "missing_successor")
        self.assertEqual(projected[0].actionability, FindingActionability.REPAIRABLE)
        self.assertEqual(projected[0].observed["nextLink"], "")

    def test_typed_lifecycle_outcome_preserves_manual_review(self) -> None:
        from nautical_core.lifecycle_application import LifecycleApplicationOutcome, LifecycleApplicationOutcomeKind
        from nautical_core.lifecycle_models import LifecycleEvent, LifecycleIdentity

        outcome = LifecycleApplicationOutcome(
            LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
            LifecycleIdentity("chain-a", "task-a", 1, 2, LifecycleEvent.COMPLETE),
            reason="parent guard changed", intent_id="intent-1",
        )
        projected = inspect_lifecycle_outcomes((outcome,))
        self.assertEqual(projected[0].code, "lifecycle.manual_review")
        self.assertEqual(projected[0].actionability, FindingActionability.MANUAL_REVIEW)
        self.assertEqual(projected[0].affected, ("task-a",))

    def test_typed_schedule_failure_maps_to_retryable_finding(self) -> None:
        from datetime import datetime, timezone
        from nautical_core.occurrence_outcomes import OccurrenceCollectionResult, UnavailableOccurrence
        from nautical_core.scheduler_cursor import OccurrenceCursor

        collection = OccurrenceCollectionResult(
            (), OccurrenceCursor(datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc),
            failure=UnavailableOccurrence("astronomy provider unavailable", "LookupError"),
        )
        projected = inspect_occurrence_collection(collection)
        self.assertEqual(projected[0].code, "schedule.unavailable")
        self.assertEqual(projected[0].actionability, FindingActionability.RETRYABLE)


if __name__ == "__main__":
    unittest.main()
