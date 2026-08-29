import json
import unittest
from types import SimpleNamespace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from nautical_core.integration_context import (
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable
from nautical_core.operator_context import (
    OperatorContextError,
    OperatorInvocationCache,
    OperatorInvocationContext,
    OperatorOutputMode,
    OperatorPresentationPolicy,
)
from nautical_core.operator_snapshot import (
    HydrationBatch,
    ChainSnapshotReader,
    OperatorSnapshotSession,
    OperatorSnapshot,
    OperatorSnapshotAssembler,
    SnapshotComponent,
    SnapshotIndexes,
    SnapshotReadRequest,
)
from nautical_core.chain_integrity_models import ChainNode, ChainSnapshot, SnapshotCoverage

from nautical_core.operator_models import (
    OperatorContractError,
    OperatorCapabilities,
    CoverageRequirement,
    CoverageKind,
    OperatorFailure,
    OperatorLimits,
    OperatorExitCode,
    OperatorOperation,
    OperatorRequest,
    OperatorResult,
    OperatorScope,
    OperatorScopeKind,
    OperatorStatus,
    OperatorCoverage,
    OperatorCursor,
    OperatorPage,
    OperatorDependency,
    CoverageRequirement,
    exit_code_for_status,
)
from nautical_core.query_models import QueryCapabilities, QueryContractError


class OperatorModelsTests(unittest.TestCase):
    def test_request_and_result_are_json_native(self) -> None:
        request = OperatorRequest(
            OperatorOperation.INSPECT,
            OperatorScope(OperatorScopeKind.CHAIN, ("abc123",)),
        )
        result = OperatorResult(
            OperatorOperation.INSPECT,
            OperatorStatus.OK,
            {"request": request.to_dict()},
        )
        encoded = json.dumps(result.to_dict(), ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema"], "nautical.operator.inspect")
        self.assertEqual(OperatorRequest.from_mapping(decoded["data"]["request"]), request)

    def test_result_round_trip_preserves_structured_failure(self) -> None:
        failure = OperatorFailure(
            "snapshot_unavailable",
            "snapshot is unavailable",
            retryable=True,
            scope=OperatorScope(OperatorScopeKind.CHAIN, ("abc123",)),
            details={"attempts": 2},
        )
        result = OperatorResult(OperatorOperation.INTEGRITY, OperatorStatus.UNAVAILABLE, failure=failure)
        restored = OperatorResult.from_mapping(result.to_dict())
        self.assertEqual(restored, result)
        self.assertEqual(result.exit_code, OperatorExitCode.UNAVAILABLE)

    def test_result_round_trip_preserves_page_contract(self) -> None:
        page = OperatorPage(({"uuid": "one"},))
        result = OperatorResult(
            OperatorOperation.INSPECT,
            OperatorStatus.OK,
            page=page,
            extensions={"future_field": {"enabled": True}},
        )
        encoded = result.to_dict()
        self.assertEqual(encoded["future_field"], {"enabled": True})
        restored = OperatorResult.from_mapping(encoded)
        self.assertEqual(restored.page, page)
        self.assertEqual(restored.extensions["future_field"], {"enabled": True})

    def test_decoding_preserves_unknown_response_fields(self) -> None:
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK)
        encoded = result.to_dict()
        encoded["future_contract_field"] = {"revision": 2}
        restored = OperatorResult.from_mapping(encoded)
        self.assertEqual(restored.extensions["future_contract_field"], {"revision": 2})
        self.assertEqual(restored.to_dict()["future_contract_field"], {"revision": 2})

    def test_request_rejects_unsupported_schema_version(self) -> None:
        request = OperatorRequest(
            OperatorOperation.INSPECT,
            OperatorScope(OperatorScopeKind.SYSTEM),
        ).to_dict()
        request["version"] = 999
        with self.assertRaises(OperatorContractError):
            OperatorRequest.from_mapping(request)

    def test_unavailable_requires_structured_failure(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorResult(OperatorOperation.OCCURRENCES, OperatorStatus.UNAVAILABLE)
        failure = OperatorFailure("snapshot_unavailable", "snapshot is unavailable", retryable=True)
        with self.assertRaises(OperatorContractError):
            OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK, failure=failure)
        with self.assertRaises(OperatorContractError):
            OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK, data={"invalid": object()})
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.UNAVAILABLE, failure=failure)
        self.assertEqual(result.to_dict()["failure"]["code"], "snapshot_unavailable")

    def test_scope_rejects_ambiguous_or_missing_values(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorScope(OperatorScopeKind.CHAIN)
        with self.assertRaises(OperatorContractError):
            OperatorScope(OperatorScopeKind.SYSTEM, ("unexpected",))
        scopes = OperatorScope(OperatorScopeKind.CHAINS, ("a", "b")).split()
        self.assertEqual(tuple(scope.kind for scope in scopes), (OperatorScopeKind.CHAIN, OperatorScopeKind.CHAIN))
        self.assertEqual(OperatorScope.system().kind, OperatorScopeKind.SYSTEM)
        self.assertEqual(OperatorScope.chains(["a"]).values, ("a",))
        self.assertEqual(OperatorScope.uuids(("u",)).kind, OperatorScopeKind.UUIDS)

    def test_scope_from_selector_normalizes_cli_shapes(self) -> None:
        self.assertEqual(
            OperatorScope.from_selector(chain_id="chain-a"),
            OperatorScope(OperatorScopeKind.CHAIN, ("chain-a",)),
        )
        self.assertEqual(
            OperatorScope.from_selector(uuid="task-a"),
            OperatorScope(OperatorScopeKind.UUID, ("task-a",)),
        )
        self.assertEqual(OperatorScope.from_selector(all_tasks=True).kind, OperatorScopeKind.SYSTEM)
        with self.assertRaises(OperatorContractError):
            OperatorScope.from_selector(chain_id="chain-a", all_tasks=True)

    def test_exit_code_mapping_is_stable(self) -> None:
        self.assertEqual(exit_code_for_status(OperatorStatus.OK), OperatorExitCode.SUCCESS)
        self.assertEqual(exit_code_for_status(OperatorStatus.REPAIRABLE), OperatorExitCode.FINDINGS)
        self.assertEqual(exit_code_for_status("unavailable"), OperatorExitCode.UNAVAILABLE)
        self.assertEqual(exit_code_for_status(OperatorStatus.MANUAL_REVIEW), OperatorExitCode.MANUAL_REVIEW)

    def test_capabilities_round_trip_is_discoverable(self) -> None:
        capabilities = OperatorCapabilities(
            taskwarrior_version="3.4.2",
            optional_dependencies={"astral": True, "rich": True},
            mutation_supported=True,
        )
        restored = OperatorCapabilities.from_mapping(capabilities.to_dict())
        self.assertEqual(restored, capabilities)
        self.assertIn("integrity", capabilities.to_dict()["operations"])
        self.assertIn("nautical.operator.integrity", capabilities.schemas)

    def test_query_capabilities_rejects_incomplete_discovery_documents(self) -> None:
        base = {
            "schema": "nautical.query.capabilities",
            "version": 2,
            "operation": "query",
            "status": "ok",
            "operations": ["occurrences", "next", "integrity"],
            "selectors": ["uuid", "chain_id", "all"],
            "omission_policies": ["exclude", "include", "report"],
            "next": {},
            "future": {"supported": True},
        }
        self.assertEqual(QueryCapabilities.from_mapping(base).to_dict(), base)
        for field, value in (("status", "warn"), ("selectors", ["uuid"]), ("next", None)):
            invalid = dict(base)
            invalid[field] = value
            with self.assertRaises(QueryContractError):
                QueryCapabilities.from_mapping(invalid)

    def test_coverage_rejects_false_completeness_and_round_trips(self) -> None:
        with self.assertRaises(OperatorContractError):
            OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", omitted_count=1)
        coverage = OperatorCoverage(
            CoverageKind.BOUNDED,
            "taskwarrior.authoritative_export",
            observed=("chain-a", "chain-a"),
            omitted_count=2,
            snapshot_id="snap-1",
            mutation_epoch="epoch-1",
        )
        self.assertEqual(OperatorCoverage.from_mapping(coverage.to_dict()), coverage)

    def test_cursor_is_bound_to_evidence_and_rejects_invalid_position(self) -> None:
        cursor = OperatorCursor("snap-1", "config-1", "epoch-1", position=4, page_size=20)
        self.assertEqual(OperatorCursor.from_mapping(cursor.to_dict()), cursor)
        cursor.assert_compatible("snap-1", "config-1", "epoch-1")
        self.assertEqual(cursor.advance().position, 24)
        self.assertEqual(cursor.advance(3).position, 7)
        with self.assertRaises(OperatorContractError):
            cursor.advance(0)
        with self.assertRaises(OperatorContractError):
            cursor.assert_compatible("snap-2", "config-1", "epoch-1")
        with self.assertRaises(OperatorContractError):
            OperatorCursor("snap-1", "config-1", "epoch-1", position=-1)

    def test_page_round_trip_requires_cursor_only_when_incomplete(self) -> None:
        cursor = OperatorCursor("snap-1", "config-1", "epoch-1", page_size=2)
        page = OperatorPage(({"uuid": "one"},), cursor=cursor, complete=False)
        self.assertEqual(OperatorPage.from_mapping(page.to_dict()), page)
        self.assertFalse(page.complete)
        self.assertEqual(page.items[0]["uuid"], "one")
        with self.assertRaises(OperatorContractError):
            OperatorPage(({"uuid": "one"},), cursor=cursor)
        with self.assertRaises(OperatorContractError):
            OperatorPage(({"uuid": "one"}, {"uuid": "two"}, {"uuid": "three"}), cursor=cursor, complete=False)
        with self.assertRaises(OperatorContractError):
            OperatorPage((object(),))

    def test_query_cursor_round_trips_with_evidence_binding(self) -> None:
        from nautical_core.query_models import OccurrenceQueryRequest

        cursor = OperatorCursor("snap-1", "config-1", "epoch-1", page_size=20)
        request = OccurrenceQueryRequest.from_mapping({
            "selector": {"all_tasks": True},
            "from": "2026-08-24",
            "count": 1,
            "cursor": cursor.to_dict(),
        })
        self.assertEqual(request.cursor, cursor)
        self.assertEqual(
            OccurrenceQueryRequest.from_mapping(request.to_dict()).cursor,
            cursor,
        )

    def test_coverage_requirement_rejects_insufficient_evidence(self) -> None:
        coverage = OperatorCoverage(CoverageKind.BOUNDED, "taskwarrior", omitted_count=1)
        self.assertTrue(CoverageRequirement(CoverageKind.BOUNDED).accepts(coverage))
        self.assertFalse(CoverageRequirement(CoverageKind.COMPLETE).accepts(coverage))

    def test_limits_include_a_bounded_file_record_budget(self) -> None:
        limits = OperatorLimits(file_records=7)
        self.assertEqual(OperatorLimits.from_mapping(limits.to_dict()), limits)
        with self.assertRaises(OperatorContractError):
            OperatorLimits(file_records=0)

    def test_every_operator_limit_has_an_enforcement_owner(self) -> None:
        limits = OperatorLimits()
        fields = (
            "taskwarrior_calls", "exported_rows", "decoded_rows", "hydration_identities",
            "sqlite_transactions", "cache_entries", "peak_memory_bytes",
            "tasks", "chains", "occurrences", "history_links", "findings",
            "outbox_rows", "file_records", "scheduler_iterations", "wall_time_seconds",
        )
        for field in fields:
            self.assertTrue(OperatorLimits.enforcement_owner(field))
            self.assertTrue(hasattr(limits, field))
        with self.assertRaises(OperatorContractError):
            OperatorLimits.enforcement_owner("unknown")

    def test_limits_round_trip_includes_resource_dimensions(self) -> None:
        limits = OperatorLimits(
            taskwarrior_calls=11,
            exported_rows=12,
            decoded_rows=13,
            hydration_identities=14,
            sqlite_transactions=15,
            cache_entries=16,
            peak_memory_bytes=17,
        )
        self.assertEqual(OperatorLimits.from_mapping(limits.to_dict()), limits)

    def test_invocation_context_captures_one_immutable_basis(self) -> None:
        configuration = ValidatedNauticalConfiguration(
            "/tmp/config", "config-1", "schedule-1", "UTC", (),
        )
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration,
            ZoneInfo("UTC"), SilentDiagnostics(), SystemClock(), "inv-1", 10,
            IntegrationAccess.READ_ONLY,
        )
        request = OperatorRequest(
            OperatorOperation.INSPECT,
            OperatorScope(OperatorScopeKind.SYSTEM),
            limits=OperatorLimits(cache_entries=3),
        )
        context = OperatorInvocationContext.from_integration(
            request, integration,
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            mutation_epoch="epoch-1",
            policy=OperatorPresentationPolicy(OperatorOutputMode.TEXT, diagnostics=True),
        )
        self.assertEqual(context.configuration_fingerprint, "config-1")
        self.assertFalse(context.mutation_capable)
        self.assertEqual(context.policy.output, OperatorOutputMode.TEXT)
        self.assertEqual(context.cache.max_entries, 3)
        context.assert_compatible(integration)
        changed = replace(integration, configuration=replace(configuration, fingerprint="config-2"))
        with self.assertRaises(OperatorContextError):
            context.assert_compatible(changed)
        with self.assertRaises(OperatorContextError):
            OperatorInvocationContext(request, integration, datetime.now())
        context = OperatorInvocationContext.from_integration(request, integration, mutation_epoch="epoch-1")
        context.assert_epoch("epoch-1")
        with self.assertRaises(OperatorContextError):
            context.assert_epoch("epoch-2")
        with self.assertRaises(OperatorContextError):
            context.assert_compatible(replace(integration, taskdata=Path("/tmp/other-taskdata")))
        with self.assertRaises(OperatorContextError):
            context.assert_compatible(replace(integration, taskdata_source="environment"))
        with self.assertRaises(OperatorContextError):
            context.assert_compatible(replace(integration, local_timezone=ZoneInfo("Europe/Bucharest")))
        bound = OperatorInvocationContext.from_unit_of_work(
            request, SimpleNamespace(context=integration, mutation_epoch=7),
        )
        self.assertEqual(bound.mutation_epoch, "7")
        other = OperatorInvocationContext.from_integration(request, integration, mutation_epoch="epoch-1")
        context.cache.put("private", 1)
        self.assertIsNone(other.cache.get("private"))
        with self.assertRaises(OperatorContextError):
            OperatorPresentationPolicy("invalid")

        apply_request = OperatorRequest(
            OperatorOperation.APPLY, OperatorScope(OperatorScopeKind.SYSTEM), apply=True,
        )
        with self.assertRaises(OperatorContextError):
            OperatorInvocationContext.from_integration(apply_request, integration)
        with self.assertRaises(OperatorContractError):
            OperatorRequest(
                OperatorOperation.APPLY,
                OperatorScope(OperatorScopeKind.SYSTEM),
                apply=True,
                coverage=CoverageRequirement(CoverageKind.BOUNDED),
            )

    def test_invocation_cache_is_bounded_and_resettable(self) -> None:
        cache = OperatorInvocationCache(max_entries=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.size, 2)
        cache.put("snapshot:chain-1", 1)
        cache.put("snapshot:chain-2", 2)
        self.assertEqual(cache.clear_prefix("snapshot:chain-"), 2)
        self.assertFalse(cache.discard("missing"))
        cache.clear()
        self.assertEqual(cache.size, 0)

    def test_repeated_invocations_do_not_leak_state(self) -> None:
        """Each invocation gets independent evidence, cursor, cache, and policy state."""
        base_configuration = ValidatedNauticalConfiguration(
            "/tmp/config", "config-0", "schedule-0", "UTC", (),
        )
        base_integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), base_configuration,
            ZoneInfo("UTC"), SilentDiagnostics(), SystemClock(), "inv-0", 10,
            IntegrationAccess.READ_ONLY,
        )
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope.system())
        contexts = []
        for index in range(3):
            configuration = replace(
                base_configuration,
                fingerprint=f"config-{index}",
                scheduler_fingerprint=f"schedule-{index}",
            )
            integration = replace(base_integration, configuration=configuration)
            context = OperatorInvocationContext.from_integration(
                request,
                integration,
                mutation_epoch=f"epoch-{index}",
                policy=OperatorPresentationPolicy(
                    OperatorOutputMode.JSON if index % 2 == 0 else OperatorOutputMode.TEXT,
                    diagnostics=bool(index % 2),
                ),
            )
            cursor = OperatorCursor(
                f"snapshot-{index}", f"config-{index}", f"epoch-{index}", position=index,
            )
            context.cache.put("invocation", index)
            contexts.append((context, cursor))

        for index, (context, cursor) in enumerate(contexts):
            self.assertEqual(context.configuration_fingerprint, f"config-{index}")
            self.assertEqual(context.mutation_epoch, f"epoch-{index}")
            self.assertEqual(context.cache.get("invocation"), index)
            cursor.assert_compatible(
                f"snapshot-{index}", f"config-{index}", f"epoch-{index}",
            )
            for other_index, (other, _) in enumerate(contexts):
                if other_index != index:
                    self.assertIsNone(other.cache.get("not-present"))
                    self.assertNotEqual(other.configuration_fingerprint, context.configuration_fingerprint)

    def test_snapshot_preserves_provenance_and_rejects_naive_time(self) -> None:
        coverage = OperatorCoverage(CoverageKind.COMPLETE, "taskwarrior", snapshot_id="snap-1", mutation_epoch="epoch-1")
        snapshot = OperatorSnapshot(
            "snap-1", coverage, datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1", "config-1", {"tasks": 2}, {"taskwarrior": "3.4.2"},
            SnapshotIndexes(task_uuids=("task-1", "task-1"), chain_ids=("chain-1",)),
            (HydrationBatch("predecessor", ("task-2",), ("task-2",)),),
            (SnapshotComponent("tasks", datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-1", coverage),),
        )
        self.assertEqual(OperatorSnapshot.from_mapping(snapshot.to_dict()), snapshot)
        self.assertEqual(snapshot.indexes.task_uuids, ("task-1",))
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope(OperatorScopeKind.SYSTEM))
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-1", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(
            request, integration, captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc), mutation_epoch="epoch-1",
        )
        self.assertIs(OperatorSnapshotAssembler.assemble(context, snapshot), snapshot)
        snapshot.assert_consistent()
        mixed = OperatorSnapshot(
            "snap-1", coverage, datetime(2026, 1, 1, tzinfo=timezone.utc),
            "epoch-1", "config-1", component_evidence=(
                SnapshotComponent("outbox", datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-2"),
            ),
        )
        with self.assertRaises(OperatorContractError):
            mixed.assert_consistent()

        unavailable = OperatorSnapshot(
            "snap-2", OperatorCoverage(CoverageKind.UNAVAILABLE, "taskwarrior", reason="timeout"),
            datetime(2026, 1, 1, tzinfo=timezone.utc), "epoch-1", "config-1",
        )
        self.assertFalse(unavailable.cacheable)
        with self.assertRaises(OperatorContractError):
            unavailable.assert_cacheable()
        with self.assertRaises(OperatorContractError):
            OperatorSnapshot("snap-1", coverage, datetime(2026, 1, 1), "epoch-1", "config-1")

    def test_hydration_batch_is_bounded_and_explicit(self) -> None:
        batch = HydrationBatch("child", ("task-1", "task-2"), ("task-1",), limit=2)
        self.assertEqual(HydrationBatch.from_mapping(batch.to_dict()), batch)
        with self.assertRaises(OperatorContractError):
            HydrationBatch("child", ("task-1",), ("task-2",))

    def test_chain_snapshot_projects_into_operator_envelope(self) -> None:
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope(OperatorScopeKind.SYSTEM))
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-1", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(
            request, integration, captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc), mutation_epoch="epoch-1",
        )
        source = ChainSnapshot(
            "chain-snap", SnapshotCoverage.CHAIN, "taskwarrior",
            (ChainNode("task-1", "chain-1", 1, "pending", ()),), "config-1", True,
        )
        projected = OperatorSnapshotAssembler.from_chain_snapshot(context, source)
        self.assertEqual(projected.indexes.chain_ids, ("chain-1",))
        self.assertEqual(projected.coverage.kind, CoverageKind.COMPLETE)
        self.assertTrue(projected.satisfies(CoverageRequirement(CoverageKind.BOUNDED)))
        self.assertEqual(projected.component_evidence[0].name, "chain")
        self.assertEqual(projected.component_evidence[0].mutation_epoch, "epoch-1")
        with self.assertRaises(OperatorContractError):
            OperatorSnapshotAssembler.from_chain_snapshot(
                context,
                replace(source, configuration_fingerprint="other-config"),
            )
        truncated = OperatorSnapshotAssembler.from_chain_snapshot(
            context,
            replace(source, coverage=SnapshotCoverage.TRUNCATED, complete_chain_history=False),
        )
        self.assertEqual(truncated.coverage.kind, CoverageKind.PARTIAL)
        with self.assertRaises(OperatorContractError):
            truncated.assert_satisfies(CoverageRequirement(CoverageKind.COMPLETE))

    def test_snapshot_read_request_round_trips_and_validates_scope(self) -> None:
        request = SnapshotReadRequest(
            OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)),
            refresh=True,
        )
        self.assertEqual(SnapshotReadRequest.from_mapping(request.to_dict()), request)
        with self.assertRaises(OperatorContractError):
            SnapshotReadRequest.from_mapping({"scope": {"kind": "system", "values": ["unexpected"]}})

    def test_chain_snapshot_reader_maps_scope_without_broadening(self) -> None:
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)))
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-1", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        source = ChainSnapshot(
            "chain-snap", SnapshotCoverage.CHAIN, "taskwarrior",
            (ChainNode("task-1", "chain-1", 1, "pending", ()),), "config-1", True,
        )
        seen = []
        from nautical_core.integration_models import Found
        def collect(source_request):
            seen.append(source_request)
            if getattr(source_request, "chain_id", None) == "chain-2":
                return Found(
                    ChainSnapshot(
                        "chain-snap-2", SnapshotCoverage.CHAIN, "taskwarrior",
                        (ChainNode("task-2", "chain-2", 1, "pending", ()),), "config-1", True,
                    ),
                    "chain read",
                )
            return Found(source, "chain read")

        reader = ChainSnapshotReader(collect)
        raw = reader.read_chain_snapshot(
            context,
            SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",))),
        )
        self.assertIs(raw, source)
        result = reader.read(context, SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",))))
        self.assertIsInstance(result, OperatorSnapshot)
        self.assertEqual(seen[0].chain_id, "chain-1")
        cached = reader.read(context, SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",))))
        self.assertIs(result, cached)
        self.assertEqual(len(seen), 1)
        reader.read(context, SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)), refresh=True))
        self.assertEqual(len(seen), 2)
        reader.read(
            context,
            SnapshotReadRequest(
                OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)),
                requirement=CoverageRequirement(CoverageKind.BOUNDED),
                refresh=True,
            ),
        )
        self.assertFalse(seen[-1].complete_chain_history)
        multi = reader.read_chain_snapshot(
            context,
            SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAINS, ("chain-1", "chain-2")), refresh=True),
        )
        self.assertIsInstance(multi, ChainSnapshot)
        self.assertEqual(len(multi.rows), 2)

        reader.read(
            context,
            SnapshotReadRequest(
                OperatorScope(OperatorScopeKind.SYSTEM),
                requirement=CoverageRequirement(CoverageKind.BOUNDED),
                refresh=True,
            ),
        )
        self.assertFalse(seen[-1].complete_chain_history)
        unsupported = reader.read(context, SnapshotReadRequest(OperatorScope(OperatorScopeKind.TEMPORAL_RANGE, ("a",))))
        self.assertIsInstance(unsupported, OperatorFailure)
        self.assertEqual(unsupported.code, "unsupported_snapshot_scope")
        limited_source = replace(source, coverage=SnapshotCoverage.TRUNCATED, complete_chain_history=False)
        limited_reader = ChainSnapshotReader(lambda _request: Found(limited_source, "chain read"))
        insufficient = limited_reader.read(
            context,
            SnapshotReadRequest(
                OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)),
                requirement=CoverageRequirement(CoverageKind.COMPLETE),
                refresh=True,
            ),
        )
        self.assertIsInstance(insufficient, OperatorFailure)
        self.assertEqual(insufficient.code, "insufficient_snapshot_coverage")
        oversized_source = replace(
            source,
            rows=(
                ChainNode("task-1", "chain-1", 1, "pending", ()),
                ChainNode("task-2", "chain-1", 2, "pending", ()),
            ),
        )
        oversized_reader = ChainSnapshotReader(lambda _request: Found(oversized_source, "chain read"))
        limited = oversized_reader.read(
            context,
            SnapshotReadRequest(
                OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)),
                limits=OperatorLimits(tasks=1),
                refresh=True,
            ),
        )
        self.assertIsInstance(limited, OperatorFailure)
        self.assertEqual(limited.code, "snapshot_limit_exceeded")
        absent_calls = []
        from nautical_core.integration_models import Absent
        absent_reader = ChainSnapshotReader(
            lambda source_request: (absent_calls.append(source_request) or Absent("chain read", "no rows"))
        )
        absent_request = SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-2",)))
        absent_reader.read(context, absent_request)
        absent_reader.read(context, absent_request)
        self.assertEqual(len(absent_calls), 2)
        session = OperatorSnapshotSession(context, absent_reader)
        self.assertIsInstance(session.read(absent_request), OperatorFailure)
        self.assertEqual(session.invalidate(absent_request), 0)
        self.assertGreaterEqual(session.invalidate(), 1)
        session2 = OperatorSnapshotSession(context, reader)
        cached_request = SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)))
        context.cache.clear()
        first = session2.read(cached_request)
        second = session2.read(cached_request)
        self.assertIsInstance(first, OperatorSnapshot)
        self.assertIs(first, second)
        self.assertEqual(len(seen), 7)
        self.assertGreaterEqual(session2.invalidate_after_mutation((cached_request,), certain=True), 1)
        self.assertGreaterEqual(session2.invalidate_after_mutation(certain=False), 0)
        unsupported_request = SnapshotReadRequest(OperatorScope(OperatorScopeKind.TEMPORAL_RANGE, ("a",)))
        batch = session2.read_many((cached_request, unsupported_request))
        self.assertEqual(len(batch), 2)
        self.assertIsInstance(batch[0], OperatorSnapshot)
        self.assertIsInstance(batch[1], OperatorFailure)

    def test_chain_snapshot_reader_rejects_overlarge_hydration_before_read(self) -> None:
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope(OperatorScopeKind.CHAINS, ("chain-1", "chain-2")))
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-1", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        calls = []
        reader = ChainSnapshotReader(lambda source_request: calls.append(source_request))
        result = reader.read_chain_snapshot(
            context,
            SnapshotReadRequest(request.scope, limits=OperatorLimits(hydration_identities=1)),
        )
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "snapshot_limit_exceeded")
        self.assertEqual(calls, [])

    def test_chain_snapshot_reader_stops_after_call_budget(self) -> None:
        request = OperatorRequest(
            OperatorOperation.INSPECT,
            OperatorScope(OperatorScopeKind.CHAINS, ("chain-1", "chain-2")),
            limits=OperatorLimits(taskwarrior_calls=1),
        )
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-1", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        source = ChainSnapshot(
            "chain-snap", SnapshotCoverage.CHAIN, "taskwarrior",
            (ChainNode("task-1", "chain-1", 1, "pending", ()),), "config-1", True,
        )
        calls = []
        from nautical_core.integration_models import Found
        reader = ChainSnapshotReader(lambda source_request: (calls.append(source_request) or Found(source, "chain read")))
        result = reader.read_chain_snapshot(context, SnapshotReadRequest(request.scope))
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "snapshot_limit_exceeded")
        self.assertEqual(len(calls), 1)

    def test_chain_snapshot_reader_enforces_exported_row_budget(self) -> None:
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope(OperatorScopeKind.CHAIN, ("chain-1",)))
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-1", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        source = ChainSnapshot(
            "chain-snap", SnapshotCoverage.CHAIN, "taskwarrior",
            (
                ChainNode("task-1", "chain-1", 1, "pending", ()),
                ChainNode("task-2", "chain-1", 2, "pending", ()),
            ), "config-1", True,
        )
        from nautical_core.integration_models import Found
        reader = ChainSnapshotReader(lambda _request: Found(source, "chain read"))
        result = reader.read(
            context,
            SnapshotReadRequest(request.scope, limits=OperatorLimits(exported_rows=1)),
        )
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "snapshot_limit_exceeded")
        self.assertEqual(result.details["resource"], "exported_rows")

    def test_large_multi_scope_does_not_claim_complete_absence(self) -> None:
        """Broad exports must fail closed when requested identities are missing."""
        from nautical_core.integration_models import Found

        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-large-scope", 32, IntegrationAccess.READ_ONLY,
        )
        request = OperatorRequest(
            OperatorOperation.INSPECT,
            OperatorScope(OperatorScopeKind.CHAINS, ("chain-a", "chain-b", "chain-c", "chain-d", "chain-e")),
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        source = ChainSnapshot(
            "broad-snapshot", SnapshotCoverage.CANDIDATES, "taskwarrior",
            (ChainNode("task-a", "chain-a", 1, "pending", ()),), "config-1", True,
        )
        reader = ChainSnapshotReader(lambda _request: Found(source, "broad export"))
        result = reader.read_chain_snapshot(context, SnapshotReadRequest(request.scope))
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "invalid_snapshot_scope")
        self.assertFalse(result.retryable)
        self.assertEqual(result.details["identity"], "chain-b")
        uuid_scope = OperatorScope(
            OperatorScopeKind.UUIDS,
            ("task-a", "task-b", "task-c", "task-d", "task-e"),
        )
        uuid_result = reader.read_chain_snapshot(context, SnapshotReadRequest(uuid_scope))
        self.assertIsInstance(uuid_result, OperatorFailure)
        self.assertEqual(uuid_result.code, "invalid_snapshot_scope")
        self.assertEqual(uuid_result.details["identity"], "task-b")

    def test_multi_scope_reader_preserves_authoritative_identity_coverage(self) -> None:
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope.system())
        from nautical_core.integration_models import Found
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-large-scope", 32, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        scope = OperatorScope(OperatorScopeKind.CHAINS, ("chain-a", "chain-b", "chain-c", "chain-d", "chain-e"))
        source = ChainSnapshot(
            "broad-snapshot", SnapshotCoverage.CANDIDATES, "taskwarrior",
            (ChainNode("task-a", "chain-a", 1, "pending", ()),), "config-1", True,
        )
        reader = ChainSnapshotReader(lambda _request: Found(source, "broad export"))
        result = reader.read_chain_snapshot(context, SnapshotReadRequest(scope))
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "invalid_snapshot_scope")
        self.assertFalse(result.retryable)
        self.assertEqual(result.details["identity"], "chain-b")

    def test_five_chain_scope_uses_one_exact_read_per_identity(self) -> None:
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "inv-five", 32, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(
            OperatorRequest(OperatorOperation.INSPECT, OperatorScope.system()), integration,
        )
        values = tuple(f"chain-{index}" for index in range(5))
        seen: list[str] = []
        from nautical_core.integration_models import Found

        def collect(source_request):
            chain_id = str(source_request.chain_id)
            seen.append(chain_id)
            return Found(
                ChainSnapshot(
                    f"snapshot-{chain_id}", SnapshotCoverage.CHAIN, "taskwarrior",
                    (ChainNode(f"task-{chain_id}", chain_id, 1, "pending", ()),), "config-1", True,
                ),
                "exact chain read",
            )

        reader = ChainSnapshotReader(collect)
        result = reader.read_chain_snapshot(
            context,
            SnapshotReadRequest(OperatorScope(OperatorScopeKind.CHAINS, values)),
        )
        self.assertIsInstance(result, ChainSnapshot)
        self.assertEqual(seen, list(values))
        self.assertEqual(tuple(row.chain_id for row in result.rows), values)

    def test_multi_scope_hydration_stops_on_unavailable_evidence(self) -> None:
        request = OperatorRequest(OperatorOperation.INSPECT, OperatorScope.system())
        configuration = ValidatedNauticalConfiguration("/tmp/config", "config-1", "schedule-1", "UTC", ())
        integration = IntegrationContext(
            Path("/tmp"), "explicit", ("task",), configuration, ZoneInfo("UTC"),
            SilentDiagnostics(), SystemClock(), "hydration-failure", 10, IntegrationAccess.READ_ONLY,
        )
        context = OperatorInvocationContext.from_integration(request, integration)
        command = TaskCommand(("task", "export"), "hydration", 1.0)
        evidence = FailureEvidence(command, CommandFailureKind.TIMEOUT, 124, 1, 1.0, True, "hydration timed out")
        reader = ChainSnapshotReader(lambda _request: Unavailable("hydration", evidence))
        scope = OperatorScope(OperatorScopeKind.CHAINS, ("a", "b", "c", "d", "e"))
        result = reader.read_chain_snapshot(context, SnapshotReadRequest(scope))
        self.assertIsInstance(result, OperatorFailure)
        self.assertEqual(result.code, "snapshot_unavailable")
        self.assertTrue(result.retryable)

    def test_dependency_evidence_requires_reason_when_unavailable(self) -> None:
        available = OperatorDependency("astral", True, version="3.2")
        self.assertEqual(OperatorDependency.from_mapping(available.to_dict()), available)
        with self.assertRaises(OperatorContractError):
            OperatorDependency("rich", False)


if __name__ == "__main__":
    unittest.main()
