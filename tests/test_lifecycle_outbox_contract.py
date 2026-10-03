from __future__ import annotations

import json
import sqlite3
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from nautical_core.lifecycle.models import ExecutionStage
from nautical_core.lifecycle.models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
from nautical_core.lifecycle.execution_policy import (
    FailureDisposition,
    MUTATION_TO_APPLICATION,
    OUTBOX_TO_APPLICATION,
    LifecycleApplicationOutcomeKind,
    classify_mutation_failure,
    classify_outbox_failure,
    remaining_drain_work,
)
from nautical_core.integration_models import MutationOutcomeKind
from nautical_core.lifecycle.outbox import (
    OUTBOX_LEGACY_SCHEMA_VERSION,
    OUTBOX_SCHEMA_VERSION,
    LifecycleOutboxError,
    _LifecycleOutboxRepository,
    OutboxFailure,
    OutboxResult,
    OutboxProcessingState,
    OutboxResultKind,
    lifecycle_outbox_path,
)


class LifecycleOutboxContractTests(unittest.TestCase):
    def test_connection_scope_does_not_mask_unexpected_operation_errors(self) -> None:
        for session in (False, True):
            with self.subTest(session=session), TemporaryDirectory() as directory:
                repository = _LifecycleOutboxRepository(Path(directory))

                def fail(_connection):
                    raise RuntimeError("injected repository defect")

                if session:
                    with repository.session():
                        with self.assertRaisesRegex(RuntimeError, "injected repository defect"):
                            repository._with_connection(fail)
                else:
                    with self.assertRaisesRegex(RuntimeError, "injected repository defect"):
                        repository._with_connection(fail)

    def test_session_startup_does_not_mask_unexpected_initialization_errors(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            with patch.object(repository, "_initialize", side_effect=RuntimeError("injected schema defect")):
                with self.assertRaises(RuntimeError) as raised:
                    with repository.session():
                        self.fail("session must not yield after initialization failure")
            self.assertIs(type(raised.exception), RuntimeError)

    def test_cached_schema_probe_does_not_mask_unexpected_errors(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            self.assertTrue(repository.open().ok)
            stat_result = repository.path.stat()
            repository._schema_identity = (
                int(stat_result.st_dev),
                int(stat_result.st_ino),
                int(stat_result.st_mtime_ns),
            )
            with patch.object(repository, "_connect", side_effect=RuntimeError("injected probe defect")):
                with self.assertRaises(RuntimeError) as raised:
                    repository.open()
            self.assertIs(type(raised.exception), RuntimeError)

    def test_open_does_not_convert_unexpected_initialization_errors(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            with patch.object(repository, "_initialize", side_effect=RuntimeError("injected open defect")):
                with self.assertRaises(RuntimeError) as raised:
                    repository.open()
            self.assertIs(type(raised.exception), RuntimeError)

    def test_integrity_work_shares_storage_without_lifecycle_claiming(self) -> None:
        from nautical_core.chain_integrity_application import RepositoryIntegrityOutboxSink
        from nautical_core.chain_integrity_models import (
            IntegrityOperation,
            IntegrityRepairPlan,
            RepairOperationKind,
            RepairSafety,
        )
        from nautical_core.integrity_outbox_envelope import IntegrityOutboxEnvelope

        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            self.assertTrue(repository.open().ok)
            operation = IntegrityOperation(
                "shared-integrity-op",
                RepairOperationKind.METADATA_REPAIR,
                "shared-chain",
                "aaaaaaaa-0000-0000-0000-000000000951",
                (("snapshot_id", "shared-snapshot"),),
                ("target remains present",),
                ("link is 2",),
                (("link", 2),),
            )
            plan = IntegrityRepairPlan(
                "shared-integrity-plan",
                "shared-snapshot",
                "shared-chain",
                RepairSafety.SAFE,
                "missing_link",
                "shared outbox test",
                (operation,),
                "cfg-shared",
            )
            envelope = IntegrityOutboxEnvelope(plan, "cfg-shared", "schedule-shared")

            self.assertEqual(
                repository.enqueue_integrity(envelope).kind, OutboxResultKind.APPLIED
            )
            self.assertEqual(
                repository.enqueue_integrity(envelope).kind,
                OutboxResultKind.ALREADY_APPLIED,
            )
            lifecycle_claim, lifecycle_records = repository.claim_batch(
                owner="lifecycle-test", lease_seconds=10, limit=10
            )
            self.assertTrue(lifecycle_claim.ok)
            self.assertEqual(lifecycle_records, ())
            integrity_claim, integrity_records = repository.claim_integrity_batch(
                owner="integrity-test", lease_seconds=10, limit=10
            )
            self.assertTrue(integrity_claim.ok)
            self.assertEqual(len(integrity_records), 1)
            self.assertTrue(
                repository.acknowledge_integrity(
                    intent_id=envelope.intent_id, owner="integrity-test"
                ).ok
            )
            self.assertEqual(
                repository.acknowledge_integrity(
                    intent_id=envelope.intent_id, owner="integrity-test"
                ).kind,
                OutboxResultKind.ALREADY_APPLIED,
            )
            sink = RepositoryIntegrityOutboxSink(
                repository,
                configuration_fingerprint="cfg-shared",
                schedule_fingerprint="schedule-shared",
            )
            self.assertTrue(sink.persist(plan).accepted)
            with sqlite3.connect(repository.path) as connection:
                work_kind = connection.execute(
                    "SELECT work_kind FROM lifecycle_outbox WHERE intent_id=?",
                    (envelope.intent_id,),
                ).fetchone()
            self.assertEqual(work_kind, ("integrity",))
            snapshot_result, snapshot_records = repository.snapshot_records()
            self.assertTrue(snapshot_result.ok)
            self.assertEqual(len(snapshot_records), 1)
            self.assertEqual(snapshot_records[0].intent_id, envelope.intent_id)

    def test_outbox_state_file_uses_dedicated_state_directory(self) -> None:
        path = lifecycle_outbox_path(Path("/tmp/taskdata"))

        self.assertEqual(path.parent.name, ".nautical-state")
        self.assertEqual(path.name, ".nautical_lifecycle_outbox.db")

    @staticmethod
    def _plan(chain: str = "contract", *, max_attempts: int = 3) -> LifecyclePlan:
        from dev_tools.golden_tests.support import task_draft as _task_draft

        parent = "00000000-0000-4000-8000-000000001001"
        child = "00000000-0000-4000-8000-000000001002"
        return LifecyclePlan.from_draft(
            identity=LifecycleIdentity(chain, parent, 1, 2, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", chain, 1, "rf-contract", "20260101T000000Z"),
            draft=_task_draft({
                "uuid": child, "description": "contract child", "status": "pending", "chain": "on",
                "chainID": chain, "link": 2, "prevLink": parent[:8], "cp": "1d",
                "due": "20260102T000000Z",
            }),
            parent_patch={"nextLink": child[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
            max_attempts=max_attempts,
        )

    def test_schema_versions_and_processing_states_are_explicit(self) -> None:
        self.assertEqual(OUTBOX_LEGACY_SCHEMA_VERSION, 1)
        self.assertEqual(OUTBOX_SCHEMA_VERSION, 2)
        self.assertEqual(
            {state.value for state in OutboxProcessingState},
            {"ready", "claimed", "retry", "manual_review", "quarantined", "acknowledged"},
        )
        self.assertEqual(
            {stage.value for stage in ExecutionStage},
            {"planned", "persisted", "child_present", "parent_linked", "verified", "finalized", "retryable", "manual_review"},
        )

    def test_outbox_database_file_permissions_are_private(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            opened = repository.open()

            self.assertTrue(opened.ok, opened.reason)
            self.assertTrue(repository.path.is_file())
            mode = stat.S_IMODE(repository.path.stat().st_mode)
            self.assertEqual(mode & 0o077, 0)

    def test_failure_round_trip_preserves_unicode_and_evidence(self) -> None:
        failure = OutboxFailure("retryable", "échec — réseau", {"attempt": 2, "note": "再試"})

        restored = OutboxFailure.from_json(failure.to_json())

        self.assertEqual(restored, failure)
        self.assertIn("échec", failure.to_json())
        self.assertEqual(json.loads(failure.to_json())["evidence"]["note"], "再試")

    def test_failure_decoder_rejects_malformed_and_non_object_payloads(self) -> None:
        for payload in ("{bad", "[]", '"text"'):
            with self.subTest(payload=payload), self.assertRaises(LifecycleOutboxError):
                OutboxFailure.from_json(payload)

    def test_schema_rejects_newer_database_version(self) -> None:
        with TemporaryDirectory() as directory:
            taskdata = Path(directory)
            repository = _LifecycleOutboxRepository(taskdata)
            connection = sqlite3.connect(":memory:")
            try:
                connection.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION + 1}")
                with self.assertRaisesRegex(LifecycleOutboxError, "newer than supported"):
                    repository._initialize(connection)
            finally:
                connection.close()

    def test_schema_owner_migrates_legacy_v1_table(self) -> None:
        from contextlib import contextmanager

        from nautical_core.lifecycle.outbox_schema import initialize

        connection = sqlite3.connect(":memory:")
        try:
            connection.execute(
                """
                CREATE TABLE lifecycle_outbox (
                    intent_id TEXT PRIMARY KEY,
                    plan_json TEXT NOT NULL,
                    plan_fingerprint TEXT NOT NULL,
                    parent_guard_json TEXT NOT NULL,
                    configuration_fingerprint TEXT NOT NULL,
                    schedule_fingerprint TEXT NOT NULL,
                    lifecycle_stage TEXT NOT NULL,
                    processing_state TEXT NOT NULL,
                    lease_owner TEXT NOT NULL DEFAULT '',
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    failure_json TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    acknowledged_at REAL NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute("PRAGMA user_version=1")
            transactions = []

            @contextmanager
            def transaction(conn: sqlite3.Connection):
                transactions.append("begin")
                conn.execute("BEGIN IMMEDIATE")
                try:
                    yield
                except Exception:
                    conn.rollback()
                    raise
                else:
                    conn.commit()

            initialize(connection, transaction=transaction, error_type=LifecycleOutboxError)

            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 2)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(lifecycle_outbox)")}
            self.assertIn("work_kind", columns)
            self.assertEqual(transactions, ["begin"])
        finally:
            connection.close()

    def test_repository_healthy_state_transition_contract(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: 100.0)
            plan = self._plan()
            intent_id = plan.identity.idempotency_key

            self.assertEqual(repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").kind, OutboxResultKind.APPLIED)
            claim, records = repository.claim_batch(owner="owner", lease_seconds=30, limit=1)
            self.assertTrue(claim.ok)
            self.assertEqual(len(records), 1)
            self.assertEqual(repository.renew_lease(intent_id=intent_id, owner="owner", lease_seconds=30).kind, OutboxResultKind.APPLIED)
            self.assertEqual(repository.advance_stage(intent_id=intent_id, owner="owner", stage=ExecutionStage.CHILD_PRESENT).kind, OutboxResultKind.APPLIED)
            self.assertEqual(repository.advance_stage(intent_id=intent_id, owner="owner", stage=ExecutionStage.PARENT_LINKED).kind, OutboxResultKind.APPLIED)
            self.assertEqual(repository.advance_stage(intent_id=intent_id, owner="owner", stage=ExecutionStage.VERIFIED).kind, OutboxResultKind.APPLIED)
            self.assertEqual(repository.acknowledge(intent_id=intent_id, owner="owner").kind, OutboxResultKind.APPLIED)
            status, payload = repository.status()
            self.assertTrue(status.ok)
            self.assertEqual(payload["states"].get("acknowledged"), 1)

    def test_repository_retry_manual_review_and_resolution_contract(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            plan = self._plan("review")
            intent_id = plan.identity.idempotency_key
            repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            repository.claim_intent(owner="owner", lease_seconds=30, intent_id=intent_id)
            self.assertEqual(
                repository.release_retry(
                    intent_id=intent_id,
                    owner="owner",
                    failure=OutboxFailure("temporary", "retry me"),
                ).kind,
                OutboxResultKind.APPLIED,
            )
            repository.claim_intent(owner="owner", lease_seconds=30, intent_id=intent_id)
            self.assertEqual(
                repository.manual_review(
                    intent_id=intent_id,
                    owner="owner",
                    failure=OutboxFailure("needs_review", "inspect me"),
                ).kind,
                OutboxResultKind.APPLIED,
            )
            self.assertEqual(repository.resolve_manual_review(intent_id=intent_id, reason="verified").kind, OutboxResultKind.APPLIED)

    def test_repository_bulk_claim_stage_renew_and_ack_contract(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: 100.0)
            plans = tuple(self._plan(f"bulk-{index}") for index in (1, 2))
            overall, persisted = repository.enqueue_many(
                plans, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            )
            self.assertTrue(overall.ok)
            ids = tuple(plan.identity.idempotency_key for plan in plans)
            claim, claimed = repository.claim_intents(intent_ids=ids, owner="bulk-owner", lease_seconds=30)
            self.assertTrue(claim.ok)
            self.assertEqual(set(claimed), set(ids))
            renewed, renewed_rows = repository.renew_leases(
                intent_ids=ids, owner="bulk-owner", lease_seconds=30
            )
            self.assertTrue(renewed.ok)
            self.assertEqual(set(renewed_rows), set(ids))
            staged, staged_rows = repository.advance_stages(
                stages={intent_id: ExecutionStage.CHILD_PRESENT for intent_id in ids},
                owner="bulk-owner",
            )
            self.assertTrue(staged.ok)
            self.assertEqual(set(staged_rows), set(ids))
            isolated, isolated_rows = repository.renew_leases(
                intent_ids=(ids[0], "missing-intent"), owner="wrong-owner", lease_seconds=30
            )
            self.assertTrue(isolated.ok)
            self.assertEqual(isolated_rows[ids[0]].kind, OutboxResultKind.CONFLICT)
            for stage in (ExecutionStage.PARENT_LINKED, ExecutionStage.VERIFIED):
                staged, _ = repository.advance_stages(
                    stages={intent_id: stage for intent_id in ids}, owner="bulk-owner"
                )
                self.assertTrue(staged.ok)
            acknowledged, rows = repository.acknowledge_many(intent_ids=ids, owner="bulk-owner")
            self.assertTrue(acknowledged.ok)
            self.assertEqual(set(rows), set(ids))

    def test_repository_prunes_old_acknowledged_rows(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: 1_000.0)
            plan = self._plan("prune")
            intent_id = plan.identity.idempotency_key
            repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            repository.claim_intent(owner="owner", lease_seconds=30, intent_id=intent_id)
            for stage in (ExecutionStage.CHILD_PRESENT, ExecutionStage.PARENT_LINKED, ExecutionStage.VERIFIED):
                repository.advance_stage(intent_id=intent_id, owner="owner", stage=stage)
            repository.acknowledge(intent_id=intent_id, owner="owner")
            path = repository.path
            with sqlite3.connect(path) as connection:
                connection.execute(
                    "UPDATE lifecycle_outbox SET acknowledged_at=?, updated_at=? WHERE intent_id=?",
                    (1.0, 1.0, intent_id),
                )
            result = repository.prune_acknowledged(retention_seconds=10.0)
            self.assertTrue(result.ok)
            self.assertEqual(result.removed, 1)

    def test_retention_prune_preserves_live_evidence_and_recovers_from_interruption(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: 1_000.0)

            def acknowledged_plan(chain: str, *, acknowledged_at: float, owner: str) -> str:
                plan = self._plan(chain)
                intent_id = plan.identity.idempotency_key
                self.assertTrue(repository.enqueue(
                    plan, configuration_fingerprint="cfg", schedule_fingerprint="sch"
                ).ok)
                claimed = repository.claim_intent(owner=owner, lease_seconds=300, intent_id=intent_id)
                self.assertTrue(claimed.ok)
                for stage in (ExecutionStage.CHILD_PRESENT, ExecutionStage.PARENT_LINKED, ExecutionStage.VERIFIED):
                    self.assertTrue(repository.advance_stage(intent_id=intent_id, owner=owner, stage=stage).ok)
                self.assertTrue(repository.acknowledge(intent_id=intent_id, owner=owner).ok)
                with sqlite3.connect(repository.path) as connection:
                    connection.execute(
                        "UPDATE lifecycle_outbox SET acknowledged_at=?, updated_at=? WHERE intent_id=?",
                        (acknowledged_at, acknowledged_at, intent_id),
                    )
                return intent_id

            acknowledged_plan("retention-old", acknowledged_at=900.0, owner="ack")
            retry = self._plan("retention-retry")
            repository.enqueue(retry, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            retry_id = retry.identity.idempotency_key
            self.assertTrue(repository.claim_intent(owner="retry", lease_seconds=30, intent_id=retry_id).ok)
            self.assertTrue(repository.release_retry(
                intent_id=retry_id, owner="retry", failure=OutboxFailure("busy", "retry later")
            ).ok)

            claimed_plan = self._plan("retention-claimed")
            repository.enqueue(claimed_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            claimed_id = claimed_plan.identity.idempotency_key
            self.assertTrue(repository.claim_intent(owner="live", lease_seconds=300, intent_id=claimed_id).ok)

            review_plan = self._plan("retention-review")
            repository.enqueue(review_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            review_id = review_plan.identity.idempotency_key
            self.assertTrue(repository.claim_intent(owner="review", lease_seconds=30, intent_id=review_id).ok)
            self.assertTrue(repository.manual_review(
                intent_id=review_id, owner="review", failure=OutboxFailure("review", "inspect")
            ).ok)

            result = repository.prune_acknowledged(retention_seconds=50.0, limit=1)
            self.assertTrue(result.ok)
            self.assertEqual(result.removed, 1)
            status_result, status = repository.status()
            self.assertTrue(status_result.ok)
            self.assertEqual(
                status["states"], {"retry": 1, "claimed": 1, "manual_review": 1}
            )
            self.assertEqual(status["states"].get("acknowledged", 0), 0)

            acknowledged_plan("retention-boundary", acknowledged_at=950.0, owner="boundary")
            boundary_status, boundary_payload = repository.status(retention_seconds=50.0)
            self.assertTrue(boundary_status.ok)
            self.assertEqual(boundary_payload["retention"]["eligible"], 1)
            self.assertEqual(repository.prune_acknowledged(retention_seconds=50.0).removed, 1)

            acknowledged_plan("retention-concurrent", acknowledged_at=900.0, owner="concurrent")
            with ThreadPoolExecutor(max_workers=2) as pool:
                status_future = pool.submit(repository.status, retention_seconds=50.0)
                prune_future = pool.submit(repository.prune_acknowledged, retention_seconds=50.0, limit=10)
                concurrent_status, concurrent_payload = status_future.result(timeout=5)
                concurrent_prune = prune_future.result(timeout=5)
            self.assertTrue(concurrent_status.ok)
            self.assertIsInstance(concurrent_payload, dict)
            self.assertTrue(concurrent_prune.ok)
            self.assertEqual(concurrent_prune.removed, 1)

            interrupted = acknowledged_plan("retention-interrupted", acknowledged_at=900.0, owner="interrupted")
            with sqlite3.connect(repository.path) as connection:
                connection.execute(
                    "CREATE TRIGGER reject_outbox_delete BEFORE DELETE ON lifecycle_outbox "
                    "BEGIN SELECT RAISE(ABORT, 'simulated interrupted cleanup'); END"
                )
            interrupted_result = repository.prune_acknowledged(retention_seconds=50.0, limit=10)
            self.assertFalse(interrupted_result.ok)
            retained_status, retained_payload = repository.status(retention_seconds=50.0)
            self.assertTrue(retained_status.ok)
            self.assertEqual(retained_payload["retention"]["acknowledged"], 1)
            with sqlite3.connect(repository.path) as connection:
                connection.execute("DROP TRIGGER reject_outbox_delete")
            recovered = repository.opportunistic_housekeeping(
                retention_seconds=50.0, interval_seconds=0.0, limit=1, checkpoint=False
            )
            self.assertTrue(recovered.ok)
            self.assertEqual(recovered.removed, 1)
            with sqlite3.connect(repository.path) as connection:
                remaining = {row[0] for row in connection.execute("SELECT intent_id FROM lifecycle_outbox")}
            self.assertNotIn(interrupted, remaining)
            cooldown = repository.opportunistic_housekeeping(retention_seconds=50.0, checkpoint=False)
            self.assertTrue(cooldown.skipped)
            self.assertEqual(cooldown.reason, "cooldown")

    def test_repository_rejects_invalid_claim_and_transition_arguments(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            self.assertEqual(
                repository.claim_batch(owner="", lease_seconds=0, limit=0)[0].kind,
                OutboxResultKind.REJECTED,
            )
            self.assertEqual(
                repository.claim_intents(intent_ids=(), owner="owner", lease_seconds=30)[0].kind,
                OutboxResultKind.REJECTED,
            )
            self.assertEqual(
                repository.renew_leases(intent_ids=(), owner="owner", lease_seconds=30)[0].kind,
                OutboxResultKind.REJECTED,
            )
            self.assertEqual(
                repository.advance_stage(intent_id="missing", owner="owner", stage="invalid").kind,
                OutboxResultKind.REJECTED,
            )
            self.assertEqual(
                repository.acknowledge_many(intent_ids=(), owner="owner")[0].kind,
                OutboxResultKind.REJECTED,
            )

    def test_batch_claim_quarantines_exhausted_and_inconsistent_rows(self) -> None:
        with TemporaryDirectory() as directory:
            now = [1_000.0]
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: now[0])

            exhausted = self._plan("claim-exhausted", max_attempts=1)
            self.assertTrue(repository.enqueue(
                exhausted, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ).ok)
            first, records = repository.claim_batch(owner="crashed", lease_seconds=5, limit=1)
            self.assertTrue(first.ok)
            self.assertEqual((len(records), records[0].attempts), (1, 1))
            now[0] += 6
            recovered, records = repository.claim_batch(owner="recovery", lease_seconds=5, limit=1)
            self.assertTrue(recovered.ok)
            self.assertEqual(records, ())
            status_result, status = repository.status()
            self.assertTrue(status_result.ok)
            self.assertEqual(status["states"].get("quarantined"), 1)
            exhausted_row = next(row for row in status["records"] if row["intent_id"] == exhausted.identity.idempotency_key)
            self.assertEqual(exhausted_row["failure"]["code"], "retry_exhausted")

            inconsistent = self._plan("claim-inconsistent")
            self.assertTrue(repository.enqueue(
                inconsistent, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ).ok)
            with sqlite3.connect(repository.path) as connection:
                connection.execute(
                    "UPDATE lifecycle_outbox SET processing_state='retry', lifecycle_stage='manual_review' "
                    "WHERE intent_id=?",
                    (inconsistent.identity.idempotency_key,),
                )
            claimed, records = repository.claim_batch(owner="poison", lease_seconds=5, limit=1)
            self.assertTrue(claimed.ok)
            self.assertEqual(records, ())
            status_result, status = repository.status()
            self.assertTrue(status_result.ok)
            self.assertEqual(status["states"].get("quarantined"), 2)
            inconsistent_row = next(
                row for row in status["records"] if row["intent_id"] == inconsistent.identity.idempotency_key
            )
            self.assertIn("active outbox state", inconsistent_row["failure"]["message"])

    def test_exact_claim_rejects_expired_lease_after_retry_budget_exhaustion(self) -> None:
        with TemporaryDirectory() as directory:
            now = [1_000.0]
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: now[0])
            plan = self._plan("claim-single-exhausted", max_attempts=1)
            intent_id = plan.identity.idempotency_key
            self.assertTrue(repository.enqueue(
                plan, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ).ok)
            first = repository.claim_intent(owner="first", lease_seconds=5, intent_id=intent_id)
            self.assertEqual(first.kind, OutboxResultKind.APPLIED)
            assert first.record is not None
            self.assertEqual(first.record.attempts, 1)
            now[0] += 6

            recovered = repository.claim_intent(owner="second", lease_seconds=5, intent_id=intent_id)

            self.assertEqual(recovered.kind, OutboxResultKind.REJECTED)
            self.assertIn("retry budget exhausted", recovered.reason)

    def test_status_on_missing_state_is_non_mutating(self) -> None:
        with TemporaryDirectory() as directory:
            result, payload = _LifecycleOutboxRepository(Path(directory)).status()
            self.assertTrue(result.ok)
            self.assertEqual(payload["schema_version"], 2)
            self.assertFalse((Path(directory) / ".nautical-state").exists())

    def test_status_filters_and_orders_records_by_state_then_update_time_and_id(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: 100.0)
            intents = {}
            for state in ("ready", "retry", "claimed", "quarantined", "manual_review"):
                plan = self._plan(f"status-{state}")
                intent_id = plan.identity.idempotency_key
                intents[state] = intent_id
                self.assertTrue(repository.enqueue(
                    plan, configuration_fingerprint="cfg", schedule_fingerprint="sch"
                ).ok)
                with sqlite3.connect(repository.path) as connection:
                    connection.execute(
                        "UPDATE lifecycle_outbox SET processing_state=?, lifecycle_stage=?, updated_at=?, "
                        "lease_owner=?, lease_expires_at=? WHERE intent_id=?",
                        (state, "finalized" if state == "manual_review" else "planned", 10.0,
                         "owner" if state == "claimed" else "", 200.0 if state == "claimed" else 0.0, intent_id),
                    )
            tied_ready = self._plan("status-ready-tie")
            tied_ready_id = tied_ready.identity.idempotency_key
            self.assertTrue(repository.enqueue(
                tied_ready, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ).ok)
            with sqlite3.connect(repository.path) as connection:
                connection.execute(
                    "UPDATE lifecycle_outbox SET updated_at=10.0 WHERE intent_id=?", (tied_ready_id,)
                )

            result, payload = repository.status(limit=2)
            self.assertTrue(result.ok)
            self.assertEqual(
                [(row["state"], row["intent_id"]) for row in payload["records"]],
                [("manual_review", intents["manual_review"]), ("quarantined", intents["quarantined"])],
            )
            filtered, filtered_payload = repository.status(intent_id=intents["retry"])
            self.assertTrue(filtered.ok)
            self.assertEqual([row["intent_id"] for row in filtered_payload["records"]], [intents["retry"]])
            all_result, all_payload = repository.status(limit=10)
            self.assertTrue(all_result.ok)
            self.assertEqual(
                [row["intent_id"] for row in all_payload["records"][-2:]],
                sorted((intents["ready"], tied_ready_id)),
            )

    def test_snapshot_is_sorted_and_rejects_lifecycle_and_integrity_poison(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            plans = (self._plan("snapshot-z"), self._plan("snapshot-a"))
            for plan in plans:
                repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            result, records = repository.snapshot_records()
            self.assertTrue(result.ok)
            self.assertEqual(
                [record.intent_id for record in records],
                sorted(plan.identity.idempotency_key for plan in plans),
            )

        for poison_kind in ("lifecycle", "integrity"):
            with self.subTest(poison_kind=poison_kind), TemporaryDirectory() as directory:
                repository = _LifecycleOutboxRepository(Path(directory))
                plans = (self._plan(f"snapshot-healthy-{poison_kind}"), self._plan(f"snapshot-poison-{poison_kind}"))
                for plan in plans:
                    repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
                poison_id = plans[1].identity.idempotency_key
                with sqlite3.connect(repository.path) as connection:
                    if poison_kind == "integrity":
                        connection.execute(
                            "UPDATE lifecycle_outbox SET work_kind='integrity', plan_json='{' WHERE intent_id=?",
                            (poison_id,),
                        )
                    else:
                        connection.execute(
                            "UPDATE lifecycle_outbox SET plan_json='{' WHERE intent_id=?", (poison_id,)
                        )
                result, records = repository.snapshot_records()
                self.assertEqual(result.kind, OutboxResultKind.REJECTED)
                self.assertEqual(records, ())

    def test_status_and_snapshot_reject_newer_schema_and_retry_on_locked_database(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))
            plan = self._plan("read-errors")
            repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            with sqlite3.connect(repository.path) as connection:
                connection.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION + 1}")
            status_result, _ = repository.status()
            snapshot_result, snapshot = repository.snapshot_records()
            self.assertEqual(status_result.kind, OutboxResultKind.REJECTED)
            self.assertEqual(snapshot_result.kind, OutboxResultKind.REJECTED)
            self.assertEqual(snapshot, ())

            with sqlite3.connect(repository.path) as connection:
                connection.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION}")
            from unittest.mock import patch

            with patch("nautical_core.lifecycle.outbox.sqlite3.connect", side_effect=sqlite3.OperationalError("database is locked")):
                status_result, _ = repository.status()
                snapshot_result, snapshot = repository.snapshot_records()
            self.assertEqual((status_result.kind, status_result.lock_busy), (OutboxResultKind.RETRYABLE, True))
            self.assertEqual((snapshot_result.kind, snapshot_result.lock_busy), (OutboxResultKind.RETRYABLE, True))
            self.assertEqual(snapshot, ())

    def test_housekeeping_respects_cooldown_bounds_and_preserves_non_acknowledged_rows(self) -> None:
        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory), clock=lambda: 1_000.0)
            states = ("ready", "retry", "claimed", "manual_review", "quarantined", "acknowledged")
            intents = {}
            for index, state in enumerate(states):
                plan = self._plan(f"housekeeping-{state}")
                intent_id = plan.identity.idempotency_key
                intents[state] = intent_id
                repository.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
                with sqlite3.connect(repository.path) as connection:
                    connection.execute(
                        "UPDATE lifecycle_outbox SET processing_state=?, acknowledged_at=?, updated_at=?, "
                        "lifecycle_stage=?, lease_owner=?, lease_expires_at=? WHERE intent_id=?",
                        (state, 1.0 if state == "acknowledged" else 0.0, float(index),
                         "finalized" if state == "acknowledged" else "planned",
                         "owner" if state == "claimed" else "", 2_000.0 if state == "claimed" else 0.0,
                         intent_id),
                    )
                if state == "ready":
                    no_work = repository.opportunistic_housekeeping(
                        retention_seconds=10.0, interval_seconds=100.0, size_threshold_bytes=2**31
                    )
                    self.assertTrue(no_work.skipped)
                    self.assertEqual(no_work.reason, "no_work")
            with sqlite3.connect(repository.path) as connection:
                connection.execute(
                    "INSERT INTO lifecycle_maintenance(key, value) VALUES('housekeeping_last_attempt', 999.0) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value"
                )

            cooldown = repository.opportunistic_housekeeping(
                retention_seconds=10.0, interval_seconds=100.0, size_threshold_bytes=0, limit=1
            )
            self.assertTrue(cooldown.skipped)
            self.assertEqual(cooldown.reason, "cooldown")

            with sqlite3.connect(repository.path) as connection:
                connection.execute(
                    "UPDATE lifecycle_maintenance SET value=0 WHERE key='housekeeping_last_attempt'"
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT processing_state, acknowledged_at FROM lifecycle_outbox WHERE intent_id=?",
                        (intents["acknowledged"],),
                    ).fetchone(),
                    ("acknowledged", 1.0),
                )
            result = repository.opportunistic_housekeeping(
                retention_seconds=10.0, interval_seconds=0.0, size_threshold_bytes=0, limit=1, checkpoint=True
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.removed, 1, result)
            self.assertIn(result.checkpoint, {"completed", "unavailable"})
            with sqlite3.connect(repository.path) as connection:
                remaining = {row[0] for row in connection.execute("SELECT intent_id FROM lifecycle_outbox")}
            self.assertEqual(remaining, {intents[state] for state in states if state != "acknowledged"})

    def test_outbox_transaction_boundary_has_no_taskwarrior_dependency(self) -> None:
        import ast

        source = Path("nautical_core/lifecycle/outbox.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        ]
        self.assertFalse(any("taskwarrior" in name.lower() for name in imports))

    def test_lifecycle_mutation_gateway_runs_outside_sqlite_transaction(self) -> None:
        from nautical_core.integration_models import (
            MutationOperation,
            MutationOutcome,
            MutationOutcomeKind,
            MutationPostcondition,
        )
        from nautical_core.lifecycle.application import (
            LifecycleApplicationOutcomeKind,
            LifecycleApplicationService,
        )
        from tests.support.lifecycle_execution import LifecycleExecutionFixture

        with TemporaryDirectory() as directory:
            repository = _LifecycleOutboxRepository(Path(directory))

            class MutationGateway:
                def __init__(self) -> None:
                    self.operations: list[MutationOperation] = []

                def apply(self, request):
                    connection = repository._session_conn
                    self.assert_transaction_closed(connection)
                    self.operations.append(request.operation)
                    postcondition = {
                        MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
                        MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED,
                    }[request.operation]
                    return MutationOutcome(
                        request.operation,
                        MutationOutcomeKind.APPLIED,
                        request.guard,
                        (postcondition,),
                    )

                @staticmethod
                def assert_transaction_closed(connection) -> None:
                    if connection is None or connection.in_transaction:
                        raise AssertionError("Taskwarrior mutation ran inside an outbox transaction")

                def compensate_imported_child(self, _request):
                    raise AssertionError("successful lifecycle execution must not compensate")

            gateway = MutationGateway()
            execution = LifecycleExecutionFixture(gateway)
            service = LifecycleApplicationService(
                unit_of_work=SimpleNamespace(mutation_epoch=0),
                mutations=gateway,
                execution=execution,
                outbox=repository,
                owner="transaction-boundary",
            )
            plan = self._plan("transaction-boundary")
            self.assertTrue(repository.enqueue(
                plan, configuration_fingerprint="cfg", schedule_fingerprint="sch"
            ).ok)

            result = service.drain(
                limit=1,
                configuration_fingerprint="cfg",
                schedule_fingerprint="sch",
            )

            self.assertTrue(result.claim.ok)
            self.assertEqual(len(result.outcomes), 1)
            self.assertIs(result.outcomes[0].kind, LifecycleApplicationOutcomeKind.APPLIED)
            self.assertEqual(
                gateway.operations,
                [MutationOperation.CHILD_IMPORT, MutationOperation.PARENT_LINK],
            )
            self.assertIsNone(repository._session_conn)
            status, payload = repository.status()
            self.assertTrue(status.ok)
            self.assertEqual(payload["states"].get("acknowledged"), 1)

    def test_claim_lease_port_exposes_only_cas_operations(self) -> None:
        from nautical_core.lifecycle.outbox_claims import RepositoryClaimLeasePort

        with TemporaryDirectory() as directory:
            port = RepositoryClaimLeasePort(_LifecycleOutboxRepository(Path(directory)))
            self.assertEqual(
                port.claim_batch(owner="", lease_seconds=0, limit=0)[0].kind,
                OutboxResultKind.REJECTED,
            )
            self.assertEqual(
                port.renew_leases(intent_ids=(), owner="owner", lease_seconds=30)[0].kind,
                OutboxResultKind.REJECTED,
            )

    def test_execution_policy_classifies_results_and_progress(self) -> None:
        self.assertEqual(
            OUTBOX_TO_APPLICATION[OutboxResultKind.RETRYABLE],
            LifecycleApplicationOutcomeKind.RETRYABLE,
        )
        self.assertEqual(
            MUTATION_TO_APPLICATION[MutationOutcomeKind.CONFLICT],
            LifecycleApplicationOutcomeKind.CONFLICT,
        )
        self.assertEqual(remaining_drain_work(ExecutionStage.PLANNED), 6)
        self.assertEqual(remaining_drain_work(ExecutionStage.PARENT_LINKED), 2)
        self.assertEqual(remaining_drain_work(ExecutionStage.VERIFIED), 1)

    def test_failure_policy_separates_retryable_and_review_decisions(self) -> None:
        self.assertEqual(
            classify_outbox_failure(OutboxResultKind.RETRYABLE),
            FailureDisposition.RETRY,
        )
        self.assertEqual(
            classify_outbox_failure(OutboxResultKind.CONFLICT),
            FailureDisposition.MANUAL_REVIEW,
        )
        self.assertEqual(
            classify_mutation_failure(MutationOutcomeKind.RETRYABLE),
            FailureDisposition.RETRY,
        )
        self.assertEqual(
            classify_mutation_failure(MutationOutcomeKind.REJECTED),
            FailureDisposition.MANUAL_REVIEW,
        )

    def test_batch_progress_reporter_accounts_action_and_terminal_work(self) -> None:
        from nautical_core.lifecycle.application import _BatchProgressReporter

        events = []
        reporter = _BatchProgressReporter(
            total=6,
            started=0.0,
            emit=lambda progress, event: events.append(event),
            progress=lambda _event: None,
        )
        state = SimpleNamespace(
            record=SimpleNamespace(stage=ExecutionStage.PLANNED, intent_id="intent-1"),
            progress_completed=0,
        )

        reporter.action(state, "child mutation", 2)
        reporter.outcome(
            state.record,
            SimpleNamespace(kind=LifecycleApplicationOutcomeKind.APPLIED),
            state,
        )

        self.assertEqual(state.progress_completed, 6)
        self.assertEqual([event.stage.value for event in events], ["processing", "complete"])
        self.assertEqual(events[-1].completed, 6)

    def test_batch_persistence_coordinator_handles_success_and_missing_rows(self) -> None:
        from nautical_core.lifecycle.application import _BatchPersistenceCoordinator

        class Outbox:
            def renew_leases(self, **_kwargs):
                return OutboxResult(OutboxResultKind.APPLIED), {
                    "ok": OutboxResult(OutboxResultKind.APPLIED),
                }

            def advance_stages(self, **_kwargs):
                return OutboxResult(OutboxResultKind.APPLIED), {}

        decisions = []
        coordinator = _BatchPersistenceCoordinator(
            outbox=Outbox(),
            owner="owner",
            retry_or_review=lambda record, result, detail, mutations: decisions.append(
                (record, result, detail, mutations)
            ) or "terminal",
        )
        state = SimpleNamespace(
            record=SimpleNamespace(intent_id="missing"),
            stage=ExecutionStage.PLANNED,
            terminal=None,
            mutations=[],
        )

        coordinator.renew([state], "child import", 30.0)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0][1].kind, OutboxResultKind.CONFLICT)

        state.terminal = None
        coordinator.advance([state], ExecutionStage.CHILD_PRESENT, "stage failed", "owner")
        self.assertEqual(len(decisions), 2)


if __name__ == "__main__":
    unittest.main()
