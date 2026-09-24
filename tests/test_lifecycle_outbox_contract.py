from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from nautical_core.lifecycle_models import ExecutionStage
from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
from nautical_core.lifecycle_execution_policy import (
    FailureDisposition,
    MUTATION_TO_APPLICATION,
    OUTBOX_TO_APPLICATION,
    LifecycleApplicationOutcomeKind,
    classify_mutation_failure,
    classify_outbox_failure,
    remaining_drain_work,
)
from nautical_core.integration_models import MutationOutcomeKind
from nautical_core.lifecycle_outbox import (
    OUTBOX_LEGACY_SCHEMA_VERSION,
    OUTBOX_SCHEMA_VERSION,
    LifecycleOutboxError,
    _LifecycleOutboxRepository,
    OutboxFailure,
    OutboxResult,
    OutboxProcessingState,
    OutboxResultKind,
)


class LifecycleOutboxContractTests(unittest.TestCase):
    @staticmethod
    def _plan(chain: str = "contract") -> LifecyclePlan:
        from dev_tools.nautical_golden_tests import _task_draft

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

        from nautical_core.lifecycle_outbox_schema import initialize

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

    def test_status_on_missing_state_is_non_mutating(self) -> None:
        with TemporaryDirectory() as directory:
            result, payload = _LifecycleOutboxRepository(Path(directory)).status()
            self.assertTrue(result.ok)
            self.assertEqual(payload["schema_version"], 2)
            self.assertFalse((Path(directory) / ".nautical-state").exists())

    def test_outbox_transaction_boundary_has_no_taskwarrior_dependency(self) -> None:
        import ast

        source = Path("nautical_core/lifecycle_outbox.py").read_text(encoding="utf-8")
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

    def test_claim_lease_port_exposes_only_cas_operations(self) -> None:
        from nautical_core.lifecycle_outbox_claims import RepositoryClaimLeasePort

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
        from nautical_core.lifecycle_application import _BatchProgressReporter

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
        from nautical_core.lifecycle_application import _BatchPersistenceCoordinator

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
