from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from nautical_core.integration_models import (
    Absent,
    CommandFailureKind,
    FailureEvidence,
    Found,
    TaskCommand,
    TaskCommandResult,
    Unavailable,
)
from nautical_core.integration_context import (
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.task_models import TaskObservation
from nautical_core.task_read_repository import (
    AuthoritativeTaskSnapshot,
    TaskQueryKind,
    TaskSnapshotScope,
)
from nautical_core.task_set_reads import (
    AuthoritativeSetReadService,
    ChainSlot,
    ChainSlotSetRequest,
    SetReadStatus,
    UUIDSetRequest,
)
from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork


def command_result() -> TaskCommandResult:
    command = TaskCommand(("task", "export"), "snapshot test", 1.0)
    return TaskCommandResult(command, 0, "[]", "", CommandFailureKind.SUCCESS, 1, 0.001)


def observations(rows: tuple[dict[str, object], ...]) -> tuple[TaskObservation, ...]:
    return tuple(TaskObservation.from_mapping(row, source_query="snapshot-test") for row in rows)


def unit_of_work(taskdata: str | Path):
    context = IntegrationContext(
        Path(taskdata).resolve(), "test", ("task",),
        ValidatedNauticalConfiguration("test", "config", "scheduler", "UTC", ()),
        timezone.utc,
        SilentDiagnostics(), SystemClock(), "task-read-contract", 256, IntegrationAccess.MUTATION,
    )
    return TaskwarriorUnitOfWork.create(context, env={})


class TaskReadSnapshotContractTests(unittest.TestCase):
    def test_lifecycle_candidate_query_distinguishes_bounded_and_full_audit(self) -> None:
        class Client:
            def __init__(self) -> None:
                self.calls: list[tuple[str, ...]] = []

            def execute(self, args, *, purpose, timeout, **_kwargs):
                self.calls.append(tuple(args))
                command = TaskCommand(tuple(args), purpose, timeout)
                row = {"uuid": "bounded-row", "status": "pending", "chain": "on"}
                return TaskCommandResult(
                    command, 0, json.dumps([row]), "", CommandFailureKind.SUCCESS,
                    1, 0.001,
                )

        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            client = Client()
            uow.client = client

            self.assertTrue(uow.repository.lifecycle_candidates(bounded=True).value)
            bounded_args = client.calls[-1]
            self.assertIn("nextLink:", bounded_args)

            self.assertTrue(uow.repository.lifecycle_candidates(bounded=False).value)
            full_args = client.calls[-1]
            self.assertNotIn("nextLink:", full_args)

    def test_repository_reuses_compatible_snapshot_and_limits_fallback_reads(self) -> None:
        class ScriptedClient:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.calls = []

            def execute(self, args, *, purpose, timeout, **_kwargs):
                self.calls.append((tuple(args), purpose))
                command = TaskCommand(("task", *args), purpose, timeout)
                return TaskCommandResult(
                    command, 0, self.outputs.pop(0), "", CommandFailureKind.SUCCESS, 1, 0.001
                )

        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            broad_rows = [
                {"uuid": "aaaaaaaa-0000-0000-0000-000000000001", "chainID": "chain-a", "link": 1, "chain": "on", "status": "pending"},
                {"uuid": "bbbbbbbb-0000-0000-0000-000000000002", "chainID": "chain-a", "link": 2, "chain": "on", "status": "completed"},
            ]
            predecessor = {"uuid": "cccccccc-0000-0000-0000-000000000003", "chainID": "chain-a", "link": 0, "status": "deleted"}
            client = ScriptedClient((json.dumps(broad_rows), json.dumps([predecessor]), json.dumps([broad_rows[0]])))
            uow.client = client
            repository = uow.repository
            broad = repository.broad_snapshot(identity="lifecycle", filters=("chain:on",), statuses=("pending", "completed"))
            self.assertIsInstance(broad, Found)
            uuid_read = repository.by_uuid("aaaaaaaa", statuses=("pending",))
            child_read = repository.exact_child_slot("chain-a", 2, statuses=("completed",))
            self.assertIsInstance(uuid_read, Found)
            self.assertIsInstance(child_read, Found)
            self.assertIs(uuid_read.value, broad.value.rows[0])
            self.assertEqual(uuid_read.value.provenance.source_query, "broad:lifecycle")
            self.assertEqual(len(client.calls), 1)

            self.assertIsInstance(repository.predecessor_slot("chain-a", 0), Found)
            self.assertIsInstance(repository.predecessor_slot("chain-a", 0), Found)
            self.assertEqual(len(client.calls), 2)
            self.assertIsInstance(repository.verification(broad_rows[0]["uuid"], statuses=("pending",)), Found)
            self.assertEqual(len(client.calls), 3)

    def test_repository_never_turns_untrusted_output_into_absence(self) -> None:
        class ScriptedClient:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.calls = 0

            def execute(self, args, *, purpose, timeout, **_kwargs):
                self.calls += 1
                kind, stdout, stderr = self.outputs.pop(0)
                command = TaskCommand(("task", *args), purpose, timeout)
                return TaskCommandResult(
                    command, 0 if kind is CommandFailureKind.SUCCESS else 1, stdout, stderr, kind, 1, 0.001
                )

        duplicate = [
            {"uuid": "aaaaaaaa-0000-0000-0000-000000000001", "status": "pending"},
            {"uuid": "aaaaaaaa-1111-0000-0000-000000000002", "status": "pending"},
        ]
        mismatched = [{"uuid": "bbbbbbbb-0000-4000-8000-000000000003", "status": "pending"}]
        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            uow.client = ScriptedClient((
                (CommandFailureKind.SUCCESS, '{"uuid":', ""),
                (CommandFailureKind.SUCCESS, json.dumps(mismatched), ""),
                (CommandFailureKind.SUCCESS, json.dumps(duplicate), ""),
                (CommandFailureKind.BUSY, "", "database is locked"),
                (CommandFailureKind.SUCCESS, "", ""),
            ))
            repository = uow.repository
            malformed = repository.by_uuid("11111111", statuses=("pending",))
            mismatched_read = repository.by_uuid("22222222", statuses=("pending",))
            duplicate_read = repository.by_uuid("aaaaaaaa", statuses=("pending",))
            busy = repository.by_uuid("33333333", statuses=("pending",))
            absent = repository.by_uuid("44444444", statuses=("pending",))
        self.assertIsInstance(malformed, Unavailable)
        self.assertIsInstance(mismatched_read, Unavailable)
        self.assertIsInstance(duplicate_read, Unavailable)
        self.assertIsInstance(busy, Unavailable)
        self.assertTrue(busy.retryable)
        self.assertIsInstance(absent, Absent)

    def test_repository_mutation_epoch_invalidates_cached_identity_read(self) -> None:
        class Client:
            def __init__(self):
                self.calls = 0

            def execute(self, args, *, purpose, timeout, **_kwargs):
                self.calls += 1
                row = {"uuid": "aaaaaaaa-0000-4000-8000-000000000001", "status": "pending", "modified": str(self.calls)}
                command = TaskCommand(("task", *args), purpose, timeout)
                return TaskCommandResult(command, 0, json.dumps([row]), "", CommandFailureKind.SUCCESS, 1, 0.001)

        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            client = Client()
            uow.client = client
            first = uow.repository.by_uuid("aaaaaaaa", statuses=("pending",))
            repeated = uow.repository.by_uuid("aaaaaaaa", statuses=("pending",))
            self.assertIsInstance(first, Found)
            self.assertIsInstance(repeated, Found)
            self.assertEqual(client.calls, 1)
            uow.record_mutation()
            fresh = uow.repository.by_uuid("aaaaaaaa", statuses=("pending",))
            self.assertIsInstance(fresh, Found)
            self.assertEqual(fresh.value.field("modified").raw_value(), "2")
            self.assertEqual(client.calls, 2)

    def test_repository_keeps_found_rows_with_decode_issues(self) -> None:
        class Client:
            def execute(self, args, *, purpose, timeout, **_kwargs):
                command = TaskCommand(("task", *args), purpose, timeout)
                row = {"uuid": "aaaaaaaa-0000-4000-8000-000000000001", "status": "pending", "link": "not-an-integer"}
                return TaskCommandResult(command, 0, json.dumps([row]), "", CommandFailureKind.SUCCESS, 1, 0.001)

        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            uow.client = Client()
            read = uow.repository.by_uuid("aaaaaaaa-0000-4000-8000-000000000001", statuses=("pending",))
        self.assertIsInstance(read, Found)
        self.assertTrue(read.value.issues)
        self.assertEqual(read.value.field("link").raw_value(), "not-an-integer")

    def test_repository_keeps_missing_status_as_malformed_found(self) -> None:
        class Client:
            def execute(self, args, *, purpose, timeout, **_kwargs):
                command = TaskCommand(("task", *args), purpose, timeout)
                row = {"uuid": "aaaaaaaa-0000-4000-8000-000000000001", "chainID": "chain-a"}
                return TaskCommandResult(command, 0, json.dumps([row]), "", CommandFailureKind.SUCCESS, 1, 0.001)

        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            uow.client = Client()
            read = uow.repository.by_uuid("aaaaaaaa-0000-4000-8000-000000000001", statuses=("pending",))
        self.assertIsInstance(read, Found)
        self.assertEqual(read.value.field("status").presence.value, "absent")

    def test_repository_exposes_typed_chain_root_and_lifecycle_reads(self) -> None:
        class Client:
            def __init__(self, payloads):
                self.payloads = list(payloads)

            def execute(self, args, *, purpose, timeout, **_kwargs):
                command = TaskCommand(("task", *args), purpose, timeout)
                return TaskCommandResult(
                    command, 0, json.dumps(self.payloads.pop(0)), "", CommandFailureKind.SUCCESS, 1, 0.001
                )

        chain_rows = [
            {"uuid": "aaaaaaaa-0000-0000-0000-000000000001", "chainID": "chain-a", "link": 1, "chain": "on", "status": "pending"},
            {"uuid": "bbbbbbbb-0000-0000-0000-000000000002", "chainID": "chain-a", "link": 2, "chain": "on", "status": "completed"},
        ]
        with TemporaryDirectory() as directory:
            uow = unit_of_work(directory)
            uow.client = Client((chain_rows, [chain_rows[0]], [chain_rows[1]]))
            repository = uow.repository
            chain = repository.chain_snapshot("chain-a", statuses=("pending", "completed"))
            active = repository.active_recurrence_roots()
            lifecycle = repository.lifecycle_candidates(statuses=("completed",))
        self.assertIsInstance(chain, Found)
        self.assertEqual(len(chain.value), 2)
        self.assertIsInstance(active, Found)
        self.assertEqual(active.value[0].field("link").value.value, 1)
        self.assertIsInstance(lifecycle, Found)
        self.assertEqual(lifecycle.value[0].field("link").value.value, 2)
    def test_authoritative_snapshot_preserves_scope_and_indexes(self) -> None:
        source = {"uuid": "aaaaaaaa-0000-0000-0000-000000000001", "chainID": "chain-a", "link": 2.0, "status": "pending"}
        sibling = {"uuid": "bbbbbbbb-0000-0000-0000-000000000002", "chainID": "chain-a", "link": 3, "status": "waiting"}
        scope = TaskSnapshotScope(
            TaskQueryKind.BROAD, "active-nautical", ("waiting", "pending", "pending"),
            complete_chain_history=False,
        )
        snapshot = AuthoritativeTaskSnapshot(scope, observations((source, sibling)), command_result())

        self.assertEqual(scope.statuses, ("pending", "waiting"))
        self.assertFalse(scope.complete_chain_history)
        self.assertEqual(snapshot.uuid_matches(source["uuid"]), (snapshot.rows[0],))
        self.assertEqual(snapshot.uuid_matches("aaaaaaaa"), (snapshot.rows[0],))
        self.assertEqual(snapshot.chain_rows("chain-a"), snapshot.rows)
        self.assertEqual(snapshot.slot_rows("chain-a", 3), (snapshot.rows[1],))
        source["status"] = "deleted"
        self.assertEqual(snapshot.rows[0].field("status").value.value, "pending")
        with self.assertRaises(TypeError):
            snapshot.rows[0].fields["status"] = snapshot.rows[0].field("status")  # type: ignore[index]

    def test_snapshot_indexes_retain_ambiguous_uuid_and_slots(self) -> None:
        rows = (
            {"uuid": "aaaaaaaa-0000-0000-0000-000000000001", "chainID": "chain-a", "link": 2},
            {"uuid": "aaaaaaaa-1111-0000-0000-000000000002", "chainID": "chain-a", "link": 2},
        )
        snapshot = AuthoritativeTaskSnapshot(
            TaskSnapshotScope(TaskQueryKind.CHAIN, "chain-a", ("pending",), complete_chain_history=True),
            observations(rows),
            command_result(),
        )
        self.assertEqual(len(snapshot.uuid_matches("aaaaaaaa")), 2)
        self.assertEqual(len(snapshot.slot_rows("chain-a", 2)), 2)

    def test_bounded_identity_set_reads_preserve_partial_and_conflict_states(self) -> None:
        first = "11111111-1111-4111-8111-111111111111"
        second = "22222222-2222-4222-8222-222222222222"

        class Repository:
            mutation_epoch = 4

            def __init__(self, rows):
                self.rows = rows
                self.calls = 0
                self.fail_after = None

            def broad_snapshot(self, *, identity, filters, statuses, refresh=False):
                del identity, filters, statuses, refresh
                self.calls += 1
                if self.fail_after is not None and self.calls > self.fail_after:
                    command = TaskCommand(("task", "export"), "set read", 1.0)
                    evidence = FailureEvidence(
                        command, CommandFailureKind.BUSY, 1, 1, 0.1, True, "locked"
                    )
                    return Unavailable("set", evidence)
                if not self.rows:
                    return Absent("set", "empty set")
                snapshot = AuthoritativeTaskSnapshot(
                    TaskSnapshotScope(TaskQueryKind.BROAD, "set", ("pending",)),
                    observations(tuple(self.rows)), command_result(),
                )
                return Found(snapshot, "set")

        found_repository = Repository((
            {"uuid": first, "status": "pending", "chainID": "abcdef12", "link": 1},
        ))
        found = AuthoritativeSetReadService(found_repository).read_uuids(
            UUIDSetRequest((first, second), expected_mutation_epoch=4)
        )
        self.assertIs(found.status, SetReadStatus.COMPLETE)
        self.assertIn(first, found.found)
        self.assertIn(second, found.absent)
        self.assertTrue(found.complete_for_requested_identities)

        duplicate = AuthoritativeSetReadService(Repository((
            {"uuid": first, "status": "pending", "chainID": "abcdef12", "link": 1},
            {"uuid": first, "status": "pending", "chainID": "abcdef12", "link": 2},
        ))).read_uuids(UUIDSetRequest((first,)))
        self.assertIs(duplicate.status, SetReadStatus.DUPLICATE)

        slot = ChainSlot("abcdef12", 2)
        contradictory = AuthoritativeSetReadService(Repository((
            {"uuid": second, "status": "pending", "chainID": "abcdef12", "link": 2, "prevLink": "deadbeef"},
        ))).read_slots(ChainSlotSetRequest((slot,), expected_predecessors={slot: "cafebabe"}))
        self.assertIs(contradictory.status, SetReadStatus.CONTRADICTORY)

        stale_repo = Repository(({"uuid": first, "status": "pending", "chainID": "abcdef12", "link": 1},))
        stale_repo.mutation_epoch = 5
        stale = AuthoritativeSetReadService(stale_repo).read_uuids(
            UUIDSetRequest((first,), expected_mutation_epoch=4)
        )
        self.assertIs(stale.status, SetReadStatus.STALE)

        partial_repo = Repository(({"uuid": first, "status": "pending", "chainID": "abcdef12", "link": 1},))
        partial_repo.fail_after = 1
        partial = AuthoritativeSetReadService(partial_repo).read_uuids(
            UUIDSetRequest((first, second), max_chunk_size=1)
        )
        self.assertIs(partial.status, SetReadStatus.PARTIAL)
        with self.assertRaises(ValueError):
            UUIDSetRequest(("deadbeef",))
