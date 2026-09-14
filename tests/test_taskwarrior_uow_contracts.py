from __future__ import annotations

from datetime import timezone
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from nautical_core.integration_context import (
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.taskwarrior_uow import QueryScope, QueryScopeKind, TaskwarriorUnitOfWork


def make_uow(taskdata: str | Path, *, access: IntegrationAccess, budget: int = 8):
    context = IntegrationContext(
        Path(taskdata), "test", (sys.executable,),
        ValidatedNauticalConfiguration("test", "config", "scheduler", "UTC", ()),
        timezone.utc, SilentDiagnostics(), SystemClock(),
        "uow-contract", budget, access,
    )
    return TaskwarriorUnitOfWork.create(context)


class TaskwarriorUnitOfWorkContractTests(unittest.TestCase):
    def test_reads_are_scoped_and_invalidated_by_mutation_epoch(self) -> None:
        with TemporaryDirectory() as directory:
            uow = make_uow(directory, access=IntegrationAccess.MUTATION, budget=3)
            scope = QueryScope(QueryScopeKind.UUID, "task-uuid", ("pending",))
            cached = uow.cache_read(scope, {"uuid": "task-uuid"})
            self.assertIs(cached.provenance.scope, scope)
            self.assertEqual(cached.provenance.covers, (scope,))
            self.assertEqual(cached.provenance.mutation_epoch, 0)
            self.assertIs(uow.cached_read(scope), cached)

            self.assertEqual(uow.record_mutation(affected=(scope,)), 1)
            self.assertIsNone(uow.cached_read(scope))
            uow.cache_read(scope, {"uuid": "task-uuid", "modified": "later"})
            uow.record_mutation(uncertain=True)
            self.assertEqual((uow.mutation_epoch, uow.reads.size), (2, 0))
            self.assertTrue(uow.mutations.observations[-1].uncertain)

    def test_broad_snapshot_serves_only_explicitly_declared_scopes(self) -> None:
        with TemporaryDirectory() as directory:
            uow = make_uow(directory, access=IntegrationAccess.READ_ONLY)
            broad = QueryScope(QueryScopeKind.BROAD, "taskdata", ("pending", "completed"))
            task = QueryScope(QueryScopeKind.UUID, "task-uuid", ("pending", "completed"))
            predecessor = QueryScope(QueryScopeKind.PREDECESSOR, "previous-uuid", ("completed", "deleted"))
            cached = uow.cache_read(broad, [{"uuid": "task-uuid", "status": "pending"}], covers=(task,))
            self.assertIs(uow.cached_read(task), cached)
            self.assertIs(cached.provenance.scope, broad)
            self.assertIsNone(uow.cached_read(predecessor))
            narrow = uow.cache_read(predecessor, {"uuid": "previous-uuid"})
            self.assertIs(uow.cached_read(predecessor), narrow)

    def test_invocations_on_same_taskdata_do_not_share_ephemeral_state(self) -> None:
        with TemporaryDirectory() as directory:
            context = IntegrationContext(
                Path(directory), "test", (sys.executable,),
                ValidatedNauticalConfiguration("test", "config", "scheduler", "UTC", ()),
                timezone.utc, SilentDiagnostics(), SystemClock(),
                "uow-isolation", 8, IntegrationAccess.MUTATION,
            )
            first = TaskwarriorUnitOfWork.create(context)
            second = TaskwarriorUnitOfWork.create(context)
            scope = QueryScope(QueryScopeKind.UUID, "task-uuid")
            first.cache_read(scope, {"uuid": "task-uuid"})
            first.record_mutation(affected=(scope,))

            self.assertIsNot(first.reads, second.reads)
            self.assertIsNot(first.mutations, second.mutations)
            self.assertIsNot(first.commands, second.commands)
            self.assertIsNot(first.diagnostics, second.diagnostics)
            self.assertEqual((second.mutation_epoch, second.reads.size), (0, 0))
            self.assertEqual(first.outbox, second.outbox)
