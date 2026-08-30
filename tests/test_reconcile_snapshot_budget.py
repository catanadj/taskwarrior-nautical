import unittest
from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult
from nautical_core.task_read_repository import TaskReadRepository

from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits
from nautical_core.reconcile_snapshot_service import ReconcileSnapshotService


class ReconcileSnapshotBudgetTests(unittest.TestCase):
    def test_broad_admission_uses_probe_limit_and_marks_truncation(self) -> None:
        class Client:
            def __init__(self):
                self.args = ()

            def execute(self, args, **kwargs):
                self.args = tuple(args)
                return TaskCommandResult(
                    command=TaskCommand(("task", *args), "test broad admission", 1.0),
                    returncode=0, stdout='[{"uuid":"00000000-0000-4000-8000-000000000001","status":"pending"}]',
                    stderr="", kind=CommandFailureKind.SUCCESS, attempt=1, duration=0.0,
                )

        class Uow:
            mutation_epoch = 0

            def __init__(self, client):
                self.client = client

            def cached_read(self, scope):
                return None

            def cache_read(self, scope, value):
                return None

        client = Client()
        repository = TaskReadRepository(Uow(client))
        result = repository.broad_snapshot(
            identity="admission", filters=("chain:on",), statuses=("pending",), max_rows=1,
        )
        self.assertEqual(client.args[-2:], ("limit:1", "export"))
        self.assertTrue(result.value.truncated)

    def test_snapshot_export_budget_blocks_repository_before_read(self) -> None:
        class Repository:
            def lifecycle_candidates(self, **kwargs):
                raise AssertionError("repository must not be called")

        budget = OperatorInvocationBudget(OperatorLimits(taskwarrior_calls=1))
        self.assertTrue(budget.consume("taskwarrior_calls"))
        service = ReconcileSnapshotService(
            Repository(), read_value=lambda value, label: (), budget=budget,
        )
        with self.assertRaisesRegex(RuntimeError, "call budget"):
            service.candidate_rows()

    def test_snapshot_task_budget_blocks_projection_after_export(self) -> None:
        class Value:
            def __init__(self, value):
                self.value = value

            def raw_value(self):
                return self.value

        class Row:
            def __init__(self, chain):
                self.chain = chain

            def field(self, name):
                return Value(self.chain if name == "chainID" else "completed")

        class Repository:
            def lifecycle_candidates(self, **kwargs):
                return object()

        budget = OperatorInvocationBudget(OperatorLimits(tasks=1))
        rows = (Row("chain-a"), Row("chain-b"))
        service = ReconcileSnapshotService(
            Repository(), read_value=lambda value, label: rows, budget=budget,
        )
        with self.assertRaisesRegex(RuntimeError, "task or chain budget"):
            service.candidate_rows()


if __name__ == "__main__":
    unittest.main()
