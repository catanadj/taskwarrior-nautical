import unittest

from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits
from nautical_core.reconcile_snapshot_service import ReconcileSnapshotService


class ReconcileSnapshotBudgetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
