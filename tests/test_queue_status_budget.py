import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits
from nautical_core.queue_status_service import QueueStatusService


class QueueStatusBudgetTests(unittest.TestCase):
    def test_status_reserves_sqlite_and_outbox_budgets_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            outbox_path = taskdata / ".nautical-state" / ".nautical_lifecycle_outbox.db"
            outbox_path.parent.mkdir()
            outbox_path.touch()
            budget = OperatorInvocationBudget(
                OperatorLimits(sqlite_transactions=1, outbox_rows=3),
            )
            with patch(
                "nautical_core.queue_status_service.LifecycleOutboxRepository.status",
                return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"schema_version": 2, "integrity": "ok", "records": []}),
            ) as status:
                summary, issues = QueueStatusService().outbox_summary(
                    outbox_path, stale_after=60.0, limit=10, budget=budget,
                )
            self.assertEqual(issues, [])
            self.assertTrue(summary["exists"])
            status.assert_called_once_with(limit=3, stale_after=60.0)
            self.assertEqual(budget.usage("sqlite_transactions"), 1)
            self.assertEqual(budget.usage("outbox_rows"), 3)

    def test_status_does_not_open_sqlite_when_transaction_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            outbox_path = Path(directory) / ".nautical-state" / ".nautical_lifecycle_outbox.db"
            outbox_path.parent.mkdir()
            outbox_path.touch()
            budget = OperatorInvocationBudget(OperatorLimits(sqlite_transactions=1))
            self.assertTrue(budget.consume("sqlite_transactions"))
            with patch("nautical_core.queue_status_service.LifecycleOutboxRepository.status") as status:
                _summary, issues = QueueStatusService().outbox_summary(
                    outbox_path, stale_after=60.0, limit=1, budget=budget,
                )
            self.assertEqual(issues, ["operator SQLite transaction budget exhausted"])
            status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
