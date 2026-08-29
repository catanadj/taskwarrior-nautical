import tempfile
import unittest
from pathlib import Path

from nautical_core.file_backed_dates import load_file_date_data
from nautical_core.operator_context import OperatorInvocationBudget
from nautical_core.operator_models import OperatorLimits


class FileBudgetTests(unittest.TestCase):
    def test_file_record_budget_is_checked_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dates.txt"
            path.write_text("2026-01-01\n", encoding="utf-8")
            budget = OperatorInvocationBudget(OperatorLimits(file_records=1))
            self.assertTrue(budget.consume("file_records"))
            with self.assertRaisesRegex(ValueError, "file-record budget"):
                load_file_date_data(str(path), label="dates", budget=budget)


if __name__ == "__main__":
    unittest.main()
