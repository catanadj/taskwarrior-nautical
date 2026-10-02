"""Direct contracts for monthly recurrence support helpers."""

import unittest

from nautical_core.monthly_support import month_doms_safe


class MonthlySupportContractTests(unittest.TestCase):
    def test_monthly_expansion_runtime_defects_propagate(self) -> None:
        def broken_expansion(_spec: str, _year: int, _month: int) -> set[int]:
            raise RuntimeError("monthly expansion defect")

        with self.assertRaisesRegex(RuntimeError, "monthly expansion defect"):
            month_doms_safe("15", 2026, 1, expand_monthly_cached=broken_expansion)


if __name__ == "__main__":
    unittest.main()
