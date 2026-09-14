from __future__ import annotations

from datetime import date
import unittest

import nautical_core as core


class CoreUtilityContractTests(unittest.TestCase):
    def test_week_count_uses_iso_week_boundaries(self) -> None:
        self.assertEqual(core._weeks_between(date(2024, 12, 31), date(2025, 1, 1)), 0)
        self.assertEqual(core._weeks_between(date(2024, 12, 29), date(2024, 12, 30)), 1)

    def test_short_uuid_handles_invalid_and_short_values(self) -> None:
        self.assertEqual(core.short_uuid(None), "")
        self.assertEqual(core.short_uuid(1234), "")
        self.assertEqual(core.short_uuid("abcd"), "abcd")

    def test_integer_coercion_rejects_values_above_supported_bounds(self) -> None:
        too_large = 2**63
        self.assertEqual(core.coerce_int(too_large, default=7), 7)
        self.assertEqual(core.coerce_int(float(too_large), default=7), 7)


if __name__ == "__main__":
    unittest.main()
