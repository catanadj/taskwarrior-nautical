from __future__ import annotations

import unittest

from nautical_core.recurrence_metadata import atom_interval


class RecurrenceMetadataContracts(unittest.TestCase):
    def test_atom_interval_defaults_for_malformed_external_values(self) -> None:
        for value in ("not-an-integer", [1], float("inf")):
            with self.subTest(value=value):
                self.assertEqual(atom_interval({"ival": value}), 1)

    def test_atom_interval_does_not_hide_unexpected_coercion_failures(self) -> None:
        class BrokenInterval:
            def __int__(self) -> int:
                raise RuntimeError("interval coercion implementation failed")

        with self.assertRaisesRegex(RuntimeError, "interval coercion implementation failed"):
            atom_interval({"ival": BrokenInterval()})


if __name__ == "__main__":
    unittest.main()
