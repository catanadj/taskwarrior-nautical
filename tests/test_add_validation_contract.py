from __future__ import annotations

import unittest
from types import SimpleNamespace

from nautical_core.add_validation import collect_anchor_time_slots, parse_chain_max, safe_parse_duration


class AddValidationContractTests(unittest.TestCase):
    def test_duration_parser_does_not_hide_internal_failures(self) -> None:
        def broken_parser(_value: object) -> None:
            raise RuntimeError("duration parser invariant failed")

        with self.assertRaisesRegex(RuntimeError, "parser invariant"):
            safe_parse_duration(
                "1d",
                "cp",
                core=SimpleNamespace(parse_cp_sequence=broken_parser),
                diag=lambda _message: None,
            )

    def test_anchor_slot_collection_does_not_hide_normalizer_defects(self) -> None:
        def broken_normalizer(_value: object) -> list[tuple[int, int]]:
            raise RuntimeError("time-slot normalizer failed")

        with self.assertRaisesRegex(RuntimeError, "time-slot normalizer failed"):
            collect_anchor_time_slots(
                [[{"mods": {"t": "09:00"}}]],
                "",
                (9, 0),
                normalize_time_slots=broken_normalizer,
                anchor_file_dir="",
            )

    def test_chain_max_accepts_positive_integral_values_and_rejects_ambiguous_caps(self) -> None:
        for value, expected in ((1, 1), (5, 5), (5.0, 5), ("5", 5), ("5.0", 5)):
            with self.subTest(value=value):
                self.assertEqual(parse_chain_max(value), (expected, None))

        for value, expected_error in (
            (0, "chainMax must be > 0"),
            (-1, "chainMax must be > 0"),
            (2.5, "chainMax must be a positive integer"),
            ("abc", "chainMax must be a positive integer"),
            (True, "chainMax must be a positive integer"),
        ):
            with self.subTest(value=value):
                self.assertEqual(parse_chain_max(value), (None, expected_error))


if __name__ == "__main__":
    unittest.main()
