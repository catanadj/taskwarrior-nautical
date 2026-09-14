from __future__ import annotations

import unittest

from nautical_core.add_validation import parse_chain_max


class AddValidationContractTests(unittest.TestCase):
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
