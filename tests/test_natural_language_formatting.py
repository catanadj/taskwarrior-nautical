from __future__ import annotations

import unittest

from nautical_core.add_formatting import format_anchor_rows


class NaturalLanguageFormattingTests(unittest.TestCase):
    def test_anchor_rows_keep_all_information_while_grouping_sections(self) -> None:
        rows = [
            ("Anchor", "every Monday"),
            ("Next anchor", "2026-09-14"),
            ("Delta", "in 5 days"),
            ("Chain cap", "12"),
        ]
        formatted = format_anchor_rows(rows)
        values = [(key, value) for key, value in formatted if key is not None]
        self.assertEqual(set(values), {row for row in rows if row[0] != "Delta"})
        self.assertIn((None, ""), formatted)

    def test_empty_rows_return_a_stable_empty_shape(self) -> None:
        self.assertEqual(format_anchor_rows([]), [])


if __name__ == "__main__":
    unittest.main()
