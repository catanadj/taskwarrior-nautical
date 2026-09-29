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

    def test_anchor_rows_number_upcoming_after_next_anchor(self) -> None:
        rows = [
            ("Pattern", "w:mon"),
            ("First due", "2025-01-01 09:00"),
            ("Next anchor", "2025-01-08 09:00"),
            ("Upcoming", "2025-01-15 09:00\n2025-01-22 09:00"),
            ("Delta", "+7d"),
            ("Chain", "enabled"),
        ]

        formatted = format_anchor_rows(rows)
        upcoming = dict(formatted)["Upcoming"]

        self.assertIn(" 3 ▸[/] 2025-01-15 09:00", upcoming)
        self.assertIn(" 4 ▸[/] 2025-01-22 09:00", upcoming)

    def test_anchor_rows_number_upcoming_without_next_anchor(self) -> None:
        rows = [
            ("Pattern", "w:mon"),
            ("First due", "2025-01-01 09:00"),
            ("Upcoming", "2025-01-08 09:00"),
            ("Delta", "+7d"),
            ("Other", "x"),
        ]

        formatted = format_anchor_rows(rows)
        by_key = dict(formatted)

        self.assertIn(" 2 ▸[/] 2025-01-08 09:00", by_key["Upcoming"])
        self.assertIn("Δ +7d", by_key["First due"])


if __name__ == "__main__":
    unittest.main()
