from __future__ import annotations

import unittest
from unittest.mock import patch

import nautical_core.expansion_support as expansion_support


class ExpansionSupportContracts(unittest.TestCase):
    def test_weekday_index_keeps_invalid_text_as_non_match(self) -> None:
        self.assertIsNone(expansion_support.wd_idx("not-a-weekday", wd_abbr=[]))

    def test_weekday_index_does_not_hide_unexpected_integer_failures(self) -> None:
        with patch.object(
            expansion_support,
            "int",
            side_effect=RuntimeError("weekday conversion implementation failed"),
            create=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "weekday conversion implementation failed"):
                expansion_support.wd_idx("9", wd_abbr=[])


if __name__ == "__main__":
    unittest.main()
