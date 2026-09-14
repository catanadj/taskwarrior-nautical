from __future__ import annotations

import unittest

from nautical_core.task_read_repository import _link_number


class LinkOverflowTests(unittest.TestCase):
    def test_non_finite_and_overflow_links_are_rejected(self) -> None:
        self.assertIsNone(_link_number("inf"))
        self.assertIsNone(_link_number("1e999"))


if __name__ == "__main__":
    unittest.main()
