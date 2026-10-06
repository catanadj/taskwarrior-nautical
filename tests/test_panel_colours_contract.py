"""Direct contracts for stable recurrence-chain display colours."""

from __future__ import annotations

import unittest

from nautical_core.panel_colours import chain_colour_root


class PanelColourContractTests(unittest.TestCase):
    def test_chain_colour_uses_complete_root_identity(self) -> None:
        root = "12345678-1234-4234-8234-00000000abcd"
        self.assertEqual(chain_colour_root("anchor", root), chain_colour_root("anchor", root))
        self.assertEqual(chain_colour_root("anchor", root.upper()), chain_colour_root("anchor", root))
        self.assertEqual(chain_colour_root("anchor", ""), "bright_cyan")
        self.assertEqual(chain_colour_root("cp", ""), "orange_red1")

        same_suffix_roots = tuple(
            f"{idx:08x}-1234-4234-8234-00000000abcd" for idx in range(256)
        )
        anchor_colours = {chain_colour_root("anchor", candidate) for candidate in same_suffix_roots}
        cp_colours = {chain_colour_root("cp", candidate) for candidate in same_suffix_roots}
        self.assertGreaterEqual(len(anchor_colours), 16)
        self.assertGreaterEqual(len(cp_colours), 15)
        legacy = "legacy/root identifier"
        self.assertEqual(chain_colour_root("anchor", legacy), chain_colour_root("anchor", legacy))
        self.assertNotEqual(chain_colour_root("anchor", root), chain_colour_root("cp", root))


if __name__ == "__main__":
    unittest.main()
