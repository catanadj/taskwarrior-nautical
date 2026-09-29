"""Direct contracts for on-add panel configuration and chain colouring."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core.hooks.add_impl as add_impl
import nautical_core.modify_models as modify_models
from nautical_core.panel_colours import chain_colour_root


class AddPanelContractTests(unittest.TestCase):
    def test_on_add_preview_uses_configured_chain_colour(self) -> None:
        rendered = []

        def panel_themes():
            return {"preview_anchor": {"border": "turquoise2", "title": "turquoise2"}}

        core = SimpleNamespace(
            panel_themes=panel_themes,
            CHAIN_COLOR_PER_CHAIN=True,
            PANEL_MODE="static",
            LIVE_PANEL_DURATION_MS=275,
            LIVE_PANEL_FOOTER="NAUTICAL",
            FAST_COLOR=True,
            render_panel=lambda *args, **kwargs: rendered.append((args, kwargs)),
        )
        task = {"chainID": "12345678", "anchor": "w:mon"}

        with patch.object(add_impl, "core", core), patch.object(
            add_impl, "_module", return_value=modify_models
        ):
            add_impl._panel("Preview", [("Pattern", "w:mon")], kind="preview_anchor", task=task)
            enabled = rendered[-1][1]
            colour = chain_colour_root("anchor", "12345678")
            self.assertEqual(
                enabled["themes"]["preview_anchor"],
                {"border": colour, "title": colour},
            )
            self.assertEqual(enabled["live_duration_ms"], 275)

            core.CHAIN_COLOR_PER_CHAIN = False
            add_impl._panel("Preview", [("Pattern", "w:mon")], kind="preview_anchor", task=task)
            disabled = rendered[-1][1]
            self.assertEqual(
                disabled["themes"],
                {"preview_anchor": {"border": "turquoise2", "title": "turquoise2"}},
            )


if __name__ == "__main__":
    unittest.main()
