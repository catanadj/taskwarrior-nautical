"""Characterization tests for the on-modify effect boundary."""

from __future__ import annotations

import importlib
import sys
import unittest


class ModifyIsolationTests(unittest.TestCase):
    def test_read_effects_import_without_hook_bootstrap(self) -> None:
        sys.modules.pop("nautical_core.hooks.modify_impl", None)
        module = importlib.import_module("nautical_core.modify_read_effects")
        self.assertIsNotNone(module)
        self.assertNotIn("nautical_core.hooks.modify_impl", sys.modules)

    def test_composition_capabilities_are_explicit_and_frozen(self) -> None:
        from nautical_core.modify_composition import ModifyHookCapabilities

        fields = getattr(ModifyHookCapabilities, "__dataclass_fields__", {})
        self.assertIn("modify_read_effects", fields)
        self.assertTrue(ModifyHookCapabilities.__dataclass_params__.frozen)


if __name__ == "__main__":
    unittest.main()
