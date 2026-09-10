"""Contract tests for immutable, typed core API bindings."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
import unittest

from nautical_core.api_bindings import ApiBinding


class ApiBindingContractTests(unittest.TestCase):
    def test_binding_is_frozen_and_exposes_declared_members(self) -> None:
        binding = ApiBinding.from_mapping({"parse": str.strip, "version": "v1"})

        self.assertTrue(is_dataclass(binding))
        self.assertEqual(binding.parse(" value "), "value")
        self.assertEqual(binding.version, "v1")
        with self.assertRaises(FrozenInstanceError):
            binding._members = {}
        with self.assertRaises(AttributeError):
            _ = binding.missing

    def test_core_api_factories_have_typed_immutable_binding_returns(self) -> None:
        root = Path(__file__).parents[1] / "nautical_core"
        modules = (
            "acf_api",
            "business_calendar_api",
            "cache_api",
            "expansion_api",
            "hint_builder_api",
            "linting_api",
            "natural_language_api",
            "parser_api",
            "parser_support_api",
            "quarter_api",
            "scheduler_api",
            "time_api",
            "token_api",
        )
        for name in modules:
            with self.subTest(name=name):
                source = (root / f"{name}.py").read_text(encoding="utf-8")
                module = __import__(f"nautical_core.{name}", fromlist=["for_core"])
                annotation = inspect.signature(module.for_core).return_annotation
                self.assertIsNot(annotation, inspect.Signature.empty)
                self.assertIn("ApiBinding", str(annotation))
                self.assertNotIn("SimpleNamespace", source)


if __name__ == "__main__":
    unittest.main()
