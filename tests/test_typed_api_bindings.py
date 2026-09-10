"""Contract tests for immutable, typed core API bindings."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, is_dataclass
import hashlib
from pathlib import Path
import unittest

from nautical_core.api_bindings import ApiBinding, core_namespace
from nautical_core import compat_api
from nautical_core.runtime_manifest import HOOK_RUNTIME_FILES


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
            ("acf_api", "acf_api.py"),
            ("business_calendar_api", "business_calendar_api.py"),
            ("cache_api", "cache_api.py"),
            ("expansion_api", "expansion_api.py"),
            ("hint_builder_api", "hint_builder_api.py"),
            ("linting_api", "linting_api.py"),
            ("natural_language_api", "natural_language_api.py"),
            ("parser_api", "parser_api.py"),
            ("parsing.parser_support_api", "parsing/parser_support_api.py"),
            ("quarter_api", "quarter_api.py"),
            ("scheduler_api", "scheduler_api.py"),
            ("time_api", "time_api.py"),
            ("token_api", "token_api.py"),
        )
        for module_name, source_name in modules:
            name = module_name.rsplit(".", 1)[-1]
            with self.subTest(name=name):
                source = (root / source_name).read_text(encoding="utf-8")
                module = __import__(f"nautical_core.{module_name}", fromlist=["for_core"])
                annotation = inspect.signature(module.for_core).return_annotation
                self.assertIsNot(annotation, inspect.Signature.empty)
                self.assertIn("ApiBinding", str(annotation))
                self.assertNotIn("SimpleNamespace", source)

    def test_core_namespace_rejects_missing_binding_source_explicitly(self) -> None:
        with self.assertRaisesRegex(TypeError, "time_api.for_core"):
            core_namespace(None, None, None, "time_api")

    def test_runtime_manifest_lists_canonical_parser_modules(self) -> None:
        required = {
            "parsing/parser_atoms.py",
            "parsing/parser_dnf.py",
            "parsing/parser_models.py",
            "parsing/parser_support_api.py",
        }
        for event, files in HOOK_RUNTIME_FILES.items():
            with self.subTest(event=event):
                self.assertTrue(required.issubset(files))

    def test_public_surface_snapshot_classifies_legacy_and_unresolved_names(self) -> None:
        self.assertEqual(len(compat_api.PUBLIC_EXPORTS), 133)
        self.assertEqual(len(set(compat_api.PUBLIC_EXPORTS)), 133)
        self.assertEqual(
            hashlib.sha256("\n".join(compat_api.PUBLIC_EXPORTS).encode()).hexdigest(),
            "37c7e605392942a8577d8e037880a39ded377b90fd86c9c5b9b66c481c0cbb3a",
        )
        self.assertIn("normalize_task_business_calendar_in_place", compat_api.PUBLIC_EXPORTS)
        self.assertNotIn("normalize_task_business_calendar", compat_api.PUBLIC_EXPORTS)

        import nautical_core as facade

        unresolved = {
            name for name in compat_api.LEGACY_UNRESOLVED_EXPORTS
            if not hasattr(facade, name)
        }
        self.assertEqual(unresolved, {"py", "random", "time"})
        for name in compat_api.PUBLIC_EXPORTS:
            if name not in compat_api.LEGACY_UNRESOLVED_EXPORTS:
                self.assertTrue(hasattr(facade, name), name)
        self.assertTrue(callable(facade.normalize_task_business_calendar))
        self.assertTrue(callable(facade.normalize_task_business_calendar_in_place))


if __name__ == "__main__":
    unittest.main()
