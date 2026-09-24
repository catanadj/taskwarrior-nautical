"""Contract tests for immutable, typed core API bindings."""

from __future__ import annotations

import inspect
import importlib
from dataclasses import FrozenInstanceError, is_dataclass
import hashlib
from types import ModuleType
from pathlib import Path
import unittest

from nautical_core.api_bindings import ApiBinding, core_namespace
from nautical_core import compat_api
from nautical_core.runtime_manifest import HOOK_RUNTIME_FILES


class ApiBindingContractTests(unittest.TestCase):
    def test_compatibility_sibling_resolution_is_lazy_and_cached(self) -> None:
        loaded: list[str] = []
        target = ModuleType("owner")
        target.value = 7

        def load(name: str) -> ModuleType:
            loaded.append(name)
            return target

        sibling = compat_api._LazySibling("owner", load)

        self.assertEqual(loaded, [])
        self.assertEqual(sibling.value, 7)
        self.assertEqual(sibling.value, 7)
        self.assertEqual(loaded, ["owner"])

    def test_lazy_api_resolution_keeps_the_registered_facade_wrapper(self) -> None:
        namespace: dict[str, object] = {}
        owner = ModuleType("owner")
        owner.for_core = lambda **_kwargs: ApiBinding.from_kwargs(value=lambda: 7)
        bundle = compat_api._LazyApiBundle(
            "owner",
            ("value",),
            core=ModuleType("facade"),
            namespace=namespace,
            import_sibling=lambda _name: owner,
            prepare=lambda: None,
        )
        compat_api._bind_lazy_api_aliases(bundle, namespace)
        facade_wrapper = namespace["value"]

        self.assertEqual(facade_wrapper(), 7)
        self.assertIs(namespace["value"], facade_wrapper)

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
        self.assertEqual(len(compat_api.PUBLIC_EXPORTS), 130)
        self.assertEqual(len(set(compat_api.PUBLIC_EXPORTS)), 130)
        self.assertEqual(
            hashlib.sha256("\n".join(compat_api.PUBLIC_EXPORTS).encode()).hexdigest(),
            "d246ffa075edc62eca043d324d362443217883880f9473e55e7d3b6e27e0d96d",
        )
        self.assertIn("normalize_task_business_calendar_in_place", compat_api.PUBLIC_EXPORTS)
        self.assertNotIn("normalize_task_business_calendar", compat_api.PUBLIC_EXPORTS)

        import nautical_core as facade

        for name in compat_api.PUBLIC_EXPORTS:
            self.assertTrue(hasattr(facade, name), name)
        wildcard: dict[str, object] = {}
        exec("from nautical_core import *", {}, wildcard)
        self.assertEqual(set(wildcard), set(compat_api.PUBLIC_EXPORTS))
        self.assertTrue(callable(facade.normalize_task_business_calendar))
        self.assertTrue(callable(facade.normalize_task_business_calendar_in_place))
        self.assertEqual(
            inspect.signature(facade.normalize_task_business_calendar),
            inspect.signature(facade.normalize_task_business_calendar_in_place),
        )

    def test_public_owner_registry_covers_surface_and_parser_group(self) -> None:
        owners = compat_api.PUBLIC_OWNER_MODULES
        self.assertEqual(set(owners), set(compat_api.PUBLIC_EXPORTS))
        self.assertEqual(owners["parse_anchor_expr_to_dnf"], "nautical_core.parser_api")
        self.assertEqual(owners["AnchorDNF"], "nautical_core.parsing.parser_models")
        self.assertEqual(owners["OccurrenceSearchExhausted"], "nautical_core.scheduler_models")
        self.assertEqual(owners["effective_config_snapshot"], "nautical_core.core_config")
        self.assertEqual(owners["cache_load"], "nautical_core.cache_api")
        self.assertEqual(owners["business_calendar_for_task"], "nautical_core.business_calendar_api")
        self.assertEqual(owners["business_calendar_displacement_for_date"], "nautical_core.business_calendar")
        self.assertEqual(owners["build_local_datetime"], "nautical_core.time_api")
        self.assertEqual(owners["TaskDict"], "nautical_core.task_models")
        self.assertEqual(owners["render_panel"], "nautical_core.ui")
        self.assertEqual(owners["_build_anchor_atom_dnf"], "nautical_core.parser_api")
        self.assertEqual(owners["_weeks_between"], "nautical_core.scheduler_api")
        self.assertEqual(owners["resolve_task_data_context"], "nautical_core.runtime")
        self.assertEqual(owners["fcntl"], "fcntl")
        self.assertEqual(owners["tempfile"], "tempfile")

    def test_public_surface_categories_cover_exports_and_legacy_alias(self) -> None:
        categories = compat_api.PUBLIC_EXPORT_CATEGORIES
        self.assertEqual(set(categories), set(compat_api.PUBLIC_EXPORTS) | {
            "normalize_task_business_calendar",
        })
        allowed = {"supported_public_api", "installed_runtime", "test_seam", "legacy_compatibility_alias"}
        self.assertTrue(set(categories.values()) <= allowed)
        self.assertEqual(categories["normalize_task_business_calendar"], "legacy_compatibility_alias")
        self.assertEqual(categories["diag"], "installed_runtime")
        self.assertEqual(categories["_build_anchor_atom_dnf"], "test_seam")

    def test_canonical_owner_modules_are_importable(self) -> None:
        owners = set(compat_api.PUBLIC_OWNER_MODULES.values())
        for module_name in sorted(owners):
            with self.subTest(module=module_name):
                self.assertIsNotNone(importlib.import_module(module_name))

    def test_legacy_alias_and_public_wrappers_preserve_callable_contracts(self) -> None:
        import nautical_core as facade

        for name in (
            "parse_anchor_expr_to_dnf",
            "parse_cp_sequence",
            "cache_load",
            "render_panel",
            "resolve_task_data_context",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(facade, name)))
        self.assertEqual(
            inspect.signature(facade.normalize_task_business_calendar),
            inspect.signature(facade.normalize_task_business_calendar_in_place),
        )

    def test_public_facade_exports_only_supported_symbols_with_stable_signatures(self) -> None:
        import nautical_core as facade

        exported = set(facade.__all__)
        for name in (
            "next_after_expr", "SchedulerService", "should_stamp_chain_id",
            "_config_paths", "_read_toml", "_cache_key_for_task_cached",
            "_parse_y_token", "_raise_if_comma_joined_anchors",
            "_validate_year_tokens_in_dnf", "parent", "_import_sibling",
        ):
            self.assertNotIn(name, exported)
        self.assertIn("resolve_task_data_context", exported)
        self.assertIn("render_panel", exported)

        contract = {
            "parse_anchor_expr_to_dnf": ("s",),
            "parse_anchor_expr_to_dnf_cached": ("s",),
            "validate_anchor_expr_strict": ("expr",),
            "parse_cp_duration": ("dur",),
            "parse_cp_sequence": ("cp",),
            "cp_sequence_interval_for_link": ("cp", "link_no", "chain_id"),
            "build_local_datetime": ("d", "hhmm"),
            "to_local": ("dt_utc",),
            "utc_to_local_naive": ("dt_utc",),
            "local_naive_to_utc": ("dt_local_naive",),
            "parse_dt_any": ("s",),
        }
        for name, expected_parameters in contract.items():
            with self.subTest(name=name):
                value = getattr(facade, name, None)
                self.assertTrue(callable(value), name)
                self.assertEqual(tuple(inspect.signature(value).parameters), expected_parameters)
        self.assertEqual(
            facade.parse_anchor_expr_to_dnf("w:mon"),
            facade.parse_anchor_expr_to_dnf_cached("w:mon"),
        )


if __name__ == "__main__":
    unittest.main()
