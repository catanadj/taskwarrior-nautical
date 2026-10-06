from __future__ import annotations

import types
import unittest
from typing import is_typeddict

from nautical_core.core_context import (
    CacheState,
    CoreContext,
    ParserDependencies,
    parser_dependencies,
)
import nautical_core.parser_api as parser_api


class CoreContextTests(unittest.TestCase):
    def test_cache_state_is_mutable_and_separate_from_dependencies(self) -> None:
        state = CacheState(memory={}, max_entries=4, ttl=30.0)
        state.memory["key"] = {"value": 1}
        self.assertEqual(state.memory["key"], {"value": 1})
        self.assertEqual((state.max_entries, state.ttl), (4, 30.0))

    def test_scheduler_and_cache_dependencies_are_immutable_snapshots(self) -> None:
        source = {"clock": object(), "limit": 8}
        from nautical_core.core_context import CacheDependencies, cache_dependencies

        cache = cache_dependencies({"ENABLE_ANCHOR_CACHE": True, **source})
        source["limit"] = 0
        self.assertTrue(cache["ENABLE_ANCHOR_CACHE"])
        with self.assertRaises(TypeError):
            cache["ENABLE_ANCHOR_CACHE"] = False

    def test_scheduler_owner_no_longer_uses_a_generic_dependency_snapshot(self) -> None:
        import nautical_core.core_context as core_context

        self.assertFalse(hasattr(core_context, "SchedulerDependencies"))

    def test_parser_and_cache_dependencies_declare_named_keys(self) -> None:
        from nautical_core.core_context import CacheDependencies

        self.assertTrue(is_typeddict(ParserDependencies))
        self.assertTrue(is_typeddict(CacheDependencies))

    def test_parser_dependencies_are_immutable_snapshots(self) -> None:
        source = {"ANCHOR_PRESETS": {"weekly": "w:mon"}}
        dependencies = parser_dependencies(source)
        source["ANCHOR_PRESETS"] = {}
        self.assertEqual(dependencies["ANCHOR_PRESETS"], {"weekly": "w:mon"})
        with self.assertRaises(TypeError):
            dependencies["ANCHOR_PRESETS"] = {}

    def test_context_owns_one_namespace(self) -> None:
        module = types.SimpleNamespace(_import_sibling=lambda _name: object(), value=1)
        context = CoreContext.from_core(module)
        self.assertEqual(context.require("value"), 1)
        context.namespace["value"] = 2
        self.assertEqual(context.require("value"), 2)

    def test_parser_factory_accepts_explicit_context(self) -> None:
        module = types.SimpleNamespace(_import_sibling=lambda _name: object())
        context = CoreContext.from_core(module, namespace={"_import_sibling": module._import_sibling})
        api = parser_api.for_core(context=context)
        self.assertIsNotNone(api)


if __name__ == "__main__":
    unittest.main()
