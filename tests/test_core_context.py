from __future__ import annotations

import types
import unittest

from nautical_core.core_context import CacheState, CoreContext, ParserDependencies
from nautical_core import parser_api


class CoreContextTests(unittest.TestCase):
    def test_cache_state_is_mutable_and_separate_from_dependencies(self) -> None:
        state = CacheState(memory={}, max_entries=4, ttl=30.0)
        state.memory["key"] = {"value": 1}
        self.assertEqual(state.memory["key"], {"value": 1})
        self.assertEqual((state.max_entries, state.ttl), (4, 30.0))

    def test_parser_dependencies_are_immutable_snapshots(self) -> None:
        source = {"value": 1}
        dependencies = ParserDependencies.from_mapping(source)
        source["value"] = 2
        self.assertEqual(dependencies["value"], 1)
        with self.assertRaises(TypeError):
            dependencies.values["value"] = 3

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
