from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nautical_core.hooks import modify_impl


class ModifyRuntimeCacheContractTests(unittest.TestCase):
    def test_query_context_get_propagates_runtime_state_failure(self) -> None:
        with patch.object(
            modify_impl,
            "_modify_runtime_state",
            side_effect=RuntimeError("query context unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "query context unavailable"):
                modify_impl._query_ctx_get("tw_get", "task-1")

    def test_query_context_set_propagates_runtime_state_failure(self) -> None:
        with patch.object(
            modify_impl,
            "_modify_runtime_state",
            side_effect=RuntimeError("query context unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "query context unavailable"):
                modify_impl._query_ctx_set("tw_get", "task-1", "payload")

    def test_read_query_get_propagates_unexpected_cached_value_failure(self) -> None:
        class BrokenCachedValue:
            def __deepcopy__(self, _memo: dict[int, object]) -> object:
                raise RuntimeError("cached value defect")

        state = SimpleNamespace(
            query_ctx={"read_query": {("chain", "chain-1"): BrokenCachedValue()}},
            diag_stats={},
        )
        with patch.object(modify_impl, "_MODIFY_RUNTIME_STATE", state):
            with self.assertRaisesRegex(RuntimeError, "cached value defect"):
                modify_impl._read_query_get("chain", "chain-1")

    def test_read_query_get_treats_expected_copy_error_as_cache_miss(self) -> None:
        class UncopyableCachedValue:
            def __deepcopy__(self, _memo: dict[int, object]) -> object:
                raise TypeError("value is not copyable")

        state = SimpleNamespace(
            query_ctx={"read_query": {("chain", "chain-1"): UncopyableCachedValue()}},
            diag_stats={},
        )
        with patch.object(modify_impl, "_MODIFY_RUNTIME_STATE", state):
            result = modify_impl._read_query_get("chain", "chain-1")

        self.assertIs(result, modify_impl._READ_QUERY_MISSING)
        self.assertEqual(state.diag_stats["read_query_cache_misses"], 1)


if __name__ == "__main__":
    unittest.main()
