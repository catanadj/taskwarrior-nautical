from __future__ import annotations

import unittest

from nautical_core.lifecycle_read_service import ChainCacheStore, LifecycleReadService
from nautical_core.task_models import TaskObservation


class LifecycleReadServiceTests(unittest.TestCase):
    def test_mutation_clear_drops_chain_evidence_and_indexes(self) -> None:
        store = ChainCacheStore()
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if value is not None else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            max_chain_walk=32,
            cache_store=store,
        )
        row = TaskObservation.from_mapping(
            {"uuid": "11111111-1111-4111-8111-111111111111", "link": 1, "status": "pending"},
            source_query="test",
        )
        service.replace_chain_cache("chain-1", [row])
        self.assertIsNotNone(service.cached_chain_rows("chain-1"))
        self.assertIsNotNone(service.lookup_uuid(str(row.get("uuid"))))

        service.clear_cache()

        self.assertIsNone(service.cached_chain_rows("chain-1"))
        self.assertIsNone(service.lookup_uuid(str(row.get("uuid"))))


if __name__ == "__main__":
    unittest.main()
