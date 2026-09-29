from __future__ import annotations

import unittest
import threading

from nautical_core.lifecycle_read_service import ChainCacheStore, LifecycleReadService
from nautical_core.task_models import TaskObservation
from nautical_core.integration_models import (
    Absent,
    CommandFailureKind,
    FailureEvidence,
    Found,
    TaskCommand,
    Unavailable,
)


class LifecycleReadServiceTests(unittest.TestCase):
    def test_chain_cache_filters_typed_repository_snapshot_in_memory(self) -> None:
        rows = (
            TaskObservation.from_mapping(
                {
                    "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "chainID": "cid",
                    "link": 1,
                    "status": "completed",
                },
                source_query="chain:cid",
            ),
            TaskObservation.from_mapping(
                {
                    "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "chainID": "cid",
                    "link": 2,
                    "status": "pending",
                },
                source_query="chain:cid",
            ),
        )
        calls = []

        class Repository:
            def chain_snapshot(self, chain_id, **_kwargs):
                calls.append(chain_id)
                return Found(rows, "chain:cid")

        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda value: str(value).split(),
            token_matcher=lambda row, token: token == f"status:{row.get('status')}",
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            repository=Repository(),
            max_chain_walk=10,
        )

        selected = service.get_chain_export("cid", extra="status:pending")

        self.assertEqual(calls, ["cid"])
        self.assertEqual([row.get("link") for row in selected or []], [2])

    def test_chain_cache_preserves_repository_unavailability(self) -> None:
        command = TaskCommand(("task", "export"), "test chain read", 1.0)
        evidence = FailureEvidence(
            command,
            CommandFailureKind.INVALID_RESPONSE,
            0,
            1,
            0.0,
            False,
            "malformed JSON",
        )

        class Repository:
            def chain_snapshot(self, _chain_id, **_kwargs):
                return Unavailable("chain:cid", evidence)

        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            repository=Repository(),
            max_chain_walk=10,
        )

        with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
            service.get_chain_export("cid")

    def test_collect_prev_two_prefers_live_statuses_over_deleted(self) -> None:
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            max_chain_walk=10,
        )
        chain_by_link = {
            2: [
                {"uuid": "deleted-2", "status": "deleted", "link": 2},
                {"uuid": "pending-2", "status": "pending", "link": 2},
            ],
            3: [
                {"uuid": "deleted-3", "status": "deleted", "link": 3},
                {"uuid": "completed-3", "status": "completed", "link": 3},
            ],
        }

        result = service.collect_prev_two(
            {"chainID": "cid", "link": 4},
            get_chain_read=lambda _chain: self.fail("provided index should be used"),
            chain_by_link=chain_by_link,
        )

        self.assertIsInstance(result, Found)
        self.assertEqual([row.get("uuid") for row in result.value], ["pending-2", "completed-3"])

    def test_chain_cache_concurrent_reads_and_replacements_keep_typed_rows(self) -> None:
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            max_chain_walk=10,
            cache_store=ChainCacheStore(),
        )
        full_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        errors: list[str] = []
        hits = 0

        def writer(chain_id: str) -> None:
            try:
                for index in range(300):
                    service.replace_chain_cache(
                        chain_id,
                        [{"uuid": full_uuid, "link": 1, "entry": f"2026-01-01T00:00:{index % 60:02d}Z"}],
                    )
            except Exception as exc:
                errors.append(f"writer: {exc}")

        def reader() -> None:
            nonlocal hits
            try:
                for _ in range(600):
                    row, _chain_id = service.lookup_short("aaaaaaaa")
                    if row is not None:
                        if not isinstance(row, TaskObservation):
                            errors.append(f"reader returned {type(row)}")
                        hits += 1
            except Exception as exc:
                errors.append(f"reader: {exc}")

        threads = [
            threading.Thread(target=writer, args=("cid-a",)),
            threading.Thread(target=writer, args=("cid-b",)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertGreater(hits, 0)

    def test_indexes_and_spawned_child_merge_preserve_chain_order_and_links(self) -> None:
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            max_chain_walk=10,
        )
        parent = TaskObservation.from_mapping(
            {"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "link": 1},
            source_query="test-parent",
        )
        successor = TaskObservation.from_mapping(
            {"uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "link": 2},
            source_query="test-successor",
        )
        indexes = service.build_indexes([parent, successor])
        self.assertEqual(indexes.by_link[1], [parent])
        self.assertIs(indexes.by_short["bbbbbbbb"], successor)
        self.assertIs(indexes.by_uuid["aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"], parent)

        child = TaskObservation.from_mapping(
            {"uuid": "cccccccc-cccc-cccc-cccc-cccccccccccc", "link": 3},
            source_query="test-child",
        )
        merged = service.merge_spawned_child(
            [parent, successor],
            parent_task=parent,
            child_task=child,
            child_short="cccccccc",
            short_uuid=lambda value: value[:8],
        )
        self.assertEqual([row.get("link") for row in merged], [1, 2, 3])
        self.assertEqual(merged[0].get("nextLink"), "cccccccc")
        self.assertEqual(merged[2].get("prevLink"), "aaaaaaaa")

    def test_safe_filtered_full_snapshot_avoids_repository_read(self) -> None:
        missing = object()
        rows = [
            TaskObservation.from_mapping(
                {"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "link": 1, "status": "completed"},
                source_query="test-snapshot",
            ),
            TaskObservation.from_mapping(
                {"uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "link": 2, "status": "pending"},
                source_query="test-snapshot",
            ),
        ]
        repository_calls: list[str] = []

        class Repository:
            def chain_snapshot(self, chain_id, **_kwargs):
                repository_calls.append(chain_id)
                return Absent(f"chain:{chain_id}", "unexpected fallback")

        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda extra: [] if not extra else str(extra).split(),
            token_matcher=lambda row, token: token in {
                f"link:{row.get('link')}", f"status:{row.get('status')}"
            },
            read_query_get=lambda kind, key: rows if kind == "chain" and key == ("chain-1", None, None, 0) else missing,
            chain_cache_get=lambda _chain: None,
            repository=Repository(),
            max_chain_walk=10,
            read_query_missing=missing,
        )

        selected = service.get_chain_export(
            "chain-1",
            extra="link:2 status:pending",
            read_query_key=lambda chain_id, since, extra, limit: (chain_id, since, extra, limit),
        )
        self.assertEqual([row.get("link") for row in selected or []], [2])
        self.assertEqual(repository_calls, [])

    def test_chain_cache_store_keeps_rows_indexes_and_chain_identity_together(self) -> None:
        store = ChainCacheStore()
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: None,
            chain_cache_get=lambda _chain: None,
            max_chain_walk=10,
            cache_store=store,
        )
        row = TaskObservation.from_mapping(
            {"uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "link": 1},
            source_query="test-cache",
        )
        indexes = service.replace_chain_cache("chain-1", [row])

        self.assertEqual([item.get("link") for item in service.cached_chain_rows("chain-1") or []], [1])
        self.assertIs(store.indexes, indexes)
        self.assertIs(indexes.by_short["aaaaaaaa"], row)
        self.assertIsNone(service.cached_chain_rows("chain-2"))

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

    def test_authoritative_empty_panel_snapshot_does_not_trigger_full_export(self) -> None:
        missing = object()
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: missing,
            chain_cache_get=lambda _chain: None,
            repository=object(),
            max_chain_walk=500,
            read_query_missing=missing,
        )
        calls = []

        result = service.collect_prev_two(
            {"chainID": "cid", "link": 5},
            get_chain_read=lambda *_args, **_kwargs: calls.append(True),
            panel_chain_by_link={},
            panel_chain_snapshot_loaded=True,
        )

        self.assertIsInstance(result, Absent)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
