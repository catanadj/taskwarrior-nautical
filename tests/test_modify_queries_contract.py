"""Best-effort display contracts for optional chain-root context."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest
from typing import Callable, get_type_hints

from nautical_core.modify_queries import QueryPorts, cached_chain_root_and_age, cached_format_root_and_age, chain_root_and_age
from nautical_core.task_models import TaskPayload


class ModifyQueriesContractTests(unittest.TestCase):
    def test_query_ports_declare_their_consumed_callable_signatures(self) -> None:
        annotations = get_type_hints(QueryPorts)
        self.assertEqual(
            annotations,
            {
                "root_uuid": Callable[[TaskPayload], str],
                "tw_get_cached": Callable[[str], str],
                "dtparse": Callable[[object], datetime | None],
                "tolocal": Callable[[datetime], datetime],
                "cache_get": Callable[[str, object], object],
                "cache_set": Callable[[str, object, object], None],
                "diag_count": Callable[[str], None],
                "diagnostic": Callable[[str], None],
            },
        )

    def test_chain_root_context_failure_uses_display_fallback(self) -> None:
        diagnostics = []
        result = chain_root_and_age(
            {"chainID": "chain-1"},
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            root_uuid_from=lambda _task: "root-1",
            tw_get_cached=lambda _ref: "20260101T090000Z",
            dtparse=lambda _value: datetime(2026, 1, 1, tzinfo=timezone.utc),
            tolocal=lambda _value: (_ for _ in ()).throw(RuntimeError("timezone adapter defect")),
            diagnostic=diagnostics.append,
        )

        self.assertEqual(result, ("—", None))
        self.assertEqual(
            diagnostics,
            ["optional chain-root context failed (RuntimeError)"],
        )

    def test_chain_root_cache_key_failure_does_not_block_context_recalculation(self) -> None:
        root_calls = 0
        diagnostics = []

        def root_uuid(_task: dict[str, object]) -> str:
            nonlocal root_calls
            root_calls += 1
            if root_calls == 1:
                raise RuntimeError("transient cache-key defect")
            return "root-1"

        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        ports = QueryPorts(
            root_uuid=root_uuid,
            tw_get_cached=lambda _ref: "entry",
            dtparse=lambda _value: datetime(2026, 1, 1, tzinfo=timezone.utc),
            tolocal=lambda value: value,
            cache_get=lambda _kind, _key: None,
            cache_set=lambda *_args: None,
            diag_count=lambda _name: None,
            diagnostic=diagnostics.append,
        )

        self.assertEqual(
            cached_chain_root_and_age(ports, {"chainID": "chain-1"}, now),
            ("root-1", 1),
        )
        self.assertEqual(
            diagnostics,
            ["optional chain-root cache key failed (RuntimeError)"],
        )

    def test_format_cache_key_failure_does_not_block_context_rendering(self) -> None:
        root_calls = 0
        diagnostics = []

        def root_uuid(_task: dict[str, object]) -> str:
            nonlocal root_calls
            root_calls += 1
            if root_calls == 1:
                raise RuntimeError("transient cache-key defect")
            return "root-1"

        now = datetime(2026, 1, 2, tzinfo=timezone.utc)
        ports = QueryPorts(
            root_uuid=root_uuid,
            tw_get_cached=lambda _ref: "entry",
            dtparse=lambda _value: now,
            tolocal=lambda value: value,
            cache_get=lambda _kind, _key: None,
            cache_set=lambda *_args: None,
            diag_count=lambda _name: None,
            diagnostic=diagnostics.append,
        )

        self.assertEqual(
            cached_format_root_and_age(ports, {"chainID": "chain-1"}, now),
            "root-1",
        )
        self.assertEqual(
            diagnostics,
            ["optional formatted-root cache key failed (RuntimeError)"],
        )

    def test_broken_diagnostic_sink_does_not_change_display_fallback(self) -> None:
        result = chain_root_and_age(
            {"chainID": "chain-1"},
            datetime(2026, 1, 2, tzinfo=timezone.utc),
            root_uuid_from=lambda _task: "root-1",
            tw_get_cached=lambda _ref: "entry",
            dtparse=lambda _value: datetime(2026, 1, 1, tzinfo=timezone.utc),
            tolocal=lambda _value: (_ for _ in ()).throw(
                RuntimeError("timezone adapter defect")
            ),
            diagnostic=lambda _message: (_ for _ in ()).throw(OSError("stderr closed")),
        )

        self.assertEqual(result, ("—", None))


if __name__ == "__main__":
    unittest.main()
