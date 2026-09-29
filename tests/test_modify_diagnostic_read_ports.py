"""Explicit-port contracts for modify read and diagnostic effects."""

from __future__ import annotations

from types import SimpleNamespace
import inspect
import unittest


class ModifyDiagnosticReadPortTests(unittest.TestCase):
    def test_parse_extra_tokens_rejects_shell_like_and_option_tokens(self) -> None:
        from nautical_core.hook_support import parse_extra_tokens as parse_task_filters
        from nautical_core.modify_read_effects import ExtraTokenPort, parse_extra_tokens

        port = ExtraTokenPort(parse_task_filters)
        for value in ("status:pending; rm -rf /", "status:pending -rc.hooks=on"):
            with self.subTest(value=value):
                self.assertIsNone(parse_extra_tokens(port, value))

    def test_collect_prev_two_preserves_repository_failure(self) -> None:
        from nautical_core.integration_models import (
            CommandFailureKind,
            FailureEvidence,
            TaskCommand,
            Unavailable,
        )
        from nautical_core.lifecycle_read_service import LifecycleReadService
        from nautical_core.modify_read_effects import PreviousChainPorts, collect_prev_two

        command = TaskCommand(("task", "export"), "test predecessor read", 1.0)
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

        missing = object()
        service = LifecycleReadService(
            coerce_int=lambda value, default: int(value) if str(value).isdigit() else default,
            parse_extra_tokens=lambda _value: [],
            token_matcher=lambda _row, _token: True,
            read_query_get=lambda _kind, _key: missing,
            chain_cache_get=lambda _chain: None,
            repository=Repository(),
            max_chain_walk=10,
            read_query_missing=missing,
        )
        ports = PreviousChainPorts(service, {}, False)

        with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
            collect_prev_two(ports, {"chainID": "cid", "link": 3})

    def test_tw_get_cached_uses_explicit_ports(self) -> None:
        from nautical_core.modify_read_effects import TwGetPorts, tw_get_cached

        counts: list[str] = []
        commands: list[list[str]] = []
        ports = TwGetPorts(
            service=SimpleNamespace(lookup_short=lambda _short: (None, "")),
            cache_get=lambda _scope, _ref: None,
            cache_set=lambda *_args: None,
            count=counts.append,
            diagnostic=lambda _message: None,
            run_task=lambda argv, **_kwargs: commands.append(argv) or SimpleNamespace(ok=True, stdout="value\n"),
            command_prefix=lambda: ["task"],
            environment=lambda: {},
        )

        self.assertEqual(tw_get_cached(ports, "abc.status"), "value")
        self.assertEqual(counts, ["tw_get_cache_misses"])
        self.assertEqual(commands[0][-2:], ["_get", "abc.status"])

    def test_diagnostic_operations_have_host_free_signatures(self) -> None:
        import nautical_core.modify_diagnostics_effects as diagnostics

        for name in (
            "lateness_stats",
            "sort_chain_for_analytics",
            "export_chain_endpoint",
            "last_n_timeline",
            "span_fields",
            "format_seconds_delta",
            "end_chain_summary",
        ):
            first = next(iter(inspect.signature(getattr(diagnostics, name)).parameters.values()))
            self.assertIn(first.name, {"port", "ports"}, name)


if __name__ == "__main__":
    unittest.main()
