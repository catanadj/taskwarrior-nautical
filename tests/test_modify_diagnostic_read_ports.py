"""Explicit-port contracts for modify read and diagnostic effects."""

from __future__ import annotations

from types import SimpleNamespace
import inspect
import unittest


class ModifyDiagnosticReadPortTests(unittest.TestCase):
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
        from nautical_core import modify_diagnostics_effects as diagnostics

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
