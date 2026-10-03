"""Explicit-port contracts for modify read and diagnostic effects."""

from __future__ import annotations

from types import SimpleNamespace
import inspect
import unittest
from datetime import datetime, timedelta
from typing import Any
from typing import Callable, get_type_hints
from unittest.mock import patch


class ModifyDiagnosticReadPortTests(unittest.TestCase):
    def test_diagnostic_datetime_parser_port_has_typed_owner_contract(self) -> None:
        from nautical_core.modify_diagnostics_effects import (
            DatetimeValuePort,
            _parse_datetime_value,
        )
        from nautical_core.task_datetime import TaskDatetimeParser

        self.assertEqual(
            get_type_hints(DatetimeValuePort)["parser"], TaskDatetimeParser
        )
        self.assertEqual(
            get_type_hints(_parse_datetime_value)["return"], datetime | None
        )

    def test_seconds_delta_port_has_concrete_humanizer_contract(self) -> None:
        from nautical_core.modify_diagnostics_effects import SecondsDeltaPort

        self.assertEqual(
            get_type_hints(SecondsDeltaPort)["humanize"],
            Callable[[datetime, datetime, bool], str],
        )

    def test_analytics_ports_match_owner_callback_contracts(self) -> None:
        from nautical_core.modify_diagnostics_effects import AnalyticsPorts

        self.assertEqual(
            get_type_hints(AnalyticsPorts),
            {
                "core": Any,
                "parse_datetime": Callable[[object], datetime | None],
                "format_delta": Callable[[timedelta], str],
                "coerce_int": Callable[[Any, Any], int | None],
                "short_uuid": Callable[[Any], str],
            },
        )

    def test_timeline_summary_ports_match_owner_callback_contracts(self) -> None:
        from nautical_core.modify_diagnostics_effects import TimelineSummaryPorts

        self.assertEqual(
            get_type_hints(TimelineSummaryPorts),
            {
                "summary": Any,
                "coerce_int": Callable[[Any, Any], int | None],
                "parse_datetime": Callable[[object], datetime | None],
                "format_local": Callable[[Any], str],
                "format_on_time_delta": Callable[[datetime, datetime, int], str],
                "short_uuid": Callable[[Any], str],
            },
        )

    def test_span_fields_ports_type_export_and_datetime_callbacks(self) -> None:
        from nautical_core.modify_diagnostics_effects import SpanFieldsPorts
        from nautical_core.task_models import TaskObservation

        self.assertEqual(
            get_type_hints(SpanFieldsPorts),
            {
                "summary": Any,
                "export_endpoint": Callable[[str, str], TaskObservation | None],
                "parse_datetime": Callable[[object], datetime | None],
                "human_delta": Any,
            },
        )

    def test_chain_export_ports_use_read_service_and_coercion_contracts(self) -> None:
        from nautical_core.modify_diagnostics_effects import ChainExportPorts

        annotations = get_type_hints(ChainExportPorts)
        self.assertIsNot(annotations["service"], Any)
        self.assertEqual(
            annotations["coerce_int"], Callable[[Any, Any], int | None]
        )

    def test_summary_export_does_not_mask_analytics_sorting_defects(self) -> None:
        import nautical_core.modify_diagnostics_effects as diagnostics
        from nautical_core.task_models import TaskObservation

        core = SimpleNamespace(
            coerce_int=lambda value, default: int(value) if value is not None else default,
            humanize_delta=lambda *_args, **_kwargs: "0s",
            anchor_preset_display=lambda *_args: None,
            describe_anchor_dnf=lambda *_args: None,
            fmt_dt_local=str,
            short_uuid=lambda value: str(value)[:8],
        )
        summary = SimpleNamespace(
            ChainSummaryRenderServices=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        modules = {
            "modify_chain_summary": summary,
            "modify_value_effects": SimpleNamespace(format_delta=lambda *_args: "0s"),
            "modify_format_effects": SimpleNamespace(
                HumanDeltaPort=lambda *_args: object(),
                human_delta=lambda *_args, **_kwargs: "0s",
                on_time_delta=lambda *_args, **_kwargs: "on time",
            ),
            "modify_read_effects": SimpleNamespace(
                ChainExportPort=lambda *_args: object(),
                export_chain_required=lambda *_args: [{"uuid": "other-task", "link": 1}],
            ),
            "modify_composition": SimpleNamespace(lifecycle_read_service_for=lambda _host: object()),
            "task_models": SimpleNamespace(TaskObservation=TaskObservation),
            "modify_task_fields": SimpleNamespace(root_uuid=lambda task: task.get("uuid")),
            "modify_queries": SimpleNamespace(
                query_ports_for=lambda _host: object(),
                cached_format_root_and_age=lambda *_args: "root",
            ),
            "modify_feedback": SimpleNamespace(format_chain_summary_rows=lambda rows: rows),
            "modify_ui_effects": SimpleNamespace(ui_ports_for=lambda _host: object(), panel=lambda *_args, **_kwargs: None),
        }
        host = SimpleNamespace(
            core=core,
            _module=lambda name: modules[name],
            _TASK_DATETIME_PARSER=SimpleNamespace(parse=lambda _value: (None, None)),
            _fmtlocal=str,
            _MAX_CHAIN_WALK=10,
            _validate_anchor_expr_cached=lambda *_args: None,
            _diag=lambda _message: None,
        )
        ports = diagnostics.end_chain_summary_ports_for(host)

        with (
            patch.object(diagnostics, "sort_chain_for_analytics", side_effect=RuntimeError("analytics sort defect")),
            self.assertRaisesRegex(RuntimeError, "analytics sort defect"),
        ):
            ports.services.export_sorted_chain("chain-1", {"uuid": "current-task"})

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
        from nautical_core.lifecycle.read_service import LifecycleReadService
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

    def test_tw_get_cached_does_not_hide_read_service_defects(self) -> None:
        from nautical_core.modify_read_effects import TwGetPorts, tw_get_cached

        class BrokenReadService:
            def lookup_short(self, _short: str) -> tuple[None, str]:
                raise RuntimeError("lookup service defect")

        ports = TwGetPorts(
            service=BrokenReadService(),
            cache_get=lambda _scope, _ref: None,
            cache_set=lambda *_args: None,
            count=lambda _name: None,
            diagnostic=lambda _message: None,
            run_task=lambda *_args, **_kwargs: None,
            command_prefix=lambda: ["task"],
            environment=lambda: {},
        )

        with self.assertRaisesRegex(RuntimeError, "lookup service defect"):
            tw_get_cached(ports, "abc.entry")

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

    def test_chain_health_advice_uses_default_completion_tolerance(self) -> None:
        from nautical_core.modify_diagnostics_effects import AnalyticsPorts, chain_health_advice

        calls: list[dict[str, object]] = []

        class Core:
            def _import_sibling(self, _name: str) -> SimpleNamespace:
                return SimpleNamespace(
                    chain_health_advice=lambda *_args, **kwargs: calls.append(kwargs) or "advice"
                )

        ports = AnalyticsPorts(
            core=Core(),
            parse_datetime=lambda _value: None,
            format_delta=str,
            coerce_int=lambda _value, default: default,
            short_uuid=str,
        )

        self.assertEqual(chain_health_advice(ports, [], "P1D", {}, style="rich"), "advice")
        self.assertEqual(calls, [{"core": ports.core, "parse_datetime": ports.parse_datetime,
                                  "format_delta": ports.format_delta, "coerce_int": ports.coerce_int,
                                  "tol_secs": 60, "style": "rich"}])


if __name__ == "__main__":
    unittest.main()
