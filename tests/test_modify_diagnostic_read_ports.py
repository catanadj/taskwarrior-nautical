"""Explicit-port contracts for modify read and diagnostic effects."""

from __future__ import annotations

from types import SimpleNamespace
from collections import abc
import inspect
import unittest
from datetime import datetime, timedelta
from typing import Any, Callable, get_type_hints
from unittest.mock import patch


class ModifyDiagnosticReadPortTests(unittest.TestCase):
    def test_tw_get_command_port_exposes_named_retry_and_execution_inputs(self) -> None:
        from nautical_core.modify_read_effects import TwGetPorts

        callback = get_type_hints(TwGetPorts)["run_task"]
        signature = inspect.signature(callback.__call__)

        self.assertEqual(
            list(signature.parameters),
            ["self", "argv", "env", "input_text", "timeout", "retries", "retry_delay", "use_tempfiles"],
        )
        self.assertEqual(
            signature.parameters["env"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

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

    def test_lateness_statistics_use_the_owner_result_model(self) -> None:
        import nautical_core.modify_analytics as analytics
        from nautical_core.modify_chain_summary import stats_rows
        from nautical_core.modify_analytics import LatenessStats
        from nautical_core.task_models import TaskObservation

        self.assertIs(get_type_hints(analytics.lateness_stats)["return"], LatenessStats)
        self.assertEqual(
            get_type_hints(stats_rows)["lateness_stats"],
            abc.Callable[[list[TaskObservation]], LatenessStats],
        )

    def test_seconds_delta_port_has_concrete_humanizer_contract(self) -> None:
        from nautical_core.modify_diagnostics_effects import (
            SecondsDeltaPort,
            seconds_delta_port_for,
        )

        self.assertEqual(
            get_type_hints(SecondsDeltaPort)["humanize"],
            Callable[[datetime, datetime, bool], str],
        )
        self.assertIsNot(get_type_hints(seconds_delta_port_for)["host"], Any)

    def test_analytics_ports_match_owner_callback_contracts(self) -> None:
        from nautical_core.modify_diagnostics_effects import AnalyticsPorts

        self.assertEqual(
            {name: annotation for name, annotation in get_type_hints(AnalyticsPorts).items()
             if name not in {"core", "service"}},
            {
                "parse_datetime": Callable[[object], datetime | None],
                "format_delta": Callable[[timedelta], str],
                "coerce_int": Callable[[Any, Any], int | None],
                "short_uuid": Callable[[Any], str],
            },
        )
        annotations = get_type_hints(AnalyticsPorts)
        self.assertIsNot(annotations["core"], Any)
        self.assertIsNot(annotations["service"], Any)

    def test_timeline_summary_ports_match_owner_callback_contracts(self) -> None:
        from nautical_core.modify_diagnostics_effects import TimelineSummaryPorts

        annotations = get_type_hints(TimelineSummaryPorts)
        self.assertIsNot(annotations["summary"], Any)
        self.assertEqual(
            {name: annotations[name] for name in annotations if name != "summary"},
            {
                "coerce_int": Callable[[Any, Any], int | None],
                "parse_datetime": Callable[[object], datetime | None],
                "format_local": Callable[[Any], str],
                "format_on_time_delta": Callable[[datetime, datetime], str],
                "short_uuid": Callable[[Any], str],
            },
        )

    def test_span_fields_ports_type_export_and_datetime_callbacks(self) -> None:
        from nautical_core.modify_diagnostics_effects import (
            SpanFieldsPorts,
            SpanSummaryService,
        )
        from nautical_core.modify_chain_summary import SpanHumanDelta
        from nautical_core.task_models import TaskObservation

        annotations = get_type_hints(SpanFieldsPorts)
        self.assertIsNot(annotations["summary"], Any)
        self.assertIs(annotations["human_delta"], SpanHumanDelta)
        self.assertIs(
            get_type_hints(SpanSummaryService.span_fields)["human_delta"],
            SpanHumanDelta,
        )
        self.assertEqual(
            {name: annotations[name] for name in annotations if name != "summary"},
            {
                "export_endpoint": Callable[[str, str], TaskObservation | None],
                "parse_datetime": Callable[[object], datetime | None],
                "human_delta": SpanHumanDelta,
            },
        )

    def test_end_chain_summary_ports_use_renderer_and_service_contracts(self) -> None:
        from nautical_core.modify_chain_summary import ChainSummaryRenderServices
        from nautical_core.modify_diagnostics_effects import EndChainSummaryPorts
        from nautical_core.modify_models import FeedbackPanelCallback

        annotations = get_type_hints(EndChainSummaryPorts)
        renderer_annotations = get_type_hints(ChainSummaryRenderServices)
        self.assertIsNot(annotations["summary"], Any)
        self.assertIs(annotations["services"], ChainSummaryRenderServices)
        self.assertIs(renderer_annotations["panel"], FeedbackPanelCallback)

    def test_diagnostic_chain_adapters_use_observation_and_datetime_types(self) -> None:
        import nautical_core.modify_diagnostics_effects as diagnostics
        from nautical_core.task_models import TaskObservation, TaskPayload

        self.assertEqual(
            get_type_hints(diagnostics.chain_integrity_warnings)["chain"],
            list[TaskObservation],
        )
        self.assertEqual(
            get_type_hints(diagnostics.lateness_stats)["chain"],
            list[TaskObservation],
        )
        self.assertEqual(
            get_type_hints(diagnostics.sort_chain_for_analytics),
            {
                "ports": diagnostics.AnalyticsPorts,
                "chain": list[TaskObservation],
                "return": list[TaskObservation],
            },
        )
        self.assertEqual(
            get_type_hints(diagnostics.span_fields)["return"],
            tuple[datetime | None, datetime | None, str],
        )
        self.assertEqual(get_type_hints(diagnostics.end_chain_summary)["now_utc"], datetime)
        self.assertEqual(get_type_hints(diagnostics.end_chain_summary)["current"], TaskPayload)

    def test_extra_token_port_uses_hook_parser_contract(self) -> None:
        from nautical_core.modify_read_effects import ExtraTokenPort

        self.assertEqual(
            get_type_hints(ExtraTokenPort)["parse"],
            abc.Callable[[str | None], list[str] | None],
        )

    def test_chain_export_ports_share_the_read_effect_owner_contract(self) -> None:
        from nautical_core.modify_diagnostics_effects import ChainExportPorts
        from nautical_core.modify_read_effects import ChainExportPort

        shared_reader = get_type_hints(ChainExportPorts)["service"]
        self.assertIsNot(shared_reader, Any)
        self.assertIs(get_type_hints(ChainExportPort)["service"], shared_reader)

    def test_seed_lookup_ports_use_lifecycle_reader_and_typed_callbacks(self) -> None:
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_read_effects import SeedLookupPorts

        annotations = get_type_hints(SeedLookupPorts)
        self.assertIs(annotations["service"], LifecycleReadService)
        self.assertIsNot(annotations["decode_row"], Any)
        self.assertEqual(annotations["cache_set"], abc.Callable[[str, Any, Any], None])

    def test_previous_chain_ports_use_typed_read_context(self) -> None:
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_read_effects import PreviousChainPorts
        from nautical_core.task_models import TaskObservation

        annotations = get_type_hints(PreviousChainPorts)
        self.assertIs(annotations["service"], LifecycleReadService)
        self.assertEqual(
            annotations["panel_chain_by_link"],
            dict[int, list[TaskObservation]],
        )
        self.assertIs(annotations["panel_chain_snapshot_loaded"], bool)

    def test_tw_get_ports_use_typed_runner_and_cache_callbacks(self) -> None:
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_read_effects import TwGetPorts, TwGetTaskCommand

        annotations = get_type_hints(TwGetPorts)
        self.assertIs(annotations["service"], LifecycleReadService)
        self.assertEqual(annotations["cache_get"], abc.Callable[[str, str], object])
        self.assertEqual(
            annotations["cache_set"], abc.Callable[[str, str, object], None]
        )
        self.assertIs(annotations["run_task"], TwGetTaskCommand)
        self.assertEqual(annotations["command_prefix"], abc.Callable[[], list[str]])
        self.assertEqual(annotations["environment"], abc.Callable[[], dict[str, str]])

    def test_lifecycle_read_capabilities_reuse_owner_types(self) -> None:
        from nautical_core.lifecycle.read_service import (
            ChainCacheStore,
            ChainSnapshotRepository,
            CoerceInt,
            Counter,
            Diagnostic,
            ReadQuery,
            TokenMatcher,
            TokenParser,
        )
        from nautical_core.modify_read_effects import LifecycleReadCapabilities

        self.assertEqual(
            get_type_hints(LifecycleReadCapabilities),
            {
                "coerce_int": CoerceInt,
                "parse_extra_tokens": TokenParser,
                "token_matcher": TokenMatcher,
                "read_query_get": ReadQuery,
                "read_query_missing": object,
                "max_chain_walk": int,
                "diag": Diagnostic,
                "record_stat": Counter,
                "cache_store": ChainCacheStore,
                "repository": ChainSnapshotRepository | None,
            },
        )

    def test_runtime_token_match_uses_owner_task_and_coercion_types(self) -> None:
        from nautical_core.lifecycle.read_service import CoerceInt
        from nautical_core.modify_read_effects import _token_match

        annotations = get_type_hints(_token_match)
        self.assertEqual(annotations["coerce_int"], CoerceInt)
        self.assertIsNot(annotations["task"], Any)
        self.assertEqual(annotations["token"], str)
        self.assertIs(annotations["return"], bool)

    def test_chain_export_adapter_has_no_rejected_environment_argument(self) -> None:
        from nautical_core.modify_read_effects import ChainExportPort, export_chain_required
        from nautical_core.task_models import TaskObservation

        row = TaskObservation.from_mapping(
            {"uuid": "task-1", "chainID": "chain-1", "link": 1},
            source_query="test chain export",
        )

        class Reader:
            def get_chain_export(self, _chain_id: str) -> list[TaskObservation]:
                return [row]

        port = ChainExportPort(Reader())
        self.assertEqual(export_chain_required(port, {"chainID": "chain-1"}), [row])
        self.assertEqual(
            get_type_hints(export_chain_required)["return"], list[TaskObservation]
        )
        with self.assertRaises(TypeError):
            export_chain_required(port, {"chainID": "chain-1"}, object())

    def test_line_preview_ports_use_task_view_formatter_and_markup_contracts(self) -> None:
        from nautical_core.modify_format_effects import LinePreviewPorts
        from nautical_core.modify_models import MarkupStripper, PreviewLineFormatter

        annotations = get_type_hints(LinePreviewPorts)
        self.assertIs(annotations["format_line_preview"], PreviewLineFormatter)
        self.assertIs(annotations["core"], MarkupStripper)
        self.assertEqual(annotations["format_local"], Callable[[Any], str])
        self.assertNotIn(Any, annotations.values())
        preview_annotations = get_type_hints(PreviewLineFormatter.__call__)
        self.assertIs(preview_annotations["core"], MarkupStripper)
        self.assertEqual(preview_annotations["format_local"], abc.Callable[[Any], str])
        self.assertEqual(
            preview_annotations["on_time_delta"], abc.Callable[[Any, Any], str]
        )
        self.assertEqual(
            preview_annotations["human_delta"],
            abc.Callable[[Any, Any, bool], str],
        )

    def test_line_preview_adapters_expose_explicit_temporal_options(self) -> None:
        import inspect
        import nautical_core.modify_format_effects as formatting

        annotations = get_type_hints(formatting.line_preview)
        self.assertEqual(annotations["child_due_utc"], datetime | None)
        self.assertEqual(annotations["now_utc"], datetime)
        self.assertEqual(annotations["return"], str)
        self.assertNotIn(
            inspect.Parameter.VAR_KEYWORD,
            [parameter.kind for parameter in inspect.signature(formatting.line_preview).parameters.values()],
        )
        self.assertEqual(get_type_hints(formatting.on_time_delta)["return"], str)

    def test_native_until_target_validator_has_typed_effects(self) -> None:
        from typing import NoReturn
        from nautical_core.modify_models import PanelCallback
        from nautical_core.modify_validation import (
            validate_native_until_after_target_or_fail,
        )

        annotations = get_type_hints(validate_native_until_after_target_or_fail)
        self.assertEqual(
            annotations["validate_anchor_mode"],
            abc.Callable[
                [object, object, object, object], tuple[bool, str | None]
            ],
        )
        self.assertEqual(
            annotations["safe_parse_datetime"],
            abc.Callable[[object], tuple[datetime | None, str | None]],
        )
        self.assertEqual(
            annotations["validate_after_target"],
            abc.Callable[
                [datetime | None, datetime | None, str], tuple[bool, str | None]
            ],
        )
        self.assertEqual(annotations["format_local"], abc.Callable[[datetime], str])
        self.assertIs(annotations["panel"], PanelCallback)
        self.assertEqual(annotations["fail"], abc.Callable[[str, str], NoReturn])
        self.assertEqual(annotations["abort"], abc.Callable[[int], NoReturn])

    def test_cp_on_modify_uses_typed_parser_contracts(self) -> None:
        from nautical_core.modify_validation import validate_cp_on_modify

        annotations = get_type_hints(validate_cp_on_modify)
        self.assertIs(annotations["chain_max_value"], object)
        self.assertIs(annotations["chain_until_value"], object)
        self.assertEqual(
            annotations["parse_cp_sequence"],
            abc.Callable[[str], list[timedelta] | None],
        )
        self.assertEqual(
            annotations["cp_sequence_parse_error"], abc.Callable[[str], str | None]
        )
        self.assertEqual(
            annotations["parse_chain_max"],
            abc.Callable[[object], tuple[int | None, str | None]],
        )
        self.assertEqual(
            annotations["parse_datetime"],
            abc.Callable[[object], datetime | None],
        )

    def test_completion_validation_services_use_typed_cp_callbacks(self) -> None:
        from typing import NoReturn
        from nautical_core.modify_validation import CompletionValidationServices

        annotations = get_type_hints(CompletionValidationServices)
        self.assertEqual(
            annotations["parse_cp_sequence"],
            abc.Callable[[str], list[timedelta] | None],
        )
        self.assertEqual(
            annotations["validate_cp"],
            abc.Callable[[str, object, object], None],
        )
        self.assertEqual(
            annotations["fail"],
            abc.Callable[[str, str], NoReturn],
        )

    def test_cp_validation_effect_keeps_raw_task_values_at_object_boundary(self) -> None:
        from nautical_core.modify_validation_effects import CPValidationPorts, validate_cp

        annotations = get_type_hints(validate_cp)
        self.assertIs(annotations["ports"], CPValidationPorts)
        self.assertIs(annotations["cp_value"], str)
        self.assertIs(annotations["chain_max_value"], object)
        self.assertIs(annotations["chain_until_value"], object)
        self.assertIs(annotations["return"], type(None))

    def test_validation_port_factories_use_owner_host_protocols(self) -> None:
        from nautical_core.modify_validation_effects import (
            AnchorValidationHost,
            ChainLimitHost,
            NativeUntilHost,
            NativeUntilSlotHost,
            OmitValidationHost,
            anchor_validation_ports_for,
            chain_limit_ports_for,
            native_until_ports_for,
            native_until_slot_ports_for,
            omit_validation_ports_for,
        )

        self.assertIs(
            get_type_hints(anchor_validation_ports_for)["host"], AnchorValidationHost
        )
        self.assertIs(
            get_type_hints(omit_validation_ports_for)["host"], OmitValidationHost
        )
        self.assertIs(
            get_type_hints(chain_limit_ports_for)["host"], ChainLimitHost
        )
        self.assertIs(
            get_type_hints(native_until_ports_for)["host"], NativeUntilHost
        )
        self.assertIs(
            get_type_hints(native_until_slot_ports_for)["host"], NativeUntilSlotHost
        )

    def test_datetime_effect_factory_uses_core_capability_host(self) -> None:
        from nautical_core.modify_datetime_effects import (
            DatetimeEffectsHost,
            datetime_effect_ports_for,
        )

        self.assertIs(
            get_type_hints(datetime_effect_ports_for)["host"], DatetimeEffectsHost
        )

    def test_modify_presentation_factories_use_owner_host_protocols(self) -> None:
        from nautical_core.modify_format_effects import (
            LinePreviewHost,
            line_preview_ports_for,
        )
        from nautical_core.modify_presentation_effects import (
            ChainStyleHost,
            LifecycleResultHost,
            chain_style_ports_for,
            lifecycle_result_port_for,
        )

        self.assertIs(
            get_type_hints(chain_style_ports_for)["host"], ChainStyleHost
        )
        self.assertIs(
            get_type_hints(lifecycle_result_port_for)["host"], LifecycleResultHost
        )
        self.assertIs(
            get_type_hints(line_preview_ports_for)["host"], LinePreviewHost
        )

    def test_command_ports_factory_uses_owner_host_protocol(self) -> None:
        from nautical_core.modify_command_effects import CommandHost, command_ports_for

        self.assertIs(get_type_hints(command_ports_for)["host"], CommandHost)

    def test_tw_get_ports_factory_uses_owner_host_protocol(self) -> None:
        from nautical_core.modify_read_effects import TwGetHost, tw_get_ports_for

        self.assertIs(get_type_hints(tw_get_ports_for)["host"], TwGetHost)

    def test_chain_export_factory_uses_narrow_host_protocol(self) -> None:
        from nautical_core.modify_diagnostics_effects import (
            ChainExportHost,
            chain_export_ports_for,
        )

        self.assertIs(
            get_type_hints(chain_export_ports_for)["host"],
            ChainExportHost,
        )

    def test_timeline_summary_factory_uses_narrow_host_protocol(self) -> None:
        from nautical_core.modify_diagnostics_effects import (
            TimelineSummaryHost,
            timeline_summary_ports_for,
        )

        self.assertIs(
            get_type_hints(timeline_summary_ports_for)["host"],
            TimelineSummaryHost,
        )

    def test_span_fields_factory_uses_narrow_host_protocol(self) -> None:
        from nautical_core.modify_diagnostics_effects import (
            SpanFieldsHost,
            span_fields_ports_for,
        )

        self.assertIs(
            get_type_hints(span_fields_ports_for)["host"],
            SpanFieldsHost,
        )

    def test_cp_on_modify_reports_non_string_chain_until_as_invalid(self) -> None:
        from nautical_core.modify_validation import validate_cp_on_modify

        with self.assertRaisesRegex(ValueError, "Invalid chainUntil '123'"):
            validate_cp_on_modify(
                "P1D",
                None,
                123,
                parse_cp_sequence=lambda _value: [timedelta(days=1)],
                cp_sequence_parse_error=lambda _value: None,
                parse_chain_max=lambda _value: (None, None),
                parse_datetime=lambda _value: None,
            )

    def test_chain_limit_validator_uses_typed_effects(self) -> None:
        from nautical_core.modify_validation import validate_chain_limits_on_modify

        annotations = get_type_hints(validate_chain_limits_on_modify)
        self.assertEqual(
            annotations["parse_chain_max"],
            abc.Callable[[object], tuple[int | None, str | None]],
        )
        self.assertEqual(
            annotations["parse_datetime"],
            abc.Callable[[object], datetime | None],
        )
        self.assertEqual(
            annotations["validate_until_not_past"],
            abc.Callable[[datetime, datetime], tuple[bool, str | None]],
        )
        self.assertEqual(annotations["now_utc"], abc.Callable[[], datetime])
        self.assertEqual(annotations["fail"], abc.Callable[[str, str], object])

    def test_native_until_slot_validator_uses_owner_contracts(self) -> None:
        from typing import NoReturn
        from nautical_core.modify_validation import (
            CollectAnchorTimeSlots,
            NormalizeTimeSlots,
            ValidationPanel,
            ValidateCalendarSlots,
            validate_native_until_anchor_slots_or_fail,
        )
        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.task_models import TaskPayload

        annotations = get_type_hints(validate_native_until_anchor_slots_or_fail)
        self.assertEqual(
            annotations["safe_parse_datetime"],
            abc.Callable[[object], tuple[datetime | None, str | None]],
        )
        self.assertEqual(annotations["validate_anchor"], abc.Callable[[str], object])
        self.assertIs(annotations["collect_time_slots"], CollectAnchorTimeSlots)
        self.assertIs(annotations["validate_time_slots"], ValidateCalendarSlots)
        self.assertIs(annotations["normalize_time_slots"], NormalizeTimeSlots)
        self.assertEqual(annotations["anchor_file_dir"], str)
        self.assertEqual(
            annotations["recurrence_context"],
            abc.Callable[[TaskPayload], RecurrenceContext],
        )
        self.assertEqual(annotations["to_local"], abc.Callable[[datetime], datetime])
        self.assertEqual(annotations["format_local"], abc.Callable[[datetime], str])
        self.assertEqual(
            annotations["astronomy_is_error"], abc.Callable[[BaseException], bool]
        )
        self.assertEqual(
            annotations["astronomy_error_message"],
            abc.Callable[[BaseException], str],
        )
        self.assertIs(annotations["panel"], ValidationPanel)
        self.assertEqual(annotations["abort"], abc.Callable[[int], NoReturn])

    def test_omit_ports_use_date_and_canonical_omit_state_contracts(self) -> None:
        import nautical_core.anchor_omit as anchor_omit
        import nautical_core.modify_anchor_effects as effects

        annotations = get_type_hints(
            effects.OmitPorts, localns={"OmitState": anchor_omit.OmitState}
        )
        self.assertNotIn(Any, annotations.values())
        self.assertIsNot(
            get_type_hints(
                effects.omit_dnf_from_parent,
                localns={"OmitState": anchor_omit.OmitState},
            )["return"],
            Any,
        )

    def test_chain_export_ports_use_read_service_and_coercion_contracts(self) -> None:
        from nautical_core.modify_diagnostics_effects import ChainExportPorts

        annotations = get_type_hints(ChainExportPorts)
        self.assertIsNot(annotations["service"], Any)
        self.assertEqual(
            annotations["coerce_int"], Callable[[Any, Any], int | None]
        )

    def test_summary_export_does_not_mask_analytics_sorting_defects(self) -> None:
        import nautical_core.modify_analytics as analytics
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
            "modify_analytics": analytics,
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
        from nautical_core.task_models import TaskObservation

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
            collect_prev_two(
                ports,
                TaskObservation.from_mapping(
                    {"chainID": "cid", "link": 3},
                    source_query="test predecessor read",
                ),
            )

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
            def cp_sequence_interval_for_link(self, *_args: object) -> timedelta | None:
                return None

        class Service:
            def chain_health_advice(self, *_args, **kwargs) -> str:
                calls.append(kwargs)
                return "advice"

        ports = AnalyticsPorts(
            core=Core(),
            service=Service(),
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
