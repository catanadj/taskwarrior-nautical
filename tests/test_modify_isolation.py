"""Characterization tests for the on-modify effect boundary."""

from __future__ import annotations

import importlib
import sys
import unittest
from datetime import date, datetime
from typing import Any, Callable, get_args, get_origin, get_type_hints


class ModifyIsolationTests(unittest.TestCase):
    def test_cp_validation_ports_have_specific_callable_contracts(self) -> None:
        from nautical_core.modify_validation_effects import CPValidationPorts

        annotations = get_type_hints(CPValidationPorts)
        self.assertEqual(
            set(annotations),
            {
                "validate",
                "parse_cp_sequence",
                "cp_sequence_error",
                "parse_chain_max",
                "parse_datetime",
            },
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_chain_limit_ports_have_specific_pipeline_and_callback_contracts(self) -> None:
        from nautical_core.modify_validation_effects import ChainLimitPorts

        annotations = get_type_hints(ChainLimitPorts)
        self.assertEqual(
            set(annotations),
            {
                "pipeline",
                "validate_limits",
                "parse_cp_sequence",
                "cp_sequence_error",
                "parse_chain_max",
                "parse_datetime",
                "validate_until_not_past",
                "now_utc",
                "fail",
            },
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_shared_validation_ports_have_specific_contracts(self) -> None:
        from nautical_core.modify_validation_effects import SharedValidationPorts

        annotations = get_type_hints(SharedValidationPorts)
        self.assertEqual(
            set(annotations),
            {"pipeline", "parse_anchor", "validate_anchor", "validate_omit"},
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_native_until_ports_have_specific_callback_contracts(self) -> None:
        from nautical_core.modify_validation_effects import NativeUntilPorts

        annotations = get_type_hints(NativeUntilPorts)
        self.assertEqual(
            set(annotations),
            {
                "validate",
                "validate_anchor_mode",
                "parse_datetime",
                "validate_after_target",
                "format_local",
                "panel",
                "fail",
                "abort",
            },
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_anchor_validation_ports_have_specific_callback_contracts(self) -> None:
        from nautical_core.modify_validation_effects import AnchorValidationPorts

        annotations = get_type_hints(AnchorValidationPorts)
        self.assertEqual(
            set(annotations),
            {
                "lint",
                "validate_strict",
                "panel",
                "is_astronomy_error",
                "astronomy_error_message",
                "fail",
            },
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_native_until_slot_ports_have_specific_callback_contracts(self) -> None:
        from nautical_core.modify_validation_effects import NativeUntilSlotPorts

        annotations = get_type_hints(NativeUntilSlotPorts)
        self.assertEqual(
            set(annotations),
            {
                "validate",
                "parse_datetime",
                "validate_anchor",
                "collect_time_slots",
                "validate_time_slots",
                "normalize_time_slots",
                "anchor_file_dir",
                "recurrence_context",
                "to_local",
                "format_local",
                "astronomy_is_error",
                "astronomy_error_message",
                "panel",
                "abort",
            },
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_omit_validation_ports_have_specific_callback_contracts(self) -> None:
        from nautical_core.modify_validation_effects import OmitValidationPorts

        annotations = get_type_hints(OmitValidationPorts)
        self.assertEqual(
            set(annotations),
            {
                "pipeline",
                "parse_anchor",
                "validate_anchor",
                "validate_omit",
                "validate_files",
                "load_anchor_file",
                "load_omit_file",
                "fail",
            },
        )
        for field_name, annotation in annotations.items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, Any)

    def test_helper_ports_do_not_use_generic_callback_protocol(self) -> None:
        from nautical_core.callback_ports import CallbackPort
        from nautical_core.modify_validation_effects import (
            AnchorModePorts,
            DurationPorts,
            UntilPorts,
        )

        for port, fields in (
            (DurationPorts, ("format_local",)),
            (UntilPorts, ("minute_delta", "compare", "humanize")),
            (AnchorModePorts, ("panel",)),
        ):
            annotations = get_type_hints(port)
            for field_name in fields:
                with self.subTest(port=port.__name__, field=field_name):
                    self.assertIsNot(annotations[field_name], CallbackPort)

    def test_datetime_effect_port_has_concrete_comparison_contract(self) -> None:
        from nautical_core.modify_value_effects import DatetimePorts, compare_datetimes

        self.assertEqual(
            get_type_hints(DatetimePorts)["compare"],
            Callable[[datetime, datetime], int],
        )
        self.assertEqual(
            get_type_hints(compare_datetimes),
            {"ports": DatetimePorts, "left": datetime, "right": datetime, "return": int},
        )

    def test_time_slot_effect_port_has_concrete_normalization_contract(self) -> None:
        from nautical_core.modify_time_effects import (
            TimeSlotPorts,
            normalize_hhmm_list,
            time_slot_ports_for,
        )

        self.assertEqual(
            get_type_hints(TimeSlotPorts)["resolve_time_slots"],
            Callable[[object, date | None], list[tuple[int, int]]],
        )
        self.assertEqual(
            get_type_hints(normalize_hhmm_list),
            {
                "ports": TimeSlotPorts,
                "value": object,
                "target_date": date | None,
                "return": list[tuple[int, int]],
            },
        )
        self.assertIsNot(get_type_hints(time_slot_ports_for)["host"], Any)

    def test_modify_ui_ports_use_typed_process_boundary_contracts(self) -> None:
        from nautical_core.callback_ports import CallbackPort
        from nautical_core.modify_ui_effects import UIEffectsPorts

        for field_name, annotation in get_type_hints(UIEffectsPorts).items():
            with self.subTest(field=field_name):
                self.assertIsNot(annotation, CallbackPort)

    def test_completion_preflight_ports_use_shared_typed_contracts(self) -> None:
        from nautical_core.modify_completion_effects import CompletionPreflightPorts
        from nautical_core.modify_completion_preflight import completion_existing_next_or_fail
        from nautical_core.modify_models import (
            CoerceIntCallback,
            EndChainSummaryCallback,
            ExistingNextLookupCallback,
            PanelCallback,
            PrintTaskCallback,
        )

        annotations = get_type_hints(CompletionPreflightPorts)
        expected = {
            "coerce_int": CoerceIntCallback,
            "short_uuid": Callable[[str | None], str],
            "panel": PanelCallback,
            "print_task": PrintTaskCallback,
            "end_chain_summary": EndChainSummaryCallback,
            "existing_next_lookup": ExistingNextLookupCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertEqual(annotations[name], contract)
        self.assertEqual(
            get_type_hints(completion_existing_next_or_fail)["existing_next_lookup"],
            ExistingNextLookupCallback,
        )

    def test_completion_caps_parser_uses_shared_datetime_contract(self) -> None:
        from nautical_core.modify_completion_effects import CompletionCapsPorts
        from nautical_core.modify_models import (
            CapFromUntilAnchorCallback,
            CapFromUntilCpCallback,
            CoerceIntCallback,
            DatetimeParserCallback,
            EstimateAnchorFinalCallback,
            EstimateCpFinalCallback,
        )

        annotations = get_type_hints(CompletionCapsPorts)
        expected = {
            "coerce_int": CoerceIntCallback,
            "parse_datetime": DatetimeParserCallback,
            "estimate_cp": EstimateCpFinalCallback,
            "estimate_anchor": EstimateAnchorFinalCallback,
            "cap_cp": CapFromUntilCpCallback,
            "cap_anchor": CapFromUntilAnchorCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertIs(annotations[name], contract)

    def test_completion_compute_ports_reuse_model_callback_contracts(self) -> None:
        from nautical_core.modify_completion_effects import CompletionComputePorts
        from nautical_core.modify_models import (
            CompletionCapGuardCallback,
            CompletionCapsCallback,
            CompletionChildDueCallback,
            CompletionChildRequiredCallback,
            CompletionComputeServices,
            CompletionDurationWarningCallback,
            CompletionUntilCallback,
            CompletionUntilGuardCallback,
        )

        annotations = get_type_hints(CompletionComputePorts)
        expected = {
            "services_type": type[CompletionComputeServices],
            "compute_child_due": CompletionChildDueCallback,
            "until_or_fail": CompletionUntilCallback,
            "until_guard_or_stop": CompletionUntilGuardCallback,
            "require_child_due_or_fail": CompletionChildRequiredCallback,
            "warn_unreasonable_duration": CompletionDurationWarningCallback,
            "caps": CompletionCapsCallback,
            "cap_guard_or_stop": CompletionCapGuardCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertEqual(annotations[name], contract)

    def test_completion_spawn_ports_reuse_model_callback_contracts(self) -> None:
        from nautical_core.modify_completion_effects import CompletionSpawnPorts
        from nautical_core.modify_models import (
            BuildChildDraftCallback,
            CompletionSpawnServices,
            DiagnosticCallback,
            PanelCallback,
            PrintTaskCallback,
            SpawnChildCallback,
        )

        annotations = get_type_hints(CompletionSpawnPorts)
        expected = {
            "services_type": type[CompletionSpawnServices],
            "build_child_draft": BuildChildDraftCallback,
            "spawn_child_atomic": SpawnChildCallback,
            "panel": PanelCallback,
            "print_task": PrintTaskCallback,
            "diagnostic": DiagnosticCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertEqual(annotations[name], contract)

    def test_completion_snapshot_ports_use_repository_and_model_contracts(self) -> None:
        from nautical_core.lifecycle.read_service import ChainSnapshotRepository
        from nautical_core.modify_completion_effects import (
            CompletionPreflightContextPorts,
            SnapshotPorts,
        )
        from nautical_core.modify_models import CompletionChainSnapshot, CompletionPreflightServices
        snapshot_annotations = get_type_hints(SnapshotPorts)
        self.assertEqual(
            snapshot_annotations,
            {
                "repository": ChainSnapshotRepository,
                "mode": Callable[[], str],
                "snapshot_type": type[CompletionChainSnapshot],
            },
        )
        context_annotations = get_type_hints(CompletionPreflightContextPorts)
        self.assertEqual(
            context_annotations["services_type"], type[CompletionPreflightServices]
        )
        self.assertEqual(
            context_annotations["snapshot_type"], type[CompletionChainSnapshot]
        )

    def test_child_due_ports_use_generation_and_task_contracts(self) -> None:
        from nautical_core.chain_generation import ChainGenerationService
        from nautical_core.modify_completion_effects import ChildDuePorts, TaskRowDecoder
        from nautical_core.task_models import NauticalTask

        annotations = get_type_hints(ChildDuePorts)
        self.assertIs(annotations["generation"], ChainGenerationService)
        self.assertIs(annotations["decode_task"], TaskRowDecoder)
        self.assertEqual(annotations["task_type"], type[NauticalTask])

    def test_completion_lifecycle_plan_ports_do_not_carry_module_bags(self) -> None:
        from nautical_core.chain_generation import ChainGenerationService
        from nautical_core.modify_completion_effects import CompletionLifecyclePlanPorts
        from nautical_core.modify_models import (
            DiagnosticCallback,
            EndChainSummaryCallback,
            InvalidRelativeCarryReasonCallback,
            PanelCallback,
            PrintTaskCallback,
        )

        annotations = get_type_hints(CompletionLifecyclePlanPorts)
        self.assertEqual(
            set(annotations),
            {
                "generation",
                "scheduler_fingerprint",
                "compare_datetimes",
                "invalid_relative_carry_reason",
                "end_chain_summary",
                "ensure_terminal_chain_off",
                "panel",
                "print_task",
                "diagnostic",
            },
        )
        self.assertIs(annotations["generation"], ChainGenerationService)
        self.assertIs(annotations["invalid_relative_carry_reason"], InvalidRelativeCarryReasonCallback)
        self.assertIs(annotations["end_chain_summary"], EndChainSummaryCallback)
        self.assertIs(annotations["panel"], PanelCallback)
        self.assertIs(annotations["print_task"], PrintTaskCallback)
        self.assertIs(annotations["diagnostic"], DiagnosticCallback)

    def test_completion_preflight_service_has_explicit_arguments(self) -> None:
        import inspect

        from nautical_core.modify_completion_effects import CompletionPreflightService
        from nautical_core.modify_models import (
            CoerceIntCallback,
            CompletionPreflightContext,
            CompletionPreflightServices,
            EndChainSummaryCallback,
            ExistingNextLookupCallback,
            PanelCallback,
            PrintTaskCallback,
            ShortUuidCallback,
        )
        from nautical_core.task_models import TaskPayload

        contracts = {
            "completion_link_numbers_or_fail": {
                "task": TaskPayload,
                "coerce_int": CoerceIntCallback,
                "max_link_number": int,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "return": tuple[int, int] | None,
            },
            "completion_kind_or_stop": {
                "task": TaskPayload,
                "now_utc": datetime,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "end_chain_summary": EndChainSummaryCallback,
                "return": str | None,
            },
            "completion_existing_next_or_fail": {
                "task": TaskPayload,
                "next_no": int,
                "existing_next_lookup": ExistingNextLookupCallback,
                "short": ShortUuidCallback,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "return": bool,
            },
            "completion_chain_id_or_fail": {
                "task": TaskPayload,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "return": str | None,
            },
            "completion_preflight_context": {
                "task": TaskPayload,
                "now_utc": datetime,
                "services": CompletionPreflightServices,
                "return": CompletionPreflightContext | None,
            },
        }
        for method_name, expected in contracts.items():
            method = getattr(CompletionPreflightService, method_name)
            with self.subTest(method=method_name):
                self.assertEqual(get_type_hints(method), expected)
                self.assertNotIn(
                    inspect.Parameter.VAR_KEYWORD,
                    (parameter.kind for parameter in inspect.signature(method).parameters.values()),
                )

    def test_completion_compute_service_has_explicit_arguments(self) -> None:
        import inspect

        from nautical_core.modify_completion_effects import CompletionComputeService
        from nautical_core.modify_models import (
            CapFromUntilAnchorCallback,
            CapFromUntilCpCallback,
            CoerceIntCallback,
            ComputeAnchorChildDueCallback,
            ComputeCpChildDueCallback,
            DatetimeParserCallback,
            DiagnosticCallback,
            EstimateAnchorFinalCallback,
            EstimateCpFinalCallback,
            EndChainSummaryCallback,
            PanelCallback,
            PrintTaskCallback,
            SafeParseDatetimeCallback,
            ValidateChainDurationCallback,
            ValidateUntilCallback,
        )
        from nautical_core.task_models import TaskPayload
        from nautical_core.scheduler_models import OccurrenceSearchExhausted

        contracts = {
            "completion_compute_child_due": {
                "task": TaskPayload,
                "kind": str,
                "compute_anchor_child_due": ComputeAnchorChildDueCallback,
                "compute_cp_child_due": ComputeCpChildDueCallback,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "diag": DiagnosticCallback | None,
                "on_terminal": Callable[[OccurrenceSearchExhausted], bool] | None,
                "return": tuple[datetime | None, dict[str, Any] | None, Any] | None,
            },
            "completion_until_or_fail": {
                "task": TaskPayload,
                "now_utc": datetime,
                "safe_parse_datetime": SafeParseDatetimeCallback,
                "validate_until_not_past": ValidateUntilCallback,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "return": datetime | None | bool,
            },
            "completion_until_guard_or_stop": {
                "task": TaskPayload,
                "child_due": datetime | None,
                "until_dt": datetime | None,
                "now_utc": datetime,
                "end_chain_summary": EndChainSummaryCallback,
                "print_task": PrintTaskCallback,
                "return": bool,
            },
            "completion_require_child_due_or_fail": {
                "task": TaskPayload,
                "child_due": datetime | None,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "return": bool,
            },
            "completion_warn_unreasonable_duration": {
                "task": TaskPayload,
                "child_due": datetime | None,
                "until_dt": datetime | None,
                "now_utc": datetime,
                "validate_chain_duration_reasonable": ValidateChainDurationCallback,
                "panel": PanelCallback,
                "return": type(None),
            },
            "completion_caps": {
                "kind": str,
                "task": TaskPayload,
                "child_due": datetime | None,
                "dnf": Any,
                "coerce_int": CoerceIntCallback,
                "dtparse": DatetimeParserCallback,
                "estimate_cp_final_by_max": EstimateCpFinalCallback,
                "estimate_anchor_final_by_max": EstimateAnchorFinalCallback,
                "cap_from_until_cp": CapFromUntilCpCallback,
                "cap_from_until_anchor": CapFromUntilAnchorCallback,
                "return": tuple[int, datetime | None, int | None, list[tuple[str, Any]], int | None],
            },
            "completion_cap_guard_or_stop": {
                "task": TaskPayload,
                "next_no": int,
                "cap_no": int | None,
                "now_utc": datetime,
                "end_chain_summary": EndChainSummaryCallback,
                "print_task": PrintTaskCallback,
                "return": bool,
            },
        }
        for method_name, expected in contracts.items():
            method = getattr(CompletionComputeService, method_name)
            with self.subTest(method=method_name):
                self.assertEqual(get_type_hints(method), expected)
                self.assertNotIn(
                    inspect.Parameter.VAR_KEYWORD,
                    (parameter.kind for parameter in inspect.signature(method).parameters.values()),
                )

    def test_completion_spawn_service_has_explicit_arguments(self) -> None:
        import inspect

        from nautical_core.lifecycle.models import LifecyclePlan
        from nautical_core.modify_completion_effects import CompletionSpawnService
        from nautical_core.modify_models import (
            CompletionSpawnResult,
            CompletionSpawnServices,
        )
        from nautical_core.task_models import TaskPayload

        method = CompletionSpawnService.completion_build_and_spawn_child
        self.assertEqual(
            get_type_hints(method),
            {
                "task": TaskPayload,
                "child_due": datetime | None,
                "child_field": str,
                "next_no": int,
                "parent_short": str,
                "kind": str,
                "cpmax": int,
                "until_dt": datetime | None,
                "lifecycle_plan": LifecyclePlan | None,
                "services": CompletionSpawnServices,
                "return": CompletionSpawnResult | None,
            },
        )
        self.assertNotIn(
            inspect.Parameter.VAR_KEYWORD,
            (parameter.kind for parameter in inspect.signature(method).parameters.values()),
        )

    def test_completion_spawn_wrapper_has_explicit_arguments(self) -> None:
        import inspect

        from nautical_core.lifecycle.models import LifecyclePlan
        from nautical_core.modify_completion_effects import (
            CompletionSpawnPorts,
            build_and_spawn_child,
        )
        from nautical_core.modify_models import CompletionSpawnResult
        from nautical_core.task_models import TaskPayload

        self.assertEqual(
            get_type_hints(build_and_spawn_child),
            {
                "ports": CompletionSpawnPorts,
                "new": TaskPayload,
                "child_due": datetime | None,
                "child_field": str,
                "next_no": int,
                "parent_short": str,
                "kind": str,
                "cpmax": int,
                "until_dt": datetime | None,
                "lifecycle_plan": LifecyclePlan | None,
                "return": CompletionSpawnResult | None,
            },
        )
        self.assertNotIn(
            inspect.Parameter.VAR_KEYWORD,
            (parameter.kind for parameter in inspect.signature(build_and_spawn_child).parameters.values()),
        )

    def test_completion_feedback_and_validation_ports_reuse_callback_contracts(self) -> None:
        from nautical_core.modify_completion_effects import (
            CompletionFeedbackPorts,
            DurationWarningPorts,
            UntilCompletionPorts,
        )
        from nautical_core.modify_models import (
            EndChainSummaryCallback,
            PanelCallback,
            PrintTaskCallback,
            SafeParseDatetimeCallback,
            ValidateChainDurationCallback,
            ValidateUntilCallback,
        )

        expectations = (
            (CompletionFeedbackPorts, {
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
                "end_chain_summary": EndChainSummaryCallback,
            }),
            (UntilCompletionPorts, {
                "parse_datetime": SafeParseDatetimeCallback,
                "validate_until_not_past": ValidateUntilCallback,
                "panel": PanelCallback,
                "print_task": PrintTaskCallback,
            }),
            (DurationWarningPorts, {
                "validate_duration": ValidateChainDurationCallback,
                "panel": PanelCallback,
            }),
        )
        for ports_type, fields in expectations:
            annotations = get_type_hints(ports_type)
            for name, contract in fields.items():
                with self.subTest(ports=ports_type.__name__, field=name):
                    self.assertIs(annotations[name], contract)

    def test_completion_child_due_ports_have_typed_runtime_callbacks(self) -> None:
        from nautical_core.modify_completion_effects import ChildDuePorts
        from nautical_core.modify_models import (
            DiagnosticCallback,
            EndChainSummaryCallback,
            PanelCallback,
            PrintTaskCallback,
        )
        from nautical_core.scheduler_models import OccurrenceSearchExhausted
        from nautical_core.task_models import TaskPayload

        annotations = get_type_hints(ChildDuePorts)
        expected = {
            "now_utc": Callable[[], datetime],
            "exhaustion_message": Callable[[OccurrenceSearchExhausted], str],
            "ensure_terminal": Callable[[TaskPayload, str | None], bool],
            "end_summary": EndChainSummaryCallback,
            "panel": PanelCallback,
            "print_task": PrintTaskCallback,
            "diag": DiagnosticCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertEqual(annotations[name], contract)

    def test_completion_preflight_context_ports_use_concrete_models_and_callbacks(self) -> None:
        from nautical_core.modify_completion_effects import CompletionPreflightContextPorts
        from nautical_core.modify_models import (
            CoerceIntCallback,
            CompletionChainSnapshot,
            CompletionPreflightServices,
            EndChainSummaryCallback,
            PanelCallback,
            PrintTaskCallback,
            ShortUuidCallback,
        )

        annotations = get_type_hints(CompletionPreflightContextPorts)
        expected = {
            "services_type": type[CompletionPreflightServices],
            "snapshot_type": type[CompletionChainSnapshot],
            "snapshot_mode": Callable[[], str],
            "coerce_int": CoerceIntCallback,
            "short_uuid": ShortUuidCallback,
            "panel": PanelCallback,
            "print_task": PrintTaskCallback,
            "end_chain_summary": EndChainSummaryCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertEqual(annotations[name], contract)

    def test_completion_lifecycle_plan_ports_have_typed_callbacks(self) -> None:
        from nautical_core.modify_completion_effects import CompletionLifecyclePlanPorts
        from nautical_core.modify_models import (
            DiagnosticCallback,
            EndChainSummaryCallback,
            InvalidRelativeCarryReasonCallback,
            PanelCallback,
            PrintTaskCallback,
        )
        from nautical_core.task_models import TaskPayload

        annotations = get_type_hints(CompletionLifecyclePlanPorts)
        expected = {
            "scheduler_fingerprint": Callable[[], str],
            "compare_datetimes": Callable[[datetime, datetime], int],
            "invalid_relative_carry_reason": InvalidRelativeCarryReasonCallback,
            "end_chain_summary": EndChainSummaryCallback,
            "ensure_terminal_chain_off": Callable[[TaskPayload, str | None], bool],
            "panel": PanelCallback,
            "print_task": PrintTaskCallback,
            "diagnostic": DiagnosticCallback,
        }
        for name, contract in expected.items():
            with self.subTest(field=name):
                self.assertEqual(annotations[name], contract)

    def test_read_effects_import_without_hook_bootstrap(self) -> None:
        sys.modules.pop("nautical_core.hooks.modify_impl", None)
        module = importlib.import_module("nautical_core.modify_read_effects")
        self.assertIsNotNone(module)
        self.assertNotIn("nautical_core.hooks.modify_impl", sys.modules)

    def test_composition_capabilities_are_explicit_and_frozen(self) -> None:
        from nautical_core.modify_composition import ModifyHookCapabilities, ModifyRuntimeServices

        fields = getattr(ModifyHookCapabilities, "__dataclass_fields__", {})
        self.assertIn("modify_read_effects", fields)
        self.assertTrue(ModifyHookCapabilities.__dataclass_params__.frozen)
        runtime_fields = getattr(ModifyRuntimeServices, "__dataclass_fields__", {})
        self.assertNotIn("capabilities", runtime_fields)
        self.assertIsNot(
            get_type_hints(ModifyHookCapabilities)["modify_generation_effects"],
            Any,
        )
        capability_annotations = get_type_hints(ModifyHookCapabilities)
        self.assertIsNot(capability_annotations["task_codec"], Any)
        self.assertIsNot(capability_annotations["task_models"], Any)
        self.assertIsNot(capability_annotations["modify_spawn_effects"], Any)
        self.assertIsNot(capability_annotations["modify_presentation_effects"], Any)
        self.assertIsNot(capability_annotations["modify_task_fields"], Any)
        self.assertIsNot(capability_annotations["modify_validation_effects"], Any)
        self.assertIsNot(capability_annotations["modify_diagnostics_effects"], Any)
        self.assertIsNot(capability_annotations["modify_completion_effects"], Any)
        self.assertIsNot(capability_annotations["modify_ui_effects"], Any)
        self.assertIsNot(capability_annotations["modify_queries"], Any)
        self.assertIsNot(capability_annotations["modify_read_effects"], Any)
        self.assertIsNot(capability_annotations["modify_lifecycle"], Any)
        self.assertIsNot(capability_annotations["modify_expiration"], Any)
        self.assertIsNot(capability_annotations["modify_transition_effects"], Any)
        self.assertIsNot(capability_annotations["modify_ordinary"], Any)
        self.assertIsNot(capability_annotations["modify_composition_adapters"], Any)
        self.assertIsNot(capability_annotations["chain_integrity_lifecycle"], Any)
        self.assertIsNot(capability_annotations["hook_results"], Any)
        self.assertIsNot(capability_annotations["hook_context"], Any)
        self.assertIsNot(capability_annotations["hook_engine"], Any)

    def test_composition_does_not_load_unused_datetime_capability_module(self) -> None:
        from nautical_core.modify_composition import ModifyHookCapabilities

        loaded: list[str] = []

        class Host:
            def _module(self, name: str, *, required: bool = True) -> object:
                loaded.append(name)
                return object()

        ModifyHookCapabilities.from_host(Host())

        self.assertNotIn("modify_datetime_effects", loaded)

    def test_completion_route_contains_only_its_consumed_capability(self) -> None:
        from nautical_core.modify_composition import CompletionRouteCapabilities

        self.assertEqual(
            set(CompletionRouteCapabilities.__dataclass_fields__),
            {"modify_completion_effects"},
        )
        self.assertIsNot(
            get_type_hints(CompletionRouteCapabilities)["modify_completion_effects"],
            Any,
        )

    def test_deletion_route_query_capability_is_typed(self) -> None:
        from nautical_core.modify_composition import DeletionRouteCapabilities

        self.assertIsNot(get_type_hints(DeletionRouteCapabilities)["modify_queries"], Any)
        self.assertIsNot(
            get_type_hints(DeletionRouteCapabilities)["modify_diagnostics_effects"],
            Any,
        )
        self.assertIsNot(get_type_hints(DeletionRouteCapabilities)["modify_expiration"], Any)

    def test_route_bundles_do_not_repeat_root_presentation_capability(self) -> None:
        from nautical_core.modify_composition import (
            DeletionRouteCapabilities,
            NonCompletionRouteCapabilities,
        )

        for route in (NonCompletionRouteCapabilities, DeletionRouteCapabilities):
            with self.subTest(route=route.__name__):
                self.assertNotIn("modify_presentation_effects", route.__dataclass_fields__)

    def test_non_completion_route_does_not_carry_unused_diagnostics_module(self) -> None:
        from nautical_core.modify_composition import NonCompletionRouteCapabilities

        self.assertNotIn(
            "modify_diagnostics_effects",
            NonCompletionRouteCapabilities.__dataclass_fields__,
        )

    def test_schedule_period_helper_uses_only_explicit_ports(self) -> None:
        from datetime import datetime, timedelta, timezone
        from nautical_core.modify_schedule_effects import SchedulePorts, cp_add_period

        ports = SchedulePorts(
            to_local=lambda value: value.astimezone(timezone.utc),
            build_local_datetime=lambda day, hm: datetime(day.year, day.month, day.day, *hm, tzinfo=timezone.utc),
        )
        result = cp_add_period(ports, datetime(2026, 1, 1, tzinfo=timezone.utc), timedelta(days=1))
        self.assertEqual(result, datetime(2026, 1, 2, tzinfo=timezone.utc))

    def test_sequence_period_helper_uses_only_sequence_port(self) -> None:
        from datetime import timedelta
        from nautical_core.modify_schedule_effects import SequencePorts, sequence_period_for_link

        ports = SequencePorts(lambda token, **kwargs: timedelta(days=int(token["days"])))
        self.assertEqual(
            sequence_period_for_link(ports, [{"days": 2}], "cp", 1),
            timedelta(days=2),
        )

    def test_cp_carry_ports_and_result_have_concrete_types(self) -> None:
        from nautical_core.modify_carry_workflow import TemporalCarryDecision
        from nautical_core.modify_transition_effects import (
            CPCarryPorts,
            preserve_cp_relative_offsets_on_due_change,
        )

        annotations = get_type_hints(CPCarryPorts)
        self.assertTrue(all(annotation is not Any for annotation in annotations.values()))
        self.assertIs(
            get_type_hints(preserve_cp_relative_offsets_on_due_change)["return"],
            TemporalCarryDecision,
        )

    def test_native_carry_ports_have_concrete_dependencies(self) -> None:
        from nautical_core.modify_transition_effects import NativeCarryPorts

        annotations = get_type_hints(NativeCarryPorts)
        self.assertTrue(all(annotation is not Any for annotation in annotations.values()))

    def test_native_preserve_ports_have_concrete_dependencies(self) -> None:
        from nautical_core.modify_transition_effects import NativePreservePorts

        annotations = get_type_hints(NativePreservePorts)
        self.assertTrue(all(annotation is not Any for annotation in annotations.values()))

    def test_completion_transition_port_discards_unused_transition_decision(self) -> None:
        from nautical_core.modify_transition_effects import CompletionValidationPorts

        apply_transition = get_type_hints(CompletionValidationPorts)["apply_transition"]
        self.assertIs(apply_transition.__args__[-1], type(None))

    def test_modify_runtime_services_do_not_use_catch_all_callback_protocol(self) -> None:
        from collections.abc import Callable as CallableOrigin
        from nautical_core.modify_composition import ModifyRuntimeServices
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_runtime import ModifyRuntimeState

        annotations = get_type_hints(
            ModifyRuntimeServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "ModifyRuntimeState": ModifyRuntimeState,
            },
        )
        callback_fields = {
            "runtime_state",
            "import_module",
            "diag_summary",
            "diagnostic",
            "chain_health_advice",
            "print_task",
            "validate_native_until",
            "validate_native_until_slots",
        }
        callback_fields.remove("chain_health_advice")
        self.assertTrue(all(get_origin(annotations[name]) is CallableOrigin for name in callback_fields))
        self.assertEqual(annotations["chain_health_advice"].__name__, "_ChainHealthAdviceCallback")

    def test_modify_runtime_services_use_runtime_state_and_read_service_owners(self) -> None:
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_composition import ModifyRuntimeServices
        from nautical_core.modify_runtime import ModifyRuntimeState

        annotations = get_type_hints(
            ModifyRuntimeServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "ModifyRuntimeState": ModifyRuntimeState,
            },
        )

        self.assertEqual(
            annotations["runtime_state"],
            Callable[[], ModifyRuntimeState],
        )
        self.assertEqual(
            annotations["lifecycle_read_service"],
            Callable[[], LifecycleReadService],
        )

    def test_completion_feedback_adapter_ports_use_typed_request_models(self) -> None:
        from nautical_core.modify_composition import _ModifyCompositionAdapters
        from nautical_core.modify_models import (
            AnchorCompletionFeedbackModel,
            CpCompletionFeedbackModel,
        )

        localns = {
            "AnchorCompletionFeedbackModel": AnchorCompletionFeedbackModel,
            "CpCompletionFeedbackModel": CpCompletionFeedbackModel,
        }
        anchor_request = get_type_hints(
            _ModifyCompositionAdapters.render_anchor_completion_feedback_for,
            localns=localns,
        )["request"]
        cp_request = get_type_hints(
            _ModifyCompositionAdapters.render_cp_completion_feedback_for,
            localns=localns,
        )["request"]

        self.assertIs(anchor_request, AnchorCompletionFeedbackModel)
        self.assertIs(cp_request, CpCompletionFeedbackModel)

    def test_completion_recurrence_callbacks_use_transition_and_carry_owners(self) -> None:
        from nautical_core.modify_carry_workflow import (
            NativeUntilDecision,
            TemporalCarryDecision,
        )
        from nautical_core.modify_composition import (
            _PrepareRecurrenceCallback,
            _PreserveCPCarryCallback,
            _PreserveNativeUntilCallback,
        )
        from nautical_core.task_changes import TaskTransition

        localns = {
            "TaskTransition": TaskTransition,
            "TemporalCarryDecision": TemporalCarryDecision,
            "NativeUntilDecision": NativeUntilDecision,
        }
        callbacks = (
            _PrepareRecurrenceCallback,
            _PreserveCPCarryCallback,
            _PreserveNativeUntilCallback,
        )
        for callback in callbacks:
            with self.subTest(callback=callback.__name__):
                annotations = get_type_hints(callback.__call__, localns=localns)
                self.assertEqual(annotations["transition"], TaskTransition | None)
        self.assertIs(
            get_type_hints(_PreserveCPCarryCallback.__call__, localns=localns)["return"],
            TemporalCarryDecision,
        )
        self.assertIs(
            get_type_hints(_PreserveNativeUntilCallback.__call__, localns=localns)["return"],
            NativeUntilDecision,
        )

    def test_transition_effect_adapter_ports_have_explicit_owner_contracts(self) -> None:
        from inspect import Parameter, signature
        from nautical_core.lifecycle.models import LifecyclePlan
        from nautical_core.modify_carry_workflow import (
            NativeUntilDecision,
            TemporalCarryDecision,
        )
        from nautical_core.modify_composition import _ModifyTransitionEffects
        from nautical_core.modify_transition_effects import (
            CPCarryPorts,
            CompletionValidationPorts,
            NativeCarryPorts,
            NativePreservePorts,
        )
        from nautical_core.task_changes import TaskTransition

        localns = {
            "LifecyclePlan": LifecyclePlan,
            "CPCarryPortsContract": CPCarryPorts,
            "CompletionValidationPortsContract": CompletionValidationPorts,
            "NativeCarryPortsContract": NativeCarryPorts,
            "NativePreservePortsContract": NativePreservePorts,
            "NativeUntilDecision": NativeUntilDecision,
            "TaskTransition": TaskTransition,
            "TemporalCarryDecision": TemporalCarryDecision,
        }
        expected = {
            "preserve_cp_relative_offsets_on_due_change": (CPCarryPorts, TemporalCarryDecision),
            "reject_native_until_carry": (NativeCarryPorts, type(None)),
            "preserve_native_until_on_target_change": (NativePreservePorts, NativeUntilDecision),
            "validate_completion_cp_and_anchor": (CompletionValidationPorts, tuple[str, str, str]),
        }
        for name, (ports_type, return_type) in expected.items():
            with self.subTest(method=name):
                method = getattr(_ModifyTransitionEffects, name)
                annotations = get_type_hints(method, localns=localns)
                self.assertIs(annotations["ports"], ports_type)
                self.assertEqual(annotations["return"], return_type)
                self.assertFalse(
                    any(
                        parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
                        for parameter in signature(method).parameters.values()
                    )
                )

    def test_chain_health_advice_uses_task_observation_and_text_result(self) -> None:
        from typing import Sequence
        from nautical_core.modify_composition import (
            _ChainHealthAdviceCallback,
            _ModifyDiagnosticsEffects,
        )
        from nautical_core.task_models import TaskObservation

        expected_chain = Sequence[TaskObservation]
        callback = get_type_hints(_ChainHealthAdviceCallback.__call__)
        effect = get_type_hints(_ModifyDiagnosticsEffects.chain_health_advice)

        self.assertEqual(callback["chain"], expected_chain)
        self.assertEqual(effect["chain"], expected_chain)
        self.assertEqual(callback["return"], str | None)
        self.assertEqual(effect["return"], str | None)

    def test_completion_effect_adapter_uses_existing_ports_and_results(self) -> None:
        from inspect import Parameter, signature
        from nautical_core.lifecycle.models import LifecyclePlan
        from nautical_core.modify_completion_effects import (
            CompletionComputePorts,
            CompletionPreflightContextPorts,
            CompletionSpawnPorts,
        )
        from nautical_core.modify_composition import _ModifyCompletionEffects
        from nautical_core.modify_models import (
            CompletionComputeResult,
            CompletionLifecycleResult,
            CompletionPreflightContext,
            CompletionSpawnResult,
        )
        from nautical_core.task_read_repository import TaskReadRepository

        localns = {
            "LifecyclePlan": LifecyclePlan,
            "CompletionComputePorts": CompletionComputePorts,
            "CompletionPreflightContextPorts": CompletionPreflightContextPorts,
            "CompletionSpawnPorts": CompletionSpawnPorts,
            "CompletionComputeResult": CompletionComputeResult,
            "CompletionLifecycleResult": CompletionLifecycleResult,
            "CompletionPreflightContext": CompletionPreflightContext,
            "CompletionSpawnResult": CompletionSpawnResult,
            "TaskReadRepository": TaskReadRepository,
        }
        expected = {
            "completion_preflight_context_ports_for": CompletionPreflightContextPorts,
            "completion_compute_ports_for": CompletionComputePorts,
            "completion_spawn_ports_for": CompletionSpawnPorts,
        }
        for name, result_type in expected.items():
            with self.subTest(factory=name):
                method = getattr(_ModifyCompletionEffects, name)
                self.assertIs(get_type_hints(method, localns=localns)["return"], result_type)

        preflight = get_type_hints(
            _ModifyCompletionEffects.preflight_context, localns=localns
        )
        compute = get_type_hints(
            _ModifyCompletionEffects.compute_next_and_limits, localns=localns
        )
        spawn = get_type_hints(
            _ModifyCompletionEffects.build_and_spawn_child, localns=localns
        )
        self.assertIs(preflight["repository"], TaskReadRepository)
        self.assertEqual(
            preflight["return"], CompletionPreflightContext | None
        )
        self.assertEqual(
            compute["return"],
            CompletionComputeResult | CompletionLifecycleResult | None,
        )
        self.assertEqual(spawn["return"], CompletionSpawnResult | None)
        for method in (
            _ModifyCompletionEffects.preflight_context,
            _ModifyCompletionEffects.compute_next_and_limits,
            _ModifyCompletionEffects.build_and_spawn_child,
        ):
            with self.subTest(method=method.__name__):
                self.assertFalse(
                    any(
                        parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
                        for parameter in signature(method).parameters.values()
                    )
                )

    def test_ordinary_modify_adapter_uses_runtime_transition_and_uow_owners(self) -> None:
        from nautical_core.modify_composition import (
            ModifyRuntimeServices,
            _ModifyCompositionAdapters,
            _ModifyLifecycle,
            _ModifyOrdinaryEffects,
        )
        from nautical_core.modify_composition_adapters import handle_non_completion
        from nautical_core.modify_ordinary import OrdinaryModifyServices
        from nautical_core.task_changes import TaskTransition
        from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork

        localns = {
            "ModifyRuntimeServices": ModifyRuntimeServices,
            "ModifyCompositionRuntimeServices": ModifyRuntimeServices,
            "TaskTransition": TaskTransition,
            "TaskwarriorUnitOfWork": TaskwarriorUnitOfWork,
        }
        port = get_type_hints(
            _ModifyCompositionAdapters.handle_non_completion,
            localns=localns,
        )
        implementation = get_type_hints(handle_non_completion, localns=localns)

        for hints in (port, implementation):
            with self.subTest(hints=hints):
                self.assertIs(hints["unit_of_work"], TaskwarriorUnitOfWork)
                self.assertEqual(hints["transition"], TaskTransition | None)
                self.assertEqual(hints["runtime"], ModifyRuntimeServices | None)

        ordinary = get_type_hints(
            _ModifyOrdinaryEffects.handle_non_completion_modify,
            localns={
                "OrdinaryModifyServices": OrdinaryModifyServices,
                "OrdinaryModifyServicesContract": OrdinaryModifyServices,
                "TaskTransition": TaskTransition,
                "_ModifyLifecycle": _ModifyLifecycle,
            },
        )
        self.assertIs(ordinary["services"], OrdinaryModifyServices)
        self.assertIs(ordinary["lifecycle"], _ModifyLifecycle)
        self.assertEqual(ordinary["transition"], TaskTransition | None)

    def test_completion_modify_adapter_uses_typed_runtime_and_lifecycle_result(self) -> None:
        from nautical_core.modify_composition import (
            ModifyRuntimeServices,
            _ModifyCompositionAdapters,
        )
        from nautical_core.modify_composition_adapters import handle_completion
        from nautical_core.modify_models import CompletionLifecycleResult
        from nautical_core.task_changes import TaskTransition
        from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork

        localns = {
            "ModifyRuntimeServices": ModifyRuntimeServices,
            "ModifyCompositionRuntimeServices": ModifyRuntimeServices,
            "CompletionLifecycleResult": CompletionLifecycleResult,
            "TaskTransition": TaskTransition,
            "TaskwarriorUnitOfWork": TaskwarriorUnitOfWork,
        }
        for method in (
            _ModifyCompositionAdapters.handle_completion,
            handle_completion,
        ):
            with self.subTest(method=method.__name__):
                annotations = get_type_hints(method, localns=localns)
                self.assertIs(annotations["unit_of_work"], TaskwarriorUnitOfWork)
                self.assertEqual(annotations["transition"], TaskTransition | None)
                self.assertEqual(annotations["runtime"], ModifyRuntimeServices | None)
                self.assertEqual(
                    annotations["return"], CompletionLifecycleResult | None
                )

    def test_capability_cache_does_not_hide_unexpected_attribute_failures(self) -> None:
        from unittest.mock import patch
        import nautical_core.modify_composition as composition

        capabilities = object()

        class Host:
            _TASK_DATETIME_PARSER = object()

            def __setattr__(self, name: str, value: object) -> None:
                if name == "_MODIFY_CAPABILITIES":
                    raise RuntimeError("broken capability cache descriptor")
                object.__setattr__(self, name, value)

        with patch.object(
            composition.ModifyHookCapabilities,
            "from_host",
            return_value=capabilities,
        ):
            with self.assertRaisesRegex(RuntimeError, "broken capability cache descriptor"):
                composition.capabilities_for(Host())

    def test_datetime_parser_cache_does_not_hide_unexpected_attribute_failures(self) -> None:
        from unittest.mock import patch
        import nautical_core.modify_composition as composition

        capabilities = object()

        class Host:
            _MODIFY_CAPABILITIES = capabilities
            _TASK_DATETIME_PARSER = None

            def __init__(self) -> None:
                self.core = object()
                self._diag = None

            def __setattr__(self, name: str, value: object) -> None:
                if name == "_TASK_DATETIME_PARSER":
                    raise RuntimeError("broken datetime parser cache descriptor")
                object.__setattr__(self, name, value)

        with patch.object(composition, "parser_for_core", return_value=object()):
            with self.assertRaisesRegex(
                RuntimeError, "broken datetime parser cache descriptor"
            ):
                composition.capabilities_for(Host())

    def test_capabilities_still_work_when_immutable_host_cannot_cache_them(self) -> None:
        from unittest.mock import patch
        import nautical_core.modify_composition as composition

        capabilities = object()

        class ImmutableHost:
            __slots__ = ()
            _TASK_DATETIME_PARSER = object()

        with patch.object(
            composition.ModifyHookCapabilities,
            "from_host",
            return_value=capabilities,
        ):
            self.assertIs(composition.capabilities_for(ImmutableHost()), capabilities)

    def test_native_carry_summary_failure_does_not_mask_primary_rejection(self) -> None:
        from nautical_core.modify_transition_effects import (
            NativeCarryPorts,
            reject_native_until_carry,
        )

        panels: list[tuple[str, list[tuple[str, str]], str]] = []

        class HookAbort(Exception):
            pass

        def broken_summary(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("optional carry summary failed")

        def panel(title: str, rows: list[tuple[str, str]], *, kind: str) -> None:
            panels.append((title, rows, kind))

        def abort(code: int) -> None:
            raise HookAbort(code)

        ports = NativeCarryPorts(
            describe_carry=broken_summary,
            parse_datetime=lambda _value: None,
            to_local=lambda value: value,
            format_local=str,
            anchor_field=lambda _task: "the next occurrence",
            panel=panel,
            abort=abort,
        )

        with self.assertRaises(HookAbort) as raised:
            reject_native_until_carry(
                ports,
                {"until": "invalid"},
                {},
                None,
                "due",
                ValueError("until must be later than due"),
            )

        self.assertEqual(raised.exception.args, (1,))
        self.assertEqual(
            panels,
            [
                (
                    "❌ Invalid expiration window",
                    [
                        ("Target", "the next occurrence"),
                        ("Required", "until must be later than due"),
                    ],
                    "error",
                )
            ],
        )

    def test_modify_runtime_services_completion_callbacks_have_named_contracts(self) -> None:
        from collections.abc import Callable as CallableOrigin
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_composition import ModifyRuntimeServices
        from nautical_core.modify_runtime import ModifyRuntimeState

        annotations = get_type_hints(
            ModifyRuntimeServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "ModifyRuntimeState": ModifyRuntimeState,
            },
        )
        expected = {
            "chain_integrity_warnings": "ChainIntegrityCallback",
            "render_anchor_completion_feedback": "AnchorCompletionRenderCallback",
            "render_cp_completion_feedback": "CpCompletionRenderCallback",
            "render_lifecycle_result": "LifecycleResultRenderCallback",
        }
        for field, protocol_name in expected.items():
            with self.subTest(field=field):
                self.assertEqual(annotations[field].__name__, protocol_name)

        seed_callback = annotations["seed_runtime_lookup_tasks"]
        self.assertIs(get_origin(seed_callback), CallableOrigin)
        seed_arguments, _return_type = get_args(seed_callback)
        self.assertIsNot(seed_arguments, Ellipsis)
        self.assertEqual(len(seed_arguments), 2)

    def test_modify_runtime_services_recurrence_callbacks_have_named_contracts(self) -> None:
        from nautical_core.lifecycle.read_service import LifecycleReadService
        from nautical_core.modify_composition import ModifyRuntimeServices
        from nautical_core.modify_runtime import ModifyRuntimeState

        annotations = get_type_hints(
            ModifyRuntimeServices,
            localns={
                "LifecycleReadService": LifecycleReadService,
                "ModifyRuntimeState": ModifyRuntimeState,
            },
        )
        expected = {
            "prepare_recurrence": "_PrepareRecurrenceCallback",
            "preserve_cp_relative_offsets": "_PreserveCPCarryCallback",
            "preserve_native_until": "_PreserveNativeUntilCallback",
            "compute_next_and_limits": "CompletionComputeCallback",
        }
        for field, protocol_name in expected.items():
            with self.subTest(field=field):
                self.assertEqual(annotations[field].__name__, protocol_name)

    def test_spawn_services_use_named_payload_callback_contracts(self) -> None:
        from nautical_core.lifecycle.models import LifecycleIdentity, LifecyclePlan
        from nautical_core.modify_spawn import SpawnServices, spawn_child_atomic
        from nautical_core.modify_models import DatetimeParserCallback
        from nautical_core.task_models import TaskPayload

        annotations = get_type_hints(SpawnServices)
        self.assertEqual(
            annotations["prepare_spawn_child_payload"].__name__,
            "_PrepareSpawnChildPayload",
        )
        self.assertEqual(
            annotations["child_uuid_for_spawn"].__name__,
            "_ChildUUIDForSpawn",
        )
        self.assertEqual(
            get_type_hints(spawn_child_atomic)["lifecycle_plan"],
            LifecyclePlan | None,
        )
        self.assertNotIn("lifecycle_models", annotations)
        self.assertEqual(annotations["fmt_isoz"], Callable[[datetime], str])
        self.assertEqual(annotations["now_utc"], Callable[[], datetime])
        self.assertEqual(
            annotations["lifecycle_spawn_identity"],
            Callable[[TaskPayload, TaskPayload], LifecycleIdentity],
        )
        self.assertEqual(
            annotations["enqueue_spawn_intent"],
            Callable[[LifecyclePlan], tuple[bool, str]],
        )
        self.assertIs(annotations["parse_datetime"], DatetimeParserCallback)

        from nautical_core.modify_spawn_effects import SpawnChildPorts, SpawnIntentPorts

        self.assertNotIn("lifecycle_models", get_type_hints(SpawnChildPorts))
        self.assertNotIn("models", get_type_hints(SpawnIntentPorts))

    def test_cp_carry_applies_typed_temporal_decision(self) -> None:
        from datetime import datetime, timezone
        from nautical_core.chain_generation import CarryFieldError
        from nautical_core.modify_carry import preserve_cp_relative_offsets_on_due_change
        import nautical_core.modify_carry_workflow as modify_carry_workflow
        from nautical_core.modify_carry_workflow import TemporalCarryDecision
        from nautical_core.modify_transition_effects import (
            CPCarryPorts,
            preserve_cp_relative_offsets_on_due_change as preserve_cp_carry,
        )
        from nautical_core.task_changes import TaskTransition
        from nautical_core.task_models import TaskObservation

        old = {
            "uuid": "00000000-0000-4000-8000-000000000001",
            "cp": "P1D",
            "due": "2026-01-01T09:00:00+00:00",
            "scheduled": "2026-01-01T10:00:00+00:00",
        }
        new = {
            **old,
            "due": "2026-01-02T09:00:00+00:00",
        }
        transition = TaskTransition.from_observations(
            TaskObservation.from_mapping(old, source_query="modify-before"),
            TaskObservation.from_mapping(new, source_query="modify-after"),
        )
        ports = CPCarryPorts(
            carry=preserve_cp_relative_offsets_on_due_change,
            field_changed=lambda _old, _new, field: field == "due",
            parse_datetime=lambda value: datetime.fromisoformat(value) if value else None,
            utc_to_local_naive=lambda value: value.replace(tzinfo=None),
            local_naive_to_utc=lambda value: value.replace(tzinfo=timezone.utc),
            format_datetime=lambda value: value.isoformat(),
            carry_error=CarryFieldError,
            workflow=modify_carry_workflow,
        )

        decision = preserve_cp_carry(ports, old, new, "P1D", transition=transition)

        self.assertIsInstance(decision, TemporalCarryDecision)
        self.assertEqual(decision.status, "adjusted")
        self.assertEqual(new["scheduled"], "2026-01-02T10:00:00Z")

    def test_completion_validation_uses_explicit_ports_and_transition(self) -> None:
        from nautical_core.modify_validation import CompletionValidationServices
        from nautical_core.modify_transition_effects import (
            CompletionValidationPorts,
            validate_completion_cp_and_anchor,
        )
        from nautical_core.task_changes import TaskTransition
        from nautical_core.task_models import TaskObservation, TaskPayload

        observed: dict[str, object] = {}

        def validate(
            old: TaskPayload,
            new: TaskPayload,
            *,
            services: CompletionValidationServices,
        ) -> tuple[str, str, str]:
            self.assertIsInstance(services, CompletionValidationServices)
            observed["changed"] = services.field_changed(old, new, "anchor")
            observed["transition"] = services.apply_transition(old, new)
            return "cp", "anchor", "omit"

        old = TaskObservation.from_mapping({"anchor": "w:mon"}, source_query="test")
        new = TaskObservation.from_mapping({"anchor": "w:tue"}, source_query="test")
        ports = CompletionValidationPorts(
            validate=validate,
            strip_quotes=lambda value: value,
            reject_conflicting_types=lambda *_args: None,
            validate_omit=lambda *_args: None,
            validate_chain_limits=lambda _task: None,
            parse_cp_sequence=lambda value: [value],
            cp_sequence_parse_error=lambda _value: None,
            field_changed=lambda *_args: False,
            validate_anchor=lambda _value: None,
            validate_cp=lambda *_args: None,
            apply_transition=lambda _old, _new: "applied",
            fail=lambda *_args: None,
            diagnostic=lambda *_args: None,
        )
        transition = TaskTransition.from_observations(old, new)

        result = validate_completion_cp_and_anchor(
            ports, {"anchor": "w:mon"}, {"anchor": "w:tue"}, transition=transition
        )

        self.assertEqual(result, ("cp", "anchor", "omit"))
        self.assertTrue(observed["changed"])
        self.assertEqual(observed["transition"], "applied")


if __name__ == "__main__":
    unittest.main()
