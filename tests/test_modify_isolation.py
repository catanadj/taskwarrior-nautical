"""Characterization tests for the on-modify effect boundary."""

from __future__ import annotations

import importlib
import sys
import unittest
from typing import Any, get_type_hints


class ModifyIsolationTests(unittest.TestCase):
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

    def test_composition_does_not_load_unused_datetime_capability_module(self) -> None:
        from nautical_core.modify_composition import ModifyHookCapabilities

        loaded: list[str] = []

        class Host:
            def _module(self, name: str, *, required: bool = True) -> object:
                loaded.append(name)
                return object()

        ModifyHookCapabilities.from_host(Host())

        self.assertNotIn("modify_datetime_effects", loaded)

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
