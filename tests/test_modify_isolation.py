"""Characterization tests for the on-modify effect boundary."""

from __future__ import annotations

import importlib
import sys
import unittest
from types import SimpleNamespace


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

    def test_completion_validation_uses_explicit_ports_and_transition(self) -> None:
        from nautical_core.modify_transition_effects import (
            CompletionValidationPorts,
            validate_completion_cp_and_anchor,
        )

        observed: dict[str, object] = {}

        def validate(old, new, *, services):
            observed["changed"] = services.field_changed(old, new, "anchor")
            observed["transition"] = services.apply_transition(old, new)
            return "cp", "anchor", "omit"

        ports = CompletionValidationPorts(
            validate=validate,
            services_type=SimpleNamespace,
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
        transition = SimpleNamespace(changed=lambda field: field == "anchor")

        result = validate_completion_cp_and_anchor(
            ports, {"anchor": "w:mon"}, {"anchor": "w:tue"}, transition=transition
        )

        self.assertEqual(result, ("cp", "anchor", "omit"))
        self.assertTrue(observed["changed"])
        self.assertEqual(observed["transition"], "applied")


if __name__ == "__main__":
    unittest.main()
