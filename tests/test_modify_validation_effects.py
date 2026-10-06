import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import nautical_core.modify_lifecycle as modify_lifecycle
import nautical_core.modify_ordinary as modify_ordinary

from nautical_core.modify_validation_effects import AnchorValidationPorts, validate_anchor
from nautical_core.modify_validation_effects import OmitValidationPorts, validate_omit
from nautical_core.parsing.parser_models import ParseError
from nautical_core.modify_validation import validate_native_until_anchor_slots_or_fail
from nautical_core.modify_datetime_effects import (
    DatetimeEffectPorts,
    local_naive_to_utc,
    safe_dt,
    utc_to_local_naive,
)


class ModifyValidationEffectsTests(unittest.TestCase):
    def _completion_validation_services(self, parse_cp_sequence, fail):
        from nautical_core.modify_validation import CompletionValidationServices

        return CompletionValidationServices(
            strip_quotes=lambda value: value,
            reject_conflicting_types=lambda *_args: None,
            validate_omit=lambda *_args: None,
            validate_chain_limits=lambda _task: None,
            parse_cp_sequence=parse_cp_sequence,
            cp_sequence_parse_error=lambda _value: None,
            field_changed=lambda *_args: False,
            validate_anchor=lambda _value: None,
            validate_cp=lambda *_args: None,
            apply_transition=lambda *_args: None,
            fail=fail,
        )

    def test_completion_cp_parser_defect_propagates(self) -> None:
        from nautical_core.modify_validation import validate_completion_cp_and_anchor

        failures = []
        services = self._completion_validation_services(
            lambda _value: (_ for _ in ()).throw(RuntimeError("parser defect")),
            lambda title, message: failures.append((title, message)),
        )

        with self.assertRaisesRegex(RuntimeError, "parser defect"):
            validate_completion_cp_and_anchor({}, {"cp": "P1D"}, services=services)

        self.assertEqual(failures, [])

    def test_completion_invalid_cp_remains_a_user_validation_failure(self) -> None:
        from nautical_core.modify_validation import validate_completion_cp_and_anchor

        failures = []
        services = self._completion_validation_services(
            lambda _value: None,
            lambda title, message: failures.append((title, message)),
        )

        validate_completion_cp_and_anchor({}, {"cp": "invalid"}, services=services)

        self.assertEqual(failures, [("Invalid CP", "invalid duration format 'invalid'")])

    def test_completion_transition_internal_failure_propagates(self) -> None:
        from nautical_core.modify_validation import validate_completion_cp_and_anchor

        failures = []

        def fail(title, message):
            failures.append((title, message))

        services = self._completion_validation_services(lambda _value: [timedelta(days=1)], fail)
        services.apply_transition = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("transition invariant defect")
        )

        with self.assertRaisesRegex(RuntimeError, "transition invariant defect"):
            validate_completion_cp_and_anchor({}, {"cp": "P1D"}, services=services)

        self.assertEqual(failures, [])

    def test_completion_transition_value_error_remains_a_safe_rejection(self) -> None:
        from nautical_core.modify_validation import validate_completion_cp_and_anchor

        class ValidationRejected(Exception):
            pass

        failures = []

        def fail(title, message):
            failures.append((title, message))
            raise ValidationRejected(message)

        services = self._completion_validation_services(lambda _value: [timedelta(days=1)], fail)
        services.apply_transition = lambda *_args: (_ for _ in ()).throw(
            ValueError("chain identity is incomplete")
        )

        with self.assertRaisesRegex(ValidationRejected, "chain identity is incomplete"):
            validate_completion_cp_and_anchor({}, {"cp": "P1D"}, services=services)

        self.assertEqual(
            failures,
            [(
                "Nautical recurrence activation failed",
                "Nautical recurrence transition failed: ValueError: chain identity is incomplete",
            )],
        )

    def test_omit_validation_maps_user_value_errors_but_surfaces_internal_defects(self) -> None:
        failures = []

        def run_omit_pipeline(expr, *, validate_omit_expr):
            validate_omit_expr(expr)

        ports = OmitValidationPorts(
            pipeline=SimpleNamespace(
                validate_omit_expression=run_omit_pipeline,
                validate_recurrence_files=lambda *_args, **_kwargs: (),
            ),
            parse_anchor=lambda _expr: None,
            validate_anchor=lambda _expr: None,
            validate_omit=lambda _expr: (_ for _ in ()).throw(ValueError("bad omit")),
            validate_files=lambda *_args, **_kwargs: (),
            load_anchor_file=lambda _name: None,
            load_omit_file=lambda _name: None,
            fail=lambda title, message: failures.append((title, message)),
        )
        validate_omit(ports, "", "", "bad", "")
        self.assertEqual(failures, [("Invalid omit", "bad omit")])

        failures.clear()
        ports = OmitValidationPorts(
            pipeline=SimpleNamespace(
                validate_omit_expression=run_omit_pipeline,
                validate_recurrence_files=lambda *_args, **_kwargs: (),
            ),
            parse_anchor=lambda _expr: None,
            validate_anchor=lambda _expr: None,
            validate_omit=lambda _expr: (_ for _ in ()).throw(RuntimeError("omit defect")),
            validate_files=lambda *_args, **_kwargs: (),
            load_anchor_file=lambda _name: None,
            load_omit_file=lambda _name: None,
            fail=lambda title, message: failures.append((title, message)),
        )
        with self.assertRaisesRegex(RuntimeError, "omit defect"):
            validate_omit(ports, "", "", "bad", "")
        self.assertEqual(failures, [])

    def test_anchor_validation_reports_parse_errors_but_surfaces_internal_defects(self) -> None:
        failures = []

        def make_ports(validate_strict):
            return AnchorValidationPorts(
                lint=lambda _expr: (None, []),
                validate_strict=validate_strict,
                panel=lambda *_args, **_kwargs: None,
                is_astronomy_error=lambda _exc: False,
                astronomy_error_message=str,
                fail=lambda title, message: failures.append((title, message)),
            )

        def reject_expression(_expression):
            raise ParseError("invalid anchor syntax")

        validate_anchor(make_ports(reject_expression), {}, {}, "malformed")
        self.assertEqual(
            failures,
            [("Invalid anchor", "invalid anchor syntax (expected an anchor such as w:mon, m:15, or y:jul)")],
        )

        failures.clear()

        def broken_validator(_expression):
            raise RuntimeError("anchor validator defect")

        with self.assertRaisesRegex(RuntimeError, "anchor validator defect"):
            validate_anchor(make_ports(broken_validator), {}, {}, "w:mon")
        self.assertEqual(failures, [])

    def test_anchor_validation_does_not_retry_strict_validation_type_errors(self) -> None:
        calls = []

        def broken_validator(_expression):
            calls.append("strict")
            raise TypeError("anchor validator contract defect")

        ports = AnchorValidationPorts(
            lint=lambda _expr: (None, []),
            validate_strict=broken_validator,
            panel=lambda *_args, **_kwargs: None,
            is_astronomy_error=lambda _exc: False,
            astronomy_error_message=str,
            fail=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected failure")),
        )

        with self.assertRaisesRegex(TypeError, "anchor validator contract defect"):
            validate_anchor(ports, {}, {}, "w:mon")

        self.assertEqual(calls, ["strict"])

    def test_native_until_slot_preflight_only_suppresses_anchor_input_errors(self) -> None:
        task = {
            "anchor": "w:mon",
            "due": "20260101T090000Z",
            "until": "20260102T090000Z",
        }

        def validate_with(anchor_validator):
            return validate_native_until_anchor_slots_or_fail(
                task,
                safe_parse_datetime=lambda _value: (datetime(2026, 1, 1, 9), None),
                validate_anchor=anchor_validator,
                collect_time_slots=lambda *_args, **_kwargs: (),
                validate_time_slots=lambda *_args, **_kwargs: (True, None),
                normalize_time_slots=lambda *_args: [],
                anchor_file_dir="",
                recurrence_context=lambda _task: None,
                to_local=lambda value: value,
                format_local=str,
                astronomy_is_error=lambda _exc: False,
                astronomy_error_message=str,
                panel=lambda *_args, **_kwargs: None,
                abort=lambda _code: None,
            )

        def malformed_anchor(_expr):
            raise ParseError("invalid anchor")

        self.assertIsNone(validate_with(malformed_anchor))

        def broken_validator(_expr):
            raise RuntimeError("anchor validator defect")

        with self.assertRaisesRegex(RuntimeError, "anchor validator defect"):
            validate_with(broken_validator)

    def test_native_until_slot_preflight_surfaces_unclassified_collection_errors(self) -> None:
        task = {
            "anchor": "w:mon",
            "due": "20260101T090000Z",
            "until": "20260102T090000Z",
        }

        def collect_broken_slots(*_args, **_kwargs):
            raise RuntimeError("slot collector defect")

        with self.assertRaisesRegex(RuntimeError, "slot collector defect"):
            validate_native_until_anchor_slots_or_fail(
                task,
                safe_parse_datetime=lambda _value: (datetime(2026, 1, 1, 9), None),
                validate_anchor=lambda _expr: [],
                collect_time_slots=collect_broken_slots,
                validate_time_slots=lambda *_args, **_kwargs: (True, None),
                normalize_time_slots=lambda *_args: [],
                anchor_file_dir="",
                recurrence_context=lambda _task: None,
                to_local=lambda value: value,
                format_local=str,
                astronomy_is_error=lambda _exc: False,
                astronomy_error_message=str,
                panel=lambda *_args, **_kwargs: None,
                abort=lambda _code: None,
            )

    def test_datetime_effects_handle_invalid_values_and_truncate_microseconds(self) -> None:
        ports = DatetimeEffectPorts(
            parse_datetime=lambda value: None if value == "bad" else datetime(2026, 1, 1, 9, 0),
            utc_to_local=lambda value: value.replace(tzinfo=None),
            local_to_utc=lambda value: value.replace(tzinfo=timezone.utc),
        )
        self.assertIsNone(safe_dt(ports, "bad"))
        expected = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(local_naive_to_utc(ports, datetime(2026, 1, 1, 9, 0, 0, 123456)), expected)
        self.assertEqual(utc_to_local_naive(ports, expected), datetime(2026, 1, 1, 9, 0))

    def test_transition_failure_is_rejected_without_mutating_candidate(self) -> None:
        services = modify_ordinary.OrdinaryModifyServices(
            field_changed=lambda old, new, field: old.get(field) != new.get(field),
            strip_quotes=lambda value: value,
            validate_anchor=lambda *_args: None,
            validate_omit=lambda *_args: None,
            reject_conflicting_types=lambda *_args: None,
            validate_chain_limits=lambda *_args: None,
            preserve_cp_offsets=lambda *_args: None,
            task_has_recurrence=lambda task: bool(str(task.get("anchor") or "").strip()),
            preserve_native_until=lambda *_args: False,
            validate_native_until=lambda *_args: None,
            validate_native_until_slots=lambda *_args: None,
            render_cp_adjustment=lambda *_args: None,
            render_timing_warning=lambda *_args: None,
            apply_transition=lambda *_args: (_ for _ in ()).throw(
                ValueError("chain identity unavailable")
            ),
            short_uuid=lambda value: str(value or "")[:8],
            recurrence_enabled_rows=lambda *_args: [],
            panel=lambda *_args, **_kwargs: None,
            render_disabled_summary=lambda *_args: None,
            semantic_diff_value=lambda old, new: f"{old} -> {new}",
            first_recurrence_target=lambda *_args: None,
            fmtlocal=lambda value: str(value),
            render_recurrence_updated=lambda *_args: None,
            print_task=lambda *_args: None,
        )
        candidate = {"uuid": "plain", "status": "pending", "anchor": "w:mon"}

        with self.assertRaisesRegex(modify_ordinary.RecurrenceActivationError, "chain identity unavailable"):
            modify_ordinary.handle_non_completion_modify(
                {"uuid": "plain", "status": "pending"},
                candidate,
                services=services,
                lifecycle=SimpleNamespace(
                    recurrence_setting_changes=lambda *_args: []
                ),
            )

        self.assertNotIn("chain", candidate)
        self.assertNotIn("chainID", candidate)

    def test_unexpected_transition_defect_propagates_to_hook_containment(self) -> None:
        services = SimpleNamespace(
            field_changed=lambda old, new, field: old.get(field) != new.get(field),
            strip_quotes=lambda value: value,
            validate_anchor=lambda *_args: None,
            validate_omit=lambda *_args: None,
            reject_conflicting_types=lambda *_args: None,
            validate_chain_limits=lambda *_args: None,
            preserve_cp_offsets=lambda *_args: None,
            task_has_recurrence=lambda _task: False,
            preserve_native_until=lambda *_args: None,
            validate_native_until=lambda *_args: None,
            validate_native_until_slots=lambda *_args: None,
            render_cp_adjustment=lambda *_args: None,
            render_timing_warning=lambda *_args: None,
            apply_transition=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("transition invariant defect")
            ),
        )

        with self.assertRaises(RuntimeError) as raised:
            modify_ordinary.handle_non_completion_modify(
                {},
                {},
                services=services,
                lifecycle=SimpleNamespace(recurrence_setting_changes=lambda *_args: []),
            )

        self.assertIs(type(raised.exception), RuntimeError)
        self.assertEqual(str(raised.exception), "transition invariant defect")

    def test_recurrence_feedback_calculation_propagates_internal_failures(self) -> None:
        services = SimpleNamespace(
            field_changed=lambda old, new, field: old.get(field) != new.get(field),
            strip_quotes=lambda value: value,
            validate_anchor=lambda *_args: None,
            validate_omit=lambda *_args: None,
            reject_conflicting_types=lambda *_args: None,
            validate_chain_limits=lambda *_args: None,
            preserve_cp_offsets=lambda *_args: None,
            task_has_recurrence=lambda _task: False,
            apply_transition=lambda *_args: None,
            render_timing_warning=lambda *_args: None,
            render_recurrence_updated=lambda *_args: None,
            print_task=lambda *_args: None,
        )
        lifecycle = SimpleNamespace(
            recurrence_setting_changes=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("recurrence feedback invariant failed")
            )
        )

        with self.assertRaisesRegex(RuntimeError, "recurrence feedback invariant failed"):
            modify_ordinary.handle_non_completion_modify(
                {"uuid": "task-1", "status": "pending"},
                {"uuid": "task-1", "status": "pending"},
                services=services,
                lifecycle=lifecycle,
            )

    def test_lifecycle_activation_requires_a_complete_unlinked_root_identity(self) -> None:
        short_uuid = lambda value: str(value or "").split("-")[0]
        invalid = (
            ({"anchor": "w:mon"}, "UUID is missing"),
            (
                {
                    "uuid": "11111111-0000-0000-0000-000000000001",
                    "anchor": "w:mon",
                    "prevLink": "aaaaaaaa",
                },
                "unlinked root",
            ),
            (
                {
                    "uuid": "11111111-0000-0000-0000-000000000001",
                    "anchor": "w:mon",
                    "link": 2,
                },
                "root link 1",
            ),
        )
        for fields, expected in invalid:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    modify_lifecycle.apply_nautical_transition(
                        {"status": "pending"},
                        {"status": "pending", **fields},
                        short_uuid=short_uuid,
                    )

        valid = {
            "uuid": "11111111-0000-0000-0000-000000000001",
            "status": "pending",
            "anchor": "w:mon",
        }
        transition = modify_lifecycle.apply_nautical_transition(
            {"status": "pending"}, valid, short_uuid=short_uuid
        )
        self.assertEqual(transition.state, "enabled")
        self.assertEqual(valid.get("chainID"), "11111111")
        self.assertEqual(valid.get("link"), 1)

    def test_datetime_parser_callback_has_stable_value_error_shape(self) -> None:
        from nautical_core.task_datetime import ConfiguredTaskDatetimeParser

        parser = ConfiguredTaskDatetimeParser(lambda value: None if value == "bad" else value)
        self.assertEqual(parser.parse(""), (None, None))
        parsed, error = parser.parse("bad")
        self.assertIsNone(parsed)
        self.assertIn("Unrecognized datetime", error)

    def test_anchor_validation_does_not_persist_unused_hints(self) -> None:
        calls = []

        ports = AnchorValidationPorts(
            lint=lambda _expr: ((), ()),
            validate_strict=lambda expr: calls.append(("validate", expr)),
            panel=lambda *_args, **_kwargs: None,
            is_astronomy_error=lambda _exc: False,
            astronomy_error_message=str,
            fail=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected failure")),
        )
        validate_anchor(ports, {}, {}, "w:mon")
        self.assertEqual(calls, [("validate", "w:mon")])


if __name__ == "__main__":
    unittest.main()
