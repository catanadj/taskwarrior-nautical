from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections.abc import Callable
from typing import Any, get_args, get_type_hints
import unittest

import nautical_core.lifecycle.reconciliation as reconciliation
from nautical_core.lifecycle.recovery_models import RecoveryResult
from nautical_core.reconcile_operator_service import ReconcileRecoveryCallbacks
from nautical_core.task_models import TaskObservation, TaskPayload


class ReconcileCallbackContractTests(unittest.TestCase):
    def test_reconcile_report_enrichment_uses_datetime_callback_contracts(self) -> None:
        from nautical_core.reconcile_report import describe_plan, describe_recovery_result

        plan_hints = get_type_hints(describe_plan)
        self.assertEqual(plan_hints["fmt_dt_local"], Callable[[datetime], str] | None)
        self.assertEqual(
            plan_hints["parse_until"],
            Callable[[object], tuple[datetime | None, str | None]] | None,
        )
        self.assertEqual(
            plan_hints["describe_carry"],
            Callable[[datetime, datetime], str | None] | None,
        )
        self.assertEqual(
            get_type_hints(describe_recovery_result)["fmt_dt_local"],
            Callable[[datetime], str] | None,
        )

    def test_lifecycle_apply_callbacks_do_not_accept_untyped_signatures(self) -> None:
        hints = get_type_hints(reconciliation.CallbackLifecycleApplyOperations)

        for name in (
            "configuration_callback",
            "refresh_callback",
            "execute_callback",
            "terminal_callback",
            "lock_callback",
        ):
            with self.subTest(callback=name):
                callback_types = get_args(hints[name])
                if callback_types:
                    self.assertIsNot(callback_types[0], Ellipsis)
                else:
                    call_hints = get_type_hints(hints[name].__call__)
                    self.assertGreater(len(call_hints), 2)
                    self.assertNotIn(Any, call_hints.values())

    def test_lifecycle_recovery_policy_uses_datetime_boundary_types(self) -> None:
        hints = get_type_hints(reconciliation.LifecycleRecoveryPolicy)
        parse_arguments, parse_result = get_args(hints["parse_datetime"])
        compare_arguments, compare_result = get_args(hints["compare_datetimes"])

        self.assertEqual([object], parse_arguments)
        self.assertEqual((datetime | None, str | None), get_args(parse_result))
        self.assertEqual([datetime, datetime], compare_arguments)
        self.assertIs(int, compare_result)

    def test_recovery_outcome_callbacks_have_exact_task_and_result_shapes(self) -> None:
        hints = get_type_hints(ReconcileRecoveryCallbacks)
        expected = {
            "next_child": ((TaskObservation, str), TaskObservation),
            "terminal_error": ((TaskObservation, Any), str),
            "recovery_error": ((TaskPayload, str), RecoveryResult),
            "recovery_partial": ((TaskPayload, str), RecoveryResult),
            "recovery_manual_review": ((TaskPayload, str), RecoveryResult),
            "recovery_terminal": ((TaskPayload, str), RecoveryResult),
            "recovery_exception": ((TaskPayload, Exception), RecoveryResult),
        }

        for name, (expected_arguments, expected_result) in expected.items():
            with self.subTest(callback=name):
                callback_arguments, callback_result = get_args(hints[name])
                self.assertIsNot(callback_arguments, Ellipsis)
                self.assertEqual(tuple(callback_arguments), expected_arguments)
                self.assertIs(callback_result, expected_result)

    def test_apply_parent_callback_rejects_unrecognized_keyword(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []

        def apply_parent(parent: object, **kwargs: object) -> tuple[object, str]:
            calls.append((parent, kwargs))
            return object(), "child"

        operation = reconciliation.CallbackLifecycleRecoveryOperations(
            apply_parent_callback=apply_parent,
            plan_parent_callback=lambda *_args, **_kwargs: object(),
            next_child_callback=lambda *_args: object(),
            virtual_child_callback=lambda *_args, **_kwargs: (None, ""),
            terminal_error_callback=lambda *_args: "",
            is_orphan_deleted_callback=lambda *_args: False,
            recovery_error_callback=lambda *_args: object(),
            recovery_partial_callback=lambda *_args: object(),
            recovery_manual_review_callback=lambda *_args: object(),
            recovery_terminal_callback=lambda *_args: object(),
            recovery_exception_callback=lambda *_args: object(),
        )

        with self.assertRaises(TypeError):
            operation.apply_parent(
                {"uuid": "parent"},
                taskdata=Path("/tmp/taskdata"),
                lease_held=False,
                verified_children={},
                generation=None,
                unrecognized=True,
            )

        self.assertEqual([], calls)

    def test_plan_parent_callback_rejects_unrecognized_keyword(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []

        def plan_parent(parent: object, **kwargs: object) -> object:
            calls.append((parent, kwargs))
            return object()

        operation = reconciliation.CallbackLifecycleRecoveryOperations(
            apply_parent_callback=lambda *_args, **_kwargs: (object(), ""),
            plan_parent_callback=plan_parent,
            next_child_callback=lambda *_args: object(),
            virtual_child_callback=lambda *_args, **_kwargs: (None, ""),
            terminal_error_callback=lambda *_args: "",
            is_orphan_deleted_callback=lambda *_args: False,
            recovery_error_callback=lambda *_args: object(),
            recovery_partial_callback=lambda *_args: object(),
            recovery_manual_review_callback=lambda *_args: object(),
            recovery_terminal_callback=lambda *_args: object(),
            recovery_exception_callback=lambda *_args: object(),
        )

        with self.assertRaises(TypeError):
            operation.plan_parent({"uuid": "parent"}, generation=None, unrecognized=True)

        self.assertEqual([], calls)

    def test_virtual_child_callback_rejects_unrecognized_keyword(self) -> None:
        calls: list[tuple[object, dict[str, object]]] = []

        def virtual_child(plan: object, **kwargs: object) -> tuple[None, str]:
            calls.append((plan, kwargs))
            return None, ""

        operation = reconciliation.CallbackLifecycleRecoveryOperations(
            apply_parent_callback=lambda *_args, **_kwargs: (object(), ""),
            plan_parent_callback=lambda *_args, **_kwargs: object(),
            next_child_callback=lambda *_args: object(),
            virtual_child_callback=virtual_child,
            terminal_error_callback=lambda *_args: "",
            is_orphan_deleted_callback=lambda *_args: False,
            recovery_error_callback=lambda *_args: object(),
            recovery_partial_callback=lambda *_args: object(),
            recovery_manual_review_callback=lambda *_args: object(),
            recovery_terminal_callback=lambda *_args: object(),
            recovery_exception_callback=lambda *_args: object(),
        )

        with self.assertRaises(TypeError):
            operation.virtual_child(
                object(), parent=object(), recovery_at=object(), unrecognized=True
            )

        self.assertEqual([], calls)


if __name__ == "__main__":
    unittest.main()
