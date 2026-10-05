from __future__ import annotations

from pathlib import Path
from datetime import datetime
from collections.abc import Callable, Sequence
from typing import Any, Callable as TypingCallable, get_args, get_type_hints
import unittest

import nautical_core.lifecycle.reconciliation as reconciliation
import nautical_core.tools.nautical_reconcile as reconcile_cli
from nautical_core.lifecycle.recovery_models import RecoveryResult
from nautical_core.lifecycle.application import (
    LifecycleApplicationOutcome,
    LifecycleApplicationService,
)
from nautical_core.reconcile_operator_service import (
    ReconcileRecoveryCallbacks,
    ReconcileRecoveryCoordinator,
)
from nautical_core.task_models import TaskObservation, TaskPayload


class ReconcileCallbackContractTests(unittest.TestCase):
    def test_reconcile_child_parser_callbacks_use_the_datetime_contract(self) -> None:
        expected = TypingCallable[[object], tuple[datetime | None, str | None]]

        for method in (
            reconciliation.LifecycleReconciliationService.plan,
            reconciliation.LifecycleReconciliationService.plan_typed,
            reconciliation.LifecycleReconciliationService.existing_children,
        ):
            with self.subTest(method=method.__name__):
                self.assertEqual(get_type_hints(method)["safe_parse_datetime"], expected)

    def test_hook_carriers_are_opaque_objects_at_reconciliation_boundary(self) -> None:
        for method in (
            reconciliation.LifecycleApplyOperations.configuration_state,
            reconciliation.CallbackLifecycleApplyOperations.configuration_state,
            reconciliation.LifecycleReconciliationService.plan,
            reconciliation.LifecycleReconciliationService.plan_typed,
        ):
            with self.subTest(method=method.__qualname__):
                self.assertIs(get_type_hints(method)["hook"], object)

        configuration_callback = get_type_hints(
            reconciliation.CallbackLifecycleApplyOperations
        )["configuration_callback"]
        self.assertEqual(get_args(configuration_callback)[0], [object])

    def test_reconciliation_service_does_not_retain_unused_unit_of_work(self) -> None:
        self.assertNotIn(
            "unit_of_work",
            reconciliation.LifecycleReconciliationService.__dataclass_fields__,
        )

    def test_lifecycle_reconciliation_application_boundary_is_typed(self) -> None:
        service_type = get_type_hints(
            reconciliation.LifecycleReconciliationService.application_service,
            globalns={**vars(reconciliation), "LifecycleApplicationService": LifecycleApplicationService},
        )["return"]
        self.assertIs(service_type, LifecycleApplicationService)

        execute_hints = get_type_hints(
            reconciliation.LifecycleReconciliationService.execute_lifecycle_plan,
            globalns={**vars(reconciliation), "LifecycleApplicationOutcome": LifecycleApplicationOutcome},
        )
        self.assertEqual(
            execute_hints["return"],
            tuple[LifecycleApplicationOutcome, LifecycleApplicationOutcome | None, str, dict[str, Any] | None],
        )

        terminal_hints = get_type_hints(
            reconciliation.LifecycleReconciliationService.apply_terminal_plan,
            globalns={**vars(reconciliation), "LifecycleApplicationOutcome": LifecycleApplicationOutcome},
        )
        self.assertIs(terminal_hints["return"], LifecycleApplicationOutcome)

    def test_parent_reconciliation_returns_the_existing_recovery_result(self) -> None:
        self.assertEqual(
            get_type_hints(
                reconciliation.LifecycleReconciliationService.apply_parent
            )["return"],
            tuple[RecoveryResult, str],
        )

    def test_verified_child_cache_uses_the_shared_task_payload_model(self) -> None:
        expected = dict[str, TaskPayload]
        owners = (
            reconciliation.LifecycleRecoveryOperations.apply_parent,
            reconciliation.ApplyParentCallback.__call__,
            reconciliation.LifecycleApplyOperations.execute_plan,
            reconciliation.CallbackLifecycleRecoveryOperations.apply_parent,
            reconciliation.LifecycleReconciliationService.execute_lifecycle_plan,
            reconciliation.LifecycleReconciliationService.apply_parent,
            reconcile_cli._execute_reconcile_lifecycle_plan,
            reconcile_cli._apply_parent_atomic,
        )

        for owner in owners:
            with self.subTest(owner=owner.__qualname__):
                globalns = {
                    **owner.__globals__,
                    "LifecycleApplicationOutcome": LifecycleApplicationOutcome,
                    "LifecycleApplicationService": LifecycleApplicationService,
                }
                hint = get_type_hints(owner, globalns=globalns)["verified_children"]
                alternatives = get_args(hint)
                payload_type = (
                    next(item for item in alternatives if item is not type(None))
                    if type(None) in alternatives
                    else hint
                )
                self.assertEqual(payload_type, expected)

    def test_reconcile_session_constructor_uses_concrete_owner_types(self) -> None:
        annotations = reconcile_cli._ReconcileSession.__init__.__annotations__
        expected = {
            "unit_of_work": "TaskwarriorUnitOfWork",
            "repository": "TaskReadRepository",
            "snapshot": "ReconcileSnapshotService",
            "control_plane": "OperatorControlPlane",
            "mutation_gateway": "TaskwarriorMutationService",
            "integrity_outbox": "LifecycleOutboxRepository",
            "lifecycle_service": "LifecycleReconciliationService",
            "lifecycle_application": "LifecycleApplicationService",
            "runtime_state": "_ReconcileRuntimeState",
            "datetime_parser": "TaskDatetimeParser",
        }
        for name, annotation in expected.items():
            with self.subTest(owner=name):
                self.assertEqual(annotations[name], annotation)

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

    def test_reconcile_recovery_datetime_and_result_contracts_are_concrete(self) -> None:
        from nautical_core.lifecycle.reconciliation import (
            CallbackLifecycleRecoveryOperations,
            LifecycleReconciliationService,
        )

        callback_hints = get_type_hints(ReconcileRecoveryCallbacks)
        callback_arguments, callback_result = get_args(callback_hints["terminal_error"])
        self.assertEqual(list(callback_arguments), [TaskObservation, datetime])
        self.assertIs(str, callback_result)
        coordinator_hints = get_type_hints(ReconcileRecoveryCoordinator.recover)
        self.assertIs(datetime, coordinator_hints["recovery_at"])
        service_hints = get_type_hints(LifecycleReconciliationService.recover_candidate)
        self.assertIs(datetime, service_hints["recovery_at"])
        result_item, = get_args(service_hints["return"])
        result_types = get_args(result_item)
        self.assertEqual(result_types, (RecoveryResult, str))
        operations_hints = get_type_hints(CallbackLifecycleRecoveryOperations.terminal_error)
        self.assertIs(datetime, operations_hints["recovery_at"])
        self.assertEqual(
            get_type_hints(LifecycleReconciliationService.preflight_wave)["parents"],
            Sequence[TaskObservation],
        )

    def test_recovery_outcome_callbacks_have_exact_task_and_result_shapes(self) -> None:
        hints = get_type_hints(ReconcileRecoveryCallbacks)
        expected = {
            "next_child": ((TaskObservation, str), TaskObservation),
            "terminal_error": ((TaskObservation, datetime), str),
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
