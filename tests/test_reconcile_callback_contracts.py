from __future__ import annotations

from pathlib import Path
import unittest

import nautical_core.lifecycle.reconciliation as reconciliation


class ReconcileCallbackContractTests(unittest.TestCase):
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
