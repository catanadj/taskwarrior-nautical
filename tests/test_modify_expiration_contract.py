from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, get_type_hints
import unittest
from unittest.mock import patch

import nautical_core.add_validation as add_validation
import nautical_core.modify_expiration as modify_expiration
from nautical_core.lifecycle.models import (
    LifecycleAction,
    LifecycleEvent,
    LifecycleIdentity,
    LifecyclePlan,
    ParentGuard,
)
from nautical_core.lifecycle.recovery_models import RecoveryPlanResult
from nautical_core.lifecycle.outbox import LifecycleOutboxError
from nautical_core.task_models import NauticalTask, TaskDraft, TaskObservation


class ModifyExpirationContractTests(unittest.TestCase):
    def test_expiration_service_dependencies_have_explicit_types(self) -> None:
        for service_type in (
            modify_expiration.ExpirationServices,
            modify_expiration.DeletedModifyServices,
        ):
            with self.subTest(service=service_type.__name__):
                dynamic_fields = [
                    name
                    for name, annotation in get_type_hints(service_type).items()
                    if annotation is Any
                ]
                self.assertEqual(dynamic_fields, [])

    def _deleted_services(self, expiration: object, *, warnings: list[tuple[dict, str]]) -> modify_expiration.DeletedModifyServices:
        return modify_expiration.DeletedModifyServices(
            expiration=expiration,
            terminal_chain_off=lambda *_args: self.fail("expiration must not stop the chain"),
            now_utc=lambda: datetime(2026, 7, 27, tzinfo=timezone.utc),
            end_chain_summary=lambda *_args, **_kwargs: self.fail("expiration must not render a stop summary"),
            format_root_and_age=lambda *_args: "root",
            short=lambda value: str(value),
            panel=lambda *_args, **_kwargs: None,
            diag=lambda _message: None,
            recovery_warning=lambda task, reason: warnings.append((task, reason)),
        )

    def _spawn_recovery_result(self) -> RecoveryPlanResult:
        parent_uuid = "00000000-0000-4000-8000-000000000111"
        parent = TaskObservation.from_mapping(
            {
                "uuid": parent_uuid,
                "status": "deleted",
                "chain": "on",
                "chainID": "expiration-chain",
                "link": 1,
                "cp": "1d",
                "due": "2026-07-27T09:00:00Z",
            },
            source_query="expiration contract",
        )
        child = NauticalTask.from_observation(
            TaskObservation.from_mapping(
                {
                    "uuid": "00000000-0000-4000-8000-000000000222",
                    "status": "pending",
                    "chain": "on",
                    "chainID": "expiration-chain",
                    "link": 2,
                    "prevLink": parent_uuid,
                    "description": "next occurrence",
                    "cp": "1d",
                    "due": "2026-07-28T09:00:00Z",
                },
                source_query="expiration contract child",
            )
        )
        plan = LifecyclePlan.from_draft(
            identity=LifecycleIdentity(
                "expiration-chain", parent_uuid, 1, 2, LifecycleEvent.EXPIRE
            ),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("deleted", "on", "expiration-chain", 1),
            draft=TaskDraft.from_task(child),
            parent_patch={"nextLink": "00000000-0000-4000-8000-000000000222"},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )
        return RecoveryPlanResult(parent, plan, child_due=datetime(2026, 7, 28, 9, tzinfo=timezone.utc))

    def test_expiration_disposition_delegates_to_recovery_owner(self) -> None:
        old = {"status": "pending", "chainID": "expiration-chain"}
        new = dict(old, status="deleted", until="20260726T235900Z", end="20260727T000000Z")
        expiration_services = object()
        warnings: list[tuple[dict, str]] = []
        called: list[tuple[dict, object]] = []
        evidence = SimpleNamespace(disposition=SimpleNamespace(value="expiration"), reason="expired")
        services = self._deleted_services(expiration_services, warnings=warnings)

        with (
            patch.object(modify_expiration, "classify_deleted_task", return_value=evidence),
            patch.object(
                modify_expiration,
                "handle_expired_deleted_modify",
                side_effect=lambda task, *, services: called.append((task, services)) or True,
            ),
        ):
            modify_expiration.handle_deleted_modify(old, new, services=services)

        self.assertEqual(called, [(new, expiration_services)])
        self.assertEqual(warnings, [])

    def test_expiration_panel_explains_native_until_carry(self) -> None:
        child_due = datetime(2026, 7, 27, 9, tzinfo=timezone.utc)
        child_until = datetime(2026, 8, 2, 23, 59, tzinfo=timezone.utc)
        rows: list[tuple[str, str]] = []
        core = SimpleNamespace(
            coerce_int=lambda value, default: int(value or default),
            fmt_dt_local=lambda value: value.isoformat(),
            _import_sibling=lambda _name: add_validation,
            to_local=lambda value: value,
        )

        def safe_parse_datetime(value: object) -> tuple[datetime | None, str | None]:
            if not isinstance(value, str):
                return None, "not text"
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")), None
            except ValueError as exc:
                return None, str(exc)

        plan = SimpleNamespace(
            plan=SimpleNamespace(
                action=LifecycleAction.SPAWN_CHILD,
                child_dict=lambda: {"until": child_until.isoformat()},
                identity=SimpleNamespace(target_link=2),
            ),
            child_due=child_due,
            reason="expired link missing next link",
        )
        services = SimpleNamespace(
            core=core,
            safe_parse_datetime=safe_parse_datetime,
            panel=lambda _title, panel_rows, **_kwargs: rows.extend(panel_rows),
        )

        modify_expiration._render_recovery_panel(
            {"description": "Take the trash out", "link": 1},
            plan,
            services=services,
            result="Next occurrence created",
            child_short="22222222",
        )

        expected_carry = add_validation.describe_native_until_carry(
            child_until, child_due, to_local=core.to_local,
        )
        self.assertIn(("Expiration", expected_carry), rows)
        self.assertTrue(any(label == "Next expires" for label, _value in rows))

    def test_optional_expiration_panel_does_not_hide_import_defects(self) -> None:
        child_due = datetime(2026, 7, 27, 9, tzinfo=timezone.utc)
        child_until = datetime(2026, 7, 28, 9, tzinfo=timezone.utc)
        core = SimpleNamespace(
            coerce_int=lambda value, default: int(value or default),
            fmt_dt_local=lambda value: value.isoformat(),
            _import_sibling=lambda _name: (_ for _ in ()).throw(
                RuntimeError("presentation import defect")
            ),
            to_local=lambda value: value,
        )
        plan = SimpleNamespace(
            plan=SimpleNamespace(
                action=LifecycleAction.SPAWN_CHILD,
                child_dict=lambda: {"until": child_until.isoformat()},
            ),
            child_due=child_due,
            next_link=2,
            reason="",
        )
        services = SimpleNamespace(
            core=core,
            safe_parse_datetime=lambda _value: (child_until, None),
            panel=lambda *_args, **_kwargs: None,
        )

        with self.assertRaisesRegex(RuntimeError, "presentation import defect"):
            modify_expiration._render_recovery_panel(
                {"link": 1}, plan, services=services
            )

    def test_unexpected_expiration_recovery_failure_propagates_without_stopping_chain(self) -> None:
        old = {
            "status": "pending",
            "chainID": "expiration-chain",
            "chain": "on",
        }
        new = dict(old, status="deleted", until="20260726T235900Z", end="20260727T000000Z")
        warnings: list[tuple[dict, str]] = []
        evidence = SimpleNamespace(disposition=SimpleNamespace(value="expiration"), reason="expired")
        services = self._deleted_services(object(), warnings=warnings)

        with (
            patch.object(modify_expiration, "classify_deleted_task", return_value=evidence),
            patch.object(
                modify_expiration,
                "handle_expired_deleted_modify",
                side_effect=RuntimeError("missing recovery module"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing recovery module"):
                modify_expiration.handle_deleted_modify(old, new, services=services)

        self.assertEqual(new["chain"], "on")
        self.assertEqual(warnings, [])

    def test_malformed_expiration_task_is_reported_as_recovery_warning(self) -> None:
        warnings: list[tuple[str, list[tuple[str, str]], dict[str, str]]] = []
        diagnostics: list[str] = []
        services = SimpleNamespace(
            reconcile=SimpleNamespace(),
            safe_parse_datetime=lambda _value: (None, "invalid"),
            compute_anchor_child_due=lambda _task: None,
            compute_cp_child_due=lambda _task: None,
            build_child_draft=lambda *_args: None,
            stage_recovery_plan=lambda _plan: (False, "unavailable"),
            panel=lambda title, rows, **kwargs: warnings.append((title, rows, kwargs)),
            short=lambda value: str(value or ""),
            diag=diagnostics.append,
        )
        task = {"uuid": object(), "status": "deleted", "chain": "on"}

        handled = modify_expiration.handle_expired_deleted_modify(task, services=services)

        self.assertTrue(handled)
        self.assertTrue(any("could not be validated" in str(rows) for _, rows, _ in warnings))
        self.assertTrue(any("expiration recovery task decode failed" in item for item in diagnostics))

    def test_expiration_task_decode_does_not_hide_unexpected_runtime_failure(self) -> None:
        services = SimpleNamespace(
            reconcile=SimpleNamespace(),
            diag=lambda _message: None,
            panel=lambda *_args, **_kwargs: None,
            short=lambda value: str(value or ""),
        )
        task = {"uuid": "valid-shape", "status": "deleted"}

        with patch.object(
            modify_expiration,
            "DEFAULT_TASK_CODEC",
            SimpleNamespace(
                decode_row=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("codec defect")
                )
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "codec defect"):
                modify_expiration.handle_expired_deleted_modify(task, services=services)

    def test_malformed_deleted_task_disposition_uses_safe_recovery_warning(self) -> None:
        warnings: list[tuple[dict, str]] = []
        services = self._deleted_services(object(), warnings=warnings)
        old = {"status": "pending", "chainID": "chain-1"}
        new = {"status": "deleted", "chainID": "chain-1", "uuid": object()}

        modify_expiration.handle_deleted_modify(old, new, services=services)

        self.assertEqual(len(warnings), 1)
        self.assertIn("could not be classified safely", warnings[0][1])

    def test_deleted_task_disposition_does_not_hide_unexpected_service_failure(self) -> None:
        services = self._deleted_services(object(), warnings=[])
        old = {"status": "pending", "chainID": "chain-1"}
        new = {"status": "deleted", "chainID": "chain-1"}

        with patch.object(
            modify_expiration,
            "classify_deleted_task",
            side_effect=RuntimeError("classification defect"),
        ):
            with self.assertRaisesRegex(RuntimeError, "classification defect"):
                modify_expiration.handle_deleted_modify(old, new, services=services)

    def test_outbox_failure_during_expiration_stage_still_warns_and_defers(self) -> None:
        warnings: list[tuple[str, list[tuple[str, str]], dict[str, str]]] = []
        diagnostics: list[str] = []
        recovery = self._spawn_recovery_result()
        services = SimpleNamespace(
            core=SimpleNamespace(),
            reconcile=SimpleNamespace(
                is_orphan_expiration_candidate=lambda *_args, **_kwargs: True,
                plan_recovery_decision=lambda *_args, **_kwargs: recovery,
            ),
            safe_parse_datetime=lambda _value: (None, "invalid"),
            compute_anchor_child_due=lambda _task: (None, None, None),
            compute_cp_child_due=lambda _task: (None, None),
            build_child_draft=lambda *_args, **_kwargs: None,
            stage_recovery_plan=lambda _plan: (_ for _ in ()).throw(
                LifecycleOutboxError("outbox unavailable")
            ),
            panel=lambda title, rows, **kwargs: warnings.append((title, rows, kwargs)),
            short=lambda value: str(value or ""),
            diag=diagnostics.append,
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "status": "deleted",
            "chain": "on",
            "chainID": "expiration-chain",
            "link": 1,
            "cp": "1d",
            "due": "2026-07-27T09:00:00Z",
        }

        handled = modify_expiration.handle_expired_deleted_modify(task, services=services)

        self.assertTrue(handled)
        self.assertTrue(any("expiration lifecycle staging failed" in item for item in diagnostics))
        self.assertTrue(any("could not be staged" in str(rows) for _, rows, _ in warnings))

    def test_expiration_stage_does_not_hide_unexpected_failure(self) -> None:
        recovery = self._spawn_recovery_result()
        services = SimpleNamespace(
            core=SimpleNamespace(),
            reconcile=SimpleNamespace(
                is_orphan_expiration_candidate=lambda *_args, **_kwargs: True,
                plan_recovery_decision=lambda *_args, **_kwargs: recovery,
            ),
            safe_parse_datetime=lambda _value: (None, "invalid"),
            compute_anchor_child_due=lambda _task: (None, None, None),
            compute_cp_child_due=lambda _task: (None, None),
            build_child_draft=lambda *_args, **_kwargs: None,
            stage_recovery_plan=lambda _plan: (_ for _ in ()).throw(
                RuntimeError("staging defect")
            ),
            panel=lambda *_args, **_kwargs: None,
            short=lambda value: str(value or ""),
            diag=lambda _message: None,
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "status": "deleted",
            "chain": "on",
            "chainID": "expiration-chain",
            "link": 1,
            "cp": "1d",
            "due": "2026-07-27T09:00:00Z",
        }

        with self.assertRaisesRegex(RuntimeError, "staging defect"):
            modify_expiration.handle_expired_deleted_modify(task, services=services)


if __name__ == "__main__":
    unittest.main()
