"""Error contracts for the reconcile operator boundary."""

import contextlib
from datetime import date, datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

import nautical_core.tools.nautical_reconcile as reconcile
import nautical_core as core
import nautical_core.modify_datetime_effects as datetime_effects
import nautical_core.timezone_facade as timezone_facade
from nautical_core.task_datetime import parser_for_core
from nautical_core.lifecycle.reconciliation import (
    LifecycleChildReadUnavailable,
    LifecycleReconciliationService,
)
from nautical_core.integration_models import (
    CommandFailureKind,
    FailureEvidence,
    TaskCommand,
    Unavailable,
)
from nautical_core.lifecycle.recovery_models import RecoveryStatus
from nautical_core.task_models import NauticalTask, TaskDraft, TaskObservation


class ReconcileErrorContracts(unittest.TestCase):
    def test_day_end_fallback_propagates_unexpected_timezone_adapter_errors(self) -> None:
        from nautical_core.chain_integrity_lifecycle import fallback_native_until_at_day_end

        observation = TaskObservation.from_mapping(
            {
                "uuid": "00000000-0000-4000-8000-000000003242",
                "status": "pending", "chain": "on", "chainID": "until-test",
                "link": 1, "due": "20260723T090000Z",
            },
            source_query="reconcile fallback adapter failure",
        )

        with self.assertRaisesRegex(RuntimeError, "timezone adapter defect"):
            fallback_native_until_at_day_end(
                observation,
                safe_parse_datetime=lambda _value: (
                    datetime(2026, 7, 23, 9, tzinfo=timezone.utc), None
                ),
                fmt_isoz=lambda _value: "unused",
                utc_to_local_naive=lambda _value: (_ for _ in ()).throw(
                    RuntimeError("timezone adapter defect")
                ),
                local_naive_to_utc=lambda _value: _value,
            )

    def test_relative_carry_verification_propagates_unexpected_datetime_errors(self) -> None:
        from nautical_core.chain_integrity_lifecycle import invalid_relative_carry_reason

        row = {
            "uuid": "00000000-0000-4000-8000-000000003243",
            "status": "pending", "chain": "on", "chainID": "until-test",
            "link": 1, "description": "carry fixture", "cp": "P7D",
            "due": "20260723T090000Z", "wait": "20260723T100000Z",
        }
        parent = TaskObservation.from_mapping(row, source_query="relative-carry-parent")
        child = TaskDraft.from_task(NauticalTask.from_observation(parent))
        parsed = datetime(2026, 7, 23, 9, tzinfo=timezone.utc)
        generation = SimpleNamespace(
            core=SimpleNamespace(
                utc_to_local_naive=lambda _value: (_ for _ in ()).throw(
                    RuntimeError("datetime adapter defect")
                )
            ),
            parse_datetime=lambda _value: (parsed, None),
        )

        with self.assertRaisesRegex(RuntimeError, "datetime adapter defect"):
            invalid_relative_carry_reason(
                parent, child, child_field="due", generation=generation
            )

    def test_native_until_carry_fallback_and_verification_contract(self) -> None:
        from nautical_core.chain_integrity_lifecycle import (
            fallback_native_until_at_day_end,
            invalid_native_until_reason,
            repair_native_until_from_previous,
        )

        parser = parser_for_core(core)
        ports = datetime_effects.datetime_effect_ports_for(SimpleNamespace(core=core))

        def stamp(day: int, hour: int) -> str:
            return core.fmt_isoz(core.build_local_datetime(date(2026, 7, day), (hour, 0)))

        def observation(**fields: object) -> TaskObservation:
            return TaskObservation.from_mapping(
                {
                    "uuid": "00000000-0000-4000-8000-000000003241",
                    "status": "pending", "chain": "on", "chainID": "until-test",
                    "link": 1, **fields,
                },
                source_query="reconcile native-until contract",
            )

        with patch.object(timezone_facade, "_local_timezone", timezone.utc):
            previous = observation(
                status="completed", due=stamp(20, 9), until=stamp(20, 23)
            )
            current = observation(due=stamp(22, 9), until=stamp(21, 23))
            self.assertTrue(invalid_native_until_reason(current, safe_parse_datetime=parser.parse))
            repaired, error = repair_native_until_from_previous(
                previous, current, kind="anchor",
                safe_parse_datetime=parser.parse,
                fmt_isoz=core.fmt_isoz,
                utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(ports, value),
                local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(ports, value),
            )
            self.assertIsNone(error)
            self.assertEqual(repaired, stamp(22, 23))

            fallback, error = fallback_native_until_at_day_end(
                observation(due=stamp(23, 9)),
                safe_parse_datetime=parser.parse,
                fmt_isoz=core.fmt_isoz,
                utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(ports, value),
                local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(ports, value),
            )
            self.assertIsNone(error)
            self.assertEqual(fallback, stamp(23, 23))

            late_fallback, error = fallback_native_until_at_day_end(
                observation(due=stamp(23, 23)),
                safe_parse_datetime=parser.parse,
                fmt_isoz=core.fmt_isoz,
                utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(ports, value),
                local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(ports, value),
            )
            self.assertIsNone(late_fallback)
            self.assertIn("at or after local 23:00", error or "")

            expected_until = stamp(23, 23)
            compact_expected = expected_until.replace("-", "").replace(":", "")
            self.assertTrue(
                reconcile._native_until_matches(
                    observation(until=compact_expected), expected_until,
                    SimpleNamespace(datetime_parser=parser),
                )
            )
            shifted_until = (parser.parse(expected_until)[0] + timedelta(hours=1))
            self.assertFalse(
                reconcile._native_until_matches(
                    observation(until=core.fmt_isoz(shifted_until)), expected_until,
                    SimpleNamespace(datetime_parser=parser),
                )
            )
            self.assertIn(
                "due",
                reconcile._native_until_guard_error(
                    observation(uuid="parent", due="20260801T090000Z"),
                    observation(uuid="parent", due="20260802T090000Z"),
                ) or "",
            )

    def test_apply_lease_is_exclusive_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            with reconcile._reconcile_apply_lock(taskdata) as first:
                self.assertTrue(first)
                with reconcile._reconcile_apply_lock(taskdata) as second:
                    self.assertFalse(second)
            with reconcile._reconcile_apply_lock(taskdata) as released:
                self.assertTrue(released)

    def test_apply_lease_conflict_returns_before_session_build(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            unit_of_work = SimpleNamespace(
                context=SimpleNamespace(
                    taskdata=taskdata,
                    command_prefix=("task",),
                )
            )
            output = io.StringIO()
            with reconcile._reconcile_apply_lock(taskdata) as held:
                self.assertTrue(held)
                with (
                    patch.object(reconcile, "_build_reconcile_session") as build_session,
                    contextlib.redirect_stdout(output),
                ):
                    result = reconcile.main(
                        ["--apply", "--json"], _unit_of_work=unit_of_work
                    )

            payload = json.loads(output.getvalue())
            self.assertEqual(result, 1)
            self.assertEqual(payload.get("stage"), "apply_lock")
            build_session.assert_not_called()

    def test_startup_config_failure_is_structured(self) -> None:
        args = SimpleNamespace(json=True, apply=True)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            result = reconcile._startup_failure(
                args, "taskdata_config", RuntimeError("invalid timezone")
            )

        payload = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertEqual(payload.get("configuration_status"), "unavailable")
        self.assertEqual(payload.get("configuration_drift"), "invalid timezone")

    def test_reconcile_plan_output_includes_safety_evidence(self) -> None:
        import nautical_core.reconcile_report as report
        from nautical_core.lifecycle.models import (
            LifecycleAction,
            LifecycleEvent,
            LifecycleIdentity,
            LifecyclePlan,
            ParentGuard,
            recurrence_fingerprint,
        )
        from nautical_core.lifecycle.recovery_models import RecoveryPlanResult

        parent = {
            "uuid": "11111111-0000-4000-8000-000000000001",
            "status": "completed",
            "description": "remote completion",
            "cp": "1d",
            "chain": "on",
            "chainID": "11111111",
            "link": 2,
            "due": "20260703T090000Z",
        }
        observation = TaskObservation.from_mapping(
            parent, source_query="reconcile output contract"
        )
        guard = ParentGuard(
            status="completed",
            chain="on",
            chain_id="11111111",
            link=2,
            recurrence_fingerprint=recurrence_fingerprint(parent),
            modified="",
        )
        identity = LifecycleIdentity(
            chain_id="11111111",
            parent_uuid=parent["uuid"],
            source_link=2,
            target_link=3,
            event=LifecycleEvent.ACTIVATE,
        )
        plan = RecoveryPlanResult(
            observation,
            LifecyclePlan(
                identity=identity,
                action=LifecycleAction.UPDATE_PARENT,
                parent_guard=guard,
            ),
            reason="next link already exists",
            child_short="22222222",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reconcile._print_plan(plan)
        rendered = output.getvalue()
        self.assertIn("backfill nextLink:", rendered)
        self.assertIn("reason: next link already exists", rendered)
        self.assertIn("existing child: 22222222", rendered)

        second_parent = {
            **parent,
            "uuid": "22222222-0000-0000-0000-000000000002",
            "link": 3,
        }
        partial = reconcile._recovery_terminal(
            second_parent,
            "expiration recovery hop limit reached at 2; native until has already elapsed",
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reconcile._print_recovery_group(
                [
                    (plan, report.describe_recovery_result(plan), "22222222"),
                    (partial, report.describe_recovery_result(partial), ""),
                ]
            )
        rendered = output.getvalue()
        self.assertIn("recover:", rendered)
        self.assertIn("advanced 1 occurrence", rendered)
        self.assertIn("result: partial", rendered)
        self.assertNotIn("spawn:", rendered)

    def test_reconcile_plan_enrichment_propagates_callback_defects(self) -> None:
        from nautical_core.lifecycle.models import (
            LifecycleAction,
            LifecycleEvent,
            LifecycleIdentity,
            LifecyclePlan,
            ParentGuard,
            recurrence_fingerprint,
        )
        from nautical_core.lifecycle.recovery_models import RecoveryPlanResult
        from nautical_core.reconcile_report import describe_plan, describe_recovery_result

        parent = {
            "uuid": "11111111-0000-4000-8000-000000000001",
            "status": "completed",
            "chain": "on",
            "chainID": "11111111",
            "link": 1,
            "cp": "1d",
        }
        observation = TaskObservation.from_mapping(
            parent, source_query="reconcile report boundary contract"
        )
        guard = ParentGuard(
            status="completed",
            chain="on",
            chain_id="11111111",
            link=1,
            recurrence_fingerprint=recurrence_fingerprint(parent),
            modified="",
        )
        identity = LifecycleIdentity(
            chain_id="11111111",
            parent_uuid=parent["uuid"],
            source_link=1,
            target_link=2,
            event=LifecycleEvent.COMPLETE,
        )
        plan = LifecyclePlan(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            child_payload=(("due", "20261004T090000Z"), ("until", "20261005T090000Z")),
        )
        result = RecoveryPlanResult(
            observation,
            plan,
            reason="next occurrence",
            child_due=datetime(2026, 10, 4, 9, tzinfo=timezone.utc),
        )

        evidence = describe_recovery_result(
            result,
            fmt_dt_local=lambda _value: (_ for _ in ()).throw(
                RuntimeError("local display unavailable")
            ),
        )
        self.assertEqual(evidence["child_due"], "2026-10-04 09:00:00+00:00")
        self.assertNotIn("child_local", evidence)

        with self.assertRaisesRegex(RuntimeError, "parse callback defect"):
            describe_plan(
                result,
                parse_until=lambda _value: (_ for _ in ()).throw(
                    RuntimeError("parse callback defect")
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "carry callback defect"):
            describe_plan(
                result,
                parse_until=lambda _value: (
                    datetime(2026, 10, 5, 9, tzinfo=timezone.utc),
                    None,
                ),
                describe_carry=lambda *_args: (_ for _ in ()).throw(
                    RuntimeError("carry callback defect")
                ),
            )

    def test_native_until_manual_review_is_not_a_hard_error(self) -> None:
        import nautical_core as core

        row = TaskObservation.from_mapping(
            {
                "uuid": "00000000-0000-4000-8000-000000003248",
                "status": "pending",
                "chain": "on",
                "chainID": "manual-until",
                "link": 1,
                "due": "20260723T230000Z",
                "until": "20260723T220000Z",
            },
            source_query="reconcile manual-until contract",
        )

        class ControlPlane:
            def audit_native_until(self, _rows: object, **_kwargs: object) -> object:
                return SimpleNamespace(
                    native_until=SimpleNamespace(
                        repairs=[{"action": "manual_review", "task": "00000000"}],
                        errors=[],
                    ),
                    candidates=[],
                )

        with patch.object(reconcile, "_reconcile_runtime_state", return_value=None):
            repairs, errors = reconcile._native_until_repairs(
                "task",
                SimpleNamespace(core=core),
                apply=False,
                snapshot=SimpleNamespace(active_rows=lambda: [row]),
                control_plane=ControlPlane(),
            )

        self.assertEqual(repairs, [{"action": "manual_review", "task": "00000000"}])
        self.assertEqual(errors, [])

    def test_parent_identity_errors_are_actionable(self) -> None:
        base = {
            "uuid": "11111111-0000-0000-0000-000000000001",
            "status": "completed",
            "chain": "on",
            "chainID": "chain001",
            "link": 2,
            "nextLink": "",
        }
        cases = (
            (dict(base, chainID=""), "parent chainID is missing"),
            (dict(base, link=""), "parent link is missing"),
            (
                dict(
                    base,
                    chainID="11111111-0000-0000-0000-000000000001",
                    link="",
                    prevLink="",
                ),
                "parent link is missing",
            ),
            (dict(base, link="not-a-number"), "parent link is invalid"),
            (dict(base, link=0), "parent link must be positive"),
        )

        for parent, expected in cases:
            with self.subTest(parent=parent), self.assertRaisesRegex(
                RuntimeError, expected
            ):
                reconcile._parent_guard_filters(parent)

    def test_reconcile_expired_pending_child_is_resumable_partial(self) -> None:
        parent = {"uuid": "11111111-0000-0000-0000-000000000001", "link": 1}

        result = reconcile._recovery_terminal(
            parent, "live recovery child native until has already elapsed"
        )

        self.assertEqual(result.status, RecoveryStatus.PARTIAL)
        self.assertIn("wait for Taskwarrior to mark the child deleted", result.reason)
        self.assertIn("rerun reconcile", result.reason)

    def test_reconcile_evidence_prefers_due_over_carried_scheduled(self) -> None:
        from nautical_core.chain_generation import AnchorChildDueResult, ChainGenerationService
        from nautical_core.chain_integrity_lifecycle import plan_recovery_decision
        from nautical_core.reconcile_report import describe_recovery_result
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask, TaskDraft
        from nautical_core.timeutil import fmt_isoz

        parent = TaskObservation.from_mapping(
            {
                "uuid": "11111111-0000-4000-8000-000000000001",
                "status": "completed",
                "description": "remote completion",
                "anchor": "w:mon@t=09:00,17:00",
                "anchor_mode": "skip",
                "chain": "on",
                "chainID": "11111111",
                "link": 1,
                "due": "20260706T060000Z",
                "scheduled": "20260706T050000Z",
                "end": "20260706T070000Z",
            },
            source_query="reconcile evidence contract",
        )

        class FakeCore:
            @staticmethod
            def coerce_int(value: object, default: int = 0) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

        class FakeGeneration(ChainGenerationService):
            def __init__(self) -> None:
                super().__init__(FakeCore())

            def safe_parse_datetime(self, _value: object) -> tuple[None, None]:
                return None, None

            def compute_anchor_child_due(
                self, _parent: object
            ) -> AnchorChildDueResult:
                return datetime(2026, 7, 6, 14, tzinfo=timezone.utc), {"target_field": "due"}, []

            def build_child_draft(
                self,
                task: NauticalTask,
                child_due: datetime,
                _child_field: str,
                next_link: int,
                parent_short: str,
                _kind: str,
                _cpmax: int,
                _until_dt: object,
            ) -> TaskDraft:
                values = {
                    "uuid": "22222222-0000-4000-8000-000000000002",
                    "description": "remote completion",
                    "status": "pending",
                    "chain": "on",
                    "chainID": task.observation.to_mapping()["chainID"],
                    "link": next_link,
                    "prevLink": parent_short,
                    "anchor": "w:mon@t=09:00,17:00",
                    "anchor_mode": "skip",
                    "due": fmt_isoz(child_due),
                    "scheduled": "20260706T130000Z",
                }
                child = NauticalTask.from_observation(
                    DEFAULT_TASK_CODEC.decode_row(
                        values, source_query="reconcile evidence child"
                    )
                )
                return TaskDraft.from_task(child)

        plan = plan_recovery_decision(
            parent, existing_children=[], hook=None, generation=FakeGeneration()
        )
        evidence = describe_recovery_result(plan)

        self.assertEqual(evidence.get("child_field"), "due")
        self.assertEqual(evidence.get("child_target"), "2026-07-06T14:00:00Z")

    def test_reconcile_plan_does_not_relabel_planning_fault_as_read_failure(self) -> None:
        parent = {"uuid": "00000000-0000-4000-8000-000000000002"}
        service = LifecycleReconciliationService(
            snapshot=SimpleNamespace(),
            repository=SimpleNamespace(),
            configuration_fingerprint="config",
            schedule_fingerprint="schedule",
        )

        with (
            patch.object(reconcile, "_configuration_state", return_value=("valid", "")),
            patch.object(
                LifecycleReconciliationService,
                "plan",
                side_effect=RuntimeError("planner invariant failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "planner invariant failed"):
                reconcile._plan_for_parent(
                    SimpleNamespace(),
                    parent,
                    generation=object(),
                    reconciliation_service=service,
                )

    def test_lifecycle_child_unavailability_uses_read_failure_type(self) -> None:
        evidence = FailureEvidence(
            command=TaskCommand(("task", "export"), "child slot", 1.0),
            kind=CommandFailureKind.TIMEOUT,
            returncode=-1,
            attempt=1,
            duration=1.0,
            retryable=True,
            detail="child slot query timed out",
        )
        service = LifecycleReconciliationService(
            snapshot=SimpleNamespace(),
            repository=SimpleNamespace(
                exact_child_slot=lambda *_args, **_kwargs: Unavailable("query", evidence)
            ),
            configuration_fingerprint="config",
            schedule_fingerprint="schedule",
        )
        parent = TaskObservation.from_mapping(
            {
                "uuid": "00000000-0000-4000-8000-000000000003",
                "status": "completed",
                "chain": "on",
                "chainID": "typed-read-failure",
                "link": 1,
            },
            source_query="reconcile-child-read-contract",
        )

        with self.assertRaisesRegex(
            LifecycleChildReadUnavailable, "child slot query timed out"
        ):
            service.existing_children(parent, safe_parse_datetime=lambda _value: (None, None))

    def test_recovery_child_lookup_does_not_reclassify_internal_faults(self) -> None:
        parent = TaskObservation.from_mapping(
            {"uuid": "00000000-0000-4000-8000-000000000001"},
            source_query="recovery-child-error-contract",
        )

        def broken_lookup(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("repository invariant failed")

        with patch.object(
            reconcile,
            "_repository",
            return_value=SimpleNamespace(by_uuid=broken_lookup),
        ):
            with self.assertRaisesRegex(RuntimeError, "repository invariant failed"):
                reconcile._next_recovery_child(parent, "child-uuid")

    def test_recovery_child_typed_unavailability_remains_retryable(self) -> None:
        parent = TaskObservation.from_mapping(
            {"uuid": "00000000-0000-4000-8000-000000000001"},
            source_query="recovery-child-error-contract",
        )
        evidence = FailureEvidence(
            command=TaskCommand(("task", "export"), "recovery child", 1.0),
            kind=CommandFailureKind.TIMEOUT,
            returncode=-1,
            attempt=1,
            duration=1.0,
            retryable=True,
            detail="query timed out",
        )

        with patch.object(
            reconcile,
            "_repository",
            return_value=SimpleNamespace(
                by_uuid=lambda *_args, **_kwargs: Unavailable("query", evidence)
            ),
        ):
            with self.assertRaisesRegex(
                reconcile._RecoveryLookupUnavailable, "query timed out"
            ):
                reconcile._next_recovery_child(parent, "child-uuid")

    def test_recovery_lookup_unavailability_stays_partial_not_error(self) -> None:
        service = LifecycleReconciliationService(
            snapshot=SimpleNamespace(),
            repository=SimpleNamespace(),
            configuration_fingerprint="config",
            schedule_fingerprint="schedule",
        )
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000004",
            "status": "completed",
            "chain": "on",
            "chainID": "recovery-read-classification",
            "link": 1,
        }
        partial_result = object()
        error_result = object()
        recovery_policy = SimpleNamespace(
            virtual_expired_child=lambda *_args, **_kwargs: (None, ""),
            terminal_error=lambda *_args: "",
        )

        with (
            patch.object(reconcile, "_recovery_policy", return_value=recovery_policy),
            patch.object(
                reconcile,
                "_apply_parent_atomic",
                side_effect=reconcile._RecoveryLookupUnavailable("child query timed out"),
            ),
            patch.object(reconcile, "_recovery_partial", return_value=partial_result),
            patch.object(reconcile, "_recovery_error", return_value=error_result),
        ):
            result = reconcile._reconcile_candidate(
                "task",
                SimpleNamespace(),
                parent,
                taskdata=Path("/tmp/reconcile-recovery-error-contract"),
                apply=True,
                max_expiration_hops=1,
                recovery_at=None,
                reconciliation_service=service,
            )

        self.assertIs(result[0][0], partial_result)

    def test_configuration_verification_fails_closed_on_unexpected_fault(self) -> None:
        def broken_verifier() -> dict[str, bool]:
            raise RuntimeError("malformed TOML")

        hook = SimpleNamespace(
            core=SimpleNamespace(configuration_drift=broken_verifier)
        )
        result = reconcile.configuration_verification(hook)

        self.assertEqual(result.status, "unavailable")
        self.assertIn("malformed TOML", result.reason)
        self.assertEqual(
            reconcile._configuration_state(hook), ("unavailable", result.reason)
        )

    def test_expiration_hop_limit_wraps_invalid_input_not_internal_faults(self) -> None:
        class BrokenIntegerConversion:
            def __int__(self) -> int:
                raise RuntimeError("integer conversion fault")

        with self.assertRaises(reconcile.argparse.ArgumentTypeError):
            reconcile._expiration_hop_limit("not-an-integer")
        with self.assertRaisesRegex(RuntimeError, "integer conversion fault"):
            reconcile._expiration_hop_limit(cast(str, BrokenIntegerConversion()))

    def test_failed_wave_preplan_is_diagnosed_and_candidate_retried_directly(self) -> None:
        taskdata = Path("/tmp/reconcile-error-contract")
        candidates = tuple(
            TaskObservation.from_mapping(
                {
                    "uuid": f"00000000-0000-4000-8000-00000000074{index}",
                    "status": "completed",
                    "chain": "on",
                    "chainID": f"error-contract-{index}",
                    "link": 1,
                },
                source_query="wave-fallback-contract",
            )
            for index in (0, 1)
        )
        lifecycle_service = SimpleNamespace(
            candidates=lambda: candidates, preflight_wave=lambda _rows: None
        )
        command_context = SimpleNamespace(command_budget=0)
        unit_of_work = SimpleNamespace(
            context=SimpleNamespace(
                taskdata=taskdata,
                command_prefix=("task",),
                configuration=SimpleNamespace(fingerprint="cfg", scheduler_fingerprint="sched"),
            ),
            commands=SimpleNamespace(
                calls=0,
                attempts=0,
                duration=0.0,
                failures=0,
                by_purpose={},
                context=command_context,
                budget_exceeded=False,
            ),
        )
        session = SimpleNamespace(
            snapshot=SimpleNamespace(_rows=None),
            control_plane=SimpleNamespace(drain_integrity=lambda *_args, **_kwargs: ()),
            mutation_gateway=object(),
            integrity_outbox=object(),
            lifecycle_service=lifecycle_service,
            lifecycle_application=object(),
            runtime_state=object(),
            audit_native_until=lambda *_args, **_kwargs: ([], [], "valid"),
        )
        calls: list[tuple[str, bool]] = []

        def candidate(*args: object, **kwargs: object) -> list[tuple[object, str]]:
            parent = args[2]
            assert isinstance(parent, dict)
            parent_uuid = str(parent["uuid"])
            applying = bool(kwargs["apply"])
            calls.append((parent_uuid, applying))
            if len(calls) == 1:
                raise RuntimeError("speculative planner failure")
            return []

        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}),
            patch.object(reconcile, "_build_reconcile_session", return_value=session),
            patch.object(reconcile, "_chain_generation_for_hook", return_value=object()),
            patch.object(reconcile, "_configuration_state", return_value=("valid", "")),
            patch.object(reconcile, "_reconcile_candidate", side_effect=candidate),
            patch.object(reconcile, "_repository", return_value=SimpleNamespace(metrics=lambda: {
                "calls": 0,
                "rows": 0,
                "seconds": 0.0,
                "slowest_seconds": 0.0,
            })),
            patch.object(reconcile, "_opportunistic_housekeeping", return_value={"status": "skipped"}),
            patch.object(reconcile, "render_result", return_value="{}"),
            contextlib.redirect_stderr(stderr),
        ):
            reconcile.main(
                ["--apply", "--json"],
                _apply_lease_held=True,
                _locked_taskdata=taskdata,
                _unit_of_work=unit_of_work,
            )

        self.assertIn((str(candidates[0].field("uuid").value), True), calls)
        self.assertIn("speculative planner failure", stderr.getvalue())

    def test_local_until_formatting_falls_back_to_raw_value(self) -> None:
        class BrokenParser:
            def parse(self, _value: object) -> tuple[None, str]:
                raise RuntimeError("parser unavailable")

        class ParsedValue:
            def parse(self, _value: object) -> tuple[object, None]:
                return object(), None

        raw = "2026-01-01T00:00:00Z"

        def fail_format(_value: object) -> str:
            raise RuntimeError("formatter unavailable")

        parser_failure = SimpleNamespace(
            datetime_parser=BrokenParser(), fmt_dt_local=lambda _value: "formatted"
        )
        formatter_failure = SimpleNamespace(
            datetime_parser=ParsedValue(), fmt_dt_local=fail_format
        )
        self.assertEqual(reconcile._format_local_until(parser_failure, raw), raw)
        self.assertEqual(reconcile._format_local_until(formatter_failure, raw), raw)

    def test_native_until_match_propagates_internal_parser_fault(self) -> None:
        class BrokenParser:
            def parse(self, _value: object) -> tuple[None, str]:
                raise RuntimeError("unexpected parser failure")

        task = TaskObservation.from_mapping(
            {"until": "2026-01-01T00:00:00Z"}, source_query="error-contract"
        )
        hook = SimpleNamespace(datetime_parser=BrokenParser())
        with self.assertRaisesRegex(RuntimeError, "unexpected parser failure"):
            reconcile._native_until_matches(task, "2026-01-02T00:00:00Z", hook)


if __name__ == "__main__":
    unittest.main()
