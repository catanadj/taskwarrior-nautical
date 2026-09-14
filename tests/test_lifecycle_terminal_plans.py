from __future__ import annotations

import unittest

from nautical_core.lifecycle_models import (
    DeletionDisposition,
    ExecutionStage,
    LifecycleAction,
    LifecycleContractError,
    LifecycleEvent,
    LifecycleIdentity,
    LifecycleOutcome,
    LifecycleOutcomeKind,
    LifecyclePlan,
    ParentGuard,
    TaskLifecycleState,
    TaskSnapshot,
)
from nautical_core.lifecycle_outbox import OutboxProcessingState
from nautical_core.lifecycle_recovery_models import RecoveryPlanResult, RecoveryRefusal, RecoveryStatus
from nautical_core.lifecycle_planner import LifecyclePlanner, RecurrenceCandidate, terminal_plan_for_snapshot
from nautical_core.chain_integrity_lifecycle import deleted_chain_disposition
from nautical_core.reconcile_report import describe_recovery_result
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.task_models import NauticalTask, TaskDraft
from nautical_core.integration_models import GuardTimestamp, GuardTimestampField, MutationGuard
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService


def task_snapshot(row: dict[str, object]) -> TaskSnapshot:
    observation = DEFAULT_TASK_CODEC.decode_row(row, source_query="lifecycle-pure-contract")
    return TaskSnapshot.from_observation(observation)


def task_draft(row: dict[str, object]) -> TaskDraft:
    observation = DEFAULT_TASK_CODEC.decode_row(row, source_query="lifecycle-pure-contract")
    task = NauticalTask.from_observation(observation)
    target_field = "due" if task.temporal.due is not None else "scheduled"
    target = task.temporal.due or task.temporal.scheduled
    if target is None:
        raise AssertionError("typed child fixture requires a target")
    excluded = {
        "id", "uuid", "status", "modified", "end", "chainID", "link", "prevLink", "nextLink",
        "description", "chain", "anchor", "anchor_file", "anchor_mode", "cp", "omit", "omit_file",
        "bc", "chainMax", "chainUntil", "due", "scheduled",
    }
    values = observation.to_mapping()
    return TaskDraft(
        identity=task.identity,
        description=task.description,
        recurrence=task.recurrence,
        target=target,
        fields={key: value for key, value in values.items() if key not in excluded},
        target_field=target_field,
    )


def snapshot() -> TaskSnapshot:
    observation = DEFAULT_TASK_CODEC.decode_row(
        {
            "uuid": "11111111-1111-4111-8111-111111111111",
            "status": "pending",
            "chain": "on",
            "chainID": "abcd1234",
            "link": 4,
            "anchor": "w:mon",
            "due": "20260824T090000Z",
        },
        source_query="terminal-plan-test",
    )
    return TaskSnapshot.from_observation(observation)


class ExhaustedService:
    def next_candidate(self, *_args: object, **_kwargs: object) -> RecurrenceCandidate:
        return RecurrenceCandidate(child_due=None, terminal_reason="scheduler_exhausted")

    def build_child(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("an exhausted candidate must not build a child")


class LifecycleTerminalPlanTests(unittest.TestCase):
    def test_parent_mutation_guard_uses_stable_terminal_timestamp(self) -> None:
        guard = MutationGuard(
            task_uuid="00000000-0000-4000-8000-000000000777",
            status="completed",
            chain_id="terminal1",
            link=1,
            recurrence_identity="rf1-terminal",
            timestamps=(
                GuardTimestamp(GuardTimestampField.END, "20260824T123939Z"),
            ),
            expected_mutation_epoch=0,
        )

        selectors = TaskwarriorMutationService._selectors(guard)

        self.assertIn("end:20260824T123939Z", selectors)
        self.assertFalse(any(item.startswith("modified:") for item in selectors))

    def test_recurrence_fingerprint_ignores_formatting_but_tracks_schedule_changes(self) -> None:
        from datetime import datetime, timezone
        from nautical_core.lifecycle_models import recurrence_fingerprint

        def parse_datetime(value: object):
            text = str(value).strip()
            if text in ("2026-08-13T09:00:00Z", "20260813T090000Z"):
                return datetime(2026, 8, 13, 9, tzinfo=timezone.utc)
            return None

        base = {
            "anchor": " w:mon ",
            "anchor_mode": "SKIP",
            "chainMax": "5",
            "due": "2026-08-13T09:00:00Z",
            "description": "presentation",
            "project": "same-plan",
        }
        equivalent = {
            **base,
            "anchor": "w:mon",
            "anchor_mode": "skip",
            "chainMax": 5,
            "due": "20260813T090000Z",
            "description": "renamed",
            "modified": "20260813T100000Z",
        }
        self.assertEqual(
            recurrence_fingerprint(base, parse_datetime=parse_datetime),
            recurrence_fingerprint(equivalent, parse_datetime=parse_datetime),
        )
        changed = {**equivalent, "anchor": "w:tue"}
        self.assertNotEqual(
            recurrence_fingerprint(base, parse_datetime=parse_datetime),
            recurrence_fingerprint(changed, parse_datetime=parse_datetime),
        )

    def test_lifecycle_planner_is_pure_and_deterministic(self) -> None:
        from nautical_core.lifecycle_planner import (
            LifecyclePlanningError,
            LifecyclePreflight,
        )

        source = {
            "uuid": "00000000-0000-4000-8000-000000000501",
            "status": "completed",
            "chain": "on",
            "chainID": "chain-1",
            "link": 4,
            "anchor": "w:mon",
        }
        lifecycle_snapshot = task_snapshot(source)

        def build_child(task: TaskSnapshot, event: LifecycleEvent) -> TaskDraft:
            self.assertIs(event, LifecycleEvent.COMPLETE)
            self.assertEqual(task.to_dict(), source)
            return task_draft({
                **source,
                "uuid": "00000000-0000-4000-8000-000000000502",
                "description": "next",
                "status": "pending",
                "link": 5,
                "prevLink": source["uuid"][:8],
                "due": "20260824T090000Z",
            })

        class Recurrence:
            def next_candidate(self, _snapshot: TaskSnapshot, _event: LifecycleEvent, _kind: str, _next_link: int) -> RecurrenceCandidate:
                return RecurrenceCandidate(child_due="20260824T090000Z", metadata=(("target_field", "due"),))

            def build_child(self, child_snapshot: TaskSnapshot, event: LifecycleEvent, _candidate: RecurrenceCandidate, _next_link: int) -> TaskDraft:
                return build_child(child_snapshot, event)

        planner = LifecyclePlanner(
            {"scheduler_fingerprint": "fp-1"},
            recurrence_service=Recurrence(),
        )
        first = planner.plan(lifecycle_snapshot, LifecycleEvent.COMPLETE)
        second = planner.plan(lifecycle_snapshot, LifecycleEvent.COMPLETE)
        self.assertEqual(first, second)
        self.assertIs(first.action, LifecycleAction.SPAWN_CHILD)
        self.assertEqual(first.child_dict()["uuid"], "00000000-0000-4000-8000-000000000502")
        self.assertTrue(first.parent_guard.recurrence_fingerprint.startswith("rf1-"))
        self.assertEqual(source, lifecycle_snapshot.to_dict())
        preflight = LifecyclePreflight.from_context(
            base_link=4,
            next_link=5,
            kind="anchor",
            chain_id="chain-1",
        )
        checked = planner.plan(lifecycle_snapshot, LifecycleEvent.COMPLETE, preflight=preflight)
        self.assertEqual(checked, first)
        with self.assertRaisesRegex(LifecyclePlanningError, "scheduled carry"):
            planner.plan(
                lifecycle_snapshot,
                LifecycleEvent.COMPLETE,
                carry_validator=lambda _snapshot, _child, _candidate: "scheduled carry is missing",
            )
        with self.assertRaisesRegex(LifecyclePlanningError, "adjacent"):
            planner.plan(
                lifecycle_snapshot,
                LifecycleEvent.COMPLETE,
                preflight=LifecyclePreflight.from_context(
                    base_link=4,
                    next_link=6,
                    kind="anchor",
                    chain_id="chain-1",
                ),
            )

        terminal = planner.plan(lifecycle_snapshot, LifecycleEvent.CHAIN_UNTIL)
        self.assertIs(terminal.action, LifecycleAction.FINALIZE_CHAIN)
        for event in (
            LifecycleEvent.DISABLE,
            LifecycleEvent.MANUAL_DELETE,
            LifecycleEvent.CHAIN_MAX,
            LifecycleEvent.CHAIN_UNTIL,
            LifecycleEvent.COMPLETE,
            LifecycleEvent.EXPIRE,
        ):
            terminal = terminal_plan_for_snapshot(lifecycle_snapshot, event)
            self.assertEqual(terminal.parent_patch_dict(), {"chain": "off"})
            self.assertIs(terminal.identity.event, event)
        linked_snapshot = task_snapshot({**source, "nextLink": "child123"})
        with self.assertRaisesRegex(LifecyclePlanningError, "persisted successor"):
            terminal_plan_for_snapshot(linked_snapshot, LifecycleEvent.CHAIN_UNTIL)
        retained = terminal_plan_for_snapshot(linked_snapshot, LifecycleEvent.MANUAL_DELETE)
        self.assertIs(retained.action, LifecycleAction.DISABLE_CHAIN)
        activation = planner.plan(lifecycle_snapshot, LifecycleEvent.RESUME)
        self.assertEqual(activation.parent_patch_dict(), {"chain": "on"})

        with self.assertRaises(LifecyclePlanningError):
            LifecyclePlanner({"scheduler_fingerprint": "fp-1"}).plan(
                task_snapshot({"uuid": "00000000-0000-4000-8000-000000000510", "status": "pending", "link": 1, "anchor": "w:mon"}),
                LifecycleEvent.COMPLETE,
            )

    def test_lifecycle_terminal_policy_routes_all_terminal_events_through_one_patch(self) -> None:
        from nautical_core.lifecycle_planner import LifecyclePlanningError
        from nautical_core.modify_lifecycle import apply_terminal_transition

        events = (
            LifecycleEvent.DISABLE,
            LifecycleEvent.MANUAL_DELETE,
            LifecycleEvent.CHAIN_MAX,
            LifecycleEvent.CHAIN_UNTIL,
            LifecycleEvent.COMPLETE,
            LifecycleEvent.EXPIRE,
        )
        for event in events:
            with self.subTest(event=event):
                task = {
                    "uuid": "77777777-0000-0000-0000-000000000007",
                    "status": "deleted" if event is LifecycleEvent.MANUAL_DELETE else "completed",
                    "chain": "on",
                    "chainID": "terminal-policy",
                    "link": 7,
                }
                self.assertTrue(apply_terminal_transition(task, event))
                self.assertEqual(task["chain"], "off")
                self.assertFalse(apply_terminal_transition(task, event))

        linked = {
            "uuid": "88888888-0000-0000-0000-000000000008",
            "status": "completed",
            "chain": "on",
            "chainID": "terminal-policy-linked",
            "link": 8,
            "nextLink": "successor",
        }
        with self.assertRaisesRegex(LifecyclePlanningError, "persisted successor"):
            terminal_plan_for_snapshot(task_snapshot(linked), LifecycleEvent.CHAIN_UNTIL)
        manual_plan = terminal_plan_for_snapshot(
            task_snapshot({**linked, "status": "deleted"}),
            LifecycleEvent.MANUAL_DELETE,
        )
        self.assertIs(manual_plan.action, LifecycleAction.DISABLE_CHAIN)

    def test_lifecycle_models_preserve_draft_identity_and_outcome_contracts(self) -> None:
        guard = ParentGuard("pending", "on", "chain-1", 4, "fp-1")
        identity = LifecycleIdentity("chain-1", "parent-uuid", 4, 5, LifecycleEvent.COMPLETE)
        draft = TaskDraft.from_task(
            NauticalTask.from_observation(
                DEFAULT_TASK_CODEC.decode_row(
                    {
                        "uuid": "22222222-0000-4000-8000-000000000902",
                        "description": "next",
                        "status": "pending",
                        "chain": "on",
                        "chainID": "chain-1",
                        "link": 5,
                        "prevLink": "11111111",
                        "cp": "1d",
                        "due": "20260824T090000Z",
                        "nested": {"unicode": "Répéter 🌊"},
                    },
                    source_query="lifecycle-model-contract",
                )
            )
        )
        plan = LifecyclePlan.from_draft(
            identity=identity,
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=guard,
            draft=draft,
            parent_patch={"nextLink": "child-uuid"},
            expected_postconditions=("child_exists", "parent_linked"),
        )

        self.assertEqual(plan.child_dict()["nested"], {"unicode": "Répéter 🌊"})
        self.assertEqual(plan.parent_patch_dict(), {"nextLink": "child-uuid"})
        self.assertTrue(plan.identity.key.endswith(":complete"))
        self.assertNotEqual(TaskLifecycleState.ACTIVE.value, OutboxProcessingState.READY.value)

        outcome = LifecycleOutcome(
            LifecycleOutcomeKind.APPLIED,
            ExecutionStage.FINALIZED,
            identity,
            "child imported and parent linked",
        )
        self.assertIs(outcome.kind, LifecycleOutcomeKind.APPLIED)

    def test_lifecycle_models_reject_invalid_transition_shapes(self) -> None:
        guard = ParentGuard("pending", "on", "chain-1", 4, "fp-1")
        identity = LifecycleIdentity("chain-1", "parent-uuid", 4, 5, LifecycleEvent.COMPLETE)
        invalid_cases = (
            lambda: LifecycleIdentity("", "parent-uuid", 0, 1, LifecycleEvent.COMPLETE),
            lambda: LifecycleIdentity("chain-1", "parent-uuid", 2, 2, LifecycleEvent.COMPLETE),
            lambda: LifecyclePlan(
                LifecycleIdentity("chain-1", "parent-uuid", 4, None, LifecycleEvent.ACTIVATE),
                LifecycleAction.FINALIZE_CHAIN,
                guard,
            ),
            lambda: LifecycleOutcome(LifecycleOutcomeKind.RETRYABLE, ExecutionStage.FINALIZED, identity),
        )
        for make_invalid in invalid_cases:
            with self.subTest(make_invalid=make_invalid), self.assertRaises(LifecycleContractError):
                make_invalid()

    def test_task_draft_drops_taskwarrior_native_urgency(self) -> None:
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "11111111-1111-4111-8111-111111111111",
                "status": "pending", "chain": "on", "chainID": "abcd1234", "link": 4,
                "description": "urgency carry regression", "anchor": "w:mon",
                "due": "20260824T090000Z", "urgency": 7.25,
            },
            source_query="task-draft-urgency-test",
        )
        draft = TaskDraft.from_task(NauticalTask.from_observation(observation))
        self.assertNotIn("urgency", draft.fields)
        self.assertNotIn("urgency", draft.to_mapping())

    def test_bound_events_preserve_terminal_kind(self) -> None:
        self.assertEqual(
            terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_MAX).terminal_kind,
            "chain_max",
        )
        self.assertEqual(
            terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_UNTIL).terminal_kind,
            "chain_until",
        )

    def test_scheduler_exhaustion_is_durable_terminal_provenance(self) -> None:
        plan = LifecyclePlanner(
            validated_configuration=object(),
            recurrence_service=ExhaustedService(),
        ).plan(snapshot(), LifecycleEvent.EXPIRE)
        self.assertEqual(plan.terminal_kind, "search_limit")

    def test_terminal_plan_replay_preserves_identity_and_provenance(self) -> None:
        plan = terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_UNTIL)
        restored = type(plan).from_dict(plan.to_dict())
        self.assertEqual(restored.identity.key, plan.identity.key)
        self.assertEqual(restored.identity.idempotency_key, plan.identity.idempotency_key)
        self.assertEqual(restored.terminal_kind, "chain_until")
        self.assertEqual(restored.action, plan.action)
        self.assertEqual(restored.expected_postconditions, plan.expected_postconditions)

    def test_shared_description_preserves_terminal_provenance(self) -> None:
        plan = terminal_plan_for_snapshot(snapshot(), LifecycleEvent.CHAIN_MAX)
        result = RecoveryPlanResult(
            snapshot().observation,
            plan,
            reason="reached chain maximum",
            terminal_kind=plan.terminal_kind,
        )
        evidence = describe_recovery_result(result)
        self.assertEqual(evidence["terminal_kind"], "chain_max")
        self.assertEqual(evidence["trigger"], "completion")

    def test_refusal_description_preserves_typed_status_and_evidence(self) -> None:
        refusal = RecoveryRefusal(
            snapshot().observation,
            RecoveryStatus.RETRYABLE,
            "scheduler evidence is unavailable",
            {"child": "deadbeef", "due": "2026-08-24T09:00:00Z"},
        )
        evidence = describe_recovery_result(refusal)
        self.assertEqual(evidence["status"], "retryable")
        self.assertEqual(evidence["reason"], "scheduler evidence is unavailable")
        self.assertEqual(evidence["child"], "deadbeef")
        self.assertEqual(evidence["due"], "2026-08-24T09:00:00Z")
        self.assertNotIn("action", evidence)
        self.assertNotIn("child_target", evidence)

    def test_refusal_statuses_remain_distinct_in_description(self) -> None:
        for status in RecoveryStatus:
            with self.subTest(status=status):
                refusal = RecoveryRefusal(snapshot().observation, status, f"{status.value} reason")
                evidence = describe_recovery_result(refusal)
                self.assertEqual(evidence["status"], status.value)
                self.assertEqual(evidence["reason"], f"{status.value} reason")

    def test_malformed_expiration_evidence_is_ambiguous(self) -> None:
        task = snapshot().observation.to_mapping()
        task.update({"status": "deleted", "until": "not-a-date", "end": "20260825T200000Z"})
        malformed = DEFAULT_TASK_CODEC.decode_row(task, source_query="malformed-expiration-test")

        def parse(value: object):
            if value == "not-a-date":
                return None, "invalid timestamp"
            return value, None

        evidence = deleted_chain_disposition(malformed, safe_parse_datetime=parse)
        self.assertEqual(evidence.disposition, DeletionDisposition.AMBIGUOUS)
        self.assertIn("reliable native-until", evidence.reason)

    def test_deleted_without_until_builds_chain_disable_terminal_plan(self) -> None:
        from nautical_core.chain_integrity_lifecycle import plan_recovery_decision

        task = snapshot().observation.to_mapping()
        task.update({"status": "deleted", "end": "20260825T200000Z"})
        deleted = DEFAULT_TASK_CODEC.decode_row(task, source_query="deleted-without-until")
        result = plan_recovery_decision(
            deleted,
            existing_children=(),
            hook=object(),
            generation=type("Generation", (), {
                "core": object(),
                "parse_datetime": staticmethod(lambda value: (None, "unused")),
            })(),
        )
        self.assertIsInstance(result, RecoveryPlanResult)
        assert isinstance(result, RecoveryPlanResult)
        self.assertEqual(result.plan.action.value, "disable_chain")
        self.assertEqual(result.plan.identity.event, LifecycleEvent.MANUAL_DELETE)

    def test_planner_owns_recurrence_candidate_and_terminal_policy(self) -> None:
        from datetime import datetime, timezone

        from nautical_core.lifecycle_planner import (
            ChainGenerationLimitPolicy,
            ChainGenerationPlanningService,
            LifecyclePlanningError,
        )

        source_values = {
            "uuid": "00000000-0000-4000-8000-000000000502",
            "status": "completed", "chain": "on", "chainID": "chain-1", "link": 4,
            "cp": "1d",
        }
        source = task_snapshot(source_values)
        candidate = RecurrenceCandidate("2026-08-13T09:00:00Z")

        class Recurrence:
            def __init__(self, value: RecurrenceCandidate) -> None:
                self.candidate = value
                self.calls = []

            def next_candidate(self, selected, event, kind, next_link):
                self.calls.append((event, kind, next_link, selected.to_dict()))
                return self.candidate

            def build_child(self, selected, _event, selected_candidate, next_link):
                values = selected.to_dict()
                return task_draft({
                    **values, "uuid": "00000000-0000-4000-8000-000000000505",
                    "description": "next", "status": "pending", "link": next_link,
                    "prevLink": values["uuid"][:8], "due": selected_candidate.child_due,
                })

        service = Recurrence(candidate)
        plan = LifecyclePlanner({"scheduler_fingerprint": "planner-contract"}, recurrence_service=service).plan(
            source, LifecycleEvent.COMPLETE,
        )
        child = plan.child_dict()
        self.assertIs(plan.action, LifecycleAction.SPAWN_CHILD)
        self.assertEqual(service.calls, [(LifecycleEvent.COMPLETE, "cp", 5, source_values)])
        self.assertEqual((child["link"], child["prevLink"], child["due"]),
                         (5, source_values["uuid"][:8], "2026-08-13T09:00:00Z"))

        limited = LifecyclePlanner(
            {"scheduler_fingerprint": "planner-contract"},
            recurrence_service=service,
            successor_limit_policy=lambda _task, _event, _candidate, link: "chainMax reached" if link > 4 else None,
        ).plan(source, LifecycleEvent.COMPLETE)
        self.assertIs(limited.action, LifecycleAction.FINALIZE_CHAIN)

        terminal = LifecyclePlanner(
            {"scheduler_fingerprint": "planner-contract"},
            recurrence_service=Recurrence(RecurrenceCandidate(None, terminal_reason="chainUntil reached")),
        ).plan(source, LifecycleEvent.EXPIRE)
        self.assertIs(terminal.action, LifecycleAction.FINALIZE_CHAIN)

        no_recurrence = task_snapshot({
            "uuid": "00000000-0000-4000-8000-000000000503", "status": "completed",
            "chain": "on", "chainID": "chain-1", "link": 4,
        })
        before_calls = len(service.calls)
        empty = LifecyclePlanner({"scheduler_fingerprint": "planner-contract"}, recurrence_service=service).plan(
            no_recurrence, LifecycleEvent.COMPLETE,
        )
        self.assertIs(empty.action, LifecycleAction.FINALIZE_CHAIN)
        self.assertEqual(len(service.calls), before_calls)

        class Generation:
            class Core:
                @staticmethod
                def coerce_int(value, default=0):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return default

            core = Core()

            def compute_cp_child_due(self, _parent):
                return datetime(2026, 8, 13, 9, tzinfo=timezone.utc), {"target_field": "due"}

            def parse_datetime(self, _value):
                return datetime(2026, 8, 12, 9, tzinfo=timezone.utc), None

        generated_source = task_snapshot({**source_values, "chainUntil": "2026-08-12T09:00:00Z"})
        generated = ChainGenerationPlanningService(Generation())
        generated_plan = LifecyclePlanner(
            {"scheduler_fingerprint": "planner-contract"},
            recurrence_service=generated,
            successor_limit_policy=ChainGenerationLimitPolicy(lambda left, right: (left > right) - (left < right)),
        ).plan(generated_source, LifecycleEvent.COMPLETE)
        self.assertIs(generated_plan.action, LifecycleAction.FINALIZE_CHAIN)
        with self.assertRaises(LifecyclePlanningError):
            LifecyclePlanner(None)

    def test_completion_and_reconcile_share_candidate_plan(self) -> None:
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from nautical_core.chain_integrity_lifecycle import plan_recovery_decision
        from nautical_core.lifecycle_planner import plan_candidate_successor

        due = datetime(2026, 8, 17, 9, tzinfo=timezone.utc)
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000508", "status": "completed",
            "chain": "on", "chainID": "planner-parity", "link": 1, "cp": "1d",
            "due": "20260816T090000Z",
        }

        class Generation:
            core = SimpleNamespace(
                coerce_int=lambda value, default=0: int(value) if str(value).isdigit() else default,
                fmt_isoz=lambda value: value.isoformat().replace("+00:00", "Z"),
            )

            def build_child_draft(self, task, child_due, child_field, next_link, parent_short, _kind, _cpmax, _until):
                values = task.observation.to_mapping()
                return task_draft({
                    "uuid": "00000000-0000-4000-8000-000000000507", "description": "parity child",
                    "status": "pending", "chain": "on", "chainID": values["chainID"],
                    "link": next_link, "prevLink": parent_short,
                    child_field: child_due.isoformat().replace("+00:00", "Z"),
                    "cp": values.get("cp", "1d"),
                })

            def compute_cp_child_due(self, _task):
                return due, {"target_field": "due"}

            def safe_parse_datetime(self, _value):
                return None, None

        generation = Generation()
        candidate = RecurrenceCandidate(due, metadata=(("target_field", "due"),))
        source = task_snapshot(parent)
        completion = plan_candidate_successor(
            source, LifecycleEvent.COMPLETE, candidate, generation=generation,
            validated_configuration={"scheduler_fingerprint": "parity"},
            compare_datetimes=lambda left, right: (left > right) - (left < right),
        )
        recovered = plan_recovery_decision(
            source.observation, existing_children=(), hook=None, generation=generation,
        )
        self.assertIsInstance(recovered, RecoveryPlanResult)
        self.assertEqual(recovered.plan, completion)
        child = recovered.plan.child_dict()
        self.assertEqual(
            (child["uuid"], child["chainID"], child["link"], child["prevLink"], child["due"]),
            ("00000000-0000-4000-8000-000000000507", "planner-parity", 2,
             "00000000", "2026-08-17T09:00:00Z"),
        )

    def test_expiration_candidate_uses_scheduled_recurrence_basis(self) -> None:
        from datetime import datetime, timezone

        from nautical_core.lifecycle_planner import LifecyclePreflight, expiration_candidate, plan_candidate_successor

        scheduled = "2026-08-16T09:00:00Z"
        parent = {
            "uuid": "00000000-0000-4000-8000-000000000504", "status": "deleted",
            "chain": "on", "chainID": "expiration-parity", "link": 4, "cp": "1d",
            "scheduled": scheduled, "end": "2026-08-20T12:00:00Z",
        }

        class Generation:
            class Core:
                @staticmethod
                def coerce_int(value, default=0):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return default

            core = Core()

            def compute_cp_child_due(self, task):
                self.asserted_end = task.observation.field("end").raw_value()
                return datetime(2026, 8, 17, 9, tzinfo=timezone.utc), {"target_field": "scheduled"}

            def build_child_draft(self, task, due, field, link, parent_short, _kind, _cpmax, _until):
                values = task.observation.to_mapping()
                return task_draft({
                    "uuid": "00000000-0000-4000-8000-000000000505", "description": "expiration child",
                    "status": "pending", "chain": "on", "chainID": values["chainID"],
                    "link": link, "prevLink": parent_short,
                    field: due.isoformat().replace("+00:00", "Z"), "cp": values.get("cp", "1d"),
                })

        generation = Generation()
        source = task_snapshot(parent)
        candidate = expiration_candidate(source, generation=generation)
        plan = plan_candidate_successor(
            source, LifecycleEvent.EXPIRE, candidate, generation=generation,
            validated_configuration={"scheduler_fingerprint": "expiration"},
            compare_datetimes=lambda left, right: (left > right) - (left < right),
            preflight=LifecyclePreflight.from_context(
                base_link=4, next_link=5, kind="cp", chain_id="expiration-parity",
            ),
        )
        self.assertEqual(generation.asserted_end, scheduled)
        child = plan.child_dict()
        self.assertEqual(child["scheduled"], "2026-08-17T09:00:00Z")
        self.assertNotIn("due", child)

    def test_lifecycle_plan_matrix_preserves_expected_recurrence_fields(self) -> None:
        from datetime import datetime, timezone

        from nautical_core.lifecycle_planner import LifecyclePreflight, plan_candidate_successor

        due = datetime(2026, 8, 17, 9, tzinfo=timezone.utc)

        class Generation:
            class Core:
                @staticmethod
                def coerce_int(value, default=0):
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return default

                @staticmethod
                def fmt_isoz(value):
                    return value.isoformat().replace("+00:00", "Z")

            core = Core()

            def build_child_draft(self, task, child_due, field, link, parent_short, _kind, cpmax, until):
                values = task.observation.to_mapping()
                child = {
                    "uuid": "00000000-0000-4000-8000-000000000506", "description": "matrix child",
                    "status": "pending", "chain": "on", "chainID": values["chainID"],
                    "link": link, "prevLink": parent_short,
                    field: child_due.isoformat().replace("+00:00", "Z"),
                }
                if values.get("until"):
                    child["until"] = values["until"]
                if cpmax:
                    child["chainMax"] = cpmax
                child.update({key: values[key] for key in ("cp", "anchor", "anchor_file", "anchor_mode") if values.get(key)})
                return task_draft(child)

        cases = (
            ({"cp": "1d", "due": "20260816T090000Z"}, "cp", "due"),
            ({"anchor": "w:mon", "due": "20260816T090000Z"}, "anchor", "due"),
            ({"anchor": "w:mon", "anchor_file": "null", "due": "20260816T090000Z"}, "anchor", "due"),
            ({"anchor_file": "dates.txt", "due": "20260816T090000Z"}, "anchor_file", "due"),
            ({"cp": "1d", "scheduled": "20260816T090000Z"}, "cp", "scheduled"),
            ({"cp": "1d", "due": "20260816T090000Z", "until": "20260818T090000Z"}, "cp", "due"),
        )
        generation = Generation()
        for index, (fields, kind, target_field) in enumerate(cases):
            with self.subTest(kind=kind, fields=fields):
                parent = {
                    "uuid": f"00000000-0000-4000-8000-0000000005{index:02d}",
                    "status": "completed", "chain": "on", "chainID": f"matrix-{index}", "link": 1,
                    **fields,
                }
                source = task_snapshot(parent)
                candidate = RecurrenceCandidate(due, metadata=(("target_field", target_field),))
                preflight = LifecyclePreflight.from_context(
                    base_link=1, next_link=2, kind=kind, chain_id=f"matrix-{index}",
                )
                plan = plan_candidate_successor(
                    source, LifecycleEvent.COMPLETE, candidate, generation=generation,
                    validated_configuration={"scheduler_fingerprint": "matrix"},
                    compare_datetimes=lambda left, right: (left > right) - (left < right),
                    preflight=preflight,
                )
                child = plan.child_dict()
                self.assertIs(plan.action, LifecycleAction.SPAWN_CHILD)
                self.assertEqual((child["chainID"], child["link"], child["prevLink"]),
                                 (f"matrix-{index}", 2, parent["uuid"][:8]))
                self.assertEqual(child[target_field], "2026-08-17T09:00:00Z")
                other_target = "scheduled" if target_field == "due" else "due"
                self.assertNotIn(other_target, child)
                for key in ("cp", "anchor", "anchor_file", "anchor_mode", "until"):
                    if fields.get(key) and fields.get(key) != "null":
                        self.assertEqual(child.get(key), fields[key])
                repeated = plan_candidate_successor(
                    source, LifecycleEvent.COMPLETE, candidate, generation=generation,
                    validated_configuration={"scheduler_fingerprint": "matrix"},
                    compare_datetimes=lambda left, right: (left > right) - (left < right),
                    preflight=preflight,
                )
                self.assertEqual(repeated, plan)

    def test_terminal_chain_patch_is_idempotent_and_validates_input(self) -> None:
        from nautical_core.modify_lifecycle import ensure_terminal_chain_off

        task = {"chain": "on", "chainID": "abcd1234"}
        self.assertTrue(ensure_terminal_chain_off(task))
        self.assertEqual(task["chain"], "off")
        self.assertFalse(ensure_terminal_chain_off(task))
        with self.assertRaisesRegex(ValueError, "task mapping"):
            ensure_terminal_chain_off(None)


if __name__ == "__main__":
    unittest.main()
