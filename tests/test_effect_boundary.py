"""Static ownership checks for the workflow effect boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from nautical_core.integration_models import (
    GuardTimestamp,
    GuardTimestampField,
    MutationGuard,
    MutationOperation,
    MutationOutcome,
    MutationOutcomeKind,
    MutationPostcondition,
    IntegrationContractError,
)
from nautical_core.taskwarrior_uow import InvocationReadCache, QueryScope, QueryScopeKind
from nautical_core.modify_feedback import lifecycle_result_feedback_facts
from nautical_core.hook_workflow_models import FeedbackFacts, FeedbackFactKind
from nautical_core.feedback_renderer import PanelView, panel_view_from_facts, render_panel_view
from nautical_core.lifecycle_application import LifecycleApplicationOutcomeKind, LifecycleApplicationService
from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard
from nautical_core.task_models import TaskObservation
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService
from nautical_core.lifecycle_models import recurrence_fingerprint
from nautical_core.operator_models import OperatorOperation, OperatorResult, OperatorStatus
from nautical_core.operator_presentation import render_result

ROOT = Path(__file__).resolve().parents[1]
PURE_WORKFLOW_MODULES = (
    "nautical_core/lifecycle_planner.py",
    "nautical_core/chain_repair_planner.py",
    "nautical_core/modify_feedback.py",
    "nautical_core/panel_diagnostics.py",
    "nautical_core/panel_colours.py",
)
FORBIDDEN_IMPORTS = {
    "subprocess",
    "nautical_core.lifecycle_outbox",
    "nautical_core.runtime_command",
    "nautical_core.taskwarrior_mutations",
    "nautical_core.task_command",
}


class EffectBoundaryTests(unittest.TestCase):
    def test_planners_and_presenters_have_no_external_effect_imports(self) -> None:
        for relative in PURE_WORKFLOW_MODULES:
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            forbidden = sorted(
                name
                for name in imported
                if name in FORBIDDEN_IMPORTS
                or any(name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS if prefix != "subprocess")
            )
            self.assertEqual(forbidden, [], f"{relative} imports external effect owners: {forbidden}")

    def test_mutation_contract_requires_guard_and_postcondition(self) -> None:
        guard = MutationGuard(
            task_uuid="11111111-1111-4111-8111-111111111111",
            status="pending",
            chain_id="abcd1234",
            link=1,
            recurrence_identity="identity",
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, "20260825T120000Z"),),
            expected_mutation_epoch=0,
        )
        outcome = MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.ALREADY_APPLIED,
            guard,
            (MutationPostcondition.CHAIN_DISABLED,),
        )
        self.assertEqual(outcome.postconditions, (MutationPostcondition.CHAIN_DISABLED,))
        with self.assertRaises(IntegrationContractError):
            MutationOutcome(
                MutationOperation.CHAIN_DISABLE,
                MutationOutcomeKind.REJECTED,
                guard,
                (MutationPostcondition.CHAIN_DISABLED,),
                reason="guard conflict",
            )

    def test_guard_timestamp_conflict_reports_expected_and_found_values(self) -> None:
        row = {"uuid": "11111111-1111-4111-8111-111111111111", "status": "pending", "chain": "on", "chainID": "abcd1234", "link": 1, "modified": "20260825T120001Z"}
        observation = TaskObservation.from_mapping(row, source_query="test")
        guard = MutationGuard(
            task_uuid=row["uuid"], status="pending", chain_id="abcd1234", link=1,
            recurrence_identity=recurrence_fingerprint(row),
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, "20260825T120000Z"),),
            expected_mutation_epoch=0,
        )
        mismatch = TaskwarriorMutationService._guard_mismatch(guard, observation)
        self.assertIn("expected 20260825T120000Z", mismatch)
        self.assertIn("found 20260825T120001Z", mismatch)

    def test_snapshot_cache_is_invalidated_by_mutation_epoch(self) -> None:
        cache = InvocationReadCache()
        scope = QueryScope(QueryScopeKind.UUID, "11111111-1111-4111-8111-111111111111")
        cache.put(scope, object(), mutation_epoch=0, command_count=1)
        self.assertIsNotNone(cache.get(scope, mutation_epoch=0))
        cache.invalidate()
        self.assertIsNone(cache.get(scope, mutation_epoch=0))

    def test_application_boundary_does_not_report_failed_mutation_as_success(self) -> None:
        identity = LifecycleIdentity(
            chain_id="abcd1234",
            parent_uuid="11111111-1111-4111-8111-111111111111",
            source_link=1,
            target_link=None,
            event=LifecycleEvent.COMPLETE,
        )
        plan = LifecyclePlan(
            identity=identity,
            action=LifecycleAction.FINALIZE_CHAIN,
            parent_guard=ParentGuard(
                status="completed",
                chain="on",
                chain_id="abcd1234",
                link=1,
                recurrence_fingerprint="identity",
                modified="20260825T120000Z",
                end="20260825T120100Z",
            ),
        )

        class UnitOfWork:
            mutation_epoch = 0

            def record_mutation(self, *, uncertain: bool = False) -> None:
                self.mutation_epoch += 1

        class RejectingGateway:
            def apply(self, request):
                return MutationOutcome(
                    request.operation,
                    MutationOutcomeKind.REJECTED,
                    request.guard,
                    reason="guard conflict",
                )

        outcome = LifecycleApplicationService(
            unit_of_work=UnitOfWork(),
            mutations=RejectingGateway(),
            outbox=object(),
        ).apply_immediate(plan)
        self.assertEqual(outcome.kind, LifecycleApplicationOutcomeKind.MANUAL_REVIEW)
        self.assertFalse(outcome.ok)

    def test_lifecycle_feedback_is_immutable_and_presentation_only(self) -> None:
        class Result:
            state = "retryable"
            reason = "link verification unavailable"

        facts = lifecycle_result_feedback_facts(Result())
        self.assertEqual(facts.warnings, ("link verification unavailable",))
        self.assertTrue(facts.recovery_guidance)
        self.assertEqual(facts.fact_kinds, (FeedbackFactKind.RECOVERY,))
        self.assertFalse(facts.chain_completed)

    def test_feedback_fact_contract_requires_actionable_failures(self) -> None:
        with self.assertRaises(ValueError):
            FeedbackFacts(fact_kinds=(FeedbackFactKind.MANUAL_REVIEW,))
        facts = FeedbackFacts(
            fact_kinds=(FeedbackFactKind.RECOVERY,),
            recovery_guidance=("Run reconcile",),
            task_uuid="task-1",
        )
        contract = facts.to_contract()
        self.assertEqual(contract["fact_kinds"], ["recovery"])
        self.assertEqual(contract["recovery_guidance"], ["Run reconcile"])

    def test_all_renderers_consume_the_same_deterministic_view(self) -> None:
        facts = FeedbackFacts(
            task_uuid="task-1",
            chain_id="chain-1",
            natural_explanation="Every Monday",
            warnings=("needs review",),
            recovery_guidance=("run reconcile",),
            next_action="retry",
        )
        first = panel_view_from_facts(facts)
        second = panel_view_from_facts(facts)
        self.assertEqual(first, second)
        self.assertEqual(first.to_diagnostic(), second.to_diagnostic())
        rendered: list[tuple[str, list[tuple[str, str]], str]] = []
        self.assertTrue(render_panel_view(first, lambda title, rows, *, kind: rendered.append((title, rows, kind))))
        self.assertEqual(rendered[0][1], list(first.rows))

    def test_rendering_failure_is_contained(self) -> None:
        view = PanelView("Nautical workflow", "note", (("Warning", "x"),))
        self.assertFalse(render_panel_view(view, lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ui"))))

    def test_rendering_failure_does_not_change_semantic_result(self) -> None:
        result = OperatorResult(OperatorOperation.INSPECT, OperatorStatus.OK, data={"decision": "schedule"})
        before = result.to_dict()

        rendered = render_result(
            result,
            "rich",
            rich_renderer=lambda _result: (_ for _ in ()).throw(RuntimeError("ui")),
        )

        self.assertIn("presentation unavailable", rendered)
        self.assertEqual(result.to_dict(), before)

    def test_production_feedback_paths_use_shared_renderer(self) -> None:
        import inspect
        from nautical_core import modify_feedback

        source = inspect.getsource(modify_feedback)
        self.assertGreaterEqual(source.count("render_panel_view("), 3)


if __name__ == "__main__":
    unittest.main()
