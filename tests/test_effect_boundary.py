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
)
from nautical_core.taskwarrior_uow import InvocationReadCache, QueryScope, QueryScopeKind

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

    def test_snapshot_cache_is_invalidated_by_mutation_epoch(self) -> None:
        cache = InvocationReadCache()
        scope = QueryScope(QueryScopeKind.UUID, "11111111-1111-4111-8111-111111111111")
        cache.put(scope, object(), mutation_epoch=0, command_count=1)
        self.assertIsNotNone(cache.get(scope, mutation_epoch=0))
        cache.invalidate()
        self.assertIsNone(cache.get(scope, mutation_epoch=0))


if __name__ == "__main__":
    unittest.main()
