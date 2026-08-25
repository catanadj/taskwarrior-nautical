from __future__ import annotations

import unittest

from nautical_core.modify_workflow import ModifyRouteKind, classify_modify_transition
from nautical_core.task_changes import TaskTransition
from nautical_core.task_models import TaskObservation


def transition(old: dict[str, object], new: dict[str, object]) -> TaskTransition:
    base = {"uuid": "11111111-1111-4111-8111-111111111111", "entry": "20260825T090000Z"}
    left = dict(base, **old)
    right = dict(base, **new)
    return TaskTransition.from_observations(
        TaskObservation.from_mapping(left, source_query="modify-test"),
        TaskObservation.from_mapping(right, source_query="modify-test"),
    )


class ModifyWorkflowTests(unittest.TestCase):
    def test_plain_transition_is_ordinary(self) -> None:
        route = classify_modify_transition(transition({"status": "pending"}, {"status": "pending", "description": "changed"}))
        self.assertEqual(route.kind, ModifyRouteKind.ORDINARY)

    def test_completion_and_recompletion_are_distinct(self) -> None:
        root = {"status": "pending", "anchor": "w:mon", "chain": "on", "chainID": "abcd1234", "link": 1}
        self.assertEqual(classify_modify_transition(transition(root, dict(root, status="completed"))).kind, ModifyRouteKind.COMPLETION)
        self.assertEqual(classify_modify_transition(transition(dict(root, status="completed"), dict(root, status="completed", modified="20260825T100000Z"))).kind, ModifyRouteKind.IDEMPOTENT_COMPLETION)

    def test_completion_with_existing_successor_is_idempotent(self) -> None:
        root = {"status": "pending", "anchor": "w:mon", "chain": "on", "chainID": "abcd1234", "link": 1}
        route = classify_modify_transition(transition(root, dict(root, status="completed", nextLink="22222222")))
        self.assertEqual(route.kind, ModifyRouteKind.IDEMPOTENT_COMPLETION)
        self.assertFalse(route.requires_spawn_evidence)
        self.assertIn("chain_slot", route.required_evidence)

    def test_activation_disable_resume_and_removal(self) -> None:
        plain = {"status": "pending"}
        active = {"status": "pending", "anchor": "w:mon", "chain": "on", "chainID": "abcd1234", "link": 1}
        self.assertEqual(classify_modify_transition(transition(plain, active)).kind, ModifyRouteKind.ACTIVATION)
        self.assertEqual(classify_modify_transition(transition(active, dict(active, chain="off"))).kind, ModifyRouteKind.MANUAL_CHAIN_OFF)
        self.assertEqual(classify_modify_transition(transition(dict(active, chain="off"), active)).kind, ModifyRouteKind.RESUME)
        self.assertEqual(classify_modify_transition(transition(active, plain)).kind, ModifyRouteKind.RECURRENCE_REMOVAL)

    def test_deletion_and_chain_identity_evidence_are_explicit(self) -> None:
        active = {"status": "pending", "anchor": "w:mon", "chain": "on", "chainID": "abcd1234", "link": 1}
        deleted = dict(active, status="deleted")
        route = classify_modify_transition(transition(active, deleted))
        self.assertEqual(route.kind, ModifyRouteKind.DELETION)
        edited = dict(active, chainID="ffff0000")
        route = classify_modify_transition(transition(active, edited))
        self.assertEqual(route.kind, ModifyRouteKind.INVALID_IDENTITY_EDIT)
        self.assertIn("chain_identity_edit", route.evidence)
        self.assertTrue(route.identity_mutation)

    def test_volatile_only_reentry_is_marked_as_evidence(self) -> None:
        active = {"status": "pending", "anchor": "w:mon", "chain": "on", "chainID": "abcd1234", "link": 1}
        route = classify_modify_transition(
            transition(active, dict(active, modified="20260825T100000Z", urgency=4.0))
        )
        self.assertIn("volatile_only", route.evidence)
        self.assertEqual(route.volatile_fields, ("modified",))
        self.assertEqual(route.user_changed_fields, ())

    def test_route_matrix_is_mutually_exclusive_and_deterministic(self) -> None:
        root = {"status": "pending", "anchor": "w:mon", "chain": "on", "chainID": "abcd1234", "link": 1}
        cases = (
            (root, dict(root, description="edit"), ModifyRouteKind.RECURRING_EDIT),
            (root, dict(root, status="completed"), ModifyRouteKind.COMPLETION),
            (root, dict(root, status="deleted"), ModifyRouteKind.DELETION),
            (root, dict(root, chain="off"), ModifyRouteKind.MANUAL_CHAIN_OFF),
            (dict(root, chain="off"), root, ModifyRouteKind.RESUME),
            (root, {"status": "pending"}, ModifyRouteKind.RECURRENCE_REMOVAL),
        )
        for old, new, expected in cases:
            first = classify_modify_transition(transition(old, new))
            second = classify_modify_transition(transition(old, new))
            self.assertEqual(first.kind, expected)
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
