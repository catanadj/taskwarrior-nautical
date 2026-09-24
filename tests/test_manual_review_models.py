import unittest

from nautical_core.manual_review_models import (
    ManualReviewAction,
    ManualReviewEvidence,
    ManualReviewItem,
    ManualReviewUnavailable,
)


class ManualReviewModelTests(unittest.TestCase):
    def test_item_serialization_is_stable_and_redacts_full_uuids(self) -> None:
        evidence = ManualReviewEvidence(
            chain_id="chain-a", source_link=19, target_link=20,
            parent_uuid="11111111-1111-4111-8111-111111111111",
            expected_child_uuid="22222222-2222-4222-8222-222222222222",
            occupants=("33333333-3333-4333-8333-333333333333",),
            reason="slot occupied",
        )
        item = ManualReviewItem(
            intent_id="intent-1", state="manual_review", evidence=evidence,
            actions=(ManualReviewAction.RETRY, ManualReviewAction.SKIP),
        )
        payload = item.to_dict()
        self.assertEqual(payload["intent_id"], "intent-1")
        self.assertEqual(payload["evidence"]["parent_uuid"], "11111111")
        self.assertEqual(payload["evidence"]["occupants"], ["33333333"])
        self.assertEqual(payload["actions"], ["retry", "skip"])
        self.assertNotIn("22222222-2222", repr(payload))

    def test_ambiguous_slot_filters_branch_acceptance(self) -> None:
        evidence = ManualReviewEvidence(
            chain_id="chain-a", source_link=19, target_link=20,
            parent_uuid="parent", expected_child_uuid="child",
            occupants=("one", "two"), reason="duplicate slot",
        )
        item = ManualReviewItem.from_evidence("intent-2", "manual_review", evidence)
        self.assertNotIn(ManualReviewAction.ACCEPT_CONNECTED.value, item.to_dict()["actions"])
        self.assertEqual(item.to_dict()["actions"], ["resolve-applied", "skip"])

    def test_unavailable_result_is_structured(self) -> None:
        result = ManualReviewUnavailable("intent-3", "chain snapshot unavailable")
        self.assertEqual(
            result.to_dict(),
            {"status": "unavailable", "intent_id": "intent-3", "reason": "chain snapshot unavailable"},
        )


if __name__ == "__main__":
    unittest.main()
