import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nautical_core import queue_status_service
from nautical_core.queue_status_service import QueueStatusService


class QueueReviewTests(unittest.TestCase):
    def test_review_filters_to_manual_states_and_keeps_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            records = [
                {"intent_id": "review-1", "state": "manual_review", "plan": {"chainID": "abcd"}},
                {"intent_id": "ready-1", "state": "ready", "plan": {"chainID": "efgh"}},
            ]
            with patch.object(queue_status_service.LifecycleOutboxRepository, "status",
                return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": records}),
            ):
                payload = QueueStatusService().review_payload(taskdata)
            self.assertEqual(payload["status"], "found")
            self.assertEqual([item["intent_id"] for item in payload["intents"]], ["review-1"])

    def test_review_exact_missing_intent_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(queue_status_service.LifecycleOutboxRepository, "status",
                return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": []}),
            ):
                payload = QueueStatusService().review_payload(Path(directory), intent_id="missing")
            self.assertEqual(payload["status"], "not_found")
            self.assertEqual(payload["failure"]["code"], "intent_not_found")

    def test_review_exact_non_reviewable_intent_is_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                queue_status_service.LifecycleOutboxRepository, "status",
                return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": [{"intent_id": "done", "state": "acknowledged"}]}),
            ):
                payload = QueueStatusService().review_payload(Path(directory), intent_id="done")
            self.assertEqual(payload["status"], "not_reviewable")
            self.assertEqual(payload["failure"]["code"], "intent_not_reviewable")


if __name__ == "__main__":
    unittest.main()
