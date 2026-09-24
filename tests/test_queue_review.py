import tempfile
import unittest
from pathlib import Path
import json
import subprocess
from contextlib import redirect_stdout
from io import StringIO
import sys
from unittest.mock import patch

from nautical_core import queue_status_service
from nautical_core.queue_status_service import QueueStatusService
from nautical_core.tools import nautical_queue_review


class QueueReviewTests(unittest.TestCase):
    def test_human_review_output_is_compact_for_integrity_finding(self) -> None:
        payload = {
            "status": "found",
            "taskdata": "/tmp/taskdata",
            "intents": [{
                "intent_id": "integrity:chain-a:continuity.child_temporal_order:child_not_after_parent",
                "state": "manual_review",
                "plan": {"chainID": "chain-a", "source_link": 14, "target_link": 15},
                "failure": {"message": "child target is not after recurrence reference"},
                "review_item": {
                    "evidence": {
                        "chain_id": "chain-a", "source_link": 14, "target_link": 15,
                        "parent_uuid": "parent123", "expected_child_uuid": "child456",
                        "occupants": ["parent123", "child456"],
                        "reason": "child target is not after recurrence reference",
                    },
                    "confirmation_token": "abc123",
                    "confirmation_available": True,
                    "actions": ["skip"],
                },
            }],
        }
        output = StringIO()
        with patch.object(nautical_queue_review.QueueStatusService, "review_payload", return_value=payload), \
             patch.object(sys, "argv", ["nautical review", "--next"]), \
             redirect_stdout(output):
            self.assertEqual(nautical_queue_review.main(), 0)
        rendered = output.getvalue()
        self.assertIn("Chain chain-a · links 14 → 15", rendered)
        self.assertIn("Problem: child target is not after recurrence reference", rendered)
        self.assertIn("Safe action: skip — leave unresolved", rendered)
        self.assertIn("Confirm: [redacted; use --json for action]", rendered)
        self.assertNotIn("abc123", rendered)
        self.assertNotIn("event=", rendered)
        self.assertNotIn("expected_none", rendered)
        self.assertNotIn("guard_current", rendered)

    def test_human_review_output_includes_task_context(self) -> None:
        payload = {
            "status": "found", "taskdata": "/tmp/taskdata", "intents": [{
                "intent_id": "integrity:chain-a:problem:reason", "state": "manual_review",
                "review_item": {
                    "evidence": {"chain_id": "chain-a", "source_link": 1, "target_link": 2,
                                  "parent_uuid": "parent123", "expected_child_uuid": "child456",
                                  "occupants": [], "reason": "needs review"},
                    "task_context": {"when": "2026-03-10 18:40", "description": "Prepare quarterly report"},
                    "confirmation_token": "abc123", "confirmation_available": True, "actions": ["skip"],
                },
            }],
        }
        output = StringIO()
        with patch.object(nautical_queue_review.QueueStatusService, "review_payload", return_value=payload), \
             patch.object(sys, "argv", ["nautical review", "--next"]), \
             redirect_stdout(output):
            self.assertEqual(nautical_queue_review.main(), 0)
        rendered = output.getvalue()
        self.assertIn("When: 2026-03-10 18:40", rendered)
        self.assertIn("Task: Prepare quarterly report", rendered)

    def test_next_human_mode_prompts_and_can_quit(self) -> None:
        payload = {
            "status": "found", "taskdata": "/tmp/taskdata", "intents": [{
                "intent_id": "integrity:chain-a:problem:reason", "state": "manual_review",
                "review_item": {"evidence": {"chain_id": "chain-a", "source_link": 1, "target_link": 2,
                                                "parent_uuid": "parent123", "expected_child_uuid": "child456",
                                                "occupants": [], "reason": "needs review"},
                                 "confirmation_token": "abc123", "confirmation_available": True, "actions": ["skip"]},
            }],
        }
        output = StringIO()
        with patch.object(nautical_queue_review.QueueStatusService, "review_payload", return_value=payload) as review, \
             patch.object(sys, "argv", ["nautical review", "--next"]), \
             patch("builtins.input", return_value="q"), redirect_stdout(output):
            self.assertEqual(nautical_queue_review.main(), 0)
        self.assertIn("[s]kip", output.getvalue())
        self.assertIn("[q]uit", output.getvalue())
        self.assertEqual(review.call_count, 2)

    def test_review_task_context_uses_authoritative_task_row(self) -> None:
        class Command:
            ok = True
            stdout = "[]"
            stderr = ""

        class Row:
            def to_mapping(self):
                return {"description": "Prepare quarterly report", "scheduled": "2026-03-10 18:40"}

        client = type("Client", (), {"execute": lambda _self, *_args, **_kwargs: Command()})()
        codec = type("Codec", (), {"decode_export": staticmethod(lambda *_args, **_kwargs: [Row()])})()
        with patch.dict(queue_status_service.__dict__, {"TaskwarriorClient": lambda *_args, **_kwargs: client,
                                                         "DEFAULT_TASK_CODEC": codec}):
            context = QueueStatusService._task_context(Path("/tmp/taskdata"), "task", ("parent123", "child456"))
        self.assertEqual(context, {"when": "2026-03-10 18:40", "description": "Prepare quarterly report"})

    def test_review_next_json_is_bounded_to_one_item(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["python3", "nautical", "review", "--next", "--json", "--taskdata", directory],
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "empty")
        self.assertEqual(payload["intents"], [])

    def test_build_review_item_projects_plan_without_exposing_full_uuids(self) -> None:
        item = QueueStatusService().build_review_item({
            "intent_id": "review-1", "state": "manual_review",
            "plan": {
                "chainID": "chain-a", "source_link": 19, "target_link": 20,
                "parent_uuid": "11111111-1111-4111-8111-111111111111",
                "child_uuid": "22222222-2222-4222-8222-222222222222",
            },
            "failure": {"message": "duplicate slot"},
            "occupants": ["33333333-3333-4333-8333-333333333333", "44444444-4444-4444-8444-444444444444"],
        })
        payload = item.to_dict()
        self.assertEqual(payload["evidence"]["chain_id"], "chain-a")
        self.assertEqual(payload["evidence"]["parent_uuid"], "11111111")
        self.assertNotIn("accept-connected", payload["actions"])

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
            self.assertEqual(payload["intents"][0]["review_item"]["evidence"]["chain_id"], "abcd")

    def test_review_exact_missing_intent_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(queue_status_service.LifecycleOutboxRepository, "status",
                return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": []}),
            ):
                payload = QueueStatusService().review_payload(Path(directory), intent_id="missing")
            self.assertEqual(payload["status"], "not_found")
            self.assertEqual(payload["failure"]["code"], "intent_not_found")

    def test_review_falls_back_to_integrity_findings_when_outbox_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            integrity = type(
                "Integrity",
                (),
                {"query": lambda _self, _request: ({
                    "status": "manual_review",
                    "findings": [{
                        "invariant_id": "slot.duplicate_occupant",
                        "status": "manual_review",
                        "chain_id": "chain-a",
                        "subject_uuids": ["11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"],
                        "message": "duplicate slot",
                        "reason_code": "duplicate_slot",
                        "evidence": {"parent_link": 1, "child_link": 2, "occupants": ["11111111", "22222222"]},
                    }],
                }, 0)},
            )()
            with patch.object(queue_status_service.LifecycleOutboxRepository, "status", return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": []})), \
                 patch.object(queue_status_service, "IntegrityQueryService", return_value=integrity):
                payload = QueueStatusService().review_payload(Path(directory), task_binary="task", runtime=object())
                self.assertEqual(payload["status"], "found")
                self.assertTrue(payload["intents"][0]["intent_id"].startswith("integrity:"))
                self.assertEqual(payload["intents"][0]["review_item"]["evidence"]["chain_id"], "chain-a")
                exact_id = payload["intents"][0]["intent_id"]
                exact = QueueStatusService().review_payload(
                    Path(directory), intent_id=exact_id, task_binary="task", runtime=object()
                )
                self.assertEqual(exact["status"], "found")
                self.assertEqual(exact["intents"][0]["intent_id"], exact_id)

    def test_review_exact_non_reviewable_intent_is_distinguished(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                queue_status_service.LifecycleOutboxRepository, "status",
                return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": [{"intent_id": "done", "state": "acknowledged"}]}),
            ):
                payload = QueueStatusService().review_payload(Path(directory), intent_id="done")
            self.assertEqual(payload["status"], "not_reviewable")
            self.assertEqual(payload["failure"]["code"], "intent_not_reviewable")

    def test_review_marks_matching_successor_as_high_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = [{
                "intent_id": "review-1", "state": "manual_review",
                "plan": {"action": "spawn_child", "parent_uuid": "parent", "child_uuid": "child1234-full", "parent_guard": {}},
            }]
            class Command:
                ok = True
                stdout = "[]"
                stderr = ""
            class Row:
                def __init__(self, value): self.value = value
                def to_mapping(self): return self.value
            def decode(_raw, *, source_query):
                return [Row({"nextLink": "child123"})] if "parent" in source_query else [Row({"uuid": "child1234-full"})]
            with patch.object(queue_status_service.LifecycleOutboxRepository, "status", return_value=(type("Result", (), {"ok": True, "reason": ""})(), {"records": records})), \
                 patch.object(queue_status_service.TaskwarriorClient, "execute", return_value=Command()), \
                 patch.object(queue_status_service, "DEFAULT_TASK_CODEC", type("Codec", (), {"decode_export": staticmethod(decode)})()):
                payload = QueueStatusService().review_payload(Path(directory), intent_id="review-1")
            self.assertEqual(payload["intents"][0]["assessment"]["status"], "already_applied")

    def test_guided_action_requires_current_confirmation_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = [{
                "intent_id": "review-1", "state": "manual_review",
                "plan": {"chainID": "chain-a", "parent_uuid": "parent", "source_link": 1, "target_link": 2},
            }]
            status_result = (type("Result", (), {"ok": True, "reason": ""})(), {"records": records})
            with patch.object(queue_status_service.LifecycleOutboxRepository, "status", return_value=status_result):
                item = QueueStatusService().build_review_item(records[0])
                token = QueueStatusService.review_confirmation_token(item)
                stale = QueueStatusService().apply_review_action(Path(directory), "review-1", "skip", "wrong")
                self.assertEqual(stale["status"], "conflict")
                applied = QueueStatusService().apply_review_action(Path(directory), "review-1", "skip", token)
                self.assertEqual(applied["status"], "skipped")
                unresolved = QueueStatusService().apply_review_action(
                    Path(directory), "review-1", "resolve-applied", token
                )
                self.assertEqual(unresolved["status"], "conflict")


if __name__ == "__main__":
    unittest.main()
