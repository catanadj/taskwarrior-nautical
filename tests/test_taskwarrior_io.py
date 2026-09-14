from __future__ import annotations

import contextlib
import io
import json
import unittest

from nautical_core import hook_protocol, hook_results
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.taskwarrior_io import TaskDocument


class HookIoContractTests(unittest.TestCase):
    """Direct contracts for Taskwarrior hook payload and response boundaries."""

    def test_on_add_preserves_unknown_task_fields_unicode_and_typed_request(self) -> None:
        task = {
            "uuid": "00000000-0000-4000-8000-000000000901",
            "description": "Répéter 🌊",
            "custom_uda": {"nested": ["значение", 3]},
            "link": 1.0,
        }
        result = hook_protocol.probe_on_add(json.dumps(task, ensure_ascii=False))
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.task, task)
        self.assertEqual(result.task.get("custom_uda"), task["custom_uda"])
        self.assertIsInstance(result.request, hook_protocol.OnAddInput)

    def test_modify_array_preserves_both_tasks_and_typed_request(self) -> None:
        old = {"uuid": "00000000-0000-4000-8000-000000000902", "description": "old", "custom": "keep"}
        new = dict(old, description="new", custom_extra="also-keep")
        result = hook_protocol.probe_on_modify(json.dumps([old, new], ensure_ascii=False))
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.old, old)
        self.assertEqual(result.new, new)
        self.assertIsInstance(result.request, hook_protocol.OnModifyInput)

    def test_on_add_rejects_trailing_json_without_partial_success(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000903", "description": "strict"}
        result = hook_protocol.probe_on_add(json.dumps(task) + " trailing")
        self.assertFalse(result.valid)
        self.assertEqual(result.error_kind, "invalid_input")
        self.assertIsInstance(result.failure, hook_protocol.ProtocolFailure)

    def test_task_response_emits_one_unescaped_json_object(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000904", "description": "Répéter 🌊"}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            hook_results.emit_task_json(task)
        text = output.getvalue()
        self.assertEqual(text.count("{"), 1)
        self.assertEqual(text.count("}"), 1)
        self.assertIn("Répéter 🌊", text)
        self.assertEqual(json.loads(text), task)

    def test_response_models_keep_legacy_names_and_typed_roles(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000906"}
        task_result = hook_results.TaskHookResponse(task)
        exit_result = hook_results.ExitHookResponse(exit_code=3, stats={"errors": 1})
        self.assertIsInstance(task_result, hook_results.HookJsonResult)
        self.assertIsInstance(exit_result, hook_results.HookExitResult)
        self.assertIs(task_result.task, task)
        self.assertEqual(exit_result.exit_code, 3)


class TaskDocumentTests(unittest.TestCase):
    """Direct contracts for lossless Taskwarrior task-document access."""

    def test_document_is_lossless_and_exposes_typed_scalar_accessors(self) -> None:
        task = {
            "uuid": "00000000-0000-4000-8000-000000000905",
            "description": "routine",
            "link": 7.0,
            "chain": "on",
            "custom_uda": {"nested": ["keep", 2]},
        }
        document = TaskDocument.from_object(task)
        self.assertIsNotNone(document)
        assert document is not None
        self.assertIs(document.as_dict(), task)
        self.assertEqual(document.text("description"), "routine")
        self.assertEqual(document.integer("link"), 7)
        self.assertTrue(document.boolean("chain"))
        self.assertEqual(document.get("custom_uda"), task["custom_uda"])
        task["description"] = "changed"
        self.assertEqual(document.text("description"), "changed")

    def test_document_rejects_non_objects_and_defaults_bad_scalars(self) -> None:
        self.assertIsNone(TaskDocument.from_object([{ "uuid": "bad" }]))
        document = TaskDocument.from_object({"link": "not-a-number", "chain": "maybe"})
        self.assertIsNotNone(document)
        assert document is not None
        self.assertEqual(document.integer("link", 4), 4)
        self.assertFalse(document.boolean("chain", False))


class TaskCodecBoundaryTests(unittest.TestCase):
    def test_sanitization_removes_controls_and_clamps_string_fields(self) -> None:
        task = {
            "description": "hi\x00there\x1f!",
            "project": "long-project-name",
            "link": 3,
        }

        DEFAULT_TASK_CODEC.sanitize_task_mapping(task, max_len=8)

        self.assertEqual(task["description"], "hithere!")
        self.assertEqual(task["project"], "long-pro")
        self.assertEqual(task["link"], 3)
