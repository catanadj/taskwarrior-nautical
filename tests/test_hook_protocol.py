from __future__ import annotations

import io
import json
import unittest

from nautical_core import hook_protocol


class HookProtocolTests(unittest.TestCase):
    """Direct contracts for the lightweight hook-input protocol."""

    def test_on_add_classifies_the_complete_nautical_field_matrix_and_rejects_bad_input(self) -> None:
        plain = {
            "uuid": "00000000-0000-4000-8000-000000000701",
            "description": "Cafe ăîșț ✅",
        }
        result = hook_protocol.probe_on_add(json.dumps(plain, ensure_ascii=False))
        self.assertTrue(result.valid, result.error)
        self.assertEqual(result.task, plain)
        self.assertFalse(result.is_nautical)

        for field, value in (
            ("anchor", "w:mon"),
            ("anchor_file", "dates.csv"),
            ("anchor_mode", "skip"),
            ("cp", "1d"),
            ("chainID", "abcd1234"),
            ("chainMax", 3),
            ("chainUntil", "20270101T000000Z"),
            ("omit", "w:sun"),
            ("omit_file", "holidays.csv"),
        ):
            with self.subTest(field=field):
                task = dict(plain, **{field: value})
                probed = hook_protocol.probe_on_add(json.dumps(task))
                self.assertTrue(probed.valid, probed.error)
                self.assertTrue(probed.is_nautical)

        self.assertFalse(hook_protocol.probe_on_add("").valid)
        self.assertFalse(hook_protocol.probe_on_add("[]").valid)
        self.assertFalse(hook_protocol.probe_on_add('{"uuid":"broken"} trailing').valid)

    def test_on_modify_accepts_concatenated_array_and_single_task_forms(self) -> None:
        uuid_str = "00000000-0000-4000-8000-000000000702"
        old = {"uuid": uuid_str, "status": "pending", "description": "old"}
        new = {"uuid": uuid_str, "status": "pending", "description": "new ăîșț"}

        concatenated = hook_protocol.probe_on_modify(
            json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False)
        )
        self.assertTrue(concatenated.valid, concatenated.error)
        self.assertEqual(concatenated.old, old)
        self.assertEqual(concatenated.new, new)
        self.assertFalse(concatenated.is_nautical)

        array_result = hook_protocol.probe_on_modify(json.dumps([old, new], ensure_ascii=False))
        self.assertTrue(array_result.valid, array_result.error)
        self.assertEqual(array_result.old, old)
        self.assertEqual(array_result.new, new)

        single_result = hook_protocol.probe_on_modify(json.dumps(new, ensure_ascii=False))
        self.assertTrue(single_result.valid, single_result.error)
        self.assertEqual(single_result.old, new)
        self.assertEqual(single_result.new, new)

    def test_on_modify_matches_recurrence_lineage_and_anchor_mode_routing_rules(self) -> None:
        uuid_str = "00000000-0000-4000-8000-000000000703"
        base = {"uuid": uuid_str, "status": "pending"}
        for field, value in (
            ("anchor", "w:mon"),
            ("anchor_file", "dates.csv"),
            ("cp", "1d"),
            ("omit", "w:sun"),
            ("omit_file", "holidays.csv"),
            ("chainID", "abcd1234"),
            ("prevLink", "11111111"),
            ("nextLink", "22222222"),
            ("link", 2),
        ):
            with self.subTest(field=field):
                task = dict(base, **{field: value})
                result = hook_protocol.probe_on_modify(json.dumps(task))
                self.assertTrue(result.valid, result.error)
                self.assertTrue(result.is_nautical)

        anchor_mode_only = hook_protocol.probe_on_modify(json.dumps(dict(base, anchor_mode="skip")))
        self.assertTrue(anchor_mode_only.valid, anchor_mode_only.error)
        self.assertFalse(anchor_mode_only.is_nautical)

    def test_modify_validation_keeps_uuid_limits_streams_and_unicode_contracts(self) -> None:
        old = {"uuid": "00000000-0000-4000-8000-000000000704", "status": "pending"}
        different = {"uuid": "00000000-0000-4000-8000-000000000705", "status": "deleted"}
        plain_mismatch = hook_protocol.probe_on_modify(json.dumps(old) + json.dumps(different))
        self.assertTrue(plain_mismatch.valid, plain_mismatch.error)
        self.assertFalse(plain_mismatch.is_nautical)

        nautical_mismatch = hook_protocol.probe_on_modify(
            json.dumps(dict(old, cp="1d")) + json.dumps(dict(different, cp="1d"))
        )
        self.assertFalse(nautical_mismatch.valid)
        self.assertEqual(nautical_mismatch.error, "Old and new task UUIDs differ")

        missing_uuid = hook_protocol.probe_on_modify(json.dumps({"status": "pending", "anchor": "w:mon"}))
        self.assertFalse(missing_uuid.valid)
        self.assertFalse(hook_protocol.probe_on_modify(json.dumps(old) + " trailing").valid)
        self.assertFalse(hook_protocol.probe_on_modify(b"12345", max_bytes=4).valid)
        self.assertFalse(hook_protocol.probe_on_add(b"12345", max_bytes=4).valid)

        add_stream = io.BytesIO(json.dumps(old).encode("utf-8"))
        self.assertTrue(hook_protocol.read_on_add(stream=add_stream).valid)
        modify_stream = io.StringIO(json.dumps(old) + "\n" + json.dumps(old))
        self.assertTrue(hook_protocol.read_on_modify(stream=modify_stream).valid)

        stream = io.StringIO()
        hook_protocol.emit_passthrough_json({"description": "Cafe ăîșț ✅"}, stream=stream)
        output = stream.getvalue()
        self.assertIn("ăîșț ✅", output)
        self.assertNotIn("\\u", output)
        self.assertEqual(json.loads(output)["description"], "Cafe ăîșț ✅")

    def test_safe_ordinary_modify_allowlist_rejects_nautical_sensitive_edits(self) -> None:
        old = {
            "uuid": "00000000-0000-4000-8000-000000000707",
            "status": "pending",
            "description": "ordinary edit",
            "project": "home",
            "due": "20270101T090000Z",
            "cp": "1d",
            "chain": "on",
            "chainID": "abcd1234",
            "link": 4,
        }
        for field, value in (
            ("description", "renamed"),
            ("project", "work"),
            ("priority", "H"),
            ("tags", ["next", "phone"]),
            ("depends", ["11111111-1111-1111-1111-111111111111"]),
            ("start", "20260101T090000Z"),
            ("parent", "22222222-2222-2222-2222-222222222222"),
            ("blocks", ["33333333-3333-3333-3333-333333333333"]),
            ("mask", "20270101T090000Z"),
            ("imask", "20270101T090000Z"),
            ("context", "work"),
        ):
            with self.subTest(ordinary_field=field):
                new = dict(old, **{field: value}, modified="20260101T000001Z")
                self.assertTrue(hook_protocol.is_safe_nautical_ordinary_modify(old, new))

        for field, value in (
            ("status", "completed"),
            ("cp", "P2D"),
            ("chain", "off"),
            ("chainMax", 8),
            ("chainUntil", "20280101T000000Z"),
            ("link", 5),
            ("nextLink", "eeeeeeee"),
            ("due", "20270102T090000Z"),
            ("scheduled", "20270101T080000Z"),
            ("wait", "20261231T090000Z"),
            ("custom_uda", "changed"),
        ):
            with self.subTest(sensitive_field=field):
                new = dict(old, **{field: value}, modified="20260101T000001Z")
                self.assertFalse(hook_protocol.is_safe_nautical_ordinary_modify(old, new))
