from __future__ import annotations

import io
import json
import builtins
from pathlib import Path
from unittest.mock import patch

import nautical_core.hook_protocol as hook_protocol
import nautical_core.hook_results as hook_results
import nautical_core.modify_protocol as modify_protocol
from tests.support.hook_process import HookSubprocessFixture


class HookProtocolTests(HookSubprocessFixture):
    def test_field_detection_does_not_hide_mapping_implementation_failures(self) -> None:
        class BrokenTask(dict):
            def get(self, _field, _default=None):
                raise RuntimeError("task mapping implementation defect")

        with self.assertRaisesRegex(RuntimeError, "task mapping implementation defect"):
            hook_protocol.task_has_add_nautical_fields(BrokenTask({"anchor": "daily"}))

    def test_panic_passthrough_uses_raw_task_after_decoder_adapter_failure(self) -> None:
        first = {"uuid": "first", "description": "first task"}
        latest = {"uuid": "latest", "description": "Cafe ăîșț ✅"}
        stream = io.StringIO()

        with patch("sys.stdout", stream):
            hook_results.panic_passthrough(
                json.dumps(first, ensure_ascii=False)
                + "\n"
                + json.dumps(latest, ensure_ascii=False),
                None,
                decode_latest_task_from_raw=lambda _raw: (_ for _ in ()).throw(
                    RuntimeError("decoder adapter failed")
                ),
            )

        self.assertEqual(json.loads(stream.getvalue()), latest)

    def test_panic_passthrough_emits_empty_json_after_raw_decoder_failure(self) -> None:
        stream = io.StringIO()
        with (
            patch("sys.stdout", stream),
            patch.object(
                hook_results,
                "decode_latest_task_from_raw",
                side_effect=RuntimeError("raw decoder failed"),
            ),
        ):
            hook_results.panic_passthrough("malformed raw input", None)

        self.assertEqual(json.loads(stream.getvalue()), {})

    def test_panic_passthrough_preserves_empty_json_when_emitter_and_flush_fail(self) -> None:
        class BrokenFlushStream(io.StringIO):
            def flush(self) -> None:
                raise OSError("stream closed during flush")

        stream = BrokenFlushStream()
        with (
            patch("sys.stdout", stream),
            patch.object(
                hook_results,
                "emit_passthrough_json",
                side_effect=RuntimeError("normal emitter failed"),
            ),
        ):
            hook_results.panic_passthrough("", None)

        self.assertEqual(json.loads(stream.getvalue()), {})

    def test_panic_passthrough_does_not_mask_hook_error_when_stdout_is_broken(self) -> None:
        class BrokenOutputStream:
            def write(self, _value: str) -> int:
                raise OSError("stdout is closed")

            def flush(self) -> None:
                raise OSError("stdout is closed")

        with (
            patch("sys.stdout", BrokenOutputStream()),
            patch.object(
                hook_results,
                "emit_passthrough_json",
                side_effect=RuntimeError("normal emitter failed"),
            ),
        ):
            hook_results.panic_passthrough("", None)

    def test_codec_import_fallback_does_not_hide_initialization_defects(self) -> None:
        real_import = builtins.__import__
        package_imports = []

        def staged_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level == 1 and name == "task_codec":
                raise ImportError("relative codec import unavailable")
            if name == "nautical_core.task_codec":
                package_imports.append(name)
                raise RuntimeError("package initializer must not run on the fast path")
            if name == "task_codec" and level == 0:
                raise RuntimeError("codec module initialization failed")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch.object(hook_protocol, "DEFAULT_TASK_CODEC", None),
            patch.object(hook_protocol, "TaskCodecError", hook_protocol._ProtocolCodecError),
            patch("builtins.__import__", side_effect=staged_import),
        ):
            with self.assertRaisesRegex(RuntimeError, "codec module initialization failed"):
                hook_protocol._codec()
        self.assertEqual(package_imports, [])

    def test_passthrough_json_does_not_hide_unexpected_flush_failure(self) -> None:
        class BrokenFlushStream(io.StringIO):
            def flush(self) -> None:
                raise RuntimeError("flush adapter defect")

        with self.assertRaisesRegex(RuntimeError, "flush adapter defect"):
            hook_protocol.emit_passthrough_json({"status": "pending"}, stream=BrokenFlushStream())

    def test_modify_json_decoder_translates_excessive_nesting(self) -> None:
        raw = "[" * 2000 + "0" + "]" * 2000

        with self.assertRaisesRegex(modify_protocol.ModifyProtocolError, "Invalid JSON input"):
            modify_protocol.decode_leading_json_objects(raw)

    def test_modify_json_decoder_does_not_hide_unexpected_decoder_failures(self) -> None:
        with patch.object(
            json.JSONDecoder,
            "raw_decode",
            side_effect=RuntimeError("JSON decoder implementation failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "JSON decoder implementation failed"):
                modify_protocol.decode_leading_json_objects("{}")

    def test_protocol_file_load_does_not_import_the_core_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        protocol = root / "nautical_core" / "hook_protocol.py"
        code = (
            "import importlib.util,sys;"
            "spec=importlib.util.spec_from_file_location('_nautical_protocol_isolated',sys.argv[1]);"
            "mod=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(mod);"
            "assert 'nautical_core' not in sys.modules"
        )

        result = self.run_python_code(
            code,
            (str(protocol),),
            cwd=root,
            timeout=5,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_hook_failure_carries_boundary_message_and_exit_code(self) -> None:
        from nautical_core.hook_results import HookFailure

        failure = HookFailure("Invalid input", "bad payload")
        self.assertEqual(failure.title, "Invalid input")
        self.assertEqual(failure.message, "bad payload")
        self.assertEqual(failure.code, 1)

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

        legacy_chainid = hook_protocol.probe_on_add(
            json.dumps(dict(plain, chainid="legacy-1234"))
        )
        self.assertTrue(legacy_chainid.valid, legacy_chainid.error)
        self.assertFalse(legacy_chainid.is_nautical)

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
