from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import unittest

from tests.support.hook_process import ROOT, HookSubprocessFixture


class HookInputContractTests(HookSubprocessFixture):
    def _run(self, hook: str, payload: str, *, diagnostics: bool = False) -> subprocess.CompletedProcess[str]:
        return self.run_hook(hook, payload, diagnostics=diagnostics)

    def test_invalid_inputs_never_traceback_or_emit_stdout(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending"}
        invalid = (
            "",
            "{\"uuid\":",
            "{not-json",
            "[]",
            json.dumps({"status": "pending", "anchor": "w:mon"}),
            json.dumps({"status": "pending", "anchor": "w:mon"})
            + "\n"
            + json.dumps({"status": "pending", "anchor": "w:mon"}),
            "  \n" + json.dumps(task) + "\n{bad",
            "x" * (10 * 1024 * 1024 + 1),
        )
        for hook in ("on-add.nautical", "on-modify.nautical"):
            for payload in invalid:
                with self.subTest(hook=hook, payload_size=len(payload)):
                    process = self._run(hook, payload)
                    self.assertNotEqual(process.returncode, 0)
                    self.assertEqual(process.stdout, "")
                    self.assertNotIn("Traceback", process.stderr)
                    self.assertNotIn("[nautical]", process.stderr)

    def test_on_modify_rejects_mismatched_nautical_task_uuids(self) -> None:
        old = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "status": "pending",
            "anchor": "w:mon",
        }
        new = {
            "uuid": "00000000-0000-4000-8000-000000000222",
            "status": "completed",
            "anchor": "w:mon",
        }

        process = self._run("on-modify.nautical", json.dumps([old, new]))

        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "")

    def test_on_modify_ignores_plain_deletes_without_uuid(self) -> None:
        tasks = (
            [{"status": "deleted"}],
            {"status": "deleted", "description": "plain Taskwarrior delete"},
        )
        for task in tasks:
            with self.subTest(task=task):
                process = self._run("on-modify.nautical", json.dumps(task))
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(len(process.stdout.splitlines()), 1)
                json.loads(process.stdout)

    def test_invalid_input_diagnostics_are_opt_in(self) -> None:
        for hook in ("on-add.nautical", "on-modify.nautical"):
            with self.subTest(hook=hook):
                quiet = self._run(hook, "{not-json")
                diagnostic = self._run(hook, "{not-json", diagnostics=True)
                self.assertNotIn("[nautical]", quiet.stderr)
                self.assertIn("[nautical]", diagnostic.stderr)

    def test_on_modify_invalid_anchor_has_no_stdout(self) -> None:
        old = {
            "uuid": "00000000-0000-4000-8000-000000000611",
            "status": "pending",
            "description": "invalid anchor test",
        }
        new = dict(old, anchor="bad")

        process = self._run(
            "on-modify.nautical",
            json.dumps(old) + "\n" + json.dumps(new),
        )

        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout.strip(), "")

    def test_on_modify_rejects_oversized_stdin_early(self) -> None:
        raw = json.dumps({"uuid": "u", "status": "pending", "description": "x" * 256})
        script = textwrap.dedent(
            """
            import importlib.util
            import io
            import json
            import sys
            from pathlib import Path

            root = Path(sys.argv[1])
            sys.path.insert(0, str(root / "nautical_core"))
            source = root / "nautical_core" / "hooks" / "modify_impl.py"
            spec = importlib.util.spec_from_file_location("_oversized_modify_input", source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module._MAX_JSON_BYTES = 32
            sys.stdin = io.TextIOWrapper(io.BytesIO(__RAW__.encode("utf-8")), encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            sys.stdout, sys.stderr = stdout, stderr
            try:
                module._read_two()
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("oversized on-modify input was accepted")
            assert stdout.getvalue() == ""
            assert "exceeds 32 bytes" in stderr.getvalue(), stderr.getvalue()
            print("ok", file=sys.__stdout__)
            """
        ).replace("__RAW__", repr(raw))
        process = subprocess.run(
            [sys.executable, "-c", script, str(ROOT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(process.stdout.strip(), "ok")

    def test_on_exit_ignores_malformed_and_oversized_input_silently(self) -> None:
        for payload in ("", "{not-json", "[]", "x" * (10 * 1024 * 1024 + 1)):
            with self.subTest(payload_size=len(payload)):
                process = self._run("on-exit.nautical", payload)
                self.assertEqual(process.returncode, 0)
                self.assertEqual(process.stdout, "")
                self.assertEqual(process.stderr, "")

    def test_unicode_task_payload_keeps_strict_json_stdout(self) -> None:
        task = {
            "uuid": "11111111-1111-1111-1111-111111111111",
            "description": "café ăîșț",
            "status": "pending",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        added = self._run("on-add.nautical", json.dumps(task, ensure_ascii=False))
        self.assertEqual(added.returncode, 0)
        self.assertEqual(json.loads(added.stdout), task)
        self.assertEqual(added.stderr, "")

        modified_task = dict(task, description="updated café ăîșț", modified="20260101T000001Z")
        modified = self._run(
            "on-modify.nautical",
            json.dumps(task, ensure_ascii=False) + "\n" + json.dumps(modified_task, ensure_ascii=False),
        )
        self.assertEqual(modified.returncode, 0)
        self.assertEqual(json.loads(modified.stdout), modified_task)
        self.assertEqual(modified.stderr, "")

    def test_diagnostics_never_contaminate_task_json_stdout(self) -> None:
        task = {
            "uuid": "22222222-2222-4222-8222-222222222222",
            "description": "plain task",
            "status": "pending",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        process = self._run(
            "on-add.nautical",
            json.dumps(task, ensure_ascii=False),
            diagnostics=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), task)
        self.assertNotIn("[nautical]", process.stdout)
        self.assertTrue(process.stderr == "" or "[nautical]" in process.stderr)

    def test_on_add_flushes_stdout_after_passthrough(self) -> None:
        task = {"uuid": "00000000-0000-4000-8000-000000000111", "status": "pending"}
        script = textwrap.dedent(
            """
            import io
            import json
            import sys
            from nautical_core.hooks import add_impl

            class FlushIO(io.StringIO):
                flush_count = 0

                def flush(self):
                    self.flush_count += 1
                    super().flush()

            expected = json.loads(__EXPECTED__)
            stdout = FlushIO()
            sys.stdout = stdout
            add_impl.main()
            if stdout.flush_count < 1 or json.loads(stdout.getvalue()) != expected:
                raise SystemExit(3)
            """
        ).replace("__EXPECTED__", repr(json.dumps(task)))
        process = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(task),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )

        self.assertEqual(process.returncode, 0, process.stderr)

    def test_on_add_anchor_routes_keep_json_stdout_and_panel_stderr(self) -> None:
        task = {
            "uuid": "44444444-4444-4444-8444-444444444444",
            "description": "weekly review",
            "status": "pending",
            "anchor": "w:mon",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        process = self._run("on-add.nautical", json.dumps(task, ensure_ascii=False))
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["chain"], "on")
        self.assertEqual(result["anchor"], "w:mon")
        self.assertNotIn("Anchor Preview", process.stdout)

    def test_on_add_invalid_anchor_is_a_clean_failure(self) -> None:
        task = {
            "uuid": "55555555-5555-4555-8555-555555555555",
            "description": "invalid recurrence",
            "status": "pending",
            "anchor": "not-a-valid-expression",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        process = self._run("on-add.nautical", json.dumps(task, ensure_ascii=False))
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "")
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
