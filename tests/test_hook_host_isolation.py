from __future__ import annotations

import unittest
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
from unittest.mock import patch

from nautical_core.modify_composition import hook_host


class HookHostIsolationTests(unittest.TestCase):
    def test_redaction_uses_local_fallback_when_core_redactor_fails(self) -> None:
        from nautical_core.hook_runtime import redact_diagnostic_message

        def broken_redactor(_message: str) -> str:
            raise RuntimeError("core redactor unavailable")

        core = type("Core", (), {"diag_log_redact": staticmethod(broken_redactor)})()
        result = redact_diagnostic_message(
            '{"description":"private text","status":"pending"}', core=core
        )

        self.assertEqual(result, '{"description":"[redacted]","status":"pending"}')

    def test_opt_in_stderr_diagnostic_failure_is_contained(self) -> None:
        from nautical_core.hook_runtime import emit_diagnostic

        class BrokenStderr:
            def write(self, _message: str) -> None:
                raise RuntimeError("diagnostic stream defect")

        with (
            patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}),
            patch("nautical_core.hook_runtime.sys.stderr", BrokenStderr()),
        ):
            emit_diagnostic("diagnostic", hook_name="on-add")

    def test_diagnostic_block_emitter_failure_is_contained(self) -> None:
        from nautical_core.hook_runtime import emit_diagnostic_block

        def broken_emitter(_message: str) -> None:
            raise RuntimeError("diagnostic block sink failed")

        emit_diagnostic_block(
            "stats",
            (("tasks", 2),),
            hook_name="on-exit",
            emit=broken_emitter,
            enabled=True,
        )

    def test_shared_runtime_state_exposes_context_and_access_metadata(self) -> None:
        from nautical_core.hook_runtime import HookRuntimeState

        state = HookRuntimeState(
            core=object(),
            target=None,
            context=type("Context", (), {"taskdata": Path("/tmp/taskdata"), "command_prefix": ("task",)})(),
            access="read_only",
        )
        self.assertEqual(state.access, "read_only")
        self.assertEqual(state.taskdata, Path("/tmp/taskdata"))
        self.assertFalse(state.uses_rc_data_location)

    def test_shared_diagnostic_redaction_preserves_unicode_and_masks_task_text(self) -> None:
        from nautical_core.hook_runtime import redact_diagnostic_message

        result = redact_diagnostic_message('{"description":"café","status":"pending"}')
        self.assertEqual(result, '{"description":"[redacted]","status":"pending"}')

    def test_shared_diagnostic_redaction_masks_scalar_text_fields_without_reencoding(self) -> None:
        from nautical_core.hook_runtime import redact_diagnostic_message

        result = redact_diagnostic_message(
            '{"note":null,"annotation":42,"description":"naïve \\"quoted\\"","safe":"keep"}'
        )
        self.assertEqual(
            result,
            '{"note":"[redacted]","annotation":"[redacted]","description":"[redacted]","safe":"keep"}',
        )

    def test_shared_diagnostic_emission_uses_core_event_sink(self) -> None:
        from nautical_core.hook_runtime import emit_diagnostic

        events = []
        core = type(
            "Core",
            (),
            {
                "DiagnosticEvent": type(
                    "DiagnosticEvent",
                    (),
                    {"from_message": staticmethod(lambda message, hook: (message, hook))},
                ),
                "diag": lambda self, event, hook, taskdata: events.append((event, hook, taskdata)),
            },
        )()
        emit_diagnostic('{"note":"secret"}', hook_name="on-add", core=core, taskdata="/tmp/taskdata")
        self.assertEqual(events, [(('{"note":"[redacted]"}', "on-add"), "on-add", "/tmp/taskdata")])

    def test_shared_profiler_is_disabled_without_profile_level(self) -> None:
        from nautical_core.hook_runtime import HookProfiler

        profiler = HookProfiler(level=0)
        self.assertFalse(profiler.enabled)

    def test_on_add_does_not_register_profiler_when_disabled(self) -> None:
        from nautical_core.hooks import add_impl

        registered = []
        with (
            patch.object(add_impl, "_PROFILE_LEVEL", 0),
            patch.object(add_impl.atexit, "register", side_effect=registered.append),
        ):
            profiler = add_impl._build_profiler()

        self.assertFalse(profiler.enabled)
        self.assertEqual(registered, [])

    def test_shared_diagnostic_block_is_bounded_and_opt_in(self) -> None:
        from nautical_core.hook_runtime import emit_diagnostic, emit_diagnostic_block

        emitted = []
        emit_diagnostic_block(
            "stats",
            (("a", 1), ("b", 2), ("c", 3)),
            hook_name="on-exit",
            emit=lambda message: emitted.append(message),
            enabled=True,
            columns=2,
        )
        self.assertEqual(emitted, ["stats:", "  a=1  b=2", "  c=3"])

        stderr = io.StringIO()
        stdout = io.StringIO()
        with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stdout(stdout), redirect_stderr(stderr):
            for hook_name, title in (
                ("on-modify", "diag stats"),
                ("on-exit", "on-exit task stats"),
            ):
                emit_diagnostic_block(
                    title,
                    (("a", 1), ("b", 2), ("c", 3), ("d", 4)),
                    hook_name=hook_name,
                    emit=lambda message: emit_diagnostic(message, hook_name=hook_name),
                    enabled=True,
                    columns=2,
                )

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(),
            "[nautical] diag stats:\n"
            "[nautical]   a=1  b=2\n"
            "[nautical]   c=3  d=4\n"
            "[nautical] on-exit task stats:\n"
            "[nautical]   a=1  b=2\n"
            "[nautical]   c=3  d=4\n",
        )

    def test_modify_lifecycle_diagnostic_is_gated_to_stderr(self) -> None:
        from nautical_core.hooks import modify_impl
        from nautical_core.modify_models import (
            CompletionLifecycleDiagnostic,
            CompletionLifecycleResult,
        )

        result = CompletionLifecycleResult(
            state="retryable",
            reason="Taskwarrior lock busy",
            diagnostic=CompletionLifecycleDiagnostic(
                transition_id="chain01:1->2",
                chain_id="chain01",
                parent_link=1,
                child_link=2,
                stage="spawn",
                attempts=1,
                failure_kind="command_error",
            ),
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(modify_impl, "_load_core", return_value=None),
            patch.object(modify_impl, "core", None),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            os.environ.pop("NAUTICAL_DIAG", None)
            modify_impl._diag_lifecycle_result(result)
            self.assertEqual(stderr.getvalue(), "")
            os.environ["NAUTICAL_DIAG"] = "1"
            modify_impl._diag_lifecycle_result(result)

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("completion lifecycle:", stderr.getvalue())
        self.assertIn("failure_kind=command_error", stderr.getvalue())

    def test_exit_outcome_diagnostics_are_bounded(self) -> None:
        from types import SimpleNamespace

        from nautical_core.exit_diagnostics import emit_outcome_diagnostics

        messages = []
        outcomes = [
            SimpleNamespace(
                intent_id=f"intent-{index}",
                kind=SimpleNamespace(value="retryable"),
                reason="busy",
            )
            for index in range(5)
        ]

        suppressed = emit_outcome_diagnostics(
            outcomes,
            diagnostic=messages.append,
            limit=2,
        )

        self.assertEqual(suppressed, 3)
        self.assertEqual(len(messages), 3)
        self.assertIn("intent-0", messages[0])
        self.assertIn("intent-1", messages[1])
        self.assertIn("suppressed 3 additional", messages[2])

    def test_exit_feedback_reaches_taskwarrior_stream_after_stdout_redirect(self) -> None:
        from nautical_core.hooks import exit_impl

        class DiscardingStream:
            def write(self, _value):
                return None

            def flush(self):
                return None

        redirected = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(exit_impl.sys, "stdout", DiscardingStream()),
            patch.object(exit_impl.sys, "stderr", stderr),
            patch.object(exit_impl.sys, "__stdout__", redirected),
        ):
            exit_impl._emit_exit_feedback("[nautical] test feedback")

        self.assertIn("[nautical] test feedback", redirected.getvalue())
        self.assertIn("[nautical] test feedback", stderr.getvalue())

    def test_exit_feedback_propagates_unexpected_stream_failures(self) -> None:
        from nautical_core.hooks import exit_impl

        class BrokenStream:
            def write(self, _value):
                raise RuntimeError("stream implementation failed")

            def flush(self):
                return None

        with (
            patch.object(exit_impl.sys, "stdout", BrokenStream()),
            patch.object(exit_impl.sys, "__stdout__", None),
            patch.object(exit_impl.sys, "stderr", None),
            self.assertRaisesRegex(RuntimeError, "stream implementation failed"),
        ):
            exit_impl._emit_exit_feedback("[nautical] test feedback")

    def test_hosts_keep_composition_namespaces_separate(self) -> None:
        first_values = {"value": "first"}
        second_values = {"value": "second"}

        first = hook_host(first_values, "first-hook")
        second = hook_host(second_values, "second-hook")

        self.assertEqual(first.value, "first")
        self.assertEqual(second.value, "second")
        self.assertEqual(first.__name__, "first-hook")
        self.assertEqual(second.__name__, "second-hook")

        first_values["value"] = "updated"
        self.assertEqual(first.value, "updated")
        self.assertEqual(second.value, "second")

    def test_missing_composition_attribute_does_not_fall_through(self) -> None:
        host = hook_host({}, "isolated-hook")
        with self.assertRaises(AttributeError):
            _ = host.not_in_composition


if __name__ == "__main__":
    unittest.main()
