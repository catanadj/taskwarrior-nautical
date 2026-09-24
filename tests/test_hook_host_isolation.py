from __future__ import annotations

import unittest
from pathlib import Path

from nautical_core.modify_composition import hook_host


class HookHostIsolationTests(unittest.TestCase):
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
            '{"note":null,"annotation":42,"description":"naïve \\"quoted\\""}'
        )
        self.assertEqual(
            result,
            '{"note":"[redacted]","annotation":"[redacted]","description":"[redacted]"}',
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

    def test_shared_diagnostic_block_is_bounded_and_opt_in(self) -> None:
        from nautical_core.hook_runtime import emit_diagnostic_block

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
