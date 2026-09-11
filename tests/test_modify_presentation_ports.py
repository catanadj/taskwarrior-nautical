"""Contract tests for host-free modify presentation effects."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import inspect
from types import SimpleNamespace
import unittest


class ModifyPresentationPortTests(unittest.TestCase):
    def test_presentation_operations_do_not_accept_hook_host(self) -> None:
        from nautical_core import modify_presentation_effects as presentation

        operation_names = (
            "chain_colour_for_task",
            "future_style_for_chain",
            "render_lifecycle_result",
        )
        for name in operation_names:
            first_parameter = next(iter(inspect.signature(getattr(presentation, name)).parameters.values()))
            self.assertIn(first_parameter.name, {"port", "ports", "services"}, name)

    def test_ui_effects_use_explicit_frozen_ports(self) -> None:
        from nautical_core.modify_ui_effects import UIEffectsPorts, print_task

        emitted: list[tuple[dict, object]] = []
        core = object()
        ports = UIEffectsPorts(
            core=lambda: core,
            load_core=lambda: None,
            override=lambda _name: None,
            emit_passthrough_json=lambda _task: self.fail("unexpected passthrough"),
            emit_task_json=lambda task, *, sanitize, core: emitted.append((task, core)),
            stderr_write=lambda _message: None,
        )

        print_task(ports, {"description": "Café"})

        self.assertEqual(emitted, [({"description": "Café"}, core)])
        with self.assertRaises(FrozenInstanceError):
            ports.core = lambda: None  # type: ignore[misc]

    def test_ui_task_output_falls_back_to_passthrough_when_core_load_fails(self) -> None:
        from nautical_core.modify_ui_effects import UIEffectsPorts, print_task

        passthrough: list[dict] = []
        task = {"description": "malformed-safe"}
        ports = UIEffectsPorts(
            core=lambda: None,
            load_core=lambda: (_ for _ in ()).throw(RuntimeError("core unavailable")),
            override=lambda _name: None,
            emit_passthrough_json=passthrough.append,
            emit_task_json=lambda *_args, **_kwargs: self.fail("unexpected task emitter"),
            stderr_write=lambda _message: None,
        )

        print_task(ports, task)

        self.assertEqual(passthrough, [task])

    def test_lifecycle_result_renderer_uses_only_panel_port(self) -> None:
        from nautical_core.modify_presentation_effects import LifecycleResultPort, render_lifecycle_result

        rendered: list[tuple[str, list[tuple[str, str]], str]] = []
        result = SimpleNamespace(
            state="manual_review",
            reason="ambiguous child",
            child_short="abc123",
            spawn_intent_id="intent-1",
        )

        render_lifecycle_result(
            LifecycleResultPort(
                panel=lambda title, rows, **options: rendered.append((title, rows, options["kind"]))
            ),
            result,
            {},
        )

        self.assertEqual(rendered[0][0], "⛓ Chain warning")
        self.assertEqual(rendered[0][2], "warning")
        self.assertIn(("Intent", "intent-1"), rendered[0][1])


if __name__ == "__main__":
    unittest.main()
