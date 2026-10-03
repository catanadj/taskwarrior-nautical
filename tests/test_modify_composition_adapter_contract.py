from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nautical_core.modify_composition_adapters as adapters
from nautical_core.modify_composition_adapters import render_disabled_chain_summary_for


class ModifyCompositionAdapterContractTests(unittest.TestCase):
    def test_expiration_warning_propagates_unexpected_renderer_failure(self) -> None:
        fallback_panels: list[str] = []

        def fail_to_render(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("recovery renderer defect")

        capabilities = SimpleNamespace(
            modify_expiration=SimpleNamespace(render_recovery_warning=fail_to_render),
            modify_ui_effects=SimpleNamespace(
                ui_ports_for=lambda _host: object(),
                panel=lambda _ports, title, *_args, **_kwargs: fallback_panels.append(title),
            ),
        )
        host = SimpleNamespace(
            _diag=lambda _message: None,
            core=SimpleNamespace(short_uuid=lambda value: str(value)),
        )

        with (
            patch.object(adapters, "_capabilities", return_value=capabilities),
            patch.object(adapters, "expiration_services_for", return_value=object()),
            self.assertRaisesRegex(RuntimeError, "recovery renderer defect"),
        ):
            adapters.expiration_recovery_warning_for(host, {"uuid": "task-1"}, "reason")

        self.assertEqual(fallback_panels, [])

    def test_disabled_chain_summary_falls_back_when_rich_summary_fails(self) -> None:
        events: list[str] = []
        diagnostics: list[str] = []
        panels: list[tuple[str, list[tuple[str, object]], str]] = []
        now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        modules = {
            "modify_models": SimpleNamespace(TaskView=SimpleNamespace(from_mapping=lambda task: task)),
            "modify_diagnostics_effects": SimpleNamespace(
                end_chain_summary_ports_for=lambda _host: object(),
                end_chain_summary=lambda *_args, **_kwargs: (
                    events.append("rich-summary"),
                    (_ for _ in ()).throw(RuntimeError("renderer defect")),
                )[1],
            ),
            "modify_ui_effects": SimpleNamespace(
                ui_ports_for=lambda _host: object(),
                panel=lambda _ports, title, rows, *, kind: (
                    events.append("fallback-panel"), panels.append((title, list(rows), kind))
                ),
            ),
            "modify_queries": SimpleNamespace(
                query_ports_for=lambda _host: object(),
                cached_format_root_and_age=lambda *_args: "root age",
            ),
        }
        host = SimpleNamespace(
            _workflow_now_utc=lambda: now,
            _module=lambda name: modules[name],
            _diag=diagnostics.append,
            core=SimpleNamespace(short_uuid=lambda value: str(value)[:8]),
        )
        old = {"uuid": "old-task", "chainID": "chain-1", "status": "pending"}
        new = {"uuid": "new-task", "chainID": "chain-1", "status": "pending"}

        render_disabled_chain_summary_for(host, old, new, "recurrence removed")

        self.assertEqual(events, ["rich-summary", "fallback-panel"])
        self.assertEqual(panels[0][0], "⛔ Nautical chain stopped")
        self.assertIn(("Root", "root age"), panels[0][1])
        self.assertEqual(panels[0][2], "summary")
        self.assertEqual(diagnostics, ["removed recurrence chain summary failed: renderer defect"])


if __name__ == "__main__":
    unittest.main()
