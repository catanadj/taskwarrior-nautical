"""Static callback contracts for completion-effect adapters."""

from __future__ import annotations

import unittest
from typing import Any, get_type_hints


class ModifyCompletionEffectPortTests(unittest.TestCase):
    def test_completion_preflight_factory_has_typed_host_contract(self) -> None:
        from nautical_core.modify_completion_effects import (
            completion_preflight_context_ports_for,
        )

        self.assertIsNot(
            get_type_hints(completion_preflight_context_ports_for)["host"], Any
        )

    def test_completion_adapter_helpers_have_concrete_callback_returns(self) -> None:
        from nautical_core.modify_completion_effects import (
            _end_summary_port_for,
            _feedback_ports_for,
            _panel_port_for,
            _print_task_port_for,
            _ui_ports_for,
        )

        for adapter in (
            _end_summary_port_for,
            _feedback_ports_for,
            _panel_port_for,
            _print_task_port_for,
            _ui_ports_for,
        ):
            self.assertIsNot(get_type_hints(adapter)["return"], Any, adapter.__name__)
