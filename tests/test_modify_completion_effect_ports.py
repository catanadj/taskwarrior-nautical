"""Static callback contracts for completion-effect adapters."""

from __future__ import annotations

import unittest
from typing import Any, Literal, get_type_hints


class ModifyCompletionEffectPortTests(unittest.TestCase):
    def test_preflight_helpers_retain_service_result_types(self) -> None:
        from nautical_core.modify_completion_effects import (
            kind_or_stop,
            link_numbers_or_fail,
        )

        hints = {
            link_numbers_or_fail.__name__: get_type_hints(link_numbers_or_fail)["return"],
            kind_or_stop.__name__: get_type_hints(kind_or_stop)["return"],
        }
        self.assertEqual(
            hints,
            {
                "link_numbers_or_fail": tuple[int, int] | None,
                "kind_or_stop": str | None,
            },
        )

    def test_completion_snapshot_operations_use_their_result_models(self) -> None:
        from nautical_core.modify_completion_effects import (
            chain_snapshot,
            existing_next_or_fail,
            preflight_context,
        )
        from nautical_core.modify_completion_effects import CompletionPreflightRepository
        from nautical_core.modify_models import CompletionChainSnapshot, CompletionPreflightContext

        self.assertIs(
            get_type_hints(chain_snapshot)["return"], CompletionChainSnapshot
        )
        self.assertEqual(
            get_type_hints(existing_next_or_fail)["chain_snapshot"],
            CompletionChainSnapshot | None,
        )
        preflight_hints = get_type_hints(preflight_context)
        self.assertIs(preflight_hints["repository"], CompletionPreflightRepository)
        self.assertEqual(
            preflight_hints["return"], CompletionPreflightContext | None
        )

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

    def test_completion_compute_adapters_match_service_contracts(self) -> None:
        from datetime import datetime

        from nautical_core.modify_completion_effects import (
            compute_child_due,
            require_child_due_or_fail,
            until_guard_or_stop,
            until_or_fail,
            warn_unreasonable_duration,
        )
        from nautical_core.modify_models import AnchorDNF

        due_result = tuple[datetime | None, dict[str, Any] | None, AnchorDNF | None] | None
        self.assertEqual(get_type_hints(compute_child_due)["return"], due_result)
        self.assertEqual(
            get_type_hints(until_or_fail)["return"], datetime | None | Literal[False]
        )
        self.assertEqual(
            get_type_hints(until_guard_or_stop)["child_due"], datetime | None
        )
        self.assertEqual(
            get_type_hints(until_guard_or_stop)["until_dt"], datetime | None
        )
        self.assertEqual(
            get_type_hints(require_child_due_or_fail)["child_due"], datetime | None
        )
        warning_hints = get_type_hints(warn_unreasonable_duration)
        self.assertEqual(warning_hints["child_due"], datetime | None)
        self.assertEqual(warning_hints["until_dt"], datetime | None)
