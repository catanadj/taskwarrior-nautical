"""Static callback contracts for completion-effect adapters."""

from __future__ import annotations

import unittest
from collections.abc import Sequence
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

    def test_completion_spawn_factory_has_typed_host_contract(self) -> None:
        from nautical_core.modify_completion_effects import completion_spawn_ports_for

        self.assertIsNot(get_type_hints(completion_spawn_ports_for)["host"], Any)

    def test_completion_compute_factory_has_typed_host_contract(self) -> None:
        from nautical_core.modify_completion_effects import completion_compute_ports_for

        self.assertIsNot(get_type_hints(completion_compute_ports_for)["host"], Any)

    def test_completion_compute_host_uses_existing_human_delta_contract(self) -> None:
        from nautical_core.modify_completion_effects import _CompletionComputeCore

        self.assertEqual(
            _CompletionComputeCore.__annotations__["humanize_delta"],
            "HumanizeUntilDelta",
        )

    def test_generation_service_contract_includes_completion_draft_builder(self) -> None:
        from nautical_core.modify_generation_effects import ChainGenerationServicePort

        self.assertTrue(hasattr(ChainGenerationServicePort, "build_child_draft"))

    def test_completion_spawn_adapter_orders_set_valued_stripped_fields(self) -> None:
        from nautical_core.modify_completion_effects import (
            _normalize_completion_spawn_result,
        )

        self.assertEqual(
            _normalize_completion_spawn_result(
                ("child123", {"zeta", "alpha"}, True, False, None, "intent-1")
            ),
            ("child123", ["alpha", "zeta"], True, False, None, "intent-1"),
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
        for adapter in (
            _ui_ports_for,
            _print_task_port_for,
            _panel_port_for,
            _end_summary_port_for,
            _feedback_ports_for,
        ):
            self.assertIsNot(get_type_hints(adapter)["host"], Any, adapter.__name__)

    def test_ui_adapter_accepts_only_completion_host_contracts(self) -> None:
        from nautical_core.modify_completion_effects import (
            CompletionComputeHost,
            CompletionPreflightHost,
            CompletionSpawnHost,
            _ui_ports_for,
        )

        self.assertEqual(
            get_type_hints(_ui_ports_for)["host"],
            CompletionComputeHost | CompletionSpawnHost | CompletionPreflightHost,
        )

    def test_completion_ui_panel_owner_has_narrow_render_arguments(self) -> None:
        from nautical_core.modify_completion_effects import _CompletionUIModule

        hints = get_type_hints(_CompletionUIModule.panel)
        self.assertIs(hints["title"], str)
        self.assertEqual(hints["rows"], Sequence[tuple[str | None, Any]])
        for style in ("border_style", "title_style", "label_style"):
            self.assertEqual(hints[style], str | None)

    def test_completion_compute_adapters_match_service_contracts(self) -> None:
        from datetime import datetime

        from nautical_core.modify_completion_effects import (
            caps,
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
        self.assertEqual(get_type_hints(caps)["child_due"], datetime | None)
        warning_hints = get_type_hints(warn_unreasonable_duration)
        self.assertEqual(warning_hints["child_due"], datetime | None)
        self.assertEqual(warning_hints["until_dt"], datetime | None)

    def test_first_recurrence_target_has_typed_callbacks_and_result(self) -> None:
        from datetime import datetime
        from typing import Callable, Mapping

        from nautical_core.modify_completion_compute import first_recurrence_target
        from nautical_core.modify_generation_effects import ChainGenerationServicePort
        from nautical_core.modify_models import DatetimeParserCallback

        hints = get_type_hints(first_recurrence_target)
        self.assertIs(hints["parse_datetime"], DatetimeParserCallback)
        self.assertEqual(hints["format_datetime"], Callable[[datetime], str])
        self.assertEqual(
            hints["generation_service"], Callable[[], ChainGenerationServicePort]
        )
        self.assertEqual(hints["return"], datetime | None)
        self.assertEqual(hints["task"], Mapping[str, Any])
