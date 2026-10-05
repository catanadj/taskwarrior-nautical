from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
import importlib
from typing import get_type_hints
import unittest

modify_generation_effects = importlib.import_module(
    "nautical_core.modify_generation_effects"
)


class ModifyGenerationEffectsContractTests(unittest.TestCase):
    def test_generation_factory_uses_narrow_host_protocol(self) -> None:
        self.assertIs(
            get_type_hints(modify_generation_effects.generation_ports_for)["host"],
            modify_generation_effects.GenerationHost,
        )

    def test_generation_ports_use_typed_service_state_and_factory(self) -> None:
        annotations = get_type_hints(modify_generation_effects.GenerationPorts)
        self.assertIs(
            annotations["state"],
            modify_generation_effects.GenerationStatePort,
        )
        self.assertIs(
            annotations["create_service"],
            modify_generation_effects.CreateChainGenerationService,
        )
        self.assertNotIn("module", annotations)

    def test_generation_due_metadata_is_typed_as_data_not_any(self) -> None:
        metadata = dict[str, object] | None
        self.assertEqual(
            get_type_hints(
                modify_generation_effects.ChainGenerationServicePort.compute_cp_child_due
            )["return"],
            tuple[datetime | None, metadata],
        )
        self.assertEqual(
            get_type_hints(
                modify_generation_effects.ChainGenerationServicePort.compute_anchor_child_due
            )["return"],
            tuple[
                datetime | None,
                metadata,
                modify_generation_effects.AnchorDNF | None,
            ],
        )

    def test_service_is_cached_and_rebuilt_when_configuration_changes(self) -> None:
        state = SimpleNamespace(chain_generation_service=None)
        core = object()
        calls: list[tuple[object, tuple[str, ...], bool, object]] = []

        def create_service(
            selected_core: object,
            *,
            recurrence_update_udas: tuple[str, ...],
            debug_wait_sched: bool,
            wait_sched_debug: object,
        ) -> object:
            calls.append(
                (selected_core, recurrence_update_udas, debug_wait_sched, wait_sched_debug)
            )
            return SimpleNamespace(
                core=selected_core,
                recurrence_update_udas=recurrence_update_udas,
            )

        ports = modify_generation_effects.GenerationPorts(
            state=state,
            create_service=create_service,
            core=core,
            recurrence_update_udas=("rappel",),
            debug_wait_sched=True,
            wait_sched_debug={},
        )

        first = modify_generation_effects.chain_generation_service(ports)
        self.assertIs(modify_generation_effects.chain_generation_service(ports), first)
        changed = modify_generation_effects.GenerationPorts(
            state=state,
            create_service=create_service,
            core=core,
            recurrence_update_udas=("rappel", "next_review"),
            debug_wait_sched=False,
            wait_sched_debug=None,
        )
        second = modify_generation_effects.chain_generation_service(changed)

        self.assertIsNot(second, first)
        self.assertIs(state.chain_generation_service, second)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][1:], (("rappel", "next_review"), False, None))

    def test_generation_factory_resolves_owner_module_at_composition(self) -> None:
        state = SimpleNamespace(chain_generation_service=None)
        core = object()
        module_loads: list[str] = []

        def from_core(
            selected_core: object,
            *,
            recurrence_update_udas: tuple[str, ...],
            debug_wait_sched: bool,
            wait_sched_debug: object,
        ) -> object:
            return SimpleNamespace(
                core=selected_core,
                recurrence_update_udas=recurrence_update_udas,
            )

        module = SimpleNamespace(
            ChainGenerationService=SimpleNamespace(from_core=from_core)
        )
        host = SimpleNamespace(
            _modify_runtime_state=lambda: state,
            _module=lambda name: module_loads.append(name) or module,
            core=core,
            _RECURRENCE_UPDATE_UDAS=("rappel",),
            _DEBUG_WAIT_SCHED=False,
            _LAST_WAIT_SCHED_DEBUG=None,
        )

        ports = modify_generation_effects.generation_ports_for(host)
        self.assertEqual(module_loads, ["chain_generation"])
        service = modify_generation_effects.chain_generation_service(ports)

        self.assertEqual(module_loads, ["chain_generation"])
        self.assertIs(state.chain_generation_service, service)


if __name__ == "__main__":
    unittest.main()
