"""Task-scoped chain-generation service binding for on-modify effects."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationPorts:
    state: Any
    module: Any
    core: Any
    recurrence_update_udas: tuple[Any, ...]
    debug_wait_sched: Any
    wait_sched_debug: Any


def chain_generation_service(ports: GenerationPorts):
    service = ports.state.chain_generation_service
    if service is None or getattr(service, "core", None) is not ports.core or tuple(getattr(service, "recurrence_update_udas", ())) != ports.recurrence_update_udas:
        service = ports.module.ChainGenerationService.from_core(
            ports.core,
            recurrence_update_udas=ports.recurrence_update_udas,
            debug_wait_sched=ports.debug_wait_sched,
            wait_sched_debug=ports.wait_sched_debug,
        )
        ports.state.chain_generation_service = service
    return service


def generation_ports_for(host: Any) -> GenerationPorts:
    return GenerationPorts(
        state=host._modify_runtime_state(),
        module=host._module("chain_generation"),
        core=host.core,
        recurrence_update_udas=tuple(host._RECURRENCE_UPDATE_UDAS or ()),
        debug_wait_sched=host._DEBUG_WAIT_SCHED,
        wait_sched_debug=host._LAST_WAIT_SCHED_DEBUG,
    )


__all__ = ("GenerationPorts", "generation_ports_for", "chain_generation_service")
