"""Task-scoped chain-generation service binding for on-modify effects."""

from __future__ import annotations

from typing import Any


def chain_generation_service(host: Any):
    state = host._modify_runtime_state()
    module = host._module("chain_generation")
    configured = tuple(host._RECURRENCE_UPDATE_UDAS or ())
    service = state.chain_generation_service
    if service is None or getattr(service, "core", None) is not host.core or tuple(getattr(service, "recurrence_update_udas", ())) != configured:
        service = module.ChainGenerationService.from_core(
            host.core,
            recurrence_update_udas=configured,
            debug_wait_sched=host._DEBUG_WAIT_SCHED,
            wait_sched_debug=host._LAST_WAIT_SCHED_DEBUG,
        )
        state.chain_generation_service = service
    return service


__all__ = ("chain_generation_service",)
