"""Task-scoped chain-generation service binding for on-modify effects."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any, Literal, Protocol

from .task_models import NauticalTask, TaskPayload


class NativeUntilGenerationService(Protocol):
    """Generation capability used to carry an occurrence's native expiration."""

    def carry_native_until(
        self,
        parent: NauticalTask,
        child: TaskPayload,
        child_due_utc: datetime,
        kind: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None: ...

class ChainGenerationServicePort(NativeUntilGenerationService, Protocol):
    """Identity needed to reuse one configured generation service."""

    core: object
    recurrence_update_udas: tuple[str, ...]


class GenerationStatePort(Protocol):
    """Mutable runtime slot for the task-scoped generation service."""

    chain_generation_service: ChainGenerationServicePort | None


class CreateChainGenerationService(Protocol):
    """Build a generation service from the composition-owned settings."""

    def __call__(
        self,
        core: object,
        *,
        recurrence_update_udas: tuple[str, ...],
        debug_wait_sched: bool,
        wait_sched_debug: MutableMapping[str, dict[str, Any]] | None,
    ) -> ChainGenerationServicePort: ...


@dataclass(frozen=True, slots=True)
class GenerationPorts:
    state: GenerationStatePort
    create_service: CreateChainGenerationService
    core: object
    recurrence_update_udas: tuple[str, ...]
    debug_wait_sched: bool
    wait_sched_debug: MutableMapping[str, dict[str, Any]] | None


class _ChainGenerationServiceFactory(Protocol):
    @staticmethod
    def from_core(
        core: object,
        *,
        recurrence_update_udas: tuple[str, ...],
        debug_wait_sched: bool,
        wait_sched_debug: MutableMapping[str, dict[str, Any]] | None,
    ) -> ChainGenerationServicePort: ...


class _ChainGenerationModule(Protocol):
    ChainGenerationService: _ChainGenerationServiceFactory


class GenerationHost(Protocol):
    core: object
    _RECURRENCE_UPDATE_UDAS: tuple[str, ...]
    _DEBUG_WAIT_SCHED: bool
    _LAST_WAIT_SCHED_DEBUG: MutableMapping[str, dict[str, Any]] | None

    def _modify_runtime_state(self) -> GenerationStatePort: ...

    def _module(self, name: Literal["chain_generation"]) -> _ChainGenerationModule: ...


def chain_generation_service(ports: GenerationPorts) -> ChainGenerationServicePort:
    service = ports.state.chain_generation_service
    if (
        service is None
        or service.core is not ports.core
        or service.recurrence_update_udas != ports.recurrence_update_udas
    ):
        service = ports.create_service(
            ports.core,
            recurrence_update_udas=ports.recurrence_update_udas,
            debug_wait_sched=ports.debug_wait_sched,
            wait_sched_debug=ports.wait_sched_debug,
        )
        ports.state.chain_generation_service = service
    return service


def generation_ports_for(host: GenerationHost) -> GenerationPorts:
    state = host._modify_runtime_state()
    module = host._module("chain_generation")

    def create_service(
        core: object,
        *,
        recurrence_update_udas: tuple[str, ...],
        debug_wait_sched: bool,
        wait_sched_debug: MutableMapping[str, dict[str, Any]] | None,
    ) -> ChainGenerationServicePort:
        return module.ChainGenerationService.from_core(
            core,
            recurrence_update_udas=recurrence_update_udas,
            debug_wait_sched=debug_wait_sched,
            wait_sched_debug=wait_sched_debug,
        )

    return GenerationPorts(
        state=state,
        create_service=create_service,
        core=host.core,
        recurrence_update_udas=tuple(host._RECURRENCE_UPDATE_UDAS or ()),
        debug_wait_sched=bool(host._DEBUG_WAIT_SCHED),
        wait_sched_debug=host._LAST_WAIT_SCHED_DEBUG,
    )


__all__ = (
    "ChainGenerationServicePort",
    "CreateChainGenerationService",
    "GenerationPorts",
    "GenerationStatePort",
    "generation_ports_for",
    "chain_generation_service",
)
