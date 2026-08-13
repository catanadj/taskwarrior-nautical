from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from nautical_core.exit_models import (
    ExitDrainStateProtocol,
    ExitOutboxBatch,
    ExitRetryReleaseResult,
    ExitDrainStats,
    LifecycleBatchPlan,
)


Entry = dict[str, Any]

TakeOutboxBatch = Callable[[], ExitOutboxBatch]
PreloadEntries = Callable[[list[Entry]], None]
PrepareLifecycleBatch = Callable[[list[Entry]], LifecycleBatchPlan | None]
ApplyLifecycleBatch = Callable[[LifecycleBatchPlan], None]
FinalizeBatch = Callable[[ExitDrainStateProtocol], None]
ProcessOutboxEntry = Callable[[int, Entry, ExitDrainStateProtocol], bool]
ReleaseRetryEntries = Callable[[list[Entry]], ExitRetryReleaseResult]
Diagnostic = Callable[[str], None]


class ExitProgressUpdate(Protocol):
    def __call__(
        self,
        *,
        advance: int = 0,
        phase: str | None = None,
        state: ExitDrainStateProtocol | None = None,
    ) -> None: ...


class ExitProgressScope(Protocol):
    def __call__(self, entries_total: int) -> AbstractContextManager[ExitProgressUpdate | None]: ...


class ExitDrainStateFactory(Protocol):
    def __call__(
        self,
        *,
        entries: list[Entry],
        entries_total: int,
    ) -> ExitDrainStateProtocol: ...


@dataclass(slots=True)
class ExitDrainServices:
    take_outbox_batch: TakeOutboxBatch
    exit_progress_scope: ExitProgressScope
    preload_export_uuids: PreloadEntries
    preload_equivalent_child_slots: PreloadEntries
    prepare_lifecycle_batch: PrepareLifecycleBatch | None
    apply_lifecycle_batch: ApplyLifecycleBatch | None
    finalize_lifecycle_batch: FinalizeBatch | None
    process_outbox_entry: ProcessOutboxEntry
    release_retry_entries_result: ReleaseRetryEntries
    drain_state_factory: ExitDrainStateFactory


def drain_outbox_result(*, services: ExitDrainServices) -> ExitDrainStats:
    import time

    drain_t0 = time.perf_counter()
    batch = services.take_outbox_batch()
    entries = batch.entries
    state = services.drain_state_factory(entries=entries, entries_total=batch.entries_total)
    lifecycle_count = len(entries)
    state.lifecycle_defer_verification = lifecycle_count > 1
    state.lifecycle_batch_discovery = lifecycle_count > 1
    with services.exit_progress_scope(batch.entries_total) as progress_update:
        if progress_update is not None:
            progress_update(phase="preload", state=state)
        services.preload_export_uuids(entries)
        services.preload_equivalent_child_slots(entries)
        lifecycle_batch_plan = None
        if services.prepare_lifecycle_batch is not None:
            lifecycle_batch_plan = services.prepare_lifecycle_batch(entries)
            if lifecycle_batch_plan is not None:
                state.lifecycle_batch_plan = lifecycle_batch_plan
        if lifecycle_batch_plan is not None and services.apply_lifecycle_batch is not None:
            services.apply_lifecycle_batch(lifecycle_batch_plan)
        if progress_update is not None:
            progress_update(phase="drain", state=state)

        for idx, entry in enumerate(entries):
            should_break = services.process_outbox_entry(idx, entry, state)
            if progress_update is not None:
                progress_update(advance=1, phase="drain", state=state)
            if should_break:
                break

        if services.finalize_lifecycle_batch is not None:
            services.finalize_lifecycle_batch(state)

        if progress_update is not None:
            progress_update(phase="finalize", state=state)

        retry_release_result = (
            services.release_retry_entries_result(state.retry_entries)
            if state.retry_entries
            else ExitRetryReleaseResult(ok=True, failed=0)
        )
    if not retry_release_result.ok:
        state.errors += retry_release_result.failed

    return state.to_stats_model(drain_t0, retry_release_result.ok, retry_release_result.failed)


__all__ = (
    "ExitDrainServices",
    "drain_outbox_result",
)
