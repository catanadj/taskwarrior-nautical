from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from nautical_core.exit_models import (
    ExitDrainStateProtocol,
    ExitQueueBatch,
    ExitQueueWriteResult,
    ExitRequeueResult,
    ExitDrainStats,
)


Entry = dict[str, Any]

QueueDbHook = Callable[[], None]
TakeQueueBatch = Callable[[], ExitQueueBatch]
LoadFinalizedIntents = Callable[[], tuple[set[str], bool]]
PreloadEntries = Callable[[list[Entry]], None]
ProcessQueueEntry = Callable[[int, Entry, ExitDrainStateProtocol], bool]
RequeueEntries = Callable[[list[Entry]], ExitRequeueResult]
AckQueueEntries = Callable[[list[tuple[int, str]]], ExitQueueWriteResult]
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
        finalized_intents: set[str],
        intent_log_ready: bool,
        intent_log_load_ms: float,
    ) -> ExitDrainStateProtocol: ...


@dataclass(slots=True)
class ExitDrainServices:
    queue_db_begin_run: QueueDbHook
    queue_db_end_run: QueueDbHook
    take_queue_batch: TakeQueueBatch
    load_finalized_intents: LoadFinalizedIntents
    exit_progress_scope: ExitProgressScope
    preload_export_uuids: PreloadEntries
    preload_equivalent_child_slots: PreloadEntries
    process_queue_entry: ProcessQueueEntry
    requeue_entries_result: RequeueEntries
    ack_queue_entries_sqlite_result: AckQueueEntries
    drain_state_factory: ExitDrainStateFactory
    diag: Diagnostic


def drain_queue_result(*, services: ExitDrainServices) -> ExitDrainStats:
    services.queue_db_begin_run()
    try:
        import time

        drain_t0 = time.perf_counter()
        batch = services.take_queue_batch()
        entries = batch.entries
        intent_t0 = time.perf_counter()
        finalized_intents, intent_log_ready = services.load_finalized_intents()
        intent_log_load_ms = (time.perf_counter() - intent_t0) * 1000.0
        state = services.drain_state_factory(
            entries=entries,
            entries_total=batch.entries_total,
            finalized_intents=finalized_intents,
            intent_log_ready=bool(intent_log_ready),
            intent_log_load_ms=float(intent_log_load_ms),
        )
        preload_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                preload_entries.append(entry)
                continue
            spawn_intent_id = str(entry.get("spawn_intent_id") or "").strip()
            if spawn_intent_id and spawn_intent_id in finalized_intents:
                continue
            preload_entries.append(entry)
        with services.exit_progress_scope(batch.entries_total) as progress_update:
            if progress_update is not None:
                progress_update(phase="preload", state=state)
            services.preload_export_uuids(preload_entries)
            services.preload_equivalent_child_slots(preload_entries)
            if progress_update is not None:
                progress_update(phase="drain", state=state)

            for idx, entry in enumerate(entries):
                should_break = services.process_queue_entry(idx, entry, state)
                if progress_update is not None:
                    progress_update(advance=1, phase="drain", state=state)
                if should_break:
                    break

            if progress_update is not None:
                progress_update(phase="finalize", state=state)

            requeue_result = (
                services.requeue_entries_result(state.requeue)
                if state.requeue
                else ExitRequeueResult(ok=True, failed=0)
            )
        if not requeue_result.ok:
            state.errors += requeue_result.failed
            services.diag(f"requeue failed for {requeue_result.failed} entries")

        if state.sqlite_acked_claims:
            ack_result = services.ack_queue_entries_sqlite_result(sorted(state.sqlite_acked_claims.items()))
            if not ack_result.ok:
                state.errors += ack_result.count
                services.diag(f"queue db ack failed for {ack_result.count} entries")

        return state.to_stats_model(drain_t0, requeue_result.ok, requeue_result.failed)
    finally:
        services.queue_db_end_run()


__all__ = (
    "ExitDrainServices",
    "drain_queue_result",
)
