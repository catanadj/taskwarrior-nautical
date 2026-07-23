from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExitDrainServices:
    queue_db_begin_run: Any
    queue_db_end_run: Any
    take_queue_batch: Any
    load_finalized_intents: Any
    exit_progress_scope: Any
    preload_export_uuids: Any
    preload_equivalent_child_slots: Any
    process_queue_entry: Any
    requeue_entries_result: Any
    ack_queue_entries_sqlite_result: Any
    drain_state_factory: Any
    exit_models: Any
    diag: Any


def drain_queue_result(*, services: ExitDrainServices):
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

            requeue_result = services.requeue_entries_result(state.requeue) if state.requeue else services.exit_models.ExitRequeueResult(ok=True, failed=0)
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
