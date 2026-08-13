#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heavy on-exit implementation loaded lazily by the executable wrapper.

Drains Nautical spawn intents after Taskwarrior releases its lock.
Reads JSONL queue entries, imports child tasks, and updates parent nextLink.
"""

from __future__ import annotations

import sys
import os
import time
import importlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path

_IMPL_CORE_DIR = Path(__file__).resolve().parent.parent
HOOK_DIR = _IMPL_CORE_DIR.parent
_TW_DIR_BOOT = _IMPL_CORE_DIR.parent
try:
    import hook_bootstrap
except ModuleNotFoundError:
    hook_bootstrap = None
    _bootstrap_paths = [
        HOOK_DIR / 'nautical_core' / 'hook_bootstrap.py',
        _TW_DIR_BOOT / 'nautical_core' / 'hook_bootstrap.py',
    ]
    _core_path_raw = (os.environ.get("NAUTICAL_CORE_PATH") or "").strip()
    if _core_path_raw:
        try:
            _core_path = Path(_core_path_raw).expanduser()
            _bootstrap_paths.extend([
                _core_path / 'hook_bootstrap.py',
                _core_path / 'nautical_core' / 'hook_bootstrap.py',
            ])
        except Exception:
            pass
    for _bootstrap_path in _bootstrap_paths:
        try:
            if not _bootstrap_path.is_file():
                continue
            _spec = importlib.util.spec_from_file_location('hook_bootstrap', _bootstrap_path)
            if _spec and _spec.loader:
                _bootstrap_mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_bootstrap_mod)
                hook_bootstrap = _bootstrap_mod
                break
        except Exception:
            continue
    if hook_bootstrap is None:
        raise
_IMPORT_T0 = time.perf_counter()
_IMPORT_MS: float | None = None
hook_bootstrap.ensure_utf8_stdio()


TW_DIR = _TW_DIR_BOOT

def _trusted_core_base(default_base: Path) -> Path:
    return hook_bootstrap.trusted_core_base(
        default_base,
        env=os.environ,
        diag_enabled=os.environ.get("NAUTICAL_DIAG") == "1",
    )


def _core_target_from_base(base: Path) -> Path | None:
    return hook_bootstrap.core_target_from_base(base)

_CORE_BASE = _trusted_core_base(TW_DIR)
_EARLY_EXIT_PROBE = None

if (
    __name__ == "__main__"
    and os.environ.get("NAUTICAL_DIAG") != "1"
    and os.environ.get("NAUTICAL_BENCH_FORCE_FULL") != "1"
):
    _path_support, _path_support_path, _path_support_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "config_support.py",
        "_nautical_exit_path_support",
    )
    _exit_probe, _exit_probe_path, _exit_probe_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "exit_probe.py",
        "_nautical_exit_probe",
    )
    if _path_support is not None and _exit_probe is not None:
        try:
            _early_taskdata = hook_bootstrap.resolve_task_data_context_light(
                path_support=_path_support,
                argv=sys.argv[1:],
                env=os.environ,
                tw_dir=str(TW_DIR),
            )
            if _early_taskdata is not None:
                _EARLY_EXIT_PROBE = _exit_probe.probe_exit_work(_early_taskdata[0])
        except Exception:
            _EARLY_EXIT_PROBE = None
        if _EARLY_EXIT_PROBE is not None and _EARLY_EXIT_PROBE.definitely_empty:
            raise SystemExit(0)


import json
import random
import sqlite3
from contextlib import contextmanager
from typing import Any

try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None


def _read_found(read) -> bool:
    return isinstance(read, _module("integration_models").Found)


def _read_unavailable(read) -> bool:
    return isinstance(read, _module("integration_models").Unavailable)


def _read_value(read):
    return read.value if _read_found(read) else None


def _read_reason(read, fallback: str) -> str:
    models = _module("integration_models")
    if isinstance(read, models.Unavailable):
        return read.evidence.detail or fallback
    if isinstance(read, models.Absent):
        return read.reason or fallback
    return fallback


core = None
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
_HOOK_SUPPORT = None
_HOOK_SUPPORT_LOAD_FAILED = False
_EXIT_QUERIES = None
_EXIT_QUERIES_LOAD_FAILED = False
_EXIT_SIDE_EFFECTS = None
_EXIT_SIDE_EFFECTS_LOAD_FAILED = False
_EXIT_ENTRY_FLOW = None
_EXIT_ENTRY_FLOW_LOAD_FAILED = False
_QUEUE_STORE = None
_QUEUE_STORE_LOAD_FAILED = False
_QUEUE_MODELS = None
_QUEUE_MODELS_LOAD_FAILED = False
_EXIT_MODELS = None
_EXIT_MODELS_LOAD_FAILED = False
_EXIT_RUNTIME = None
_EXIT_RUNTIME_LOAD_FAILED = False
_EXIT_DRAIN_FLOW = None
_EXIT_DRAIN_FLOW_LOAD_FAILED = False
_HOOK_CONTEXT = None
_HOOK_CONTEXT_LOAD_FAILED = False
_HOOK_ENGINE = None
_HOOK_ENGINE_LOAD_FAILED = False
_HOOK_RESULTS = None
_HOOK_RESULTS_LOAD_FAILED = False
_HOOK_RUNTIME = None
_HOOK_RUNTIME_LOAD_FAILED = False
_INTEGRATION_CONTEXT_MODULE = None
_INTEGRATION_CONTEXT_MODULE_LOAD_FAILED = False
_INTEGRATION_MODELS = None
_INTEGRATION_MODELS_LOAD_FAILED = False
_TASKWARRIOR_MUTATIONS = None
_TASKWARRIOR_MUTATIONS_LOAD_FAILED = False
_INTEGRATION_CONTEXT = None
_CORE_READY = False
_HOOK_MODULE_ACCESS = None
_LIFECYCLE_MODELS = None
_LIFECYCLE_MODELS_LOAD_FAILED = False
_LIFECYCLE_PLANNER = None
_LIFECYCLE_PLANNER_LOAD_FAILED = False
_LIFECYCLE_EXECUTOR = None
_LIFECYCLE_EXECUTOR_LOAD_FAILED = False
_LIFECYCLE_OUTBOX = None
_LIFECYCLE_OUTBOX_LOAD_FAILED = False
_MODULE_SPECS = {
    "hook_runtime": (
        "_HOOK_RUNTIME",
        "_HOOK_RUNTIME_LOAD_FAILED",
        "hook_runtime.py",
        "nautical_core.hook_runtime",
    ),
    "integration_context": (
        "_INTEGRATION_CONTEXT_MODULE",
        "_INTEGRATION_CONTEXT_MODULE_LOAD_FAILED",
        "integration_context.py",
        "nautical_core.integration_context",
    ),
    "integration_models": (
        "_INTEGRATION_MODELS",
        "_INTEGRATION_MODELS_LOAD_FAILED",
        "integration_models.py",
        "nautical_core.integration_models",
    ),
    "taskwarrior_mutations": (
        "_TASKWARRIOR_MUTATIONS",
        "_TASKWARRIOR_MUTATIONS_LOAD_FAILED",
        "taskwarrior_mutations.py",
        "nautical_core.taskwarrior_mutations",
    ),
    "hook_support": (
        "_HOOK_SUPPORT",
        "_HOOK_SUPPORT_LOAD_FAILED",
        "hook_support.py",
        "nautical_core.hook_support",
    ),
    "exit_side_effects": (
        "_EXIT_SIDE_EFFECTS",
        "_EXIT_SIDE_EFFECTS_LOAD_FAILED",
        "exit_side_effects.py",
        "nautical_core.exit_side_effects",
    ),
    "exit_entry_flow": (
        "_EXIT_ENTRY_FLOW",
        "_EXIT_ENTRY_FLOW_LOAD_FAILED",
        "exit_entry_flow.py",
        "nautical_core.exit_entry_flow",
    ),
    "queue_store": (
        "_QUEUE_STORE",
        "_QUEUE_STORE_LOAD_FAILED",
        "queue_store.py",
        "nautical_core.queue_store",
    ),
    "queue_models": (
        "_QUEUE_MODELS",
        "_QUEUE_MODELS_LOAD_FAILED",
        "queue_models.py",
        "nautical_core.queue_models",
    ),
    "exit_models": (
        "_EXIT_MODELS",
        "_EXIT_MODELS_LOAD_FAILED",
        "exit_models.py",
        "nautical_core.exit_models",
    ),
    "lifecycle_models": (
        "_LIFECYCLE_MODELS",
        "_LIFECYCLE_MODELS_LOAD_FAILED",
        "lifecycle_models.py",
        "nautical_core.lifecycle_models",
    ),
    "lifecycle_planner": (
        "_LIFECYCLE_PLANNER",
        "_LIFECYCLE_PLANNER_LOAD_FAILED",
        "lifecycle_planner.py",
        "nautical_core.lifecycle_planner",
    ),
    "lifecycle_executor": (
        "_LIFECYCLE_EXECUTOR",
        "_LIFECYCLE_EXECUTOR_LOAD_FAILED",
        "lifecycle_executor.py",
        "nautical_core.lifecycle_executor",
    ),
    "lifecycle_outbox": (
        "_LIFECYCLE_OUTBOX",
        "_LIFECYCLE_OUTBOX_LOAD_FAILED",
        "lifecycle_outbox.py",
        "nautical_core.lifecycle_outbox",
    ),
    "exit_runtime": (
        "_EXIT_RUNTIME",
        "_EXIT_RUNTIME_LOAD_FAILED",
        "exit_runtime.py",
        "nautical_core.exit_runtime",
    ),
    "exit_drain_flow": (
        "_EXIT_DRAIN_FLOW",
        "_EXIT_DRAIN_FLOW_LOAD_FAILED",
        "exit_drain_flow.py",
        "nautical_core.exit_drain_flow",
    ),
    "hook_context": (
        "_HOOK_CONTEXT",
        "_HOOK_CONTEXT_LOAD_FAILED",
        "hook_context.py",
        "nautical_core.hook_context",
    ),
    "hook_engine": (
        "_HOOK_ENGINE",
        "_HOOK_ENGINE_LOAD_FAILED",
        "hook_engine.py",
        "nautical_core.hook_engine",
    ),
    "hook_results": (
        "_HOOK_RESULTS",
        "_HOOK_RESULTS_LOAD_FAILED",
        "hook_results.py",
        "nautical_core.hook_results",
    ),
}
core = None
_CORE_IMPORT_TARGET = None
_CORE_IMPORT_ERROR = None


def _resolve_task_data_context() -> tuple[str, bool]:
    return hook_bootstrap.resolve_task_data_context_lazy(
        core=core,
        core_import_error=_CORE_IMPORT_ERROR,
        core_import_target=_CORE_IMPORT_TARGET,
        core_base=_CORE_BASE,
        tw_dir=str(TW_DIR),
        argv=sys.argv[1:],
        env=os.environ,
    )

_TASKDATA_RAW = ""
_USE_RC_DATA_LOCATION = False
TW_DATA_DIR = Path(TW_DIR).expanduser()
_IMPORT_MS = (time.perf_counter() - _IMPORT_T0) * 1000.0


def _load_core() -> None:
    """Load the configured core only when the exit lifecycle is entered."""
    global core, _CORE_IMPORT_TARGET, _CORE_IMPORT_ERROR, _IMPORT_MS, _CORE_READY
    if core is not None and _CORE_READY:
        return
    _initialize_integration_context()
    _IMPORT_MS = (time.perf_counter() - _IMPORT_T0) * 1000.0
    globals()["_QUEUE_MAX_LINES"] = _env_int(
        "NAUTICAL_SPAWN_QUEUE_MAX_LINES",
        int(getattr(core, "SPAWN_QUEUE_DRAIN_MAX_ITEMS", 200)),
        min_value=1,
        max_value=100000,
    )
    _CORE_READY = True


def _tw_data_dir_path() -> Path:
    td = TW_DATA_DIR
    if isinstance(td, Path):
        return td
    try:
        return Path(str(td)).expanduser()
    except Exception:
        return Path(".")


def _hook_runtime_module():
    global _HOOK_RUNTIME
    if _HOOK_RUNTIME is None:
        _HOOK_RUNTIME = importlib.import_module("nautical_core.hook_runtime")
    return _HOOK_RUNTIME


def _hook_module_access():
    global _HOOK_MODULE_ACCESS
    if _HOOK_MODULE_ACCESS is None:
        hook_runtime = _hook_runtime_module()
        _HOOK_MODULE_ACCESS = hook_runtime.HookModuleAccess(globals(), _MODULE_SPECS)
    return _HOOK_MODULE_ACCESS


def _module(name: str, *, required: bool = True):
    return _hook_module_access().module(name, required=required)

def _initialize_integration_context() -> None:
    global core, _CORE_IMPORT_TARGET, _INTEGRATION_CONTEXT
    global _TASKDATA_RAW, _USE_RC_DATA_LOCATION, TW_DATA_DIR
    if _INTEGRATION_CONTEXT is not None:
        return
    hook_runtime = _hook_runtime_module()
    core, target, context = hook_runtime.initialize_integration_context(
        module_access=_hook_module_access(),
        hook_bootstrap=hook_bootstrap,
        core_base=_CORE_BASE,
        argv=tuple(sys.argv[1:]),
        tw_dir=str(TW_DIR),
        access="mutation",
    )
    _CORE_IMPORT_TARGET = target
    _INTEGRATION_CONTEXT = context
    TW_DATA_DIR = context.taskdata
    _TASKDATA_RAW = str(context.taskdata)
    _USE_RC_DATA_LOCATION = len(context.command_prefix) > 1





def _build_hook_runtime_context():
    hook_runtime = _hook_runtime_module()
    return hook_runtime.build_hook_runtime_context(
        module_access=_hook_module_access(),
        hook_name="on-exit",
        integration_context=_INTEGRATION_CONTEXT,
        hook_dir=str(HOOK_DIR),
        import_ms=_IMPORT_MS,
    )

def _nautical_state_dir_path() -> Path:
    tw_data_dir = _tw_data_dir_path()
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        return queue_store.nautical_state_dir_path(tw_data_dir)
    return tw_data_dir / ".nautical-state"

def _nautical_lock_dir_path() -> Path:
    tw_data_dir = _tw_data_dir_path()
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        return queue_store.nautical_lock_dir_path(tw_data_dir)
    return tw_data_dir / ".nautical-locks"

# Do not import queue_store merely to construct defaults.  run_hook refreshes
# these paths after the lightweight TASKDATA resolver has run.
_QUEUE_DB_PATH = TW_DATA_DIR / ".nautical-state" / ".nautical_queue.db"
_DEAD_LETTER_PATH = TW_DATA_DIR / ".nautical-state" / ".nautical_dead_letter.jsonl"
_DEAD_LETTER_LOCK = TW_DATA_DIR / ".nautical-locks" / ".nautical_dead_letter.lock"
_ORPHAN_CLEANUP_EVIDENCE_PATH = TW_DATA_DIR / ".nautical-state" / ".nautical_orphan_cleanup.jsonl"
_ORPHAN_CLEANUP_EVIDENCE_LOCK = TW_DATA_DIR / ".nautical-locks" / ".nautical_orphan_cleanup.lock"
HOOK_IMPL_API = 1
NAUTICAL_HOOK_VERSION = "updateG-20260328"
_QUEUE_LOCK_FAIL_MARKER = TW_DATA_DIR / ".nautical-locks" / ".nautical_spawn_queue.lock_failed"
_QUEUE_LOCK_FAIL_COUNT = TW_DATA_DIR / ".nautical-locks" / ".nautical_spawn_queue.lock_failed.count"
_DURABLE_QUEUE = os.environ.get("NAUTICAL_DURABLE_QUEUE") == "1"
# When set, exit 1 if any spawns were dead-lettered or errored (for scripting/monitoring).
_EXIT_STRICT = (os.environ.get("NAUTICAL_EXIT_STRICT") or "").strip().lower() in ("1", "true", "yes", "on")

def _migrate_legacy_nautical_state() -> None:
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        issues = queue_store.migrate_nautical_state(
            tw_data_dir=TW_DATA_DIR,
            extra_file_pairs=((_intent_log_path(), TW_DATA_DIR / ".nautical_spawn_intents.jsonl"),),
        )
        for issue in issues:
            _diag(f"queue state migration failed: {issue.current} from {issue.legacy}: {issue.error}")
        globals()["_QUEUE_DB_PATH"] = queue_store.queue_db_path(TW_DATA_DIR)
        globals()["_DEAD_LETTER_PATH"] = queue_store.dead_letter_path(TW_DATA_DIR)
        globals()["_DEAD_LETTER_LOCK"] = queue_store.dead_letter_lock_path(TW_DATA_DIR)
        return

def _env_float(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    return hook_bootstrap.env_float(
        name,
        default,
        min_value=min_value,
        max_value=max_value,
    )


def _env_int(
    name: str,
    default: int,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    return hook_bootstrap.env_int(
        name,
        default,
        min_value=min_value,
        max_value=max_value,
    )


_DEAD_LETTER_RETENTION_DAYS = _env_int("NAUTICAL_DEAD_LETTER_RETENTION_DAYS", 30, min_value=0, max_value=3650)
_QUEUE_MAX_LINES = _env_int(
    "NAUTICAL_SPAWN_QUEUE_MAX_LINES",
    200,
    min_value=1,
    max_value=100000,
)
_DEAD_LETTER_MAX_BYTES = _env_int("NAUTICAL_DEAD_LETTER_MAX_BYTES", 524288, min_value=0, max_value=100 * 1024 * 1024)
_QUEUE_RETRY_MAX = _env_int("NAUTICAL_QUEUE_RETRY_MAX", 6, min_value=0, max_value=100)
_TASK_TIMEOUT_EXPORT = _env_float("NAUTICAL_TASK_TIMEOUT_EXPORT", 3.0, min_value=0.1, max_value=300.0)
_TASK_TIMEOUT_IMPORT = _env_float("NAUTICAL_TASK_TIMEOUT_IMPORT", 8.0, min_value=0.1, max_value=300.0)
_TASK_TIMEOUT_MODIFY = _env_float("NAUTICAL_TASK_TIMEOUT_MODIFY", 4.0, min_value=0.1, max_value=300.0)
_TASK_RETRIES_EXPORT = _env_int("NAUTICAL_TASK_RETRIES_EXPORT", 2, min_value=0, max_value=20)
_TASK_RETRIES_MODIFY = _env_int("NAUTICAL_TASK_RETRIES_MODIFY", 2, min_value=0, max_value=20)
_TASK_RETRY_DELAY = _env_float("NAUTICAL_TASK_RETRY_DELAY", 0.2, min_value=0.0, max_value=10.0)
_QUEUE_LOCK_RETRIES = _env_int("NAUTICAL_QUEUE_LOCK_RETRIES", 6, min_value=0, max_value=100)
_QUEUE_LOCK_SLEEP_BASE = _env_float("NAUTICAL_QUEUE_LOCK_SLEEP_BASE", 0.03, min_value=0.0, max_value=10.0)
_QUEUE_LOCK_STALE_AFTER = _env_float("NAUTICAL_QUEUE_LOCK_STALE_AFTER", 30.0, min_value=0.0, max_value=86400.0)
_INTENT_LOG_MAX_BYTES = _env_int("NAUTICAL_INTENT_LOG_MAX_BYTES", 524288, min_value=0, max_value=100 * 1024 * 1024)
_INTENT_LOG_MAX_ENTRIES = _env_int("NAUTICAL_INTENT_LOG_MAX_ENTRIES", 20000, min_value=0, max_value=1000000)
_LOCK_STORM_THRESHOLD = _env_int("NAUTICAL_LOCK_STORM_THRESHOLD", 8, min_value=0, max_value=1000)
_LOCK_BACKOFF_BASE = _env_float("NAUTICAL_LOCK_BACKOFF_BASE", 0.05, min_value=0.0, max_value=60.0)
_LOCK_BACKOFF_MAX = _env_float("NAUTICAL_LOCK_BACKOFF_MAX", 1.0, min_value=0.0, max_value=300.0)
_QUEUE_PROCESSING_STALE_AFTER = _env_float(
    "NAUTICAL_QUEUE_PROCESSING_STALE_AFTER",
    300.0,
    min_value=0.0,
    max_value=7 * 86400.0,
)
_QUEUE_DB_CONNECT_RETRIES = _env_int("NAUTICAL_QUEUE_DB_CONNECT_RETRIES", 3, min_value=1, max_value=20)
_QUEUE_DB_CONNECT_TIMEOUT_MAX = _env_float(
    "NAUTICAL_QUEUE_DB_CONNECT_TIMEOUT_MAX",
    60.0,
    min_value=1.0,
    max_value=300.0,
)
_QUEUE_DB_CONNECT_BACKOFF_BASE = _env_float(
    "NAUTICAL_QUEUE_DB_CONNECT_BACKOFF_BASE",
    0.05,
    min_value=0.0,
    max_value=10.0,
)

_EXIT_RUNTIME_STATE = None
_EXIT_PRELOAD_CHUNK_SIZE = 32
_EXIT_EQUIV_PRELOAD_CHUNK_SIZE = 8
_EXIT_DIAG_STATS: dict[str, Any] = {}


def _exit_runtime_state():
    global _EXIT_RUNTIME_STATE
    if _EXIT_RUNTIME_STATE is None:
        exit_runtime = _module("exit_runtime")
        _EXIT_RUNTIME_STATE = exit_runtime.new_runtime_state()
        _EXIT_RUNTIME_STATE.diag_stats = _EXIT_DIAG_STATS
    return _EXIT_RUNTIME_STATE


def _reset_exit_runtime_state() -> None:
    global _EXIT_RUNTIME_STATE, _EXIT_DIAG_STATS
    exit_runtime = _module("exit_runtime")
    _EXIT_RUNTIME_STATE = exit_runtime.new_runtime_state()
    _EXIT_DIAG_STATS = _EXIT_RUNTIME_STATE.diag_stats


def _reset_exit_diag_stats() -> None:
    global _EXIT_DIAG_STATS
    _EXIT_DIAG_STATS = {}
    _exit_runtime_state().diag_stats = _EXIT_DIAG_STATS


def _diag_count_exit(key: str, inc: float | int = 1) -> None:
    try:
        state = _exit_runtime_state()
        stats = state.diag_stats
        stats[key] = stats.get(key, 0) + inc
    except Exception:
        pass


@contextmanager
def _task_phase(name: str):
    state = _exit_runtime_state()
    previous = state.task_phase
    state.task_phase = str(name or "unclassified")
    try:
        yield
    finally:
        state.task_phase = previous


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "yes", "true", "on"}:
        return True
    if raw in {"0", "no", "false", "off"}:
        return False
    return bool(default)


def _exit_progress_enabled(entries_total: int) -> bool:
    if int(entries_total or 0) < 2:
        return False
    if not sys.stderr.isatty():
        return False
    if str(os.environ.get("TERM") or "").strip().lower() == "dumb":
        return False
    return _env_bool("NAUTICAL_EXIT_PROGRESS", bool(getattr(core, "EXIT_PROGRESS", True)))


def _exit_progress_counts(state: object | None) -> str:
    if state is None:
        return "ok:0 skip:0 rq:0 dead:0"
    try:
        ok = int(getattr(state, "processed", 0) or 0)
        skip = int(getattr(state, "skipped_idempotent", 0) or 0)
        rq = len(getattr(state, "requeue", []) or [])
        dead = int(getattr(state, "dead_lettered", 0) or 0)
        return f"ok:{ok} skip:{skip} rq:{rq} dead:{dead}"
    except Exception:
        return "ok:0 skip:0 rq:0 dead:0"


@contextmanager
def _exit_progress_scope(entries_total: int):
    if not _exit_progress_enabled(entries_total):
        yield None
        return
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except Exception:
        yield None
        return
    try:
        console = Console(file=sys.stderr, force_terminal=True)
        with Progress(
            TextColumn("[bold cyan]Nautical[/]"),
            TextColumn("[dim]{task.fields[phase]}[/]"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[dim]{task.fields[counts]}[/]"),
            console=console,
            transient=True,
            refresh_per_second=10,
        ) as progress:
            task_id = progress.add_task(
                "drain",
                total=max(1, int(entries_total or 0)),
                phase="preload",
                counts=_exit_progress_counts(None),
            )

            def _advance(*, advance: int = 0, phase: str | None = None, state: object | None = None) -> None:
                try:
                    kwargs = {
                        "advance": max(0, int(advance or 0)),
                        "counts": _exit_progress_counts(state),
                    }
                    if phase is not None:
                        kwargs["phase"] = str(phase)
                    progress.update(task_id, **kwargs)
                except Exception:
                    pass

            yield _advance
    except Exception:
        yield None


def _chunked(items: list[Any], size: int):
    size = max(1, int(size or 1))
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def _preload_export_uuids(entries: list[dict]) -> None:
    relevant = any(
        isinstance(entry, dict)
        and (
            str(entry.get("parent_uuid") or "").strip()
            or str((entry.get("child") or {}).get("uuid") or "").strip()
        )
        for entry in entries or []
    )
    if not relevant:
        return
    repository = _exit_runtime_state().repository
    if repository is None:
        raise RuntimeError("on-exit task read repository is unavailable")
    with _task_phase("preload_uuid"):
        repository.broad_snapshot(
            identity="exit-chain-state",
            filters=("chain:on",),
            statuses=("completed", "deleted", "pending", "waiting"),
            complete_chain_history=True,
        )
    _diag_count_exit("preload_export_chunks")


def _queue_db_begin_run() -> None:
    state = _exit_runtime_state()
    state.run_queue_db_active = True
    state.run_queue_db_conn = None
    state.queue_db_open_count = 0
    state.queue_db_reuse_count = 0
    state.queue_lock_failures_this_run = 0
    state.last_queue_lock_diag_ts = 0.0
    state.task_phase = ""
    _reset_exit_diag_stats()
    state.lifecycle_parent_preflight.clear()
    state.lifecycle_batch_imported.clear()
    state.lifecycle_batch_import_failed.clear()


def _queue_db_end_run() -> None:
    state = _exit_runtime_state()
    state.run_queue_db_active = False
    conn = state.run_queue_db_conn
    state.run_queue_db_conn = None
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def _local_lock_sleep_once(sleep_base: float) -> None:
    try:
        delay = float(sleep_base or 0.0)
    except Exception:
        delay = 0.0
    if delay > 0:
        time.sleep(delay)


def _local_lock_age(path_str: str) -> float | None:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            head = f.read(64)
        parts = head.strip().split()
        if len(parts) >= 2:
            return time.time() - float(parts[1])
    except Exception:
        pass
    try:
        st = os.stat(path_str)
        return time.time() - float(st.st_mtime)
    except Exception:
        return None


def _local_lock_stale_pid(path_str: str, stale_after: float | None) -> bool:
    try:
        with open(path_str, "r", encoding="utf-8") as f:
            head = f.read(64)
        parts = head.strip().split()
        pid_str = parts[0] if parts else ""
        pid = int(pid_str)
        if pid <= 0:
            return True
        if stale_after is not None and len(parts) >= 2:
            try:
                age = time.time() - float(parts[1])
                if age < float(stale_after):
                    return False
            except Exception:
                pass
        try:
            os.kill(pid, 0)
            return False
        except PermissionError:
            return False
        except ProcessLookupError:
            return True
        except Exception:
            return False
    except Exception:
        return False


def _local_lock_ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _mount_path_unescape(path: str) -> str:
    return (
        str(path or "")
        .replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _path_looks_network_mount(path: Path) -> bool:
    network_fs = {
        "nfs",
        "nfs4",
        "cifs",
        "smbfs",
        "fuse.sshfs",
        "9p",
        "afpfs",
        "davfs",
        "glusterfs",
        "ceph",
    }
    try:
        target = str(path.resolve())
    except Exception:
        target = str(path)
    if not target:
        return False
    best_mount = ""
    best_fs = ""
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = _mount_path_unescape(parts[1]).rstrip("/") or "/"
                fs_type = str(parts[2] or "").strip().lower()
                if not fs_type:
                    continue
                if target == mount_point or target.startswith(mount_point + "/"):
                    if len(mount_point) > len(best_mount):
                        best_mount = mount_point
                        best_fs = fs_type
    except Exception:
        return False
    if not best_fs:
        return False
    return best_fs in network_fs or best_fs.startswith("nfs")


@contextmanager
def _local_lock_fcntl_context(path: Path, path_str: str, *, tries: int, sleep_base: float):
    lf = None
    acquired = False
    _local_lock_ensure_parent(path)
    try:
        fd = os.open(path_str, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        lf = os.fdopen(fd, "a", encoding="utf-8")
        for _ in range(tries):
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except Exception:
                _local_lock_sleep_once(sleep_base)
    except Exception:
        lf = None
    try:
        yield acquired
    finally:
        try:
            if acquired and lf is not None:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            if lf is not None:
                lf.close()
        except Exception:
            pass


@contextmanager
def _local_lock_excl_context(
    path: Path,
    path_str: str,
    *,
    tries: int,
    sleep_base: float,
    stale_after: float | None,
):
    fd = None
    acquired = False
    for _ in range(tries):
        _local_lock_ensure_parent(path)
        try:
            fd = os.open(path_str, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            try:
                payload = f"{os.getpid()} {int(time.time())}\n"
                os.write(fd, payload.encode("ascii", "replace"))
            except Exception:
                pass
            acquired = True
            break
        except FileExistsError:
            pid_stale = _local_lock_stale_pid(path_str, stale_after)
            age_stale = False
            if stale_after is not None:
                age = _local_lock_age(path_str)
                if age is not None and age >= float(stale_after):
                    age_stale = True
            if pid_stale and age_stale:
                try:
                    os.unlink(path_str)
                except Exception:
                    pass
            else:
                _local_lock_sleep_once(sleep_base)
        except Exception:
            break
    try:
        yield acquired
    finally:
        try:
            if acquired and fd is not None:
                os.close(fd)
        except Exception:
            pass
        try:
            if acquired and fd is not None:
                os.unlink(path_str)
        except Exception:
            pass


@contextmanager
def _local_safe_lock(path: Path, *, retries: int = 6, sleep_base: float = 0.05, stale_after: float | None = 60.0):
    path_str = str(path) if path else ""
    if not path_str:
        yield False
        return

    tries = max(1, int(retries or 0))
    if fcntl is None and _path_looks_network_mount(path):
        _diag(f"queue lock fallback disabled on network mount: {path}")
        yield False
        return
    if fcntl is not None:
        with _local_lock_fcntl_context(path, path_str, tries=tries, sleep_base=sleep_base) as acquired:
            yield acquired
        return

    with _local_lock_excl_context(
        path,
        path_str,
        tries=tries,
        sleep_base=sleep_base,
        stale_after=stale_after,
    ) as acquired:
        yield acquired


_DIAG_REDACT_KEYS = frozenset({"description", "annotation", "annotations", "note", "notes"})


def _diag_redact_msg(msg: object) -> str:
    raw = msg if isinstance(msg, str) else str(msg)
    redactor = getattr(core, "diag_log_redact", None) if core is not None else None
    if callable(redactor):
        try:
            red = redactor(raw)
            return red if isinstance(red, str) else str(red)
        except Exception:
            pass
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for k in list(data.keys()):
                if k in _DIAG_REDACT_KEYS:
                    data[k] = "[redacted]"
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        pass
    return raw


def _diag(msg: str) -> None:
    safe_msg = _diag_redact_msg(msg)
    if core is not None:
        event_factory = getattr(core, "DiagnosticEvent", None)
        event = event_factory.from_message(safe_msg, hook="on-exit") if event_factory is not None else safe_msg
        core.diag(event, "on-exit", str(TW_DATA_DIR))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {safe_msg}\n")
        except Exception:
            pass


def _diag_block(title: str, items, *, columns: int = 3) -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        pairs = [f"{k}={v}" for k, v in (items or ())]
        _diag(f"{title}:")
        step = max(1, int(columns or 1))
        for idx in range(0, len(pairs), step):
            _diag("  " + "  ".join(pairs[idx:idx + step]))
    except Exception:
        pass


def _emit_exit_feedback(msg: str) -> None:
    """Write failing-hook feedback for Taskwarrior and keep stderr diagnostics."""
    seen: set[int] = set()
    for stream in (getattr(sys, "__stdout__", None), getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is None:
            continue
        ident = id(stream)
        if ident in seen:
            continue
        seen.add(ident)
        try:
            stream.write(msg + "\n")
            stream.flush()
        except Exception:
            pass


def _tw_lock_path() -> Path:
    return _tw_data_dir_path() / "lock"

def _tw_lock_recent(max_age_s: float = 5.0) -> bool:
    try:
        p = _tw_lock_path()
        if not p.exists():
            return False
        age = time.time() - p.stat().st_mtime
        return age >= 0 and age <= max_age_s
    except Exception:
        return False

def _sleep(secs: float) -> None:
    time.sleep(secs)

def _record_queue_lock_failure() -> None:
    state = _exit_runtime_state()
    state.queue_lock_failures_this_run += 1
    now = time.time()
    if now - state.last_queue_lock_diag_ts >= 60.0:
        _diag("queue lock not acquired; drain deferred")
        state.last_queue_lock_diag_ts = now
    try:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": "queue lock busy",
        }
        _QUEUE_LOCK_FAIL_MARKER.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(_QUEUE_LOCK_FAIL_MARKER), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass
    try:
        count = 0
        if _QUEUE_LOCK_FAIL_COUNT.exists():
            try:
                data = json.loads(_QUEUE_LOCK_FAIL_COUNT.read_text(encoding="utf-8") or "{}")
                count = int(data.get("count") or 0)
            except Exception:
                count = 0
        count += 1
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": count,
        }
        _QUEUE_LOCK_FAIL_COUNT.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(_QUEUE_LOCK_FAIL_COUNT), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


@contextmanager
def _lock_dead_letter():
    lock_fn = core.safe_lock if core is not None and hasattr(core, "safe_lock") else _local_safe_lock
    with lock_fn(
        _DEAD_LETTER_LOCK,
        retries=_QUEUE_LOCK_RETRIES,
        sleep_base=_QUEUE_LOCK_SLEEP_BASE,
        stale_after=_QUEUE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


@contextmanager
def _lock_orphan_cleanup_evidence():
    lock_fn = core.safe_lock if core is not None and hasattr(core, "safe_lock") else _local_safe_lock
    with lock_fn(
        _ORPHAN_CLEANUP_EVIDENCE_LOCK,
        retries=_QUEUE_LOCK_RETRIES,
        sleep_base=_QUEUE_LOCK_SLEEP_BASE,
        stale_after=_QUEUE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


def _record_orphan_cleanup_evidence(child_uuid: str, spawn_intent_id: str, reason: str) -> None:
    """Persist cleanup failures separately from the original lifecycle conflict."""
    queue_store = _module("queue_store", required=False)
    if queue_store is None:
        _diag(f"orphan cleanup evidence unavailable: {reason}")
        return
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hook": "on-exit",
        "hook_version": NAUTICAL_HOOK_VERSION,
        "kind": "orphan_cleanup",
        "child_uuid": str(child_uuid or "").strip(),
        "spawn_intent_id": str(spawn_intent_id or "").strip(),
        "reason": str(reason or "cleanup unavailable").strip(),
    }
    ok = queue_store.append_dead_letter_jsonl(
        path=_ORPHAN_CLEANUP_EVIDENCE_PATH,
        payload=payload,
        durable=True,
        acquire_lock=_lock_orphan_cleanup_evidence,
        diag=_diag,
    )
    if not ok:
        _diag(f"orphan cleanup evidence write failed (child={child_uuid[:8]}): {reason}")


def _intent_log_path() -> Path:
    return _nautical_state_dir_path() / ".nautical_spawn_intents.jsonl"


def _intent_log_lock_path() -> Path:
    return _nautical_lock_dir_path() / ".nautical_spawn_intents.lock"


@contextmanager
def _lock_intent_log():
    lock_fn = core.safe_lock if core is not None and hasattr(core, "safe_lock") else _local_safe_lock
    with lock_fn(
        _intent_log_lock_path(),
        retries=_QUEUE_LOCK_RETRIES,
        sleep_base=_QUEUE_LOCK_SLEEP_BASE,
        stale_after=_QUEUE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


def _parent_nextlink_lock_path(parent_uuid: str) -> Path:
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        return queue_store.parent_nextlink_lock_path(_tw_data_dir_path(), parent_uuid)
    raw = (parent_uuid or "").strip().lower()
    safe = "".join(ch for ch in raw if ch.isalnum())
    if not safe:
        safe = "unknown"
    if len(safe) > 64:
        safe = safe[:64]
    return _nautical_lock_dir_path() / f".nautical_parent_nextlink.{safe}.lock"


@contextmanager
def _lock_parent_nextlink(parent_uuid: str):
    lock_fn = core.safe_lock if core is not None and hasattr(core, "safe_lock") else _local_safe_lock
    with lock_fn(
        _parent_nextlink_lock_path(parent_uuid),
        retries=_QUEUE_LOCK_RETRIES,
        sleep_base=_QUEUE_LOCK_SLEEP_BASE,
        stale_after=_QUEUE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


def _intent_log_collect_final_states(path: Path) -> dict[str, str] | None:
    final_states: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                sid = (obj.get("spawn_intent_id") or "").strip()
                status = (obj.get("status") or "").strip().lower()
                if not sid or status not in {"done", "dead"}:
                    continue
                if sid in final_states:
                    final_states.pop(sid, None)
                final_states[sid] = status
        return final_states
    except Exception as e:
        _diag(f"intent log read failed: {e}")
        return None


def _intent_log_needs_compact(path: Path, final_states: dict[str, str]) -> bool:
    try:
        st_size = path.stat().st_size
    except Exception:
        st_size = 0
    return bool(
        (_INTENT_LOG_MAX_BYTES > 0 and st_size > _INTENT_LOG_MAX_BYTES)
        or (_INTENT_LOG_MAX_ENTRIES > 0 and len(final_states) > _INTENT_LOG_MAX_ENTRIES)
    )


def _intent_log_trim_states(final_states: dict[str, str]) -> None:
    if _INTENT_LOG_MAX_ENTRIES <= 0:
        return
    if len(final_states) <= _INTENT_LOG_MAX_ENTRIES:
        return
    drop_n = len(final_states) - _INTENT_LOG_MAX_ENTRIES
    for sid in list(final_states.keys())[:drop_n]:
        final_states.pop(sid, None)


def _intent_log_compact(path: Path, final_states: dict[str, str]) -> None:
    tmp_path = path.with_suffix(".staging")
    try:
        fd = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for sid, status in final_states.items():
                payload = {
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "hook": "on-exit",
                    "hook_version": NAUTICAL_HOOK_VERSION,
                    "status": status,
                    "spawn_intent_id": sid,
                    "reason": "compacted",
                }
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            if _DURABLE_QUEUE:
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
        os.replace(tmp_path, path)
        if _DURABLE_QUEUE:
            _fsync_dir(path.parent)
    except Exception as e:
        _diag(f"intent log compaction failed: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


def _load_finalized_intents() -> tuple[set[str], bool]:
    """Return finalized spawn_intent_id set, with best-effort compaction."""
    p = _intent_log_path()
    with _lock_intent_log() as locked:
        if not locked:
            _diag("intent log lock busy; idempotency disabled for this drain")
            return set(), False
        try:
            if not p.exists():
                return set(), True
        except Exception:
            return set(), True
        final_states = _intent_log_collect_final_states(p)
        if final_states is None:
            return set(), False

        needs_compact = _intent_log_needs_compact(p, final_states)
        _intent_log_trim_states(final_states)
        if needs_compact:
            _intent_log_compact(p, final_states)
    return set(final_states.keys()), True


def _mark_intent_status(spawn_intent_id: str, status: str, reason: str = "") -> bool:
    sid = (spawn_intent_id or "").strip()
    st = (status or "").strip().lower()
    if not sid or st not in {"done", "dead"}:
        return False
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hook": "on-exit",
        "hook_version": NAUTICAL_HOOK_VERSION,
        "status": st,
        "spawn_intent_id": sid,
        "reason": reason,
    }
    p = _intent_log_path()
    with _lock_intent_log() as locked:
        if not locked:
            _diag(f"intent log lock busy; could not mark {sid} as {st}")
            return False
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            fd = os.open(str(p), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                if _DURABLE_QUEUE:
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                        _fsync_dir(p.parent)
                    except Exception:
                        pass
            return True
        except Exception as e:
            _diag(f"intent log write failed ({sid}={st}): {e}")
            return False


def _lock_backoff_delay(streak: int) -> float:
    if streak <= 0:
        return 0.0
    base = max(0.0, float(_LOCK_BACKOFF_BASE or 0.0))
    cap = max(0.0, float(_LOCK_BACKOFF_MAX or 0.0))
    if base <= 0.0:
        return 0.0
    exp = min(int(streak), 8)
    delay = base * (2 ** (exp - 1))
    delay = min(delay, cap if cap > 0 else delay)
    jitter = random.uniform(0.0, base) if base > 0 else 0.0
    return delay + jitter


def _write_dead_letter(entry: dict, reason: str) -> bool:
    queue_store = _module("queue_store")
    payload = queue_store.build_dead_letter_payload(
        hook="on-exit",
        hook_version=NAUTICAL_HOOK_VERSION,
        entry=entry,
        reason=reason,
    )
    return bool(queue_store.append_dead_letter_jsonl(
        path=_DEAD_LETTER_PATH,
        payload=payload,
        durable=_DURABLE_QUEUE,
        acquire_lock=_lock_dead_letter,
        diag=_diag,
        max_bytes=_DEAD_LETTER_MAX_BYTES,
        retention_days=_DEAD_LETTER_RETENTION_DAYS,
    ))

def _fsync_dir(path: Path) -> None:
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        queue_store.fsync_dir(path)
        return
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
    except Exception:
        return
    try:
        os.fsync(fd)
    except Exception:
        pass
    finally:
        try:
            os.close(fd)
        except Exception:
            pass

def _queue_db_connect_result():
    state = _exit_runtime_state()
    queue_store = _module("queue_store")
    db_path = _QUEUE_DB_PATH
    timeout_base = max(1.0, _QUEUE_LOCK_SLEEP_BASE * max(1, _QUEUE_LOCK_RETRIES) * 4.0)
    timeout_max = max(timeout_base, float(_QUEUE_DB_CONNECT_TIMEOUT_MAX or timeout_base))
    result = queue_store.connect_queue_db_result(
        db_path,
        attempts=max(1, int(_QUEUE_DB_CONNECT_RETRIES or 1)),
        timeout_base=timeout_base,
        timeout_max=timeout_max,
        backoff_base=float(_QUEUE_DB_CONNECT_BACKOFF_BASE or 0.0),
        durable=_DURABLE_QUEUE,
        row_factory=sqlite3.Row,
        diag=_diag,
        sleep_fn=_sleep,
    )
    if result.conn is not None:
        state.queue_db_open_count += 1
    return result


def _queue_db_init(conn: sqlite3.Connection) -> None:
    queue_store = _module("queue_store")
    queue_store.init_queue_db(conn)


def _queue_db_open_ready() -> sqlite3.Connection | None:
    state = _exit_runtime_state()
    if state.run_queue_db_active and state.run_queue_db_conn is not None:
        try:
            state.run_queue_db_conn.execute("SELECT 1")
            state.queue_db_reuse_count += 1
            return state.run_queue_db_conn
        except Exception:
            _queue_close_silent(state.run_queue_db_conn)
            state.run_queue_db_conn = None
    queue_store = _module("queue_store")
    db_path = _QUEUE_DB_PATH
    result = queue_store.open_ready_queue_db_result(
        db_path,
        connect_fn=_queue_db_connect_result,
        init_fn=_queue_db_init,
        close_fn=_queue_close_silent,
        diag=_diag,
    )
    conn = result.conn
    if state.run_queue_db_active and conn is not None:
        state.run_queue_db_conn = conn
    return conn


def _queue_db_open_fresh_ready() -> sqlite3.Connection | None:
    queue_store = _module("queue_store")
    db_path = _QUEUE_DB_PATH
    result = queue_store.open_ready_queue_db_result(
        db_path,
        connect_fn=_queue_db_connect_result,
        init_fn=_queue_db_init,
        close_fn=_queue_close_silent,
        diag=_diag,
    )
    return result.conn


def _queue_close_silent(conn: sqlite3.Connection) -> None:
    if conn is None:
        return
    state = _exit_runtime_state()
    if state.run_queue_db_active and conn is state.run_queue_db_conn:
        return
    queue_store = _module("queue_store")
    queue_store.close_silent(conn)


def _take_queue_entries_sqlite_batch():
    conn = _queue_db_open_ready()
    exit_models = _module("exit_models")
    if conn is None:
        return exit_models.ExitQueueBatch(entries=[])
    token = f"drain-{os.getpid()}-{os.urandom(8).hex()}"
    now = time.time()
    queue_store = _module("queue_store")
    claim = queue_store.claim_rows_sqlite_result(
        conn,
        token=token,
        now=now,
        processing_stale_after=_QUEUE_PROCESSING_STALE_AFTER,
        max_lines=_QUEUE_MAX_LINES,
        diag=_diag,
        on_lock_busy=_record_queue_lock_failure,
    )
    entries = queue_store.rows_to_entries_result(claim.rows).entries
    _queue_close_silent(conn)
    return exit_models.ExitQueueBatch(entries=entries)


def _lifecycle_outbox_repository():
    outbox = _module("lifecycle_outbox")
    return outbox.LifecycleOutboxRepository(TW_DATA_DIR, durable=_DURABLE_QUEUE)


def _outbox_execution_entry(record) -> dict:
    """Expose one claimed immutable plan to the lifecycle executor."""
    plan = record.plan.with_stage(record.stage)
    child = plan.child_dict()
    child_uuid = str(child.get("uuid") or "").strip()
    parent_patch = plan.parent_patch_dict()
    child_short = str(parent_patch.get("nextLink") or _short_uuid(child_uuid)).strip()
    return {
        "parent_uuid": plan.identity.parent_uuid,
        "parent_nextlink": "",
        "child_short": child_short,
        "child": child,
        "spawn_intent_id": record.intent_id,
        "parent_guard": plan.parent_guard.to_dict(),
        "lifecycle_plan": plan.to_dict(),
        "attempts": record.attempts,
        "__outbox_backend": "lifecycle",
        "__outbox_intent_id": record.intent_id,
        "__outbox_owner": record.lease_owner,
    }


def _take_lifecycle_outbox_batch():
    exit_models = _module("exit_models")
    token = f"exit-{os.getpid()}-{os.urandom(8).hex()}"
    result, records = _lifecycle_outbox_repository().claim_batch(
        owner=token,
        lease_seconds=max(1.0, _QUEUE_PROCESSING_STALE_AFTER),
        limit=_QUEUE_MAX_LINES,
    )
    if not result.ok:
        if result.lock_busy:
            _record_queue_lock_failure()
        else:
            _diag(f"lifecycle outbox claim failed: {result.reason or 'unknown error'}")
        return exit_models.ExitQueueBatch(entries=[])
    return exit_models.ExitQueueBatch(entries=[_outbox_execution_entry(record) for record in records])


def _ack_queue_entries_sqlite_result(entry_claims: list[tuple[int, str]]):
    conn = _queue_db_open_fresh_ready()
    exit_models = _module("exit_models")
    claims = [
        (int(raw_id), str(raw_token or "").strip())
        for raw_id, raw_token in (entry_claims or [])
        if str(raw_id).isdigit() and int(raw_id) > 0 and str(raw_token or "").strip()
    ]
    if conn is None:
        return exit_models.ExitQueueWriteResult(ok=False, count=len(claims))
    try:
        queue_store = _module("queue_store")
        result = queue_store.ack_entry_claims_sqlite_result(
            conn,
            claims,
            diag=_diag,
            on_lock_busy=_record_queue_lock_failure,
        )
        return exit_models.ExitQueueWriteResult(ok=result.ok, count=result.count)
    finally:
        _queue_close_silent(conn)


def _requeue_entries_sqlite_result(entries: list[dict]):
    conn = _queue_db_open_fresh_ready()
    exit_models = _module("exit_models")
    items = [
        entry
        for entry in (entries or [])
        if isinstance(entry, dict) and entry.get("__queue_backend") == "sqlite" and entry.get("__queue_id")
    ]
    if conn is None:
        return exit_models.ExitQueueWriteResult(ok=False, count=len(items))
    try:
        queue_store = _module("queue_store")
        result = queue_store.requeue_entries_sqlite_result(
            conn,
            items,
            now=time.time(),
            diag=_diag,
            on_lock_busy=_record_queue_lock_failure,
        )
        return exit_models.ExitQueueWriteResult(ok=result.ok, count=result.count)
    finally:
        _queue_close_silent(conn)


def _advance_lifecycle_stage(entry: dict, stage):
    """Advance one claimed lifecycle outbox plan."""
    exit_models = _module("exit_models")
    intent_id = str(entry.get("__outbox_intent_id") or "").strip() if isinstance(entry, dict) else ""
    owner = str(entry.get("__outbox_owner") or "").strip() if isinstance(entry, dict) else ""
    if not intent_id or not owner:
        return exit_models.ExitQueueWriteResult(False, 1, "outbox claim identity is missing")
    lifecycle_models = _module("lifecycle_models")
    try:
        target = lifecycle_models.ExecutionStage(stage)
    except (TypeError, ValueError):
        return exit_models.ExitQueueWriteResult(False, 1, "invalid lifecycle stage")
    repository = _lifecycle_outbox_repository()
    if target is lifecycle_models.ExecutionStage.FINALIZED:
        result = repository.acknowledge(intent_id=intent_id, owner=owner)
    else:
        result = repository.advance_stage(intent_id=intent_id, owner=owner, stage=target)
    if result.ok and isinstance(entry.get("lifecycle_plan"), dict):
        try:
            plan = lifecycle_models.LifecyclePlan.from_dict(entry["lifecycle_plan"])
            entry["lifecycle_plan"] = plan.with_stage(target).to_dict()
        except Exception:
            pass
    return exit_models.ExitQueueWriteResult(
        result.ok,
        1 if result.ok else 0,
        result.reason,
        result.lock_busy,
    )


def _enqueue_entries_sqlite_result(entries: list[dict]):
    conn = _queue_db_open_ready()
    exit_models = _module("exit_models")
    items = [entry for entry in (entries or []) if isinstance(entry, dict)]
    if conn is None:
        return exit_models.ExitQueueWriteResult(ok=False, count=len(items))
    try:
        queue_store = _module("queue_store")
        result = queue_store.enqueue_entries_sqlite_result(
            conn,
            items,
            now=time.time(),
            diag=_diag,
            on_lock_busy=_record_queue_lock_failure,
        )
        return exit_models.ExitQueueWriteResult(ok=result.ok, count=result.count)
    finally:
        _queue_close_silent(conn)


def _normalize_queue_entry(entry: dict) -> dict:
    return dict(entry) if isinstance(entry, dict) else {}


def _validate_queue_entry(entry: dict) -> tuple[bool, str]:
    try:
        if not isinstance(entry, dict) or str(entry.get("__outbox_backend") or "") != "lifecycle":
            return False, "entry was not claimed from the lifecycle outbox"
        lifecycle_models = _module("lifecycle_models")
        plan = lifecycle_models.LifecyclePlan.from_dict(entry.get("lifecycle_plan") or {})
        if str(entry.get("__outbox_intent_id") or "").strip() != plan.identity.idempotency_key:
            return False, "outbox intent identity differs from lifecycle plan"
        if not str(entry.get("__outbox_owner") or "").strip():
            return False, "outbox claim owner is missing"
        return True, ""
    except Exception as e:
        return False, str(e)

def _bump_attempts(entry: dict) -> int:
    try:
        attempts = int(entry.get("attempts") or 0)
    except Exception:
        attempts = 0
    attempts += 1
    entry["attempts"] = attempts
    return attempts

def _short_uuid(value: str) -> str:
    text = str(value or "").strip()
    shortener = getattr(core, "short_uuid", None)
    if callable(shortener):
        try:
            shortened = str(shortener(text) or "").strip()
            if shortened:
                return shortened
        except Exception:
            pass
    return text[:8]


def _preload_equivalent_child_slots(entries: list[dict]) -> None:
    del entries
    # The chain-wide authoritative snapshot loaded by
    # ``_preload_export_uuids`` already indexes exact child slots.


def _export_uuid(uuid_str: str, *, prefer_cache: bool = True):
    repository = _exit_runtime_state().repository
    if repository is None:
        raise RuntimeError("on-exit task read repository is unavailable")
    return repository.by_uuid(
        uuid_str,
        statuses=("completed", "deleted", "pending", "waiting"),
        refresh=not prefer_cache,
    )


def _existing_equivalent_child(child: dict, parent_uuid: str = ""):
    repository = _exit_runtime_state().repository
    if repository is None:
        raise RuntimeError("on-exit task read repository is unavailable")
    return repository.exact_child_slot(
        str(child.get("chainID") or ""),
        int(float(child.get("link"))),
        statuses=("pending", "waiting", "completed"),
        expected_prev_link=str(child.get("prevLink") or _short_uuid(parent_uuid)),
    )


def _exit_mutation_guard(parent_uuid: str, parent_guard: dict[str, Any] | None):
    models = _module("integration_models")
    guard = parent_guard if isinstance(parent_guard, dict) else {}
    modified = str(guard.get("modified") or "").strip()
    recurrence_identity = str(guard.get("recurrence_fingerprint") or "").strip()
    chain_id = str(guard.get("chainID") or "").strip()
    if not modified or not recurrence_identity or not chain_id:
        raise ValueError("exit mutation requires complete parent guard evidence")
    try:
        link = int(float(str(guard.get("link") or "").strip()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("exit mutation guard requires an integer parent link") from exc
    return models.MutationGuard(
        task_uuid=parent_uuid,
        status=str(guard.get("status") or ""),
        chain_id=chain_id,
        link=link,
        recurrence_identity=recurrence_identity,
        timestamps=(models.GuardTimestamp(models.GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=_exit_runtime_state().unit_of_work.mutation_epoch,
        chain=str(guard.get("chain") or "on"),
    )


def _exit_row_mutation_guard(row: dict[str, Any]):
    """Build a complete guard from one authoritative Taskwarrior row."""
    lifecycle_models = _module("lifecycle_models")
    if not isinstance(row, dict):
        raise ValueError("mutation target is not a task object")
    task_uuid = str(row.get("uuid") or "").strip()
    modified = str(row.get("modified") or "").strip()
    chain_id = str(row.get("chainID") or "").strip()
    status = str(row.get("status") or "").strip()
    if not task_uuid or not modified or not chain_id or not status:
        raise ValueError("mutation target has incomplete identity")
    try:
        link = int(float(str(row.get("link") or "").strip()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mutation target has no integer link") from exc
    return _module("integration_models").MutationGuard(
        task_uuid=task_uuid,
        status=status,
        chain_id=chain_id,
        link=link,
        recurrence_identity=lifecycle_models.recurrence_fingerprint(
            row,
            parse_datetime=getattr(core, "parse_dt_any", None),
        ),
        timestamps=(
            _module("integration_models").GuardTimestamp(
                _module("integration_models").GuardTimestampField.MODIFIED,
                modified,
            ),
        ),
        expected_mutation_epoch=_exit_runtime_state().unit_of_work.mutation_epoch,
        chain=str(row.get("chain") or "on"),
    )


def _mutation_exit_result(outcome, *, import_result: bool = False):
    exit_models = _module("exit_models")
    models = _module("integration_models")
    if import_result:
        if outcome.kind in {models.MutationOutcomeKind.APPLIED, models.MutationOutcomeKind.ALREADY_APPLIED}:
            return exit_models.ExitImportResult(True, "")
        return exit_models.ExitImportResult(
            False,
            outcome.reason or outcome.kind.value,
            outcome.kind is models.MutationOutcomeKind.RETRYABLE,
        )
    if outcome.kind is models.MutationOutcomeKind.APPLIED:
        return exit_models.ExitParentUpdateResult(True, "", "ok")
    if outcome.kind is models.MutationOutcomeKind.ALREADY_APPLIED:
        return exit_models.ExitParentUpdateResult(True, "", "already")
    return exit_models.ExitParentUpdateResult(
        False,
        outcome.reason or outcome.kind.value,
        "locked" if outcome.kind is models.MutationOutcomeKind.RETRYABLE else "conflict",
        outcome.kind is models.MutationOutcomeKind.RETRYABLE,
    )


def _import_child(ctx):
    exit_models = _module("exit_models")
    state = _exit_runtime_state()
    if state.unit_of_work is None:
        return exit_models.ExitImportResult(False, "on-exit mutation unit of work is unavailable")
    try:
        guard = _exit_mutation_guard(ctx.parent_uuid, ctx.parent_guard)
        payload = _module("integration_models").ChildImportPayload.from_mapping(
            dict(ctx.child),
            parent_uuid=ctx.parent_uuid,
        )
        request = _module("integration_models").MutationRequest(
            _module("integration_models").MutationOperation.CHILD_IMPORT,
            guard,
            payload,
        )
        outcome = _module("taskwarrior_mutations").TaskwarriorMutationService(state.unit_of_work).apply(request)
        return _mutation_exit_result(outcome, import_result=True)
    except Exception as exc:
        return exit_models.ExitImportResult(False, str(exc).strip() or type(exc).__name__)


def _lifecycle_batch_link_token(value: Any) -> str:
    token = str(value or "").strip()
    try:
        return str(int(float(token)))
    except (TypeError, ValueError, OverflowError):
        return token


def _lifecycle_batch_child_matches(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    for field in ("chainID", "link", "prevLink"):
        expected_value = str(expected.get(field) or "").strip()
        actual_value = str(actual.get(field) or "").strip()
        if field == "link":
            expected_value = _lifecycle_batch_link_token(expected_value)
            actual_value = _lifecycle_batch_link_token(actual_value)
        if expected_value and actual_value != expected_value:
            return f"child {field} changed"
    return ""


def _prepare_lifecycle_batch(entries: list[dict]):
    """Classify claimed lifecycle entries without mutating Taskwarrior."""
    state = _exit_runtime_state()
    decisions = []
    exit_models = _module("exit_models")
    lifecycle_models = _module("lifecycle_models")
    flow = _module("exit_entry_flow")
    use_preload = sum(
        isinstance(item, dict) and isinstance(item.get("lifecycle_plan"), dict)
        for item in entries or []
    ) > 1
    for entry in entries or []:
        if not isinstance(entry, dict) or not isinstance(entry.get("lifecycle_plan"), dict):
            continue
        sid = str(entry.get("spawn_intent_id") or "").strip()
        if not sid:
            continue
        try:
            plan = lifecycle_models.LifecyclePlan.from_dict(entry["lifecycle_plan"])
        except Exception as exc:
            _diag(f"lifecycle batch plan rejected: {exc}")
            continue
        with _task_phase("batch_preflight"):
            parent_res = _export_uuid(plan.identity.parent_uuid, prefer_cache=use_preload)
        if _read_unavailable(parent_res):
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.UNAVAILABLE,
                reason=_read_reason(parent_res, "parent preflight unavailable"),
            ))
            continue
        parent = _read_value(parent_res)
        if not isinstance(parent, dict):
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING,
                reason="parent task is missing",
            ))
            continue
        expected_short = str(plan.parent_patch_dict().get("nextLink") or "").strip()
        parent_linked = (
            expected_short
            and _lifecycle_batch_link_token(parent.get("nextLink"))
            == _lifecycle_batch_link_token(expected_short)
        )
        mismatch = flow._parent_guard_mismatch(
            parent,
            plan.parent_guard.to_dict(),
            recurrence_fingerprint=lambda task: lifecycle_models.recurrence_fingerprint(
                task,
                parse_datetime=getattr(core, "parse_dt_any", None),
            ),
            check_modified=not parent_linked,
        )
        if mismatch:
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING,
                parent=parent, reason=mismatch,
            ))
            continue
        state.lifecycle_parent_preflight[sid] = parent
        child = plan.child_dict()
        child_uuid = str(child.get("uuid") or "").strip()
        if not child_uuid:
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING,
                parent=parent, reason="lifecycle child has no UUID",
            ))
            continue
        with _task_phase("batch_preflight"):
            exact = _export_uuid(child_uuid, prefer_cache=use_preload)
        child_obj = _read_value(exact)
        if _read_unavailable(exact):
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.UNAVAILABLE,
                parent=parent, reason=_read_reason(exact, "child preflight unavailable"),
            ))
            continue
        if child_obj is not None:
            child_reason = _lifecycle_batch_child_matches(child, child_obj)
            if child_reason:
                decisions.append(exit_models.LifecycleBatchDecision(
                    sid, entry, plan, exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING,
                    parent=parent, child=child_obj, reason=child_reason,
                ))
                continue
        else:
            equivalent = _existing_equivalent_child(child, plan.identity.parent_uuid)
            if _read_unavailable(equivalent):
                decisions.append(exit_models.LifecycleBatchDecision(
                    sid, entry, plan, exit_models.LifecycleBatchDecisionKind.UNAVAILABLE,
                    parent=parent, reason=_read_reason(equivalent, "equivalent child preflight unavailable"),
                ))
                continue
            if _read_found(equivalent):
                child_obj = _read_value(equivalent)
                child_reason = _lifecycle_batch_child_matches(child, child_obj)
                if child_reason:
                    decisions.append(exit_models.LifecycleBatchDecision(
                        sid, entry, plan, exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING,
                        parent=parent, child=child_obj, reason=child_reason,
                    ))
                    continue
        actual_next = _lifecycle_batch_link_token(parent.get("nextLink"))
        expected_next = _lifecycle_batch_link_token(expected_short)
        if child_obj is not None:
            kind = (
                exit_models.LifecycleBatchDecisionKind.ALREADY_SATISFIED
                if expected_next and actual_next == expected_next
                else exit_models.LifecycleBatchDecisionKind.READY_TO_APPLY
            )
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, kind, parent=parent, child=child_obj,
            ))
        elif expected_next and actual_next == expected_next:
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING,
                parent=parent, reason="parent points to a missing child",
            ))
        else:
            decisions.append(exit_models.LifecycleBatchDecision(
                sid, entry, plan, exit_models.LifecycleBatchDecisionKind.MISSING_CHILD,
                parent=parent, child=child,
            ))
    return exit_models.LifecycleBatchPlan(tuple(decisions))


def _apply_lifecycle_batch(plan) -> None:
    """Apply only the missing-child subset of a read-only batch plan."""
    exit_models = _module("exit_models")
    state = _exit_runtime_state()
    missing = plan.for_kind(exit_models.LifecycleBatchDecisionKind.MISSING_CHILD)
    if not missing:
        return
    for decision in missing:
        child = decision.plan.child_dict()
        child_uuid = str(child.get("uuid") or "").strip()
        if not child_uuid:
            state.lifecycle_batch_import_failed.add(decision.spawn_intent_id)
            continue
        context = _build_exit_entry_context(
            decision.entry,
            -1,
            state,
            parent_uuid=decision.plan.identity.parent_uuid,
            child_short=str(decision.plan.parent_patch_dict().get("nextLink") or _short_uuid(child_uuid)).strip(),
            expected_parent_nextlink=str(decision.entry.get("parent_nextlink") or "").strip() or None,
            parent_guard=decision.plan.parent_guard.to_dict(),
            child=child,
            child_uuid=child_uuid,
            spawn_intent_id=decision.spawn_intent_id,
        )
        result = _import_child(context)
        if not result.ok:
            _diag(f"lifecycle child batch import failed: {result.err or 'unknown error'}")
            state.lifecycle_batch_import_failed.add(decision.spawn_intent_id)
            continue
        state.lifecycle_batch_imported.add(decision.spawn_intent_id)


def _cleanup_lifecycle_batch(state) -> None:
    """Safely batch-compensate imported children after multi-entry conflicts."""
    requests = getattr(state, "lifecycle_orphan_cleanup", {})
    if not requests:
        return
    entries = [
        {"parent_uuid": parent_uuid, "child": {"uuid": child_uuid}}
        for child_uuid, (parent_uuid, _child_short, _sid) in requests.items()
    ]
    _preload_export_uuids(entries)
    safe: list[tuple[str, str, str]] = []
    for child_uuid, (parent_uuid, child_short, sid) in requests.items():
        parent_res = _export_uuid(parent_uuid)
        child_res = _export_uuid(child_uuid)
        if _read_unavailable(parent_res):
            reason = "parent unavailable while compensating orphan child"
            state.lifecycle_cleanup_failures[child_uuid] = reason
            _record_orphan_cleanup_evidence(child_uuid, sid, reason)
            continue
        parent = _read_value(parent_res)
        if isinstance(parent, dict) and str(parent.get("nextLink") or "").strip() == str(child_short or "").strip():
            reason = "cleanup skipped because parent link is now present"
            state.lifecycle_cleanup_failures[child_uuid] = reason
            _record_orphan_cleanup_evidence(child_uuid, sid, reason)
            continue
        if _read_unavailable(child_res):
            reason = "child unavailable while compensating orphan"
            state.lifecycle_cleanup_failures[child_uuid] = reason
            _record_orphan_cleanup_evidence(child_uuid, sid, reason)
            continue
        if not _read_found(child_res):
            continue
        safe.append((child_uuid, parent_uuid, sid))
    for child_uuid, _parent, sid in safe:
        result = _cleanup_orphan_child(child_uuid, sid)
        if not result.ok:
            reason = result.err or "batched orphan cleanup failed"
            state.lifecycle_cleanup_failures[child_uuid] = reason
            _record_orphan_cleanup_evidence(child_uuid, sid, reason)


def _finalize_lifecycle_batch(state) -> None:
    """Verify all deferred lifecycle postconditions with bounded exports."""
    _cleanup_lifecycle_batch(state)
    pending = getattr(state, "lifecycle_pending_verification", {})
    if not pending:
        return
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for ctx, plan, _child in pending.values():
        for uuid_str in (ctx.parent_uuid, ctx.child_uuid):
            uuid_str = str(uuid_str or "").strip()
            if uuid_str and uuid_str not in seen:
                seen.add(uuid_str)
                specs.append((uuid_str, f"uuid:{uuid_str}"))
    filters: list[str] = []
    for index, (_uuid, clause) in enumerate(specs):
        if index:
            filters.append("or")
        filters.append(clause)
    repository = _exit_runtime_state().repository
    if repository is None:
        raise RuntimeError("on-exit task read repository is unavailable")
    with _task_phase("batch_verify"):
        read = repository.broad_snapshot(
            identity="exit-lifecycle-verification:" + ",".join(uuid_str for uuid_str, _clause in specs),
            filters=tuple(filters),
            statuses=("completed", "deleted", "pending", "waiting"),
            refresh=True,
        )
    if _read_unavailable(read):
        reason = _read_reason(read, "batch verification unavailable")
        for ctx, _plan, _child in pending.values():
            _handle_lifecycle_postcondition_failure(ctx.entry, ctx.idx, state, reason)
        return
    snapshot = _read_value(read)
    rows = list(snapshot.rows) if snapshot is not None else []
    by_uuid = {str(row.get("uuid") or "").strip(): row for row in rows if isinstance(row, dict)}
    flow = _module("exit_entry_flow")
    lifecycle_models = _module("lifecycle_models")
    for sid, (ctx, plan, expected_child) in pending.items():
        child = by_uuid.get(ctx.child_uuid)
        parent = by_uuid.get(ctx.parent_uuid)
        reason = ""
        if not isinstance(child, dict):
            reason = "child postcondition missing after lifecycle batch"
        elif not isinstance(parent, dict):
            reason = "parent postcondition missing after lifecycle batch"
        else:
            for field in ("chainID", "link", "prevLink"):
                expected = str(expected_child.get(field) or "").strip()
                actual = str(child.get(field) or "").strip()
                if field == "link":
                    try:
                        expected = str(int(float(expected)))
                        actual = str(int(float(actual)))
                    except (TypeError, ValueError, OverflowError):
                        pass
                if expected and actual != expected:
                    reason = f"child {field} changed"
                    break
            if not reason and str(parent.get("nextLink") or "").strip() != ctx.child_short:
                reason = "parent linkage postcondition not satisfied"
            if not reason:
                mismatch = flow._parent_guard_mismatch(
                    parent,
                    plan.parent_guard.to_dict(),
                    recurrence_fingerprint=lambda task: lifecycle_models.recurrence_fingerprint(
                        task,
                        parse_datetime=getattr(core, "parse_dt_any", None),
                    ),
                    check_modified=False,
                )
                if mismatch:
                    reason = mismatch
        if reason:
            _handle_lifecycle_postcondition_failure(ctx.entry, ctx.idx, state, reason)
            continue
        verified = _advance_lifecycle_stage(ctx.entry, lifecycle_models.ExecutionStage.VERIFIED)
        if not verified.ok:
            _handle_lifecycle_stage_failure(ctx.entry, ctx.idx, state, verified)
            continue
        finalized = _advance_lifecycle_stage(ctx.entry, lifecycle_models.ExecutionStage.FINALIZED)
        if not finalized.ok:
            _handle_lifecycle_stage_failure(ctx.entry, ctx.idx, state, finalized)
            continue
        if not state.mark_final(ctx.entry, "done", "processed"):
            state.errors += 1
            state.requeue.append(ctx.entry)
            continue
        state.processed += 1
        state.reset_lock_streak()


def _update_parent_nextlink(
    parent_uuid: str,
    child_short: str,
    expected_prev: str | None = None,
    *,
    parent_guard: dict[str, Any] | None = None,
    parent_snapshot: dict[str, Any] | None = None,
):
    exit_models = _module("exit_models")
    state = _exit_runtime_state()
    if state.unit_of_work is None:
        return exit_models.ExitParentUpdateResult(False, "on-exit mutation unit of work is unavailable")
    if not isinstance(parent_guard, dict):
        return exit_models.ExitParentUpdateResult(False, "parent guard evidence is unavailable", "conflict")
    try:
        guard = _exit_mutation_guard(parent_uuid, parent_guard)
        models = _module("integration_models")
        request = models.MutationRequest(
            models.MutationOperation.PARENT_LINK,
            guard,
            models.ParentLinkPayload(parent_uuid, child_short, str(expected_prev or "").strip()),
        )
        with _lock_parent_nextlink(parent_uuid) as locked:
            if not locked:
                return exit_models.ExitParentUpdateResult(False, "parent lock busy", "locked", True)
            with _task_phase("parent_update"):
                outcome = _module("taskwarrior_mutations").TaskwarriorMutationService(
                    state.unit_of_work,
                    timeout=_TASK_TIMEOUT_MODIFY,
                ).apply(request)
        return _mutation_exit_result(outcome)
    except Exception as exc:
        return exit_models.ExitParentUpdateResult(False, str(exc).strip() or type(exc).__name__, "conflict")


def _clear_parent_nextlink_if_matches(parent_uuid: str, child_short: str):
    exit_models = _module("exit_models")
    state = _exit_runtime_state()
    if state.unit_of_work is None:
        return exit_models.ExitParentUpdateResult(False, "on-exit mutation unit of work is unavailable")
    if not parent_uuid or not child_short:
        return exit_models.ExitParentUpdateResult(False, "missing parent or child", "invalid")
    with _lock_parent_nextlink(parent_uuid) as locked:
        if not locked:
            return exit_models.ExitParentUpdateResult(False, "parent lock busy", "locked", True)
        parent_res = _export_uuid(parent_uuid, prefer_cache=False)
        if _read_unavailable(parent_res):
            return exit_models.ExitParentUpdateResult(
                False,
                _read_reason(parent_res, "parent export unavailable"),
                "locked",
                True,
            )
        parent = _read_value(parent_res)
        if not isinstance(parent, dict):
            return exit_models.ExitParentUpdateResult(True, "", "already")
        current = str(parent.get("nextLink") or "").strip()
        if current != child_short:
            return exit_models.ExitParentUpdateResult(True, "", "already")
        try:
            models = _module("integration_models")
            request = models.MutationRequest(
                models.MutationOperation.PARENT_LINK_CLEAR,
                _exit_row_mutation_guard(parent),
                models.ParentLinkClearPayload(parent_uuid, child_short),
            )
            outcome = _module("taskwarrior_mutations").TaskwarriorMutationService(
                state.unit_of_work,
                timeout=_TASK_TIMEOUT_MODIFY,
            ).apply(request)
            return _mutation_exit_result(outcome)
        except Exception as exc:
            return exit_models.ExitParentUpdateResult(False, str(exc).strip() or type(exc).__name__, "conflict")


def _parent_nextlink_state(
    parent_uuid: str,
    child_short: str,
    expected_prev: str | None = None,
    *,
    prefer_cache: bool = True,
    parent_guard: dict[str, Any] | None = None,
    guard_mismatch_fn=None,
):
    exit_side_effects = _module("exit_side_effects")
    return exit_side_effects.parent_nextlink_state(
        parent_uuid,
        child_short,
        expected_prev=expected_prev,
        export_uuid=lambda uuid_str: _export_uuid(uuid_str, prefer_cache=prefer_cache),
        parent_guard=parent_guard,
        guard_mismatch_fn=guard_mismatch_fn,
    )


def _cleanup_orphan_child(child_uuid: str, spawn_intent_id: str = ""):
    exit_models = _module("exit_models")
    state = _exit_runtime_state()
    if state.unit_of_work is None:
        return exit_models.ExitImportResult(False, "on-exit mutation unit of work is unavailable")
    child_res = _export_uuid(child_uuid, prefer_cache=False)
    if _read_unavailable(child_res):
        reason = _read_reason(child_res, "child export unavailable")
        _diag(f"orphan cleanup unavailable (intent={spawn_intent_id or '-'} child={child_uuid[:8]}): {reason}")
        return exit_models.ExitImportResult(False, reason, True)
    child = _read_value(child_res)
    if not isinstance(child, dict):
        return exit_models.ExitImportResult(True, "")
    try:
        models = _module("integration_models")
        request = models.MutationRequest(
            models.MutationOperation.CHILD_COMPENSATION,
            _exit_row_mutation_guard(child),
            models.ChildCompensationPayload(child_uuid, str(child.get("status") or "pending")),
        )
        outcome = _module("taskwarrior_mutations").TaskwarriorMutationService(
            state.unit_of_work,
            timeout=_TASK_TIMEOUT_MODIFY,
        ).apply(request)
        result = _mutation_exit_result(outcome, import_result=True)
        if not result.ok:
            _diag(f"orphan cleanup failed (intent={spawn_intent_id or '-'} child={child_uuid[:8]}): {result.err}")
        return result
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        _diag(f"orphan cleanup failed (intent={spawn_intent_id or '-'} child={child_uuid[:8]}): {reason}")
        return exit_models.ExitImportResult(False, reason)


def _cleanup_orphan_children(child_uuids: list[str]):
    exit_models = _module("exit_models")
    for child_uuid in dict.fromkeys(str(value or "").strip() for value in child_uuids or []):
        if not child_uuid:
            continue
        result = _cleanup_orphan_child(child_uuid)
        if not result.ok:
            return result
    return exit_models.ExitImportResult(True, "")


def _take_queue_batch():
    return _take_lifecycle_outbox_batch()


def _take_queue_entries():
    return _take_queue_batch().entries


def _requeue_entries_result(entries: list[dict]):
    exit_models = _module("exit_models")
    items = [e for e in (entries or []) if isinstance(e, dict)]
    if not items:
        return exit_models.ExitRequeueResult(ok=True, failed=0)
    outbox_failed = 0
    outbox = _module("lifecycle_outbox")
    repository = _lifecycle_outbox_repository()
    for entry in items:
        intent_id = str(entry.get("__outbox_intent_id") or "").strip()
        owner = str(entry.get("__outbox_owner") or "").strip()
        result = repository.release_retry(
            intent_id=intent_id,
            owner=owner,
            failure=outbox.OutboxFailure("deferred", "lifecycle execution deferred"),
        )
        if not result.ok:
            outbox_failed += 1
            if result.lock_busy:
                _record_queue_lock_failure()
            else:
                _diag(f"lifecycle outbox retry release failed: {result.reason or 'unknown error'}")
    return exit_models.ExitRequeueResult(ok=outbox_failed == 0, failed=outbox_failed)


class _DrainState:
    def __init__(
        self,
        entries: list[dict],
        entries_total: int,
    ) -> None:
        self.entries = entries
        self.entries_total = entries_total
        self.processed = 0
        self.errors = 0
        self.requeue: list[dict] = []
        self.dead_lettered = 0
        self.skipped_idempotent = 0
        self.lock_events = 0
        self.lock_streak = 0
        self.lock_streak_max = 0
        self.circuit_breaks = 0
        self.lifecycle_defer_verification = False
        self.lifecycle_batch_discovery = False
        self.lifecycle_pending_verification: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
        self.lifecycle_batch_plan = None
        self.lifecycle_orphan_cleanup: dict[str, tuple[str, str, str]] = {}
        self.lifecycle_cleanup_failures: dict[str, str] = {}

    def mark_final(self, entry: dict, status: str, reason: str) -> bool:
        return True

    def queue_backend(self, entry: dict) -> str:
        return "outbox"

    def entry_clean(self, entry: dict) -> dict:
        if not isinstance(entry, dict):
            return {}
        out = dict(entry)
        out.pop("__outbox_backend", None)
        out.pop("__outbox_intent_id", None)
        out.pop("__outbox_owner", None)
        return out

    def dead_letter(self, entry: dict, reason: str) -> None:
        outbox = _module("lifecycle_outbox")
        result = _lifecycle_outbox_repository().manual_review(
            intent_id=str(entry.get("__outbox_intent_id") or ""),
            owner=str(entry.get("__outbox_owner") or ""),
            failure=outbox.OutboxFailure("manual_review", str(reason or "lifecycle execution failed")),
        )
        if not result.ok:
            self.requeue.append(entry)
            self.errors += 1
            _diag(f"lifecycle outbox manual-review transition failed: {result.reason or 'unknown error'}")
            return
        self.dead_lettered += 1
        self.errors += 1

    def record_lock_event(self, idx: int) -> bool:
        self.lock_events += 1
        self.lock_streak += 1
        if self.lock_streak > self.lock_streak_max:
            self.lock_streak_max = self.lock_streak
        delay = _lock_backoff_delay(self.lock_streak)
        if delay > 0:
            _sleep(delay)
        if _LOCK_STORM_THRESHOLD > 0 and self.lock_streak >= _LOCK_STORM_THRESHOLD and (idx + 1) < self.entries_total:
            self.circuit_breaks += 1
            self.requeue.extend(self.entries[idx + 1:])
            _diag(
                f"lock storm detected (streak={self.lock_streak}); "
                f"requeued remaining {self.entries_total - (idx + 1)} entries"
            )
            return True
        return False

    def reset_lock_streak(self) -> None:
        self.lock_streak = 0

    def to_stats_model(self, drain_t0: float, requeue_ok: bool, requeue_failed: int):
        exit_models = _module("exit_models")
        return exit_models.ExitDrainStats(
            processed=self.processed,
            errors=self.errors,
            requeued=len(self.requeue) if requeue_ok else 0,
            requeue_failed=requeue_failed,
            dead_lettered=self.dead_lettered,
            queue_lock_failures=_exit_runtime_state().queue_lock_failures_this_run,
            entries_total=self.entries_total,
            entries_skipped_idempotent=self.skipped_idempotent,
            lock_events=self.lock_events,
            lock_streak_max=self.lock_streak_max,
            circuit_breaks=self.circuit_breaks,
            intent_log_ready=1,
            intent_log_size=0,
            intent_log_load_ms=0.0,
            intent_mark_ok=0,
            intent_mark_fail=0,
            queue_db_opens=0,
            queue_db_reuses=0,
            preload_export_uuids=int(_exit_runtime_state().diag_stats.get("preload_export_uuids", 0)),
            preload_export_hits=int(_exit_runtime_state().diag_stats.get("preload_export_hits", 0)),
            preload_export_misses=int(_exit_runtime_state().diag_stats.get("preload_export_misses", 0)),
            preload_export_chunks=int(_exit_runtime_state().diag_stats.get("preload_export_chunks", 0)),
            drain_ms=round((time.perf_counter() - drain_t0) * 1000.0, 3),
        )


def _requeue_or_dead_letter_for_lock(entry: dict, idx: int, state: _DrainState) -> bool:
    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
        state.dead_letter(entry, "exceeded retry budget")
    else:
        state.requeue.append(entry)
    return state.record_lock_event(idx)


def _handle_lifecycle_stage_failure(entry: dict, idx: int, state: _DrainState, result) -> bool:
    """Keep a claimed plan retryable, or quarantine a structurally invalid one."""
    error = str(getattr(result, "err", "") or "lifecycle stage update failed")
    _diag(f"lifecycle stage update failed for queue entry {entry.get('__queue_id')}: {error}")
    if bool(getattr(result, "lock_busy", False)):
        return _requeue_or_dead_letter_for_lock(entry, idx, state)
    if "claim ownership lost" in error.lower():
        # Another worker owns recovery now; requeueing would itself fail the
        # ownership check and could obscure the original race.
        state.errors += 1
        _diag(f"lifecycle stage update abandoned after claim loss: {error}")
        return True
    state.dead_letter(entry, f"lifecycle stage update failed: {error}")
    return False


def _verify_lifecycle_postconditions(ctx) -> tuple[str, str]:
    """Freshly verify the child and reciprocal parent link before finalizing."""
    if not isinstance(ctx.entry.get("lifecycle_plan"), dict):
        return "ok", ""
    child_res = _export_uuid(ctx.child_uuid, prefer_cache=False)
    if _read_unavailable(child_res):
        return "retry", _read_reason(child_res, "child verification unavailable")
    if not _read_found(child_res):
        return "retry", "child postcondition missing after lifecycle apply"
    link_res = _parent_nextlink_state(
        ctx.parent_uuid,
        ctx.child_short,
        ctx.expected_parent_nextlink,
        prefer_cache=False,
    )
    if link_res.state == "already":
        return "ok", ""
    if link_res.state == "locked":
        return "retry", link_res.err or "parent linkage verification unavailable"
    return "retry", link_res.err or "parent linkage postcondition not satisfied"


def _handle_lifecycle_postcondition_failure(entry: dict, idx: int, state: _DrainState, reason: str) -> bool:
    """Retry a failed final read, bounded by the normal queue retry budget."""
    message = str(reason or "lifecycle postcondition verification failed")
    _diag(f"lifecycle postcondition verification deferred: {message}")
    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
        state.dead_letter(entry, message)
    else:
        state.requeue.append(entry)
    return False


def _handle_entry_gate(entry: dict, state: _DrainState) -> bool:
    valid, reason = _validate_queue_entry(entry)
    if not valid:
        _diag(f"queue entry rejected before lifecycle execution: {reason}")
        state.dead_letter(entry, reason)
        state.reset_lock_streak()
        return True
    return False


def _build_exit_entry_context(
    entry: dict,
    idx: int,
    state: _DrainState,
    *,
    parent_uuid: str,
    child_short: str,
    expected_parent_nextlink: str | None,
    parent_guard: dict[str, str] | None,
    child: dict,
    child_uuid: str,
    spawn_intent_id: str,
):
    exit_models = _module("exit_models")
    return exit_models.ExitEntryContext(
        entry=entry,
        idx=idx,
        state=state,
        parent_uuid=parent_uuid,
        child_short=child_short,
        expected_parent_nextlink=expected_parent_nextlink,
        parent_guard=parent_guard,
        child=child,
        child_uuid=child_uuid,
        spawn_intent_id=spawn_intent_id,
    )


def _exit_runtime_services():
    exit_runtime = _module("exit_runtime")
    return exit_runtime.ExitRuntimeServices(
        state=_exit_runtime_state(),
        parent_nextlink_state=_parent_nextlink_state,
        requeue_or_dead_letter_for_lock=_requeue_or_dead_letter_for_lock,
        export_uuid=_export_uuid,
        import_child=_import_child,
        diag=_diag,
        update_parent_nextlink=_update_parent_nextlink,
        clear_parent_nextlink_if_matches=_clear_parent_nextlink_if_matches,
        cleanup_orphan_child=_cleanup_orphan_child,
    )


def _precheck_parent_link_state(ctx) -> tuple[str, bool]:
    exit_entry_flow = _module("exit_entry_flow")
    exit_runtime = _module("exit_runtime")
    services = exit_runtime.build_precheck_services(_exit_runtime_services())
    return exit_entry_flow.precheck_parent_link_state(ctx, services=services)


def _precheck_parent_guard(ctx) -> str:
    exit_entry_flow = _module("exit_entry_flow")
    exit_runtime = _module("exit_runtime")
    services = exit_runtime.build_precheck_services(_exit_runtime_services())
    lifecycle_models = _module("lifecycle_models")
    parse_datetime = getattr(core, "parse_dt_any", None)
    services.recurrence_fingerprint = lambda task: lifecycle_models.recurrence_fingerprint(
        task,
        parse_datetime=parse_datetime,
    )
    return exit_entry_flow.precheck_parent_guard(ctx, services=services)


def _ensure_child_exists_for_entry(ctx, *, initial_export_res=None) -> tuple[str, bool]:
    exit_entry_flow = _module("exit_entry_flow")
    exit_runtime = _module("exit_runtime")
    services = exit_runtime.build_ensure_child_services(_exit_runtime_services())
    return exit_entry_flow.ensure_child_exists_for_entry(
        ctx,
        services=services,
        initial_export_res=initial_export_res,
    )


def _apply_parent_update_for_entry(
    ctx,
    *,
    parent_linked_already: bool,
    imported: bool,
) -> str:
    exit_entry_flow = _module("exit_entry_flow")
    exit_runtime = _module("exit_runtime")
    services = exit_runtime.build_apply_parent_update_services(_exit_runtime_services())
    services.recheck_parent_guard = _precheck_parent_guard
    return exit_entry_flow.apply_parent_update_for_entry(
        ctx,
        parent_linked_already=parent_linked_already,
        imported=imported,
        services=services,
    )


def _lifecycle_operation_result(state, *, value=None, reason=""):
    lifecycle_executor = _module("lifecycle_executor")
    return lifecycle_executor.OperationResult(state, value=value, reason=reason)


def _execute_lifecycle_queue_entry(ctx, state):
    """Execute a validated queue plan through the shared typed executor."""
    lifecycle_models = _module("lifecycle_models")
    lifecycle_executor = _module("lifecycle_executor")
    exit_models = _module("exit_models")
    try:
        plan = lifecycle_models.LifecyclePlan.from_dict(ctx.entry["lifecycle_plan"])
    except Exception as exc:
        _diag(f"lifecycle executor rejected plan: {exc}")
        state.dead_letter(ctx.entry, f"invalid lifecycle plan: {exc}")
        state.reset_lock_streak()
        return False

    batch_plan = getattr(state, "lifecycle_batch_plan", None)
    batch_decision = None
    if batch_plan is not None:
        batch_decision = batch_plan.by_intent().get(ctx.spawn_intent_id)
    if batch_decision is not None:
        if batch_decision.kind is exit_models.LifecycleBatchDecisionKind.STALE_CONFLICTING:
            state.dead_letter(
                ctx.entry,
                f"lifecycle batch preflight conflict: {batch_decision.reason or 'state changed'}",
            )
            state.reset_lock_streak()
            return False
        if batch_decision.kind is exit_models.LifecycleBatchDecisionKind.UNAVAILABLE:
            reason = batch_decision.reason or "lifecycle batch preflight unavailable"
            _diag(f"lifecycle batch preflight deferred: {reason}")
            if _bump_attempts(ctx.entry) > _QUEUE_RETRY_MAX:
                state.dead_letter(ctx.entry, reason)
            else:
                state.requeue.append(ctx.entry)
            return False
        # A partial batch import may have created a child before Taskwarrior
        # returned failure. On retry, promote that durable plan before the
        # executor advances it to linkage/verification.
        if (
            isinstance(batch_decision.child, dict)
            and plan.stage is lifecycle_models.ExecutionStage.PERSISTED
            and not (
                batch_decision.kind is exit_models.LifecycleBatchDecisionKind.MISSING_CHILD
                and ctx.spawn_intent_id in _exit_runtime_state().lifecycle_batch_imported
            )
        ):
            child_stage = _advance_lifecycle_stage(ctx.entry, lifecycle_models.ExecutionStage.CHILD_PRESENT)
            if not child_stage.ok:
                return _handle_lifecycle_stage_failure(ctx.entry, ctx.idx, state, child_stage)
        if (
            batch_decision.kind is exit_models.LifecycleBatchDecisionKind.MISSING_CHILD
            and ctx.spawn_intent_id in _exit_runtime_state().lifecycle_batch_import_failed
        ):
            reason = "lifecycle child batch import unavailable"
            _diag(f"lifecycle batch import deferred (intent={ctx.spawn_intent_id})")
            if _bump_attempts(ctx.entry) > _QUEUE_RETRY_MAX:
                state.dead_letter(ctx.entry, reason)
            else:
                state.requeue.append(ctx.entry)
            return False

    def planned_decision():
        if batch_plan is None:
            return None
        return batch_plan.by_intent().get(ctx.spawn_intent_id)

    def stage(stage):
        result = _advance_lifecycle_stage(ctx.entry, stage)
        if result.ok:
            return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED)
        operation_state = (
            lifecycle_executor.OperationState.UNAVAILABLE
            if bool(getattr(result, "lock_busy", False))
            else lifecycle_executor.OperationState.FAILED
        )
        return _lifecycle_operation_result(operation_state, reason=getattr(result, "err", "") or "queue stage update failed")

    class Services:
        def validate_parent(self, current_plan):
            decision = planned_decision()
            if decision is not None and isinstance(decision.parent, dict):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.APPLIED,
                    value=decision.parent,
                )
            if ctx.spawn_intent_id in _exit_runtime_state().lifecycle_parent_preflight:
                return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED)
            result = _export_uuid(current_plan.identity.parent_uuid, prefer_cache=False)
            if _read_unavailable(result):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.UNAVAILABLE,
                    reason=_read_reason(result, "parent export unavailable"),
                )
            parent = _read_value(result)
            if not isinstance(parent, dict):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason="parent task is unavailable",
                )
            flow = _module("exit_entry_flow")
            mismatch = flow._parent_guard_mismatch(
                parent,
                current_plan.parent_guard.to_dict(),
                recurrence_fingerprint=lambda task: lifecycle_models.recurrence_fingerprint(
                    task,
                    parse_datetime=getattr(core, "parse_dt_any", None),
                ),
            )
            if mismatch:
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason=mismatch,
                )
            return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED)

        def find_equivalent_child(self, current_plan):
            decision = planned_decision()
            if decision is not None and isinstance(decision.child, dict):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.FOUND,
                    value=decision.child,
                )
            if (
                decision is not None
                and decision.kind is exit_models.LifecycleBatchDecisionKind.MISSING_CHILD
                and ctx.spawn_intent_id in _exit_runtime_state().lifecycle_batch_imported
            ):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.FOUND,
                    value=current_plan.child_dict(),
                )
            child = current_plan.child_dict()
            child_uuid = str(child.get("uuid") or "").strip()
            if child_uuid:
                # Discovery may use the authoritative drain preload. Fresh
                # reads remain mandatory for parent guards and every
                # post-mutation verification below.
                exact = _export_uuid(child_uuid)
                if _read_unavailable(exact):
                    return _lifecycle_operation_result(
                        lifecycle_executor.OperationState.UNAVAILABLE,
                        reason=_read_reason(exact, "child export unavailable"),
                    )
                exact_task = _read_value(exact)
                if isinstance(exact_task, dict):
                    return _lifecycle_operation_result(
                        lifecycle_executor.OperationState.FOUND,
                        value=exact_task,
                    )
            equivalent = _existing_equivalent_child(child, current_plan.identity.parent_uuid)
            if _read_unavailable(equivalent):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.UNAVAILABLE,
                    reason=_read_reason(equivalent, "equivalent child lookup unavailable"),
                )
            equivalent_task = _read_value(equivalent)
            if isinstance(equivalent_task, dict):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.FOUND,
                    value=equivalent_task,
                )
            return _lifecycle_operation_result(lifecycle_executor.OperationState.ABSENT)

        def import_child(self, current_plan):
            child = current_plan.child_dict()
            child_uuid = str(child.get("uuid") or "").strip()
            import_ctx = _build_exit_entry_context(
                ctx.entry,
                ctx.idx,
                state,
                parent_uuid=current_plan.identity.parent_uuid,
                child_short=ctx.child_short or _short_uuid(child_uuid),
                expected_parent_nextlink=ctx.expected_parent_nextlink,
                parent_guard=current_plan.parent_guard.to_dict(),
                child=child,
                child_uuid=child_uuid,
                spawn_intent_id=ctx.spawn_intent_id,
            )
            imported = _import_child(import_ctx)
            if imported.ok:
                stage_result = stage(lifecycle_models.ExecutionStage.CHILD_PRESENT)
                return stage_result
            operation_state = (
                lifecycle_executor.OperationState.UNAVAILABLE
                if imported.retryable
                else lifecycle_executor.OperationState.FAILED
            )
            return _lifecycle_operation_result(operation_state, reason=imported.err or "child import failed")

        def verify_child(self, current_plan, child):
            if getattr(state, "lifecycle_defer_verification", False):
                return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED, value=child)
            child_uuid = str(child.get("uuid") or "").strip()
            if not child_uuid:
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason="child verification has no UUID",
                )
            result = _export_uuid(child_uuid, prefer_cache=False)
            if _read_unavailable(result):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.UNAVAILABLE,
                    reason=_read_reason(result, "child verification unavailable"),
                )
            verified_child = _read_value(result)
            if not isinstance(verified_child, dict):
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason="child postcondition is missing",
                )
            child.clear()
            child.update(verified_child)
            return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED)

        def apply_parent_patch(self, current_plan, child):
            decision = planned_decision()
            if (
                decision is not None
                and decision.kind is exit_models.LifecycleBatchDecisionKind.ALREADY_SATISFIED
            ):
                return _lifecycle_operation_result(lifecycle_executor.OperationState.ALREADY)
            child_short = str(
                current_plan.parent_patch_dict().get("nextLink")
                or ctx.child_short
                or _short_uuid(str(child.get("uuid") or ""))
            ).strip()
            if not child_short:
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason="parent patch has no child identity",
                )
            expected_prev = str(ctx.entry.get("parent_nextlink") or "").strip() or None
            updated = _update_parent_nextlink(
                current_plan.identity.parent_uuid,
                child_short,
                expected_prev,
                parent_guard=current_plan.parent_guard.to_dict(),
                parent_snapshot=(
                    planned_decision().parent
                    if planned_decision() is not None and isinstance(planned_decision().parent, dict)
                    else None
                ),
            )
            if updated.ok:
                if updated.state == "already":
                    return _lifecycle_operation_result(lifecycle_executor.OperationState.ALREADY)
                stage_result = stage(lifecycle_models.ExecutionStage.PARENT_LINKED)
                return stage_result
            if updated.state == "locked":
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.UNAVAILABLE,
                    reason=updated.err or "parent linkage unavailable",
                )
            if updated.state in {"conflict", "missing", "invalid"}:
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason=updated.err or "parent nextLink changed",
                )
            operation_state = (
                lifecycle_executor.OperationState.UNAVAILABLE
                if updated.retryable
                else lifecycle_executor.OperationState.FAILED
            )
            return _lifecycle_operation_result(operation_state, reason=updated.err or "parent update failed")

        def verify_linkage(self, current_plan, child):
            if getattr(state, "lifecycle_defer_verification", False):
                return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED, value=child)
            child_short = str(
                current_plan.parent_patch_dict().get("nextLink")
                or ctx.child_short
                or _short_uuid(str(child.get("uuid") or ""))
            ).strip()
            state_result = _parent_nextlink_state(
                current_plan.identity.parent_uuid,
                child_short,
                str(ctx.entry.get("parent_nextlink") or "").strip() or None,
                prefer_cache=False,
            )
            if state_result.state == "locked":
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.UNAVAILABLE,
                    reason=state_result.err or "parent linkage verification unavailable",
                )
            if state_result.state != "already":
                return _lifecycle_operation_result(
                    lifecycle_executor.OperationState.CONFLICT,
                    reason=state_result.err or "parent linkage postcondition not satisfied",
                )
            return stage(lifecycle_models.ExecutionStage.VERIFIED)

        def compensate_child(self, _current_plan, child):
            child_uuid = str(child.get("uuid") or "").strip()
            if child_uuid:
                if getattr(state, "lifecycle_defer_verification", False):
                    state.lifecycle_orphan_cleanup.setdefault(
                        child_uuid,
                        (ctx.parent_uuid, ctx.child_short, ctx.spawn_intent_id),
                    )
                else:
                    cleanup = _cleanup_orphan_child(child_uuid, ctx.spawn_intent_id)
                    if not cleanup.ok:
                        return _lifecycle_operation_result(
                            lifecycle_executor.OperationState.UNAVAILABLE
                            if cleanup.retryable
                            else lifecycle_executor.OperationState.FAILED,
                            reason=cleanup.err or "orphan cleanup failed",
                        )
            return _lifecycle_operation_result(lifecycle_executor.OperationState.APPLIED)

    if ctx.spawn_intent_id in _exit_runtime_state().lifecycle_batch_imported:
        child_stage = _advance_lifecycle_stage(ctx.entry, lifecycle_models.ExecutionStage.CHILD_PRESENT)
        if not child_stage.ok:
            return _handle_lifecycle_stage_failure(ctx.entry, ctx.idx, state, child_stage)
    outcome = lifecycle_executor.LifecycleTransitionExecutor(Services()).execute(plan)
    _diag(
        f"lifecycle transition intent={ctx.spawn_intent_id} stage={plan.stage.value} "
        f"outcome={outcome.kind.value} reason={outcome.reason or 'none'}"
    )
    if outcome.kind in {
        lifecycle_models.LifecycleOutcomeKind.RETRYABLE,
    }:
        _diag(f"lifecycle transition retry: {outcome.reason or 'reason unavailable'}")
        if _bump_attempts(ctx.entry) > _QUEUE_RETRY_MAX:
            state.dead_letter(ctx.entry, outcome.reason or "lifecycle transition retry limit reached")
        else:
            state.requeue.append(ctx.entry)
        return False
    if outcome.kind is lifecycle_models.LifecycleOutcomeKind.MANUAL_REVIEW:
        state.dead_letter(ctx.entry, outcome.reason or "lifecycle transition requires manual review")
        state.reset_lock_streak()
        return False
    if getattr(state, "lifecycle_defer_verification", False):
        state.lifecycle_pending_verification[ctx.spawn_intent_id] = (ctx, plan, plan.child_dict())
        return False
    final_stage = _advance_lifecycle_stage(ctx.entry, lifecycle_models.ExecutionStage.FINALIZED)
    if not final_stage.ok:
        return _handle_lifecycle_stage_failure(ctx.entry, ctx.idx, state, final_stage)
    if not state.mark_final(ctx.entry, "done", "processed"):
        state.errors += 1
        state.requeue.append(ctx.entry)
        _diag("finalized intent write failed; retaining queue entry for retry")
        return False
    state.processed += 1
    state.reset_lock_streak()
    return False


def _process_queue_entry(idx: int, entry: dict, state: _DrainState) -> bool:
    if _handle_entry_gate(entry, state):
        return False
    queue_entry = entry
    entry = _normalize_queue_entry(entry)

    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip()
    parent_uuid = (entry.get("parent_uuid") or "").strip()
    expected_parent_nextlink = (entry.get("parent_nextlink") or "").strip()
    parent_guard = entry.get("parent_guard")
    child = entry.get("child") or {}
    child_short = (entry.get("child_short") or "").strip()
    child_uuid = (child.get("uuid") or "").strip()
    ctx = _build_exit_entry_context(
        queue_entry,
        idx,
        state,
        parent_uuid=parent_uuid,
        child_short=child_short,
        expected_parent_nextlink=expected_parent_nextlink or None,
        parent_guard=parent_guard if isinstance(parent_guard, dict) else None,
        child=child,
        child_uuid=child_uuid,
        spawn_intent_id=spawn_intent_id,
    )

    if not isinstance(queue_entry.get("lifecycle_plan"), dict):
        state.dead_letter(queue_entry, "outbox entry is missing a lifecycle plan")
        return False
    return _execute_lifecycle_queue_entry(ctx, state)

    # The lifecycle outbox always carries a typed plan.  The former raw queue
    # executor remains below only until its focused deletion pass.
    guard_action = _precheck_parent_guard(ctx)
    if guard_action == "break":
        return True
    if guard_action == "continue":
        return False

    exact_child = _export_uuid(child_uuid)
    if _read_unavailable(exact_child):
        return _requeue_or_dead_letter_for_lock(entry, idx, state)
    child_already_exists = _read_found(exact_child)
    if not child_already_exists:
        equivalent = _existing_equivalent_child(child, parent_uuid)
        if _read_unavailable(equivalent):
            return _requeue_or_dead_letter_for_lock(entry, idx, state)
        existing_obj = _read_value(equivalent)
        if isinstance(existing_obj, dict):
            child_uuid = (existing_obj.get("uuid") or "").strip()
            child_short = _short_uuid(child_uuid)
            child_already_exists = bool(child_uuid)
            if child_short:
                if spawn_intent_id:
                    _diag(
                        f"equivalent child already exists; binding intent {spawn_intent_id} "
                        f"to child {child_short}"
                    )
                else:
                    _diag(f"equivalent child already exists; binding to child {child_short}")

    ctx.child_short = child_short
    ctx.child_uuid = child_uuid

    link_action, parent_linked_already = _precheck_parent_link_state(ctx)
    if link_action == "break":
        return True
    if link_action == "continue":
        return False

    if child_already_exists:
        child_action, imported = ("ok", False)
    else:
        guard_action = _precheck_parent_guard(ctx)
        if guard_action == "break":
            return True
        if guard_action == "continue":
            return False
        child_action, imported = _ensure_child_exists_for_entry(ctx, initial_export_res=exact_child)
    if child_action == "break":
        return True
    if child_action == "continue":
        return False

    lifecycle_models = _module("lifecycle_models")
    child_stage = _advance_lifecycle_stage(queue_entry, lifecycle_models.ExecutionStage.CHILD_PRESENT)
    if not child_stage.ok:
        return _handle_lifecycle_stage_failure(queue_entry, idx, state, child_stage)

    parent_action = _apply_parent_update_for_entry(
        ctx,
        parent_linked_already=parent_linked_already,
        imported=imported,
    )
    if parent_action == "break":
        return True
    if parent_action == "continue":
        return False

    parent_stage = _advance_lifecycle_stage(queue_entry, lifecycle_models.ExecutionStage.PARENT_LINKED)
    if not parent_stage.ok:
        return _handle_lifecycle_stage_failure(queue_entry, idx, state, parent_stage)

    verification_action, verification_reason = _verify_lifecycle_postconditions(ctx)
    if verification_action != "ok":
        return _handle_lifecycle_postcondition_failure(queue_entry, idx, state, verification_reason)
    verified_stage = _advance_lifecycle_stage(queue_entry, lifecycle_models.ExecutionStage.VERIFIED)
    if not verified_stage.ok:
        return _handle_lifecycle_stage_failure(queue_entry, idx, state, verified_stage)
    finalized_stage = _advance_lifecycle_stage(queue_entry, lifecycle_models.ExecutionStage.FINALIZED)
    if not finalized_stage.ok:
        return _handle_lifecycle_stage_failure(queue_entry, idx, state, finalized_stage)

    if not state.mark_final(queue_entry, "done", "processed"):
        state.errors += 1
        state.requeue.append(queue_entry)
        _diag("finalized intent write failed; retaining queue entry for retry")
        return False
    state.processed += 1
    state.reset_lock_streak()
    return False


def _drain_queue_result(unit_of_work):
    _exit_runtime_state().unit_of_work = unit_of_work
    _exit_runtime_state().repository = unit_of_work.repository
    unit_of_work.repository.configure_commands(
        timeout=_TASK_TIMEOUT_EXPORT,
        attempts=_TASK_RETRIES_EXPORT,
        retry_delay=_TASK_RETRY_DELAY,
    )
    exit_drain_flow = _module("exit_drain_flow")
    return exit_drain_flow.drain_queue_result(
        services=exit_drain_flow.ExitDrainServices(
            take_queue_batch=_take_queue_batch,
            exit_progress_scope=_exit_progress_scope,
            preload_export_uuids=_preload_export_uuids,
            preload_equivalent_child_slots=_preload_equivalent_child_slots,
            prepare_lifecycle_batch=_prepare_lifecycle_batch,
            apply_lifecycle_batch=_apply_lifecycle_batch,
            finalize_lifecycle_batch=_finalize_lifecycle_batch,
            process_queue_entry=_process_queue_entry,
            requeue_entries_result=_requeue_entries_result,
            drain_state_factory=_DrainState,
        )
    )


def _drain_queue(unit_of_work) -> dict:
    return _drain_queue_result(unit_of_work).to_dict()


def _redirect_stdout_to_devnull() -> None:
    hook_results = _module("hook_results")
    hook_results.redirect_stdout_to_devnull()


def _emit_drain_stats_diag(stats: dict) -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    startup_stats = _exit_runtime_state().startup_stats
    if startup_stats:
        _diag_block("on-exit startup", startup_stats.items(), columns=2)
    drain_items = [
        ("entries_total", stats.get("entries_total", 0)),
        ("idempotent_skipped", stats.get("entries_skipped_idempotent", 0)),
        ("processed", stats.get("processed", 0)),
        ("errors", stats.get("errors", 0)),
        ("requeued", stats.get("requeued", 0)),
        ("requeue_failed", stats.get("requeue_failed", 0)),
        ("dead_lettered", stats.get("dead_lettered", 0)),
        ("queue_lock_failures", stats.get("queue_lock_failures", 0)),
        ("lock_events", stats.get("lock_events", 0)),
        ("lock_streak_max", stats.get("lock_streak_max", 0)),
        ("circuit_breaks", stats.get("circuit_breaks", 0)),
        ("intent_log_ready", stats.get("intent_log_ready", 0)),
        ("intent_log_size", stats.get("intent_log_size", 0)),
        ("intent_mark_ok", stats.get("intent_mark_ok", 0)),
        ("intent_mark_fail", stats.get("intent_mark_fail", 0)),
        ("intent_log_load_ms", stats.get("intent_log_load_ms", 0)),
        ("queue_db_opens", stats.get("queue_db_opens", 0)),
        ("queue_db_reuses", stats.get("queue_db_reuses", 0)),
        ("preload_export_uuids", stats.get("preload_export_uuids", 0)),
        ("preload_export_hits", stats.get("preload_export_hits", 0)),
        ("preload_export_misses", stats.get("preload_export_misses", 0)),
        ("preload_export_chunks", stats.get("preload_export_chunks", 0)),
        ("drain_ms", stats.get("drain_ms", 0)),
    ]
    _diag_block("on-exit drain", drain_items, columns=3)
    diag_stats = _exit_runtime_state().diag_stats
    task_stats = {
        "run_task_calls": diag_stats.get("run_task_calls", 0),
        "run_task_failures": diag_stats.get("run_task_failures", 0),
        "run_task_calls_export_uuid": diag_stats.get("run_task_calls_export_uuid", 0),
        "run_task_calls_export_equivalent_child": diag_stats.get("run_task_calls_export_equivalent_child", 0),
        "run_task_calls_import": diag_stats.get("run_task_calls_import", 0),
        "run_task_calls_modify_parent_nextlink": diag_stats.get("run_task_calls_modify_parent_nextlink", 0),
        "run_task_calls_modify_cleanup": diag_stats.get("run_task_calls_modify_cleanup", 0),
        "run_task_calls_modify_other": diag_stats.get("run_task_calls_modify_other", 0),
        "run_task_calls_export_other": diag_stats.get("run_task_calls_export_other", 0),
        "run_task_calls_other": diag_stats.get("run_task_calls_other", 0),
        "run_task_failures_export_uuid": diag_stats.get("run_task_failures_export_uuid", 0),
        "run_task_failures_export_equivalent_child": diag_stats.get("run_task_failures_export_equivalent_child", 0),
        "run_task_failures_import": diag_stats.get("run_task_failures_import", 0),
        "run_task_failures_modify_parent_nextlink": diag_stats.get("run_task_failures_modify_parent_nextlink", 0),
        "run_task_failures_modify_cleanup": diag_stats.get("run_task_failures_modify_cleanup", 0),
        "run_task_failures_modify_other": diag_stats.get("run_task_failures_modify_other", 0),
        "run_task_failures_export_other": diag_stats.get("run_task_failures_export_other", 0),
        "run_task_failures_other": diag_stats.get("run_task_failures_other", 0),
        "run_task_seconds_export_uuid": round(float(diag_stats.get("run_task_seconds_export_uuid", 0.0)), 4),
        "run_task_seconds_export_equivalent_child": round(float(diag_stats.get("run_task_seconds_export_equivalent_child", 0.0)), 4),
        "run_task_seconds_import": round(float(diag_stats.get("run_task_seconds_import", 0.0)), 4),
        "equivalent_child_cache_hits": diag_stats.get("equivalent_child_cache_hits", 0),
        "equivalent_child_cache_misses": diag_stats.get("equivalent_child_cache_misses", 0),
        "equivalent_child_cache_seeded": diag_stats.get("equivalent_child_cache_seeded", 0),
        "equivalent_child_preload_slots": diag_stats.get("equivalent_child_preload_slots", 0),
        "equivalent_child_preload_hits": diag_stats.get("equivalent_child_preload_hits", 0),
        "equivalent_child_preload_misses": diag_stats.get("equivalent_child_preload_misses", 0),
        "equivalent_child_preload_chunks": diag_stats.get("equivalent_child_preload_chunks", 0),
        "run_task_seconds_modify_parent_nextlink": round(float(diag_stats.get("run_task_seconds_modify_parent_nextlink", 0.0)), 4),
        "run_task_seconds_modify_cleanup": round(float(diag_stats.get("run_task_seconds_modify_cleanup", 0.0)), 4),
        "run_task_seconds_modify_other": round(float(diag_stats.get("run_task_seconds_modify_other", 0.0)), 4),
        "run_task_seconds_export_other": round(float(diag_stats.get("run_task_seconds_export_other", 0.0)), 4),
        "run_task_seconds_other": round(float(diag_stats.get("run_task_seconds_other", 0.0)), 4),
        "run_task_seconds": round(float(diag_stats.get("run_task_seconds", 0.0)), 4),
    }
    _diag_block("on-exit task stats", task_stats.items(), columns=3)


def _strict_exit_feedback_message(stats: dict) -> str | None:
    errors = stats.get("errors", 0)
    dead_lettered = stats.get("dead_lettered", 0)
    queue_lock_failures = stats.get("queue_lock_failures", 0)
    if not (_EXIT_STRICT and (errors > 0 or dead_lettered > 0 or queue_lock_failures > 0)):
        return None
    return (
        f"[nautical] on-exit: {dead_lettered} dead-lettered, {errors} errors, {queue_lock_failures} queue lock failures. "
        "Check .nautical-state/.nautical_dead_letter.jsonl (set NAUTICAL_EXIT_STRICT=0 to disable)"
    )


def _render_exit_drain_failure_panel(stats: dict) -> None:
    if not isinstance(stats, dict) or core is None:
        return

    def count(key: str) -> int:
        try:
            return max(0, int(stats.get(key, 0) or 0))
        except Exception:
            return 0

    errors = count("errors")
    dead_lettered = count("dead_lettered")
    requeue_failed = count("requeue_failed")
    if not (errors or dead_lettered or requeue_failed):
        return

    problems = []
    if dead_lettered:
        problems.append(f"{dead_lettered} dead-lettered")
    if requeue_failed:
        suffix = "" if requeue_failed == 1 else "s"
        problems.append(f"{requeue_failed} requeue write{suffix} failed")
    other_errors = max(0, errors - dead_lettered - requeue_failed)
    if other_errors:
        suffix = "" if other_errors == 1 else "s"
        problems.append(f"{other_errors} other drain error{suffix}")

    rows = [
        ("Action", "Run nautical queue-status"),
        ("Problems", "; ".join(problems) or f"{errors} drain errors"),
    ]
    if dead_lettered:
        rows.append(("State", ".nautical-state/.nautical_dead_letter.jsonl"))
    requeued = count("requeued")
    if requeued:
        rows.append(("Retrying", str(requeued)))
    queue_lock_failures = count("queue_lock_failures")
    if queue_lock_failures:
        rows.append(("Lock events", str(queue_lock_failures)))

    core.render_panel(
        "⚠ Nautical spawn drain failed",
        rows,
        kind="warning",
        panel_mode=core.PANEL_MODE,
        live_duration_ms=getattr(core, "LIVE_PANEL_DURATION_MS", 160),
        live_footer=getattr(core, "LIVE_PANEL_FOOTER", "NAUTICAL"),
        fast_color=core.FAST_COLOR,
        themes=core.panel_themes(),
        allow_line=True,
        label_width_min=6,
        label_width_max=14,
    )


def main() -> int:
    # Queue draining and diagnostics use the configured core/UI services;
    # defer that package import until the lifecycle is actually entered.
    _load_core()
    _reset_exit_runtime_state()
    startup_t0 = time.perf_counter()
    module_t0 = time.perf_counter()
    hook_context = _module("hook_context")
    hook_results = _module("hook_results")
    hook_engine = _module("hook_engine")
    module_ms = round((time.perf_counter() - module_t0) * 1000.0, 3)
    request_t0 = time.perf_counter()
    request = hook_context.build_on_exit_request(runtime=_build_hook_runtime_context())
    request_ms = round((time.perf_counter() - request_t0) * 1000.0, 3)
    _exit_runtime_state().startup_stats = {
        "startup_import_ms": round(float(_IMPORT_MS or 0.0), 3),
        "startup_module_ms": module_ms,
        "startup_request_ms": request_ms,
        "startup_total_ms": round((time.perf_counter() - startup_t0) * 1000.0, 3),
    }
    result = hook_engine.handle_on_exit(
        request,
        exit_result_cls=hook_results.ExitHookResponse,
        redirect_stdout_to_devnull=_redirect_stdout_to_devnull,
        drain_queue=_drain_queue,
        strict_exit_result=_strict_exit_feedback_message,
    )
    stats_path = (os.environ.get("NAUTICAL_BENCH_STATS_FILE") or "").strip()
    if stats_path:
        try:
            Path(stats_path).write_text(
                json.dumps(
                    {"task_stats": dict(_exit_runtime_state().diag_stats)},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            _diag(f"benchmark stats write failed: {type(exc).__name__}: {exc}")
    _render_exit_drain_failure_panel(result.stats or {})
    return hook_results.emit_exit_result(
        result,
        emit_exit_feedback=_emit_exit_feedback,
        emit_stats_diag=_emit_drain_stats_diag,
    )


def run_hook(
    *,
    argv: tuple[str, ...],
    hook_dir: str,
    core_base: str,
) -> int:
    """Run the extracted implementation with context captured by the wrapper."""
    global HOOK_DIR, TW_DIR, _CORE_BASE
    global _TASKDATA_RAW, _USE_RC_DATA_LOCATION, TW_DATA_DIR
    global _QUEUE_DB_PATH, _DEAD_LETTER_PATH, _DEAD_LETTER_LOCK
    global _ORPHAN_CLEANUP_EVIDENCE_PATH, _ORPHAN_CLEANUP_EVIDENCE_LOCK
    global _QUEUE_LOCK_FAIL_MARKER, _QUEUE_LOCK_FAIL_COUNT

    HOOK_DIR = Path(hook_dir)
    TW_DIR = HOOK_DIR.parent
    _CORE_BASE = Path(core_base)
    sys.argv = [sys.argv[0], *argv]
    try:
        _initialize_integration_context()
    except _hook_runtime_module().HookIntegrationContextError as exc:
        globals()["core"] = exc.core
        _emit_exit_feedback(f"[nautical] on-exit: {exc.stage}: {exc.detail}")
        return 1

    state_dir = TW_DATA_DIR / ".nautical-state"
    lock_dir = TW_DATA_DIR / ".nautical-locks"
    _QUEUE_DB_PATH = state_dir / ".nautical_queue.db"
    _DEAD_LETTER_PATH = state_dir / ".nautical_dead_letter.jsonl"
    _DEAD_LETTER_LOCK = lock_dir / ".nautical_dead_letter.lock"
    _ORPHAN_CLEANUP_EVIDENCE_PATH = state_dir / ".nautical_orphan_cleanup.jsonl"
    _ORPHAN_CLEANUP_EVIDENCE_LOCK = lock_dir / ".nautical_orphan_cleanup.lock"
    _QUEUE_LOCK_FAIL_MARKER = lock_dir / ".nautical_spawn_queue.lock_failed"
    _QUEUE_LOCK_FAIL_COUNT = lock_dir / ".nautical_spawn_queue.lock_failed.count"
    return main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        _diag(f"on-exit unexpected error: {e}")
        err_text = _diag_redact_msg(f"{type(e).__name__}: {e}")
        _emit_exit_feedback(f"[nautical] on-exit: unexpected error: {err_text}")
        try:
            _write_dead_letter({"error": "unexpected_error"}, "on-exit exception")
        except Exception:
            pass
        raise SystemExit(1)
