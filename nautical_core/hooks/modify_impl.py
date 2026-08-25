#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heavy on-modify implementation loaded lazily by the executable wrapper.

- Works for classic cp (cp/chainMax/chainUntil) and anchors (anchor/anchor_mode).
- Cap logic unified (chainMax, chainUntil -> numeric cap_no).
- Queues child spawn intent; on-exit hook performs `task import -`.
- Timeline is capped and marks (last link).
"""

import sys, json, os, importlib, importlib.util
import time as _ptime
import copy
from contextlib import nullcontext
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

# Ensure hook IO supports Unicode (emoji, symbols) in JSON output.
hook_bootstrap.ensure_utf8_stdio()

_IMPORT_T0 = _ptime.perf_counter()
_IMPORT_MS = None

_MAX_JSON_BYTES = 10 * 1024 * 1024
HOOK_IMPL_API = 1
NAUTICAL_HOOK_VERSION = "updateG-20260328"

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
_EARLY_PROTOCOL_RESULT = None
_PROTOCOL = None
_PROBE_UNSET = object()

if __name__ == "__main__":
    _protocol, _protocol_path, _protocol_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "hook_protocol.py",
        "_nautical_hook_protocol_modify",
    )
    _PROTOCOL = _protocol
    if _protocol is not None:
        try:
            _EARLY_PROTOCOL_RESULT = _protocol.read_on_modify(max_bytes=_MAX_JSON_BYTES)
        except Exception:
            _EARLY_PROTOCOL_RESULT = None
        if (
            _EARLY_PROTOCOL_RESULT is not None
            and _EARLY_PROTOCOL_RESULT.valid
            and not _EARLY_PROTOCOL_RESULT.is_nautical
            and os.environ.get("NAUTICAL_BENCH_FORCE_FULL") != "1"
        ):
            _protocol.emit_passthrough_json(_EARLY_PROTOCOL_RESULT.task)
            raise SystemExit(0)


import atexit
import hashlib
import random
import re
import stat
import tempfile
import time as _time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NoReturn, Optional

if TYPE_CHECKING:
    from nautical_core.modify_models import CompletionLifecycleResult


# set config show_analytics=false to disable analytics panel entry.

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

_MAX_CHAIN_WALK = 500  # cap for chain summaries/analytics
_MAX_ITERATIONS = 2000  # prevent infinite loops in stepping functions
_MIN_FUTURE_WARN = 365 * 2  # warn if chain extends >2 years


_MAX_SPAWN_ATTEMPTS = 3
_SPAWN_RETRY_DELAY = 0.1  # seconds between retries
_STABLE_CHILD_UUID_NAMESPACE = uuid.UUID("1f4b2396-df58-5a32-a879-33f0d3fe711f")
# Panel chain index and chain caches live in the per-run modify runtime state.
_MODIFY_RUNTIME_STATE = None
_HOOK_CONTEXT = None
_HOOK_CONTEXT_LOAD_FAILED = False
_HOOK_ENGINE = None
_HOOK_ENGINE_LOAD_FAILED = False
_HOOK_RESULTS = None
_HOOK_RESULTS_LOAD_FAILED = False
# ------------------------------------------------------------------------------
# Debug: wait/scheduled carry-forward
# Set debug_wait_sched=true to include carry computations in the feedback panel.
# ------------------------------------------------------------------------------
_DEFAULT_DEBUG_WAIT_SCHED = False
_LAST_WAIT_SCHED_DEBUG: OrderedDict[str, dict[str, Any]] = OrderedDict()
_MAX_WAIT_SCHED_DEBUG = 32


def _diag(msg: str) -> None:
    try:
        _load_core()
    except Exception:
        pass
    if core is not None:
        event_factory = getattr(core, "DiagnosticEvent", None)
        event = event_factory.from_message(msg, hook="on-modify") if event_factory is not None else msg
        core.diag(event, "on-modify", str(TW_DATA_DIR))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass


def _modify_runtime_state():
    global _MODIFY_RUNTIME_STATE
    if _MODIFY_RUNTIME_STATE is None:
        modify_runtime = _module("modify_runtime")
        _MODIFY_RUNTIME_STATE = modify_runtime.new_runtime_state()
    return _MODIFY_RUNTIME_STATE


def _reset_modify_runtime_state() -> None:
    global _MODIFY_RUNTIME_STATE
    modify_runtime = _module("modify_runtime")
    _MODIFY_RUNTIME_STATE = modify_runtime.new_runtime_state()


def _workflow_now_utc() -> datetime:
    workflow = getattr(_modify_runtime_state(), "workflow_context", None)
    value = getattr(workflow, "now_utc", None)
    if isinstance(value, datetime):
        return value
    return core.now_utc()


def _anchor_file_provider_for(
    anchor_file: str,
    *,
    fallback_hhmm: tuple[int, int],
    seed_base: str,
):
    return _module("modify_runtime").anchor_file_provider_for(
        anchor_file,
        fallback_hhmm=fallback_hhmm,
        seed_base=seed_base,
        state=_modify_runtime_state(),
        core=core,
    )


def _anchor_file_fallback_hhmm(task: dict, default_local: datetime) -> tuple[int, int]:
    """Keep provider fallback time stable across completion projection stages."""
    for field in ("due", "scheduled"):
        parsed, error = _safe_parse_datetime(task.get(field))
        if not error and parsed is not None:
            local = _to_local_cached(parsed)
            return local.hour, local.minute
    return default_local.hour, default_local.minute


def _modify_chain_state():
    return _modify_runtime_state()


def _diag_count(key: str, inc: int = 1) -> None:
    try:
        state = _modify_runtime_state()
        stats = state.diag_stats
        stats[key] = stats.get(key, 0) + inc
    except Exception:
        pass


def _run_task_diag_bucket(cmd: list[str]) -> str:
    try:
        parts = []
        for p in (cmd or ()):
            parts.extend(str(p).split())
        parts = tuple(parts)
    except Exception:
        return "other"
    if not parts:
        return "other"
    if "_get" in parts:
        return "get"
    if "import" in parts:
        return "import"
    if "count" in parts:
        return "count"
    if "export" in parts:
        if any(p.startswith("chainID:") for p in parts):
            return "export_chain"
        return "other"
    return "other"


def _diag_record_run_task(cmd: list[str], *, ok: bool, elapsed: float) -> None:
    bucket = _run_task_diag_bucket(cmd)
    _diag_count(f"run_task_calls_{bucket}")
    _diag_count(f"run_task_seconds_{bucket}", float(elapsed or 0.0))
    if not ok:
        _diag_count(f"run_task_failures_{bucket}")


def _emit_diag_block(title: str, items, *, columns: int = 3) -> None:
    try:
        pairs = [f"{k}={v}" for k, v in (items or ())]
        sys.stderr.write(f"[nautical] {title}:\n")
        step = max(1, int(columns or 1))
        for idx in range(0, len(pairs), step):
            sys.stderr.write("[nautical]   " + "  ".join(pairs[idx:idx + step]) + "\n")
    except Exception:
        pass


def _dump_diag_stats() -> None:
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            state = _modify_runtime_state()
            stats = state.diag_stats
            elapsed = _ptime.perf_counter() - state.diag_start_ts
            stats["hook_seconds"] = round(elapsed, 4)
            stats["run_task_seconds"] = round(stats.get("run_task_seconds", 0.0), 4)
            _emit_diag_block("diag stats", stats.items(), columns=3)
        except Exception:
            pass


def _query_ctx_get(bucket: str, key):
    try:
        store = _modify_runtime_state().query_ctx.get(bucket)
        if isinstance(store, dict):
            return store.get(key)
    except Exception:
        pass


def _write_bench_stats() -> None:
    """Persist opt-in workflow counters for the performance harness."""
    path = str(os.environ.get("NAUTICAL_BENCH_STATS_FILE") or "").strip()
    if not path:
        return
    try:
        stats = dict(_modify_runtime_state().diag_stats)
        Path(path).write_text(
            json.dumps({"task_stats": stats}, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        _diag(f"benchmark stats write failed: {type(exc).__name__}: {exc}")
    return None


def _query_ctx_set(bucket: str, key, value) -> None:
    try:
        state = _modify_runtime_state()
        store = state.query_ctx.get(bucket)
        if isinstance(store, dict):
            store[key] = value
            state.diag_stats[f"query_ctx_{bucket}_entries"] = len(store)
    except Exception:
        pass


_READ_QUERY_MISSING = object()


def _read_query_get(kind: str, key):
    """Return a defensive copy of a read-only Taskwarrior query result."""
    try:
        state = _modify_runtime_state()
        bucket = state.query_ctx.get("read_query")
        cache_key = (str(kind), key)
        if not isinstance(bucket, dict) or cache_key not in bucket:
            _diag_count("read_query_cache_misses")
            return _READ_QUERY_MISSING
        _diag_count("read_query_cache_hits")
        return copy.deepcopy(bucket[cache_key])
    except Exception:
        _diag_count("read_query_cache_misses")
        return _READ_QUERY_MISSING


def _read_query_set(kind: str, key, value) -> None:
    try:
        state = _modify_runtime_state()
        bucket = state.query_ctx.get("read_query")
        if not isinstance(bucket, dict):
            return
        stored = copy.deepcopy(value)
        if str(kind) in {"chain", "chain_snapshot"} and isinstance(stored, list):
            if len(stored) > _MAX_CHAIN_WALK:
                stored = stored[:_MAX_CHAIN_WALK]
                state.diag_stats["chain_snapshot_truncations"] = (
                    state.diag_stats.get("chain_snapshot_truncations", 0) + 1
                )
        bucket[(str(kind), key)] = stored
        state.diag_stats["read_query_cache_entries"] = len(bucket)
    except Exception:
        pass


def _read_query_delete(kind: str, key) -> None:
    try:
        state = _modify_runtime_state()
        bucket = state.query_ctx.get("read_query")
        if isinstance(bucket, dict):
            bucket.pop((str(kind), key), None)
            state.diag_stats["read_query_cache_entries"] = len(bucket)
    except Exception:
        pass


def _invalidate_read_query_cache() -> None:
    """Invalidate all request-scoped reads after a Taskwarrior mutation."""
    try:
        state = _modify_runtime_state()
        bucket = state.query_ctx.get("read_query")
        if isinstance(bucket, dict):
            bucket.clear()
        state.diag_stats["read_query_cache_entries"] = 0
        state.diag_stats["read_query_cache_invalidations"] = (
            state.diag_stats.get("read_query_cache_invalidations", 0) + 1
        )
    except Exception:
        pass
    try:
        service = getattr(_modify_chain_state(), "lifecycle_read_service", None)
        clear_cache = getattr(service, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
        else:
            _module("lifecycle_read_service").clear_cached_chain_exports()
    except Exception:
        pass


def _record_chain_snapshot_stat(name: str, inc: int = 1) -> None:
    try:
        state = _modify_runtime_state()
        state.diag_stats[name] = state.diag_stats.get(name, 0) + inc
    except Exception:
        pass


def _task_args_cacheable(args) -> bool:
    try:
        parts = tuple(str(a) for a in (args or ()))
    except Exception:
        return False
    return ('_get' in parts) or ('export' in parts) or ('count' in parts)


def _diag_summary() -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        parts = [
            ("spawn_deferred", _modify_runtime_state().diag_stats.get("spawn_deferred", 0)),
            ("queue_lock_failures", _modify_runtime_state().diag_stats.get("queue_lock_failures", 0)),
        ]
        _emit_diag_block("diag summary", parts, columns=2)
    except Exception:
        pass


def _diag_lifecycle_result(result) -> None:
    """Write structured lifecycle diagnostics only to gated stderr output."""
    if os.environ.get("NAUTICAL_DIAG") != "1" or result is None:
        return
    try:
        diagnostic = getattr(result, "diagnostic", None)
        items = [
            ("state", getattr(result, "state", "")),
            ("reason", str(getattr(result, "reason", "") or "").replace("\n", " ")),
        ]
        if diagnostic is not None:
            items.extend(
                [
                    ("transition", diagnostic.transition_id),
                    ("chain", diagnostic.chain_id),
                    ("parent_link", diagnostic.parent_link),
                    ("child_link", diagnostic.child_link),
                    ("stage", diagnostic.stage),
                    ("attempts", diagnostic.attempts),
                    ("failure_kind", diagnostic.failure_kind),
                ]
            )
        _emit_diag_block("completion lifecycle", items, columns=2)
    except Exception:
        pass


atexit.register(_dump_diag_stats)


def _fmt_td_dd_hhmm(delta: timedelta) -> str:
    """Format a timedelta as ±Dd HHh:MMm (UTC-seconds based; seconds omitted)."""
    try:
        total = int(delta.total_seconds())
    except Exception:
        return str(delta)
    sign = "-" if total < 0 else "+"
    total = abs(total)
    # truncate seconds
    total_minutes = total // 60
    dd, rem_m = divmod(total_minutes, 1440)  # 24*60
    hh, mm = divmod(rem_m, 60)
    return f"{sign}{dd}d {hh:02}h:{mm:02}m"


def _append_next_wait_sched_rows(
    fb: list[tuple[str, str]],
    nxt: dict,
    nxt_due_utc: datetime,
    *,
    anchor_field: str = "due",
) -> None:
    """Compatibility adapter for wait/scheduled feedback presentation."""
    _module("modify_feedback").append_next_wait_sched_rows(
        fb,
        _module("modify_models").TaskView.from_mapping(nxt),
        nxt_due_utc,
        anchor_field=anchor_field,
        format_local=core.fmt_dt_local,
        compare_datetimes=_compare_datetimes,
        format_delta=_fmt_td_dd_hhmm,
    )

core = None
_DATETIME_COMPARATOR = None
_CORE_READY = False
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
_HOOK_SUPPORT = None
_HOOK_SUPPORT_LOAD_FAILED = False
_MODIFY_QUERIES = None
_MODIFY_QUERIES_LOAD_FAILED = False
_CHAIN_GENERATION = None
_CHAIN_GENERATION_LOAD_FAILED = False
_MODIFY_ORDINARY = None
_MODIFY_ORDINARY_LOAD_FAILED = False
_MODIFY_SPAWN_PREP = None
_MODIFY_SPAWN = None
_MODIFY_SPAWN_LOAD_FAILED = False
_MODIFY_SPAWN_PREP_LOAD_FAILED = False
_MODIFY_COMPLETION_PREFLIGHT = None
_MODIFY_COMPLETION_PREFLIGHT_LOAD_FAILED = False
_MODIFY_COMPLETION_COMPUTE = None
_MODIFY_COMPLETION_COMPUTE_LOAD_FAILED = False
_MODIFY_COMPLETION_SPAWN = None
_MODIFY_COMPLETION_SPAWN_LOAD_FAILED = False
_MODIFY_MODELS = None
_MODIFY_MODELS_LOAD_FAILED = False
_LIFECYCLE_MODELS = None
_LIFECYCLE_MODELS_LOAD_FAILED = False
_LIFECYCLE_PLANNER = None
_LIFECYCLE_PLANNER_LOAD_FAILED = False
_LIFECYCLE_APPLICATION = None
_LIFECYCLE_APPLICATION_LOAD_FAILED = False
_LIFECYCLE_OUTBOX = None
_LIFECYCLE_OUTBOX_LOAD_FAILED = False
_MODIFY_FEEDBACK = None
_MODIFY_FEEDBACK_LOAD_FAILED = False
_MODIFY_TIMELINE = None
_MODIFY_TIMELINE_LOAD_FAILED = False
_MODIFY_EXPIRATION = None
_MODIFY_EXPIRATION_LOAD_FAILED = False
_HOOK_RUNTIME = None
_HOOK_RUNTIME_LOAD_FAILED = False
_INTEGRATION_CONTEXT_MODULE = None
_INTEGRATION_CONTEXT_MODULE_LOAD_FAILED = False
_INTEGRATION_CONTEXT = None
_HOOK_MODULE_ACCESS = None
_RECURRENCE_EVALUATOR = None
_RECURRENCE_EVALUATOR_LOAD_FAILED = False
_ADD_ANCHOR_COMPUTE = None
_ADD_ANCHOR_COMPUTE_LOAD_FAILED = False
_MODIFY_PROTOCOL = None
_MODIFY_PROTOCOL_LOAD_FAILED = False
_MODIFY_CHAIN_SUMMARY = None
_MODIFY_CHAIN_SUMMARY_LOAD_FAILED = False
_MODIFY_WORKFLOW = None
_MODIFY_WORKFLOW_LOAD_FAILED = False
_MODIFY_COMPOSITION = None
_MODIFY_COMPOSITION_LOAD_FAILED = False
_MODIFY_EFFECTS = None
_MODIFY_EFFECTS_LOAD_FAILED = False
_MODIFY_COMPLETION_EFFECTS = None
_MODIFY_COMPLETION_EFFECTS_LOAD_FAILED = False
_ASTRONOMY_VALIDATION = None
_ASTRONOMY_VALIDATION_LOAD_FAILED = False
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
    "hook_support": (
        "_HOOK_SUPPORT",
        "_HOOK_SUPPORT_LOAD_FAILED",
        "hook_support.py",
        "nautical_core.hook_support",
    ),
    "modify_queries": (
        "_MODIFY_QUERIES",
        "_MODIFY_QUERIES_LOAD_FAILED",
        "modify_queries.py",
        "nautical_core.modify_queries",
    ),
    "lifecycle_read_service": (
        "_LIFECYCLE_READ_SERVICE",
        "_LIFECYCLE_READ_SERVICE_LOAD_FAILED",
        "lifecycle_read_service.py",
        "nautical_core.lifecycle_read_service",
    ),
    "modify_spawn_prep": (
        "_MODIFY_SPAWN_PREP",
        "_MODIFY_SPAWN_PREP_LOAD_FAILED",
        "modify_spawn_prep.py",
        "nautical_core.modify_spawn_prep",
    ),
    "modify_spawn": (
        "_MODIFY_SPAWN",
        "_MODIFY_SPAWN_LOAD_FAILED",
        "modify_spawn.py",
        "nautical_core.modify_spawn",
    ),
    "chain_generation": (
        "_CHAIN_GENERATION",
        "_CHAIN_GENERATION_LOAD_FAILED",
        "chain_generation.py",
        "nautical_core.chain_generation",
    ),
    "modify_ordinary": (
        "_MODIFY_ORDINARY",
        "_MODIFY_ORDINARY_LOAD_FAILED",
        "modify_ordinary.py",
        "nautical_core.modify_ordinary",
    ),
    "modify_completion_preflight": (
        "_MODIFY_COMPLETION_PREFLIGHT",
        "_MODIFY_COMPLETION_PREFLIGHT_LOAD_FAILED",
        "modify_completion_preflight.py",
        "nautical_core.modify_completion_preflight",
    ),
    "modify_completion_compute": (
        "_MODIFY_COMPLETION_COMPUTE",
        "_MODIFY_COMPLETION_COMPUTE_LOAD_FAILED",
        "modify_completion_compute.py",
        "nautical_core.modify_completion_compute",
    ),
    "modify_completion_spawn": (
        "_MODIFY_COMPLETION_SPAWN",
        "_MODIFY_COMPLETION_SPAWN_LOAD_FAILED",
        "modify_completion_spawn.py",
        "nautical_core.modify_completion_spawn",
    ),
    "modify_models": (
        "_MODIFY_MODELS",
        "_MODIFY_MODELS_LOAD_FAILED",
        "modify_models.py",
        "nautical_core.modify_models",
    ),
    "lifecycle_models": (
        "_LIFECYCLE_MODELS",
        "_LIFECYCLE_MODELS_LOAD_FAILED",
        "lifecycle_models.py",
        "nautical_core.lifecycle_models",
    ),
    "task_codec": (
        "_TASK_CODEC",
        "_TASK_CODEC_LOAD_FAILED",
        "task_codec.py",
        "nautical_core.task_codec",
    ),
    "task_models": (
        "_TASK_MODELS",
        "_TASK_MODELS_LOAD_FAILED",
        "task_models.py",
        "nautical_core.task_models",
    ),
    "lifecycle_planner": (
        "_LIFECYCLE_PLANNER",
        "_LIFECYCLE_PLANNER_LOAD_FAILED",
        "lifecycle_planner.py",
        "nautical_core.lifecycle_planner",
    ),
    "chain_integrity_lifecycle": (
        "_CHAIN_INTEGRITY_LIFECYCLE",
        "_CHAIN_INTEGRITY_LIFECYCLE_LOAD_FAILED",
        "chain_integrity_lifecycle.py",
        "nautical_core.chain_integrity_lifecycle",
    ),
    "lifecycle_application": (
        "_LIFECYCLE_APPLICATION",
        "_LIFECYCLE_APPLICATION_LOAD_FAILED",
        "lifecycle_application.py",
        "nautical_core.lifecycle_application",
    ),
    "lifecycle_outbox": (
        "_LIFECYCLE_OUTBOX",
        "_LIFECYCLE_OUTBOX_LOAD_FAILED",
        "lifecycle_outbox.py",
        "nautical_core.lifecycle_outbox",
    ),
    "modify_feedback": (
        "_MODIFY_FEEDBACK",
        "_MODIFY_FEEDBACK_LOAD_FAILED",
        "modify_feedback.py",
        "nautical_core.modify_feedback",
    ),
    "modify_lifecycle": (
        "_MODIFY_LIFECYCLE",
        "_MODIFY_LIFECYCLE_LOAD_FAILED",
        "modify_lifecycle.py",
        "nautical_core.modify_lifecycle",
    ),
    "modify_runtime": (
        "_MODIFY_RUNTIME",
        "_MODIFY_RUNTIME_LOAD_FAILED",
        "modify_runtime.py",
        "nautical_core.modify_runtime",
    ),
    "modify_timeline": (
        "_MODIFY_TIMELINE",
        "_MODIFY_TIMELINE_LOAD_FAILED",
        "modify_timeline.py",
        "nautical_core.modify_timeline",
    ),
    "modify_expiration": (
        "_MODIFY_EXPIRATION",
        "_MODIFY_EXPIRATION_LOAD_FAILED",
        "modify_expiration.py",
        "nautical_core.modify_expiration",
    ),
    "modify_analytics": (
        "_MODIFY_ANALYTICS",
        "_MODIFY_ANALYTICS_LOAD_FAILED",
        "modify_analytics.py",
        "nautical_core.modify_analytics",
    ),
    "anchor_omit": (
        "_ANCHOR_OMIT",
        "_ANCHOR_OMIT_LOAD_FAILED",
        "anchor_omit.py",
        "nautical_core.anchor_omit",
    ),
    "add_anchor_compute": (
        "_ADD_ANCHOR_COMPUTE",
        "_ADD_ANCHOR_COMPUTE_LOAD_FAILED",
        "add_anchor_compute.py",
        "nautical_core.add_anchor_compute",
    ),
    "panel_diagnostics": (
        "_PANEL_DIAGNOSTICS",
        "_PANEL_DIAGNOSTICS_LOAD_FAILED",
        "panel_diagnostics.py",
        "nautical_core.panel_diagnostics",
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
    "recurrence_evaluator": (
        "_RECURRENCE_EVALUATOR",
        "_RECURRENCE_EVALUATOR_LOAD_FAILED",
        "recurrence_evaluator.py",
        "nautical_core.recurrence_evaluator",
    ),
    "modify_protocol": (
        "_MODIFY_PROTOCOL",
        "_MODIFY_PROTOCOL_LOAD_FAILED",
        "modify_protocol.py",
        "nautical_core.modify_protocol",
    ),
    "modify_chain_summary": (
        "_MODIFY_CHAIN_SUMMARY",
        "_MODIFY_CHAIN_SUMMARY_LOAD_FAILED",
        "modify_chain_summary.py",
        "nautical_core.modify_chain_summary",
    ),
    "modify_workflow": (
        "_MODIFY_WORKFLOW",
        "_MODIFY_WORKFLOW_LOAD_FAILED",
        "modify_workflow.py",
        "nautical_core.modify_workflow",
    ),
    "modify_composition": (
        "_MODIFY_COMPOSITION",
        "_MODIFY_COMPOSITION_LOAD_FAILED",
        "modify_composition.py",
        "nautical_core.modify_composition",
    ),
    "modify_effects": (
        "_MODIFY_EFFECTS",
        "_MODIFY_EFFECTS_LOAD_FAILED",
        "modify_effects.py",
        "nautical_core.modify_effects",
    ),
    "modify_completion_effects": (
        "_MODIFY_COMPLETION_EFFECTS",
        "_MODIFY_COMPLETION_EFFECTS_LOAD_FAILED",
        "modify_completion_effects.py",
        "nautical_core.modify_completion_effects",
    ),
    "modify_transition_effects": (
        "_MODIFY_TRANSITION_EFFECTS",
        "_MODIFY_TRANSITION_EFFECTS_LOAD_FAILED",
        "modify_transition_effects.py",
        "nautical_core.modify_transition_effects",
    ),
    "modify_validation": (
        "_MODIFY_VALIDATION",
        "_MODIFY_VALIDATION_LOAD_FAILED",
        "modify_validation.py",
        "nautical_core.modify_validation",
    ),
    "astronomy_validation": (
        "_ASTRONOMY_VALIDATION",
        "_ASTRONOMY_VALIDATION_LOAD_FAILED",
        "astronomy_validation.py",
        "nautical_core.astronomy_validation",
    ),
    "modify_carry": (
        "_MODIFY_CARRY",
        "_MODIFY_CARRY_LOAD_FAILED",
        "modify_carry.py",
        "nautical_core.modify_carry",
    ),
    "modify_carry_workflow": (
        "_MODIFY_CARRY_WORKFLOW",
        "_MODIFY_CARRY_WORKFLOW_LOAD_FAILED",
        "modify_carry_workflow.py",
        "nautical_core.modify_carry_workflow",
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


def _build_hook_runtime_context(task=None):
    hook_runtime = _hook_runtime_module()
    business_calendar = None
    if task is not None:
        resolver = getattr(core, "business_calendar_for_task", None)
        if callable(resolver):
            business_calendar = resolver(task)
    return hook_runtime.build_hook_runtime_context(
        module_access=_hook_module_access(),
        hook_name="on-modify",
        integration_context=_INTEGRATION_CONTEXT,
        hook_dir=str(HOOK_DIR),
        import_ms=_IMPORT_MS,
        business_calendar=business_calendar,
    )


def _task_cmd_prefix() -> list[str]:
    from nautical_core.runtime_command import command_prefix

    return command_prefix(_INTEGRATION_CONTEXT, hook_name="on-modify")


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
        access="read_only",
    )
    _CORE_IMPORT_TARGET = target
    _INTEGRATION_CONTEXT = context
    TW_DATA_DIR = context.taskdata
    _TASKDATA_RAW = str(context.taskdata)
    _USE_RC_DATA_LOCATION = len(context.command_prefix) > 1

def _load_core() -> None:
    global core, _MAX_JSON_BYTES, _CORE_READY, _IMPORT_MS
    if core is not None and _CORE_READY:
        return
    _initialize_integration_context()
    try:
        core._warn_once_per_day_any("core_path", f"[nautical] core loaded: {getattr(core, '__file__', 'unknown')}")
    except Exception:
        pass
    try:
        _MAX_JSON_BYTES = int(getattr(core, "MAX_JSON_BYTES", _MAX_JSON_BYTES))
    except Exception:
        pass
    _apply_core_config()
    if _IMPORT_MS is None:
        _IMPORT_MS = (_ptime.perf_counter() - _IMPORT_T0) * 1000.0
    _CORE_READY = True

def _require_core() -> bool:
    try:
        _load_core()
        return core is not None
    except Exception:
        return False

# ------------------------------------------------------------------------------
# Config-driven toggles (env overrides still supported via core config helpers)
# ------------------------------------------------------------------------------
_CHAIN_COLOR_PER_CHAIN = True
_SHOW_TIMELINE_GAPS = False
_SHOW_ANALYTICS = False
_ANALYTICS_STYLE = "compact"
_ANALYTICS_ONTIME_TOL_SECS = 3600
_CHECK_CHAIN_INTEGRITY = False
_DEBUG_WAIT_SCHED = _DEFAULT_DEBUG_WAIT_SCHED
_RECURRENCE_UPDATE_UDAS: tuple[str, ...] = ()
_MAX_CHAIN_WALK = _MAX_CHAIN_WALK

def _apply_core_config() -> None:
    global _CHAIN_COLOR_PER_CHAIN, _SHOW_TIMELINE_GAPS, _SHOW_ANALYTICS, _ANALYTICS_STYLE
    global _ANALYTICS_ONTIME_TOL_SECS, _CHECK_CHAIN_INTEGRITY
    global _DEBUG_WAIT_SCHED, _RECURRENCE_UPDATE_UDAS, _MAX_CHAIN_WALK
    if core is None:
        return
    _CHAIN_COLOR_PER_CHAIN = core.CHAIN_COLOR_PER_CHAIN
    _SHOW_TIMELINE_GAPS = core.SHOW_TIMELINE_GAPS
    _SHOW_ANALYTICS = core.SHOW_ANALYTICS
    _ANALYTICS_STYLE = core.ANALYTICS_STYLE
    _ANALYTICS_ONTIME_TOL_SECS = core.ANALYTICS_ONTIME_TOL_SECS
    _CHECK_CHAIN_INTEGRITY = core.CHECK_CHAIN_INTEGRITY
    _DEBUG_WAIT_SCHED = core.DEBUG_WAIT_SCHED if hasattr(core, "DEBUG_WAIT_SCHED") else _DEFAULT_DEBUG_WAIT_SCHED
    _RECURRENCE_UPDATE_UDAS = tuple(core.RECURRENCE_UPDATE_UDAS) if hasattr(core, "RECURRENCE_UPDATE_UDAS") else ()
    _MAX_CHAIN_WALK = core.MAX_CHAIN_WALK
# ------------------------------------------------------------------------------
# Small cached helpers for speed + consistency
# ------------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _parse_dt_any_cached(s: str):
    return core.parse_dt_any(s)


@lru_cache(maxsize=512)
def _fmt_dt_local_cached(dt):
    return core.fmt_dt_local(dt)


@lru_cache(maxsize=512)
def _to_local_cached(dt):
    # Accept either datetime or (datetime, meta) tuples from helper parsers.
    if isinstance(dt, (tuple, list)) and dt:
        dt0 = dt[0]
        if isinstance(dt0, datetime):
            dt = dt0
    return core.to_local(dt)


@lru_cache(maxsize=256)
def _validate_anchor_expr_cached(expr: str) -> list[list[dict]]:
    return core.validate_anchor_expr_strict(expr)


@lru_cache(maxsize=256)
def _validate_omit_expr_cached(expr: str) -> list[list[dict]]:
    anchor_omit = _module("anchor_omit")
    return anchor_omit.validate_omit_expr_strict(
        expr,
        validate_anchor_expr_cached=_validate_anchor_expr_cached,
        resolve_omit_presets=core.resolve_omit_presets,
    )


def _load_omit_file_dates(name: str):
    omit_files = core._import_sibling("omit_files")
    return omit_files.load_omit_file_dates(name, getattr(core, "OMIT_FILE_DIR", ""))


def _load_anchor_file_dates(name: str):
    anchor_files = core._import_sibling("anchor_files")
    return anchor_files.load_anchor_file_dates(name, getattr(core, "ANCHOR_FILE_DIR", ""))


def _dtparse(s):
    return _parse_dt_any_cached(s)


def _fmtlocal(dt):
    return _fmt_dt_local_cached(dt)


def _tolocal(dt):
    return _to_local_cached(dt)


def _compare_datetimes(left: datetime, right: datetime) -> int:
    """Compare aware datetimes by instant, preserving DST fold ordering."""
    global _DATETIME_COMPARATOR
    if _DATETIME_COMPARATOR is None:
        _DATETIME_COMPARATOR = core._import_sibling("timeutil").compare_datetimes
    return _DATETIME_COMPARATOR(left, right)


# ------------------------------------------------------------------------------
# Basic IO and panel
# ------------------------------------------------------------------------------
def _fail_and_exit(title: str, msg: str) -> NoReturn:
    _panel(f"❌ {title}", [("Message", msg)], kind="error")
    sys.exit(1)

_RAW_INPUT_TEXT = ""
_PARSED_NEW = None
_PARSED_OLD_OBSERVATION = None
_PARSED_NEW_OBSERVATION = None


def _fail_protocol_error(msg: str) -> NoReturn:
    _fail_and_exit("Protocol error", msg)


def _fail_invalid_input(msg: str) -> None:
    _fail_and_exit("Invalid input", msg)


def _task_uuid_or_empty(task: dict) -> str:
    return _module("modify_protocol").task_uuid_or_empty(task)


def _validate_modify_pair(old: dict, new: dict) -> tuple[dict, dict]:
    protocol = _module("modify_protocol")
    try:
        return protocol.validate_modify_pair(
            old,
            new,
            has_nautical_fields=_task_has_nautical_fields,
        )
    except protocol.ModifyProtocolError as exc:
        _fail_protocol_error(str(exc))


def _validate_single_modify_task(task: dict) -> tuple[dict, dict]:
    protocol = _module("modify_protocol")
    try:
        return protocol.validate_single_modify_task(
            task,
            has_nautical_fields=_task_has_nautical_fields,
        )
    except protocol.ModifyProtocolError as exc:
        _fail_protocol_error(str(exc))


def _decode_leading_json_objects(raw: str, max_objects: int = 2) -> tuple[list[object], int]:
    protocol = _module("modify_protocol")
    try:
        return protocol.decode_leading_json_objects(raw, max_objects=max_objects)
    except protocol.ModifyProtocolError as exc:
        _fail_protocol_error(str(exc))


def _read_two():
    global _RAW_INPUT_TEXT, _PARSED_NEW, _PARSED_OLD_OBSERVATION, _PARSED_NEW_OBSERVATION
    if _EARLY_PROTOCOL_RESULT is not None:
        _RAW_INPUT_TEXT = _EARLY_PROTOCOL_RESULT.raw_text
        _PARSED_NEW = _EARLY_PROTOCOL_RESULT.new
        _PARSED_OLD_OBSERVATION = getattr(_EARLY_PROTOCOL_RESULT, "old_observation", None)
        _PARSED_NEW_OBSERVATION = getattr(_EARLY_PROTOCOL_RESULT, "new_observation", None)
        if not _EARLY_PROTOCOL_RESULT.valid:
            if _EARLY_PROTOCOL_RESULT.error_kind == "protocol":
                _fail_protocol_error(_EARLY_PROTOCOL_RESULT.error)
            _fail_invalid_input(_EARLY_PROTOCOL_RESULT.error)
        request = getattr(_EARLY_PROTOCOL_RESULT, "request", None)
        old = getattr(request, "old", None) or _EARLY_PROTOCOL_RESULT.old
        new = getattr(request, "new", None) or _EARLY_PROTOCOL_RESULT.new
        if isinstance(old, dict) and isinstance(new, dict):
            return old, new
        _fail_invalid_input("on-modify must receive two JSON tasks")

    hook_results = _module("hook_results")
    raw_bytes, raw = hook_results.read_stdin_text(_MAX_JSON_BYTES)
    if len(raw_bytes) > _MAX_JSON_BYTES:
        _fail_invalid_input(f"on-modify input exceeds {_MAX_JSON_BYTES} bytes")
    _RAW_INPUT_TEXT = raw
    if not raw or not raw.strip():
        _fail_invalid_input("on-modify must receive two JSON tasks")

    if _PROTOCOL is not None:
        result = _PROTOCOL.probe_on_modify(raw_bytes, max_bytes=_MAX_JSON_BYTES)
        if not result.valid:
            if result.error_kind == "protocol":
                _fail_protocol_error(result.error)
            _fail_invalid_input(result.error)
        request = getattr(result, "request", None)
        old = getattr(request, "old", None) or result.old
        new = getattr(request, "new", None) or result.new
        if isinstance(old, dict) and isinstance(new, dict):
            _PARSED_OLD_OBSERVATION = getattr(result, "old_observation", None)
            _PARSED_NEW_OBSERVATION = getattr(result, "new_observation", None)
            _PARSED_NEW = new
            return old, new
        _fail_invalid_input("on-modify must receive two JSON tasks")

    objs, idx = _decode_leading_json_objects(raw, max_objects=2)

    if len(objs) == 1 and isinstance(objs[0], list):
        if raw[idx:].strip():
            _fail_protocol_error("Invalid JSON input: trailing content")
        arr = [o for o in objs[0] if isinstance(o, dict)]
        if len(arr) >= 2:
            _PARSED_NEW = arr[-1]
            old, new = _validate_modify_pair(arr[0], arr[-1])
            return old, new
        if len(arr) == 1:
            _PARSED_NEW = arr[0]
            only, _ = _validate_single_modify_task(arr[0])
            return only, only

    objs = [o for o in objs if isinstance(o, dict)]
    if len(objs) >= 2:
        if raw[idx:].strip():
            _fail_protocol_error("Invalid JSON input: trailing content")
        _PARSED_NEW = objs[-1]
        old, new = _validate_modify_pair(objs[0], objs[-1])
        return old, new
    if len(objs) == 1:
        _PARSED_NEW = objs[0]
        only, _ = _validate_single_modify_task(objs[0])
        return only, only

    _fail_invalid_input("on-modify must receive two JSON tasks")


def _apply_description_uda_aliases(old: dict, new: dict) -> None:
    """Expand enabled short UDA directives before on-modify dispatch."""
    if not bool(getattr(core, "ENABLE_UDA_ALIASES", False)):
        return
    description = new.get("description")
    if not isinstance(description, str) or not description:
        return
    try:
        validation = core._import_sibling("hook_validation_pipeline")
        validation.normalize_description_uda_aliases(
            new,
            previous=old,
            enabled=True,
        )
    except ValueError as exc:
        _fail_and_exit("Invalid UDA alias", str(exc))


def _panic_passthrough() -> None:
    hook_results = _module("hook_results")
    hook_results.panic_passthrough(
        _RAW_INPUT_TEXT,
        _PARSED_NEW,
    )


def _task_has_nautical_fields(old: dict, new: dict) -> bool:
    modify_lifecycle = _module("modify_lifecycle")
    return modify_lifecycle.task_has_nautical_fields(old) or modify_lifecycle.task_has_nautical_fields(new)


def _print_task(task):
    hook_results = _module("hook_results")
    if core is None:
        try:
            _load_core()
        except Exception:
            hook_results.emit_passthrough_json(task)
            return
    hook_results.emit_task_json(task, sanitize=True, core=core)




def _panel(
    title,
    rows,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    label_style: str | None = None,
):
    if core is None:
        try:
            _load_core()
        except Exception:
            try:
                sys.stderr.write(f"[nautical] {title}\n")
            except Exception:
                pass
            return
    themes = core.panel_themes()
    theme = dict(themes.get(kind, themes.get("info", {})))
    if border_style:
        theme["border"] = border_style
    if title_style:
        theme["title"] = title_style
    if label_style:
        theme["label"] = label_style
    themes[kind] = theme
    core.render_panel(
        title,
        rows,
        kind=kind,
        panel_mode=core.PANEL_MODE,
        live_duration_ms=getattr(core, "LIVE_PANEL_DURATION_MS", 160),
        live_footer=getattr(core, "LIVE_PANEL_FOOTER", "NAUTICAL"),
        fast_color=core.FAST_COLOR,
        themes=themes,
        allow_line=True,
        line_force_rich_kinds={"summary"},
        label_width_min=6,
        label_width_max=14,
    )


def _panel_line(
    title: str,
    line: str,
    *,
    kind: str = "info",
    border_style: str | None = None,
    title_style: str | None = None,
    markup_body: bool = False,
) -> None:
    core.panel_line(
        title,
        line,
        kind=kind,
        themes=core.panel_themes(),
        border_style=border_style,
        title_style=title_style,
        markup_body=markup_body,
    )


def _text_line(
    line: str,
    *,
    kind: str = "info",
    markup_body: bool = False,
) -> None:
    core.text_line(
        line,
        kind=kind,
        markup_body=markup_body,
    )

def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') else s


def _format_chain_summary_rows(
    rows: list[tuple[str, str]]
) -> list[tuple[str | None, str]]:
    return _module("modify_feedback").format_chain_summary_rows(rows)



def _format_next_anchor_rows(
    rows: list[tuple[str, str]]
) -> list[tuple[str | None, str]]:
    return _module("modify_feedback").format_next_anchor_rows(rows)

def _format_next_cp_rows(
    rows: list[tuple[str, str]]
) -> list[tuple[str | None, str]]:
    return _module("modify_feedback").format_next_cp_rows(rows)




# ------------------------------------------------------------------------------
# Taskwarrior integration
# ------------------------------------------------------------------------------
def _short(u):
    return (u or "")[:8]


def _run_task_result(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
):
    """Execute one on-modify Taskwarrior command through the shared client."""
    from nautical_core.runtime_command import run_task_result

    started = _ptime.perf_counter()
    result = run_task_result(
        cmd,
        env=env,
        input_text=input_text,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        use_tempfiles=use_tempfiles,
        purpose=f"on-modify {_run_task_diag_bucket(cmd)}",
    )
    elapsed = _ptime.perf_counter() - started
    _diag_count("run_task_calls")
    _diag_count("run_task_seconds", elapsed)
    _diag_record_run_task(cmd, ok=result.ok, elapsed=elapsed)
    if not result.ok:
        _diag_count("run_task_failures")
    return result


def _reserve_child_uuid(env: dict) -> str:
    candidate = str(uuid.uuid4())
    while True:
        result = _run_task_result(
            _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{candidate}", "count"],
            env=env,
            timeout=2.5,
            retries=2,
        )
        if result.ok:
            if (result.stdout or "").strip() == "0":
                return candidate
            candidate = str(uuid.uuid4())
            continue
        _diag(f"uuid availability check failed (uuid={candidate[:8]}): {result.stderr.strip()}")
        return candidate


def _stable_child_uuid(parent_task: dict | None, child_task: dict | None) -> str:
    modify_spawn_prep = _module("modify_spawn_prep")
    return modify_spawn_prep.stable_child_uuid(
        parent_task,
        child_task,
        task_uuid_or_empty=_task_uuid_or_empty,
        coerce_int=core.coerce_int,
        stable_child_uuid_namespace=_STABLE_CHILD_UUID_NAMESPACE,
    )


def _child_uuid_for_spawn(parent_task: dict | None, child_task: dict | None, env: dict) -> str:
    modify_spawn_prep = _module("modify_spawn_prep")
    return modify_spawn_prep.child_uuid_for_spawn(
        parent_task,
        child_task,
        env,
        stable_child_uuid=_stable_child_uuid,
        reserve_child_uuid=_reserve_child_uuid,
    )


def _sanitize_unknown_attrs(stderr: str, payload: dict) -> set[str]:
    return _module("modify_spawn_prep").sanitize_unknown_attrs(stderr, payload)


def _format_line_preview(
    link_no: int,
    task: dict,
    child_due_utc: datetime,
    child_short: str,
    now_utc: datetime,
    child_field: str = "due",
    cap_no: int | None = None,
    until_dt: datetime | None = None,
    until_no: int | None = None,
    child_until_dt: datetime | None = None,
    kind: str = "cp",
    minimal: bool = False,
) -> str:
    task = _module("modify_models").TaskView.from_mapping(task)
    return _module("modify_feedback").format_line_preview(
        link_no,
        task,
        child_due_utc,
        child_short,
        now_utc,
        child_field=child_field,
        cap_no=cap_no,
        until_dt=until_dt,
        until_no=until_no,
        child_until_dt=child_until_dt,
        kind=kind,
        minimal=minimal,
        core=core,
        format_local=_fmtlocal,
        on_time_delta=_fmt_on_time_delta,
        human_delta=_human_delta,
    )


# Helper to categorize subprocess failures
def _categorize_spawn_error(returncode: int, stderr: str) -> tuple[str, bool]:
    return _module("modify_spawn_prep").categorize_spawn_error(returncode, stderr)


def _enqueue_spawn_intent(plan) -> tuple[bool, str]:
    """Stage one immutable lifecycle plan through the shared application service."""
    if _INTEGRATION_CONTEXT is None:
        return False, "validated integration context is unavailable"
    lifecycle_models = _module("lifecycle_models")
    if not isinstance(plan, lifecycle_models.LifecyclePlan):
        return False, "invalid lifecycle plan"
    configuration = _INTEGRATION_CONTEXT.configuration
    lifecycle_application = _module("lifecycle_application")
    outbox = _module("lifecycle_outbox").LifecycleOutboxRepository(TW_DATA_DIR)
    # Staging-only: no unit_of_work/mutations here by design. On-modify must
    # not construct a command-capable unit of work, to avoid re-entering
    # Taskwarrior while it still holds the datastore lock for this task.
    service = lifecycle_application.LifecycleApplicationService(outbox=outbox, owner="on-modify")
    result = service.stage(
        plan,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    if result.ok:
        return True, ""
    return False, result.reason or "lifecycle outbox staging failed"


def _lifecycle_spawn_identity(parent: dict, child: dict):
    """Build one retry-stable transition identity for a child slot."""
    lifecycle_models = _module("lifecycle_models")
    chain_id = str(parent.get("chainID") or "").strip()
    parent_uuid = str(parent.get("uuid") or "").strip()
    try:
        source_link = int(parent.get("link"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("lifecycle transition requires a numeric parent link") from exc
    try:
        target_link = int(child.get("link") or (source_link + 1))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("lifecycle transition requires a numeric child link") from exc
    status = str(parent.get("status") or "").strip().lower()
    event = (
        lifecycle_models.LifecycleEvent.EXPIRE
        if status == "deleted"
        else lifecycle_models.LifecycleEvent.COMPLETE
    )
    identity = lifecycle_models.LifecycleIdentity(
        chain_id=chain_id,
        parent_uuid=parent_uuid,
        source_link=source_link,
        target_link=target_link,
        event=event,
    )
    return identity


def _spawn_child_atomic(
    child_task,
    parent_task_with_nextlink: dict,
    *,
    lifecycle_plan=None,
) -> tuple[str, set[str], bool, bool, str | None, str | None]:
    modify_spawn = _module("modify_spawn")
    if hasattr(child_task, "to_mapping"):
        child_task = child_task.to_mapping()
    return modify_spawn.spawn_child_atomic(
        child_task,
        parent_task_with_nextlink,
        lifecycle_plan=lifecycle_plan,
        services=modify_spawn.SpawnServices(
            prepare_spawn_child_payload=_module("modify_spawn_prep").prepare_spawn_child_payload,
            child_uuid_for_spawn=_child_uuid_for_spawn,
            fmt_isoz=core.fmt_isoz,
            now_utc=core.now_utc,
            lifecycle_models=_module("lifecycle_models"),
            lifecycle_spawn_identity=_lifecycle_spawn_identity,
            enqueue_spawn_intent=_enqueue_spawn_intent,
            parse_datetime=getattr(core, "parse_dt_any", None),
            diag_count=_diag_count,
        ),
    )



def _root_uuid_from(task: dict) -> str:
    """Return the stable chain seed.

    ChainID is the only source of truth.
    """
    return (task.get("chainID") or "").strip()

# --- Chain export: chainID is mandatory --------------------------------------
def _task(args, env=None) -> str:
    """
    Thin wrapper around 'task' returning stdout as text.
    Always disables hooks; caller should provide rc.json.array flag when needed.
    """
    cache_key = None
    if env is None and _task_args_cacheable(args):
        try:
            cache_key = tuple(str(a) for a in args)
        except Exception:
            cache_key = None
        if cache_key is not None:
            cached = _query_ctx_get("task_text", cache_key)
            if isinstance(cached, str):
                _diag_count("task_text_cache_hits")
                return cached
            _diag_count("task_text_cache_misses")
    modify_queries = _module("modify_queries")
    out = modify_queries.task_text(
        args,
        run_task=_run_task_result,
        task_cmd_prefix=_task_cmd_prefix(),
        env=(env or os.environ.copy()),
        timeout=3.0,
        retries=2,
        diag=_diag,
    )
    if cache_key is not None:
        _query_ctx_set("task_text", cache_key, out or "")
    return out

def tw_export_chain_required(seed_task, env=None):
    """Return full chain export for a task.

    Policy: chainID is mandatory.
    """
    chain_id = seed_task.get('chainID')
    if not chain_id:
        raise RuntimeError(
            "ChainID is required (legacy chain traversal removed). "
            "Run dev_tools/nautical_backfill_chainid.py, then retry."
        )
    if env is not None:
        raise RuntimeError("chain reads must use the invocation Taskwarrior repository")
    rows = _lifecycle_read_service().get_chain_export(chain_id)
    if rows is None:
        raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
    return rows
def _tw_get_cached(ref: str) -> str:
    """Return `task _get <ref>` stdout stripped. Cached within one hook run."""
    try:
        if ref.endswith(".entry"):
            short = ref[:-6].strip()
            cached, cache_chain_id = (
                _lifecycle_read_service().lookup_short(short) if short else (None, "")
            )
            if short and hasattr(cached, "get"):
                _diag_count("tw_get_cache_hits")
                return (str(cached.get("entry") or "")).strip()
            if short and cache_chain_id:
                _diag_count("unexpected_cache_misses")
                _diag(f"cache miss: _get {ref} (chainID={cache_chain_id})")
        cached = _query_ctx_get("tw_get", ref)
        if isinstance(cached, str):
            _diag_count("tw_get_cache_hits")
            return cached
        _diag_count("tw_get_cache_misses")
        modify_queries = _module("modify_queries")
        out = modify_queries.tw_get(
            ref,
            task_text=lambda args: _task(args, env=None),
        )
        _query_ctx_set("tw_get", ref, out or "")
        return out
    except Exception:
        return ""

def _chain_root_and_age(task: dict, now_utc: datetime) -> tuple[str, int | None]:
    """Get chain root (chainID) and age in days.
    Returns (root_short, age_days). age_days is None if unavailable."""
    try:
        cache_key = (_root_uuid_from(task), str(_tolocal(now_utc).date()))
    except Exception:
        cache_key = None
    if cache_key is not None:
        cached = _query_ctx_get("chain_root_age", cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            _diag_count("chain_root_age_cache_hits")
            return cached
        _diag_count("chain_root_age_cache_misses")
    modify_queries = _module("modify_queries")
    result = modify_queries.chain_root_and_age(
        task,
        now_utc,
        root_uuid_from=_root_uuid_from,
        tw_get_cached=_tw_get_cached,
        dtparse=_dtparse,
        tolocal=_tolocal,
    )
    if cache_key is not None:
        _query_ctx_set("chain_root_age", cache_key, result)
    return result

def _format_root_and_age(task: dict, now_utc: datetime) -> str:
    """Format root and age as a single string.
    Returns root (age) or just root if age is 0 or unavailable."""
    try:
        cache_key = (_root_uuid_from(task), str(_tolocal(now_utc).date()))
    except Exception:
        cache_key = None
    if cache_key is not None:
        cached = _query_ctx_get("format_root_age", cache_key)
        if isinstance(cached, str):
            _diag_count("format_root_age_cache_hits")
            return cached
        _diag_count("format_root_age_cache_misses")
    modify_queries = _module("modify_queries")
    result = modify_queries.format_root_and_age(
        task,
        now_utc,
        chain_root_and_age=_chain_root_and_age,
    )
    if cache_key is not None:
        _query_ctx_set("format_root_age", cache_key, result)
    return result

# ------------------------------------------------------------------------------
# On modify-without-completion helpers
# ------------------------------------------------------------------------------


def _canon_for_compare(v):
    """Canonicalize values so 5 == 5.0, strings are trimmed, and
    dict/list comparisons are stable."""
    from decimal import Decimal, InvalidOperation
    if v is None:
        return None
    # Booleans/numbers
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return Decimal(str(v))  # 5 and 5.0 normalize equal
    # Strings that might be numeric
    if isinstance(v, str):
        s = v.strip()
        try:
            return Decimal(s)  # if numeric string, compare numerically
        except (InvalidOperation, ValueError):
            return s  # non-numeric string
    # Collections → stable JSON
    try:
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    except Exception:
        return str(v)

def _field_changed(old: dict, new: dict, key: str) -> bool:
    ov = old.get(key)
    nv = new.get(key)
    return _canon_for_compare(ov) != _canon_for_compare(nv)





def _validate_shared_anchor_on_modify(expr: str) -> None:
    core._import_sibling("hook_validation_pipeline").validate_anchor_expression(
        expr,
        parse_anchor_expr=core.parse_anchor_expr_to_dnf,
        validate_anchor_expr=_validate_anchor_expr_cached,
    )


def _validate_shared_omit_on_modify(expr: str) -> None:
    core._import_sibling("hook_validation_pipeline").validate_omit_expression(
        expr,
        validate_omit_expr=_validate_omit_expr_cached,
    )


def _validate_cp_on_modify(cp_str: str, chain_max_val, chain_until_val):
    add_validation = core._import_sibling("add_validation")
    return _module("modify_validation").validate_cp_on_modify(
        cp_str,
        chain_max_val,
        chain_until_val,
        parse_cp_sequence=core.parse_cp_sequence,
        cp_sequence_parse_error=core.cp_sequence_parse_error,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=core.parse_dt_any,
    )


def _validate_chain_limits_on_modify(task: dict) -> None:
    add_validation = core._import_sibling("add_validation")
    pipeline = core._import_sibling("hook_validation_pipeline")
    cpmax, _until_dt, findings = pipeline.validate_recurrence_limits(
        task.get("cp"), task.get("chainMax"), task.get("chainUntil"),
        parse_cp_sequence=core.parse_cp_sequence,
        cp_sequence_parse_error=core.cp_sequence_parse_error,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=core.parse_dt_any,
    )
    if findings:
        finding = findings[0]
        _fail_and_exit(f"Invalid {finding.field}", finding.reason)
    if cpmax is not None:
        task["chainMax"] = cpmax
    return _module("modify_validation").validate_chain_limits_on_modify(
        task,
        parse_chain_max=add_validation.parse_chain_max,
        parse_datetime=core.parse_dt_any,
        validate_until_not_past=_validate_until_not_past,
        now_utc=core.now_utc,
        fail=_fail_and_exit,
    )


def _validate_native_until_after_target_or_fail(task: dict) -> None:
    add_validation = core._import_sibling("add_validation")
    _module("modify_validation").validate_native_until_after_target_or_fail(
        task,
        validate_anchor_mode=add_validation.validate_native_until_anchor_mode,
        safe_parse_datetime=_safe_parse_datetime,
        validate_after_target=add_validation.validate_native_until_after_target,
        format_local=core.fmt_dt_local,
        panel=_panel,
        fail=_fail_and_exit,
        abort=sys.exit,
    )


def _validate_native_until_anchor_slots_or_fail(task: dict) -> None:
    add_validation = core._import_sibling("add_validation")
    astronomy = core._import_sibling("astronomy")
    native_until = core._import_sibling("native_until")
    recurrence_context = core._import_sibling("recurrence_context").RecurrenceContext
    until_dt, until_err = _safe_parse_datetime(task.get("until"))
    if until_err or until_dt is None:
        return
    anchor = str(task.get("anchor") or "").strip()
    dnf = _validate_anchor_expr_cached(anchor) if anchor else None
    try:
        valid, reason, slots = core._import_sibling("astronomy_validation").validate_native_until_slots(
            until_dt=until_dt,
            target_dt=core.parse_dt_any(task.get("due") or task.get("scheduled")),
            dnf=dnf,
            anchor_file_value=str(task.get("anchor_file") or "").strip(),
            fallback_hhmm=(0, 0),
            collect_time_slots=add_validation.collect_anchor_time_slots,
            normalize_time_slots=_norm_hhmm_list,
            resolve_time_slots=lambda value, target_date: _norm_hhmm_list(value, target_date),
            anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
            recurrence_context=recurrence_context.from_task(task),
            to_local=_tolocal,
            validate_time_slots=native_until.validate_calendar_slots,
        )
    except Exception as exc:
        if astronomy.is_astronomy_error(exc):
            _panel("❌ Invalid astronomy time", [("Required", astronomy.scheduling_error_message(exc))], kind="error")
            sys.exit(1)
        return
    if valid:
        return
    _panel(
        "❌ Invalid expiration window",
        [("Expires", core.fmt_dt_local(until_dt)),
         ("Anchor slots", ", ".join(f"{hh:02d}:{mm:02d}" for hh, mm in slots) or "none"),
         ("Required", reason or "calendar expiration must be later than every anchor slot")],
        kind="error",
    )
    sys.exit(1)


# ------------------------------------------------------------------------------
# Pretty helpers
# ------------------------------------------------------------------------------
def _chain_colour_for_task(task: dict, kind: str) -> str:
    """
    Get the chain colour for this task (uses root uuid, cached).
    """
    root = _root_uuid_from(task)
    return core.chain_colour_root(kind, root)


def _future_style_for_chain(task: dict, kind: str) -> str:
    """
    Style for FUTURE links in the timeline.
    - When per-chain mode is OFF → static colour per kind (fast path)
    - When per-chain mode is ON  → cached colour per chain root
    """
    if not _CHAIN_COLOR_PER_CHAIN:
        # Static behaviour
        return "dark_orange" if kind == "cp" else "cyan"

    return _chain_colour_for_task(task, kind)




def _fmt_on_time_delta(due_dt, end_dt, tol_secs: int = 60):
    if not (due_dt and end_dt):
        return ""
    diff = (end_dt - due_dt).total_seconds()
    if diff > tol_secs:
        human = core.humanize_delta(due_dt, end_dt, use_months_days=False)
        return f"[yellow](+{human.replace('overdue by ','').replace('in ','')} late)[/]"
    if diff < -tol_secs:
        human = core.humanize_delta(end_dt, due_dt, use_months_days=False)
        return f"[cyan](-{human.replace('in ','')} early)[/]"
    return "[green](on time)[/]"


def _collect_prev_two(current_task: dict, chain_by_link=None):
    from nautical_core.integration_models import Absent, Found, Unavailable

    service = _lifecycle_read_service()
    read = service.collect_prev_two(
        current_task,
        get_chain_read=lambda chain_id: service.get_chain_read(chain_id),
        panel_chain_by_link=_modify_chain_state().panel_chain_by_link,
        panel_chain_snapshot_loaded=_modify_chain_state().panel_chain_snapshot_loaded,
        chain_by_link=chain_by_link,
    )
    if isinstance(read, Unavailable):
        raise RuntimeError(read.evidence.detail or "lifecycle predecessor read unavailable")
    if isinstance(read, Absent):
        return []
    if not isinstance(read, Found):
        raise RuntimeError("lifecycle predecessor read returned an invalid result")
    return list(read.value)


def _cached_chain_token_match(task, token: str) -> bool:
    if not hasattr(task, "get") or not isinstance(token, str) or not token:
        return False
    if token.startswith("+"):
        want = token[1:].strip().lower()
        tags = task.get("tags")
        if isinstance(tags, (list, tuple, set)):
            return want in {str(t).strip().lower() for t in tags}
        return False
    if ":" not in token:
        return False
    key, value = token.split(":", 1)
    negate = False
    if key.endswith(".not"):
        negate = True
        key = key[:-4]
    actual = task.get(key)
    matched = False
    if key in {"link", "id"}:
        matched = str(core.coerce_int(actual, None) if actual is not None else "") == value
    else:
        matched = str(actual or "").strip().lower() == value.strip().lower()
    return (not matched) if negate else matched


def _lifecycle_read_service():
    """Build the focused chain-read service for this hook invocation."""
    state = _modify_chain_state()
    existing = getattr(state, "lifecycle_read_service", None)
    if existing is not None:
        repository = getattr(_modify_runtime_state(), "task_repository", None)
        if repository is not None:
            bind_repository = getattr(existing, "bind_repository", None)
            if callable(bind_repository):
                bind_repository(repository)
        return existing
    lifecycle_read_service = _module("lifecycle_read_service")
    if getattr(state, "chain_cache_store", None) is None:
        state.chain_cache_store = lifecycle_read_service.ChainCacheStore()
    repository = getattr(_modify_runtime_state(), "task_repository", None)

    service = lifecycle_read_service.LifecycleReadService(
        coerce_int=core.coerce_int,
        parse_extra_tokens=_parse_extra_tokens,
        token_matcher=_cached_chain_token_match,
        read_query_get=_read_query_get,
        chain_cache_get=lambda _chain_id: None,
        repository=repository,
        max_chain_walk=_MAX_CHAIN_WALK,
        diag=_diag,
        record_stat=_record_chain_snapshot_stat,
        cache_store=state.chain_cache_store,
        read_query_missing=_READ_QUERY_MISSING,
    )
    state.lifecycle_read_service = service
    return service


def _seed_runtime_lookup_task(task: dict | None, *, lookup_short: str | None = None) -> dict | None:
    if not isinstance(task, dict):
        return None
    uuid_str = str(task.get("uuid") or "").strip()
    if not uuid_str:
        return None
    short = uuid_str[:8]
    service = _lifecycle_read_service()
    task_codec = _module("task_codec")
    observation = task_codec.DEFAULT_TASK_CODEC.decode_row(
        task,
        source_query="on-modify lookup seed",
    )
    task_obj = service.seed_lookup_task(observation, short_uuid=short)
    requested_short = str(lookup_short or "").strip()
    if requested_short and requested_short != short:
        task_obj = service.seed_lookup_task(task_obj, short_uuid=requested_short)
    entry = task_obj.get("entry")
    if short and entry:
        _query_ctx_set("tw_get", f"{short}.entry", str(entry).strip())
    return task_obj.to_mapping()


def _seed_runtime_lookup_tasks(*tasks: dict | None) -> None:
    for task in tasks:
        _seed_runtime_lookup_task(task)


# ------------------------------------------------------------------------------
# Multi-time occurrence helpers (hook-level)
# ------------------------------------------------------------------------------

def _recurrence_seed_base(task: dict) -> str:
    """Resolve the task recurrence identity at the hook input boundary."""
    return str(task.get("chainID") or task.get("uuid") or "preview").strip()

def _norm_hhmm_list(v, target_date=None) -> list[tuple[int, int]]:
    """Normalize various core representations of @t into a sorted list of (hh, mm)."""
    if v is None:
        return []
    time_slots = core._import_sibling("time_slots")
    return time_slots.resolve_time_slots(
        v,
        target_date,
        config=getattr(core, "ASTRONOMY_CONFIG", {}),
        to_local=core.to_local,
    )


def _next_occurrence_after_local_dt(
    dnf,
    after_local_dt: datetime,
    default_seed_date,
    seed_base: str,
    omit_dnf=None,
    fallback_hhmm: tuple[int, int] | None = None,
):
    """Return the next occurrence using the shared add-side scheduler.

    Keep this adapter while completion/evaluator callers migrate to the
    evaluator-owned service.  The stable callback signature avoids changing
    the completion orchestration in the same pass.
    """
    if not dnf:
        return None
    scheduler = _module("add_anchor_compute")
    return scheduler.anchor_next_occurrence_after_local_dt(
        dnf,
        after_local_dt,
        fallback_hhmm=fallback_hhmm or (0, 0),
        interval_seed=default_seed_date,
        seed_base=seed_base,
        omit_dnf=omit_dnf,
        default_seed_date=default_seed_date,
        core=core,
    )


def _human_delta(a, b, prefer_months=True):
    try:
        return core.humanize_delta(a, b, use_months_days=bool(prefer_months))
    except TypeError:
        return core.humanize_delta(a, b)


def _cp_add_td(dt: datetime, td: timedelta) -> datetime:
    secs = int(td.total_seconds())
    if secs % 86400 == 0:
        dl = _tolocal(dt)
        return core.build_local_datetime(
            (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
        ).astimezone(timezone.utc)
    return (dt + td).replace(microsecond=0)


def _cp_sequence_period_for_link(
    tokens: list[dict],
    cp_str: str,
    link_no: int,
    chain_id: str | None = None,
) -> timedelta:
    idx = (max(1, int(link_no)) - 1) % len(tokens)
    td = core.cp_sequence_interval_for_token(
        tokens[idx],
        cp=cp_str,
        link_no=link_no,
        token_index=idx,
        chain_id=chain_id,
    )
    return td or timedelta()


# ------------------------------------------------------------------------------
# Due calculators
# ------------------------------------------------------------------------------


def _chain_generation_service():
    """Return the task-scoped shared chain-generation service."""
    state = _modify_runtime_state()
    generation_module = _module("chain_generation")
    configured = tuple(_RECURRENCE_UPDATE_UDAS or ())
    service = state.chain_generation_service
    if (
        service is None
        or getattr(service, "core", None) is not core
        or tuple(getattr(service, "recurrence_update_udas", ())) != configured
    ):
        service = generation_module.ChainGenerationService.from_core(
            core,
            recurrence_update_udas=configured,
            debug_wait_sched=_DEBUG_WAIT_SCHED,
            wait_sched_debug=_LAST_WAIT_SCHED_DEBUG,
        )
        state.chain_generation_service = service
    return service


def _safe_parse_datetime(dt_str: str) -> tuple[datetime | None, str | None]:
    """
    Parse datetime safely.
    Returns (datetime, error_msg).
    error_msg is None on success, or a user-friendly explanation on failure.
    """
    if not (dt_str or "").strip():
        return (None, None)

    try:
        dt = core.parse_dt_any(dt_str)
        if dt is None:
            return (None, f"Unrecognized datetime format '{dt_str}'")
        return (dt, None)
    except ValueError as e:
        _diag(f"datetime parse value error: {e}")
        return (None, "DateTime parsing error")
    except TypeError as e:
        _diag(f"datetime parse type error: {e}")
        return (None, "DateTime type error")
    except Exception as e:
        _diag(f"datetime parse unexpected error: {e}")
        return (None, "Unexpected error parsing datetime")


def _validate_anchor_mode(mode_str: str) -> tuple[str, str | None]:
    """
    Validate and normalize anchor_mode. Returns (normalized_mode, error_msg).
    """
    raw = (mode_str or "").strip()
    if not raw:
        return ("", None)
    mode = raw.lower()
    if mode not in ("skip", "all", "flex"):
        return (
            "skip",
            f"anchor_mode must be 'skip', 'all', or 'flex' (got '{raw}'). Defaulting to 'skip'.",
        )
    return (mode, None)


def _omit_dnf_from_parent(parent: dict):
    expr_str = (parent.get("omit") or "").strip()
    omit_file = (parent.get("omit_file") or "").strip()
    omit_dnf = None
    omit_dates: frozenset[Any] = frozenset()
    omit_descriptions: dict[Any, str] = {}
    if expr_str:
        try:
            omit_dnf = _validate_omit_expr_cached(expr_str)
        except Exception as e:
            raise ValueError(f"Invalid omit expression '{expr_str}': {str(e)}")
    if omit_file:
        try:
            omit_files = core._import_sibling("omit_files")
            omit_dates, omit_descriptions = omit_files.load_omit_file_data(
                omit_file,
                getattr(core, "OMIT_FILE_DIR", ""),
            )
        except Exception as e:
            raise ValueError(f"Invalid omit_file '{omit_file}': {str(e)}")
    if not omit_dnf and not omit_dates and not omit_descriptions:
        return "", None
    anchor_omit = _module("anchor_omit")
    return expr_str, anchor_omit.combine_omit_state(
        omit_dnf=omit_dnf,
        omit_dates=omit_dates,
        omit_descriptions=omit_descriptions,
    )


def _recurrence_evaluator_for_task(task: dict):
    """Build the task-scoped evaluator used by completion projections."""
    return _scheduler_service_for_task(task).session.evaluator


def _scheduler_service_for_task(task: dict):
    return _module("modify_runtime").scheduler_service_for_task(
        task,
        state=_modify_runtime_state(),
        core=core,
        recurrence_seed_base=_recurrence_seed_base,
    )


def _anchor_included_occurrences(
    parent: dict,
    *,
    after_local_dt: datetime,
    inclusive: bool,
    limit: int,
    fallback_hhmm: tuple[int, int],
    omit_dnf,
    seed_base: str,
    default_seed_date,
    dnf,
    anchor_file_provider: Any | None = None,
) -> list[datetime]:
    service = _scheduler_service_for_task(parent)
    return service.included_occurrences_after(
        after_local_dt,
        inclusive=inclusive,
        limit=limit,
    )


def _estimate_cp_final_by_max(task: dict, next_due_utc):
    return _module("modify_completion_compute").estimate_cp_final_by_max(
        task,
        next_due_utc,
        coerce_int=core.coerce_int,
        parse_cp_sequence_tokens=core.parse_cp_sequence_tokens,
        sequence_period_for_link=_cp_sequence_period_for_link,
        add_period=_cp_add_td,
        max_iterations=_MAX_ITERATIONS,
        diagnostic=_diag,
    )


def _estimate_anchor_final_by_max(task: dict, next_due_utc, dnf):
    return _module("modify_completion_compute").estimate_anchor_final_by_max(
        task,
        next_due_utc,
        dnf,
        coerce_int=core.coerce_int,
        recurrence_seed_base=_recurrence_seed_base,
        to_local_cached=_to_local_cached,
        safe_parse_datetime=_safe_parse_datetime,
        anchor_file_fallback_hhmm=_anchor_file_fallback_hhmm,
        omit_dnf_from_parent=_omit_dnf_from_parent,
        recurrence_evaluator_for_task=_recurrence_evaluator_for_task,
        anchor_file_provider_for=_anchor_file_provider_for,
        anchor_included_occurrences=_anchor_included_occurrences,
        diagnostic=_diag,
        max_iterations=_MAX_ITERATIONS,
    )


# Helper to validate chainUntil is in the future
def _validate_until_not_past(
    until_dt: datetime, now_utc: datetime
) -> tuple[bool, str | None]:
    """
    Check if chainUntil is in the past.
    Returns (is_valid, error_msg).
    """
    if not until_dt:
        return (True, None)

    # Allow small grace period (1 minute) for race conditions
    grace = timedelta(minutes=1)
    if _compare_datetimes(until_dt, now_utc - grace) < 0:
        past_by = now_utc - until_dt
        past_s = core.humanize_delta(until_dt, now_utc, use_months_days=False)
        return (False, f"chainUntil is in the past (was {past_s} ago)")

    return (True, None)


# ------------------------------------------------------------------------------
# Child build (copy almost everything; override minimal set)
# ------------------------------------------------------------------------------
def _validate_chain_duration_reasonable(
    child_due: datetime, until_dt: datetime, now_utc: datetime
) -> tuple[bool, str | None]:
    if not until_dt:
        return (True, None)
    days = (until_dt - now_utc).days
    if days > _MIN_FUTURE_WARN:
        years = days / 365.25
        return (True, f"Chain extends {years:.1f} years into future (until {core.fmt_dt_local(until_dt)})")
    return (True, None)


_RESERVED_DROP = {
    "id",
    "uuid",
    "urgency",
    "status",
    "modified",
    "start",
    "end",
    "mask",
    "imask",
    "parent",
    "recur",
    "rc",
    "nextLink",  # set on parent, not copied
}

_RESERVED_OVERRIDE = {"due", "entry", "status", "chain", "prevLink", "link"}


# ------------------------------------------------------------------------------
# wait/scheduled carry-forward (relative to due)
# ------------------------------------------------------------------------------
def _utc_to_local_naive(dt_utc: datetime) -> datetime:
    """UTC -> local naive (wall-clock)."""
    if not isinstance(dt_utc, datetime):
        raise TypeError("dt_utc must be datetime")
    return core.utc_to_local_naive(dt_utc)


def _local_naive_to_utc(dt_local_naive: datetime) -> datetime:
    """Local naive (wall-clock) -> UTC using the shared DST policy."""
    if not isinstance(dt_local_naive, datetime):
        raise TypeError("dt_local_naive must be datetime")
    naive = dt_local_naive.replace(microsecond=0)
    return core.local_naive_to_utc(naive)


def _recurrence_anchor_field(task: dict | None) -> str:
    if isinstance(task, dict):
        if task.get("due"):
            return "due"
        if task.get("scheduled"):
            return "scheduled"
    return "due"


# ------------------------------------------------------------------------------
# End-of-chain summary + stats
# ------------------------------------------------------------------------------
def _lateness_stats(chain: list[dict], tol_secs: int = 60) -> dict:
    return _module("modify_analytics").lateness_stats(
        chain,
        parse_datetime=_dtparse,
        tol_secs=tol_secs,
    )


def _sort_chain_for_analytics(chain: list[dict]) -> list[dict]:
    analytics = _module("modify_analytics")
    return analytics.sort_chain_for_analytics(
        chain,
        coerce_int=core.coerce_int,
        parse_datetime=_dtparse,
    )


def _chain_health_advice(
    chain: list[dict],
    kind: str,
    task: dict,
    tol_secs: int = _ANALYTICS_ONTIME_TOL_SECS,
    style: str = _ANALYTICS_STYLE,
) -> str | None:
    return _module("modify_analytics").chain_health_advice(
        chain,
        kind,
        task,
        core=core,
        parse_datetime=_dtparse,
        format_delta=_fmt_td_dd_hhmm,
        coerce_int=core.coerce_int,
        tol_secs=tol_secs,
        style=style,
    )


def _chain_integrity_warnings(chain: list[dict], expected_chain_id: str | None = None) -> list[str]:
    if core is None:
        _load_core()
    return _module("modify_analytics").chain_integrity_warnings(
        chain,
        expected_chain_id=expected_chain_id,
        coerce_int=core.coerce_int,
        short=_short,
    )


def _fmt_secs_delta(now_ref, secs: float | None) -> str:
    if secs is None:
        return "—"
    base = datetime(2000, 1, 1, tzinfo=timezone.utc)
    tgt = base + timedelta(seconds=secs)
    s = (
        core.humanize_delta(base, tgt, use_months_days=False)
        .replace("in ", "")
        .replace("overdue by ", "")
    )
    if secs > 0:
        return f"[yellow]+{s}[/]"
    if secs < 0:
        return f"[cyan]-{s}[/]"
    return "[green]±0[/]"


def _last_n_timeline(chain: list[dict], n: int = 6) -> list[str]:
    return _module("modify_chain_summary").last_n_timeline(
        chain,
        n,
        coerce_int=core.coerce_int,
        parse_datetime=_dtparse,
        format_local=_fmtlocal,
        format_on_time_delta=_fmt_on_time_delta,
        short_uuid=_short,
    )

def _end_summary_current(current: dict, current_task: dict | None) -> dict:
    return _module("modify_chain_summary").summary_current(current, current_task)


def _end_summary_chain_id_row(actual_current: dict) -> str:
    return _module("modify_chain_summary").summary_chain_id(actual_current)


def _end_summary_sorted_chain(chain_id: str, actual_current: dict) -> list:
    chain = tw_export_chain_required(actual_current)
    if actual_current and chain:
        for i, task in enumerate(chain):
            if task.get("uuid") == actual_current.get("uuid"):
                chain[i] = _module("task_models").TaskObservation.from_mapping(
                    actual_current, source_query=f"chain:{chain_id}:current"
                )
                break
    try:
        chain = _sort_chain_for_analytics(chain)
    except Exception:
        pass
    return chain


def _end_summary_span_fields(
    chain_id: str,
    chain: list[dict],
    *,
    stop_at=None,
    stopped_by_delete: bool = False,
) -> tuple[datetime | None, datetime | None, str]:
    return _module("modify_chain_summary").span_fields(
        chain_id,
        chain,
        stop_at=stop_at,
        stopped_by_delete=stopped_by_delete,
        export_endpoint=_export_chain_endpoint,
        parse_datetime=_dtparse,
        human_delta=_human_delta,
    )


def _end_summary_kind_rows(rows: list[tuple[str, str]], kind: str, current: dict) -> None:
    summary = _module("modify_chain_summary")
    summary.kind_rows(
        rows,
        kind,
        current,
        anchor_preset_display=core.anchor_preset_display,
        validate_anchor=_validate_anchor_expr_cached,
        describe_anchor=core.describe_anchor_dnf,
    )


def _end_summary_stats_rows(rows: list[tuple[str, str]], chain: list[dict], now_utc) -> None:
    _module("modify_chain_summary").stats_rows(
        rows,
        chain,
        now_utc,
        lateness_stats=_lateness_stats,
        format_seconds_delta=_fmt_secs_delta,
    )


def _end_summary_limits_row(rows: list[tuple[str, str]], current: dict) -> None:
    _module("modify_chain_summary").limits_row(
        rows,
        current,
        coerce_int=core.coerce_int,
        parse_datetime=_dtparse,
        format_local=core.fmt_dt_local,
    )


def _end_chain_summary(current: dict, reason: str, now_utc, current_task: dict = None) -> None:
    summary = _module("modify_chain_summary")
    summary.render_chain_summary(
        current,
        reason,
        now_utc,
        current_task,
        export_sorted_chain=_end_summary_sorted_chain,
        root_uuid_from=_root_uuid_from,
        short_uuid=_short,
        format_root_and_age=_format_root_and_age,
        kind_rows=_end_summary_kind_rows,
        span_fields=_end_summary_span_fields,
        stats_rows=_end_summary_stats_rows,
        limits_row=_end_summary_limits_row,
        last_n_timeline_rows=_last_n_timeline,
        format_rows=_format_chain_summary_rows,
        coerce_int=core.coerce_int,
        format_local=core.fmt_dt_local,
        max_chain_walk=_MAX_CHAIN_WALK,
        panel=_panel,
        diagnostic=_diag,
    )



# ------------------------------------------------------------------------------
# Timeline (capped) — no dependency on core.next_anchor_after
# ------------------------------------------------------------------------------

def _timeline_lines(
    kind: str,
    task: dict,
    child_due_utc,
    child_short: str,
    dnf,
    next_count: int = 3,
    cap_no: int | None = None,
    cur_no: int | None = None,
    show_gaps: bool = True,
    round_anchor_gaps: bool = True,
) -> list[str]:
    """Compatibility adapter for task-scoped timeline presentation."""
    if not _require_core():
        return []
    return _module("modify_timeline").timeline_lines_for_task(
        kind,
        task,
        child_due_utc,
        child_short,
        dnf,
        next_count=next_count,
        cap_no=cap_no,
        cur_no=cur_no,
        show_gaps=show_gaps,
        round_anchor_gaps=round_anchor_gaps,
        core=core,
        max_iterations=_MAX_ITERATIONS,
        future_style_for_chain=_future_style_for_chain,
        collect_prev_two=_collect_prev_two,
        dtparse=_dtparse,
        fmt_on_time_delta=_fmt_on_time_delta,
        fmtlocal=_fmtlocal,
        short=_short,
        tolocal=_tolocal,
        next_occurrence_after_local_dt=_next_occurrence_after_local_dt,
        to_local_cached=_to_local_cached,
        safe_parse_datetime=_safe_parse_datetime,
        format_gap=_module("modify_timeline").format_gap,
        module_loader=_module,
        omit_dnf_from_parent=_omit_dnf_from_parent,
        recurrence_evaluator_for_task=_recurrence_evaluator_for_task,
        scheduler_service_for_task=_scheduler_service_for_task,
    )

def _got_anchor_invalid(msg: str) -> None:
    _fail_and_exit("Invalid anchor", msg)


# chainUntil -> numeric cap and final permitted occurrence
def _cap_from_until_cp(task, next_due_utc):
    return _module("modify_completion_compute").cap_from_until_cp(
        task,
        next_due_utc,
        parse_datetime=_dtparse,
        parse_cp_sequence_tokens=core.parse_cp_sequence_tokens,
        coerce_int=core.coerce_int,
        sequence_period_for_link=_cp_sequence_period_for_link,
        add_period=_cp_add_td,
        max_iterations=_MAX_ITERATIONS,
    )


def _cap_from_until_anchor(task, next_due_utc, dnf):
    return _module("modify_completion_compute").cap_from_until_anchor(
        task,
        next_due_utc,
        dnf,
        parse_datetime=_dtparse,
        coerce_int=core.coerce_int,
        recurrence_seed_base=_recurrence_seed_base,
        to_local_cached=_to_local_cached,
        safe_parse_datetime=_safe_parse_datetime,
        anchor_file_fallback_hhmm=_anchor_file_fallback_hhmm,
        omit_dnf_from_parent=_omit_dnf_from_parent,
        recurrence_evaluator_for_task=_recurrence_evaluator_for_task,
        anchor_file_provider_for=_anchor_file_provider_for,
        anchor_included_occurrences=_anchor_included_occurrences,
        compare_datetimes=_compare_datetimes,
        max_iterations=_MAX_ITERATIONS,
    )

def _safe_dt(v):
    try:
        return _dtparse(v) if isinstance(v, str) else v
    except Exception:
        return None

def _parse_extra_tokens(extra: str | None) -> list[str] | None:
    """Parse extra Taskwarrior filters in strict token form: key:value."""
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.parse_extra_tokens(extra)
    if extra is None:
        return []
    if not isinstance(extra, str):
        return None
    s = extra.strip()
    if not s:
        return []
    out: list[str] = []
    for tok in s.split():
        if tok.startswith("+"):
            tag = tok[1:]
            if not tag or re.fullmatch(r"[A-Za-z0-9_.-]+", tag) is None:
                return None
            out.append(tok)
            continue
        if tok.startswith("-"):
            return None
        if ":" not in tok:
            return None
        key, value = tok.split(":", 1)
        if not key or not value:
            return None
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None:
            return None
        if re.fullmatch(r"[A-Za-z0-9_.:@%+,-]+", value) is None:
            return None
        out.append(f"{key}:{value}")
    return out

def _export_chain_endpoint(chain_id: str, direction: str):
    """Return a chain endpoint from the invocation's authoritative snapshot."""
    rows = _lifecycle_read_service().get_chain_export(chain_id)
    if rows is None:
        raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
    with_links = [
        (core.coerce_int(row.get("link"), None), row)
        for row in rows
    ]
    with_links = [(link, row) for link, row in with_links if link is not None]
    if not with_links:
        return None
    with_links.sort(key=lambda item: item[0])
    return with_links[0 if direction == "first" else -1][1]

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def _modify_runtime_services():
    modify_runtime = _module("modify_runtime")
    return modify_runtime.ModifyRuntimeServices(
        state=_modify_runtime_state(),
        core=core,
        debug_wait_sched=_DEBUG_WAIT_SCHED,
        last_wait_sched_debug=_LAST_WAIT_SCHED_DEBUG,
        diag_enabled=os.environ.get("NAUTICAL_DIAG") == "1",
        format_root_and_age=_format_root_and_age,
        append_next_wait_sched_rows=_append_next_wait_sched_rows,
        timeline_lines=_timeline_lines,
        show_timeline_gaps=_SHOW_TIMELINE_GAPS,
        root_uuid_from=_root_uuid_from,
        short=_short,
        format_next_anchor_rows=_format_next_anchor_rows,
        format_next_cp_rows=_format_next_cp_rows,
        format_line_preview=_format_line_preview,
        panel_line=_panel_line,
        text_line=_text_line,
        panel=_panel,
        print_task=_print_task,
        diag=_diag,
        chain_color_per_chain=_CHAIN_COLOR_PER_CHAIN,
        chain_colour_for_task=_chain_colour_for_task,
        strip_quotes=_strip_quotes,
        human_delta=_human_delta,
    )


def _render_anchor_completion_feedback(
    *,
    new: dict,
    child: dict,
    child_due,
    child_short: str,
    next_no: int,
    parent_short: str,
    cap_no: int | None,
    finals: list[tuple[str, object]],
    now_utc,
    until_dt,
    until_cap_no: int | None,
    dnf,
    meta: dict,
    stripped_attrs: list[str],
    deferred_spawn: bool,
    spawn_intent_id: str | None,
    lifecycle_result=None,
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
) -> None:
    calendar_feedback = importlib.import_module("nautical_core.calendar_feedback")
    modify_feedback = _module("modify_feedback")
    modify_models = _module("modify_models")
    modify_feedback.orchestrate_anchor_completion_feedback(
        new=modify_models.TaskView.from_mapping(new),
        child=modify_models.TaskView.from_mapping(child),
        child_due=child_due,
        child_short=child_short,
        next_no=next_no,
        parent_short=parent_short,
        cap_no=cap_no,
        finals=finals,
        now_utc=now_utc,
        until_dt=until_dt,
        until_cap_no=until_cap_no,
        dnf=dnf,
        meta=meta,
        stripped_attrs=stripped_attrs,
        deferred_spawn=deferred_spawn,
        spawn_intent_id=spawn_intent_id,
        lifecycle_result=lifecycle_result,
        chain_by_short=chain_by_short,
        analytics_advice=analytics_advice,
        integrity_warnings=integrity_warnings,
        base_no=base_no,
        core=core,
        panel=_panel,
        calendar_feedback=calendar_feedback,
        panel_diagnostics=_module("panel_diagnostics"),
        modify_models=_module("modify_models"),
        modify_runtime=_module("modify_runtime"),
        build_runtime_services=_modify_runtime_services,
    )


def _render_cp_completion_feedback(
    *,
    new: dict,
    child: dict,
    child_due,
    child_short: str,
    next_no: int,
    parent_short: str,
    cap_no: int | None,
    finals: list[tuple[str, object]],
    now_utc,
    until_dt,
    until_cap_no: int | None,
    meta: dict,
    deferred_spawn: bool,
    spawn_intent_id: str | None,
    lifecycle_result=None,
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
) -> None:
    modify_feedback = _module("modify_feedback")
    modify_models = _module("modify_models")
    modify_feedback.orchestrate_cp_completion_feedback(
        new=modify_models.TaskView.from_mapping(new),
        child=modify_models.TaskView.from_mapping(child),
        child_due=child_due,
        child_short=child_short,
        next_no=next_no,
        parent_short=parent_short,
        cap_no=cap_no,
        finals=finals,
        now_utc=now_utc,
        until_dt=until_dt,
        until_cap_no=until_cap_no,
        meta=meta,
        deferred_spawn=deferred_spawn,
        spawn_intent_id=spawn_intent_id,
        lifecycle_result=lifecycle_result,
        chain_by_short=chain_by_short,
        analytics_advice=analytics_advice,
        integrity_warnings=integrity_warnings,
        base_no=base_no,
        core=core,
        panel_diagnostics=_module("panel_diagnostics"),
        modify_models=_module("modify_models"),
        modify_runtime=_module("modify_runtime"),
        build_runtime_services=_modify_runtime_services,
    )


def _render_lifecycle_result(result, task) -> None:
    """Render one finalized non-success outcome without deciding its state."""
    state = str(getattr(result, "state", "retryable") or "retryable").strip().lower()
    title = "⛓ Chain warning" if state == "manual_review" else "⛓ Chain error"
    rows = [("Result", state.replace("_", " ").title())]
    reason = str(getattr(result, "reason", "") or "").strip()
    if reason:
        rows.append(("Reason", reason))
    child_short = str(getattr(result, "child_short", "") or "").strip()
    if child_short:
        rows.append(("Child", child_short))
    intent_id = str(getattr(result, "spawn_intent_id", "") or "").strip()
    if intent_id:
        rows.append(("Intent", intent_id))
    _panel(title, rows, kind="warning" if state == "manual_review" else "error")


def _non_completion_anchor_error_message(anchor_expr: str, default_msg: str) -> str:
    has_type_colon = bool(
        re.search(r"(?:^|[^A-Za-z])(w|m|y)(?:/\d+)?:", anchor_expr, re.IGNORECASE)
    )
    if has_type_colon:
        return default_msg
    if re.match(r"^(mon|tue|wed|thu|fri|sat|sun)\b", anchor_expr, re.IGNORECASE):
        return (
            "Weekly anchors must start with 'w:'. "
            "Examples: 'w:mon..fri' or 'w:mon,tue,wed,thu,fri'."
        )
    return (
        "Anchors must start with 'w:', 'm:' or 'y:'. "
        "Examples: 'w:mon', 'm:-1', 'y:06-01'."
    )


def _non_completion_anchor_mode(old: dict, new: dict) -> str:
    anchor_mode_raw = (new.get("anchor_mode") or old.get("anchor_mode") or "").strip()
    mode_norm, warn_msg = _validate_anchor_mode(anchor_mode_raw)
    if warn_msg:
        _panel("⚠ Anchor mode", [("Warning", warn_msg)], kind="warning")
        new["anchor_mode"] = mode_norm
    elif (new.get("anchor_mode") or "").strip():
        new["anchor_mode"] = mode_norm
    return ((mode_norm or anchor_mode_raw or "").strip().upper() or "ALL")


def _non_completion_validate_anchor_cache(new: dict, old: dict, anchor_expr: str) -> None:
    _, warns = core.lint_anchor_expr(anchor_expr)
    if warns:
        _panel("ℹ️  Lint", [("Hint", w) for w in warns], kind="note")

    anchor_mode = _non_completion_anchor_mode(old, new)
    due_dt = _safe_dt(new.get("due") or old.get("due"))
    if core.ENABLE_ANCHOR_CACHE:
        _ = core.build_and_cache_hints(
            anchor_expr,
            anchor_mode,
            default_due_dt=due_dt,
            include_per_year=False,
        )
    else:
        _ = core.validate_anchor_expr_strict(anchor_expr)


def _non_completion_validate_anchor(old: dict, new: dict, new_anchor: str) -> None:
    try:
        _non_completion_validate_anchor_cache(new, old, new_anchor)
    except TypeError:
        _ = core.validate_anchor_expr_strict(new_anchor)
    except Exception as e:
        astronomy = core._import_sibling("astronomy")
        if astronomy.is_astronomy_error(e):
            _got_anchor_invalid(astronomy.scheduling_error_message(e))
        _got_anchor_invalid(_non_completion_anchor_error_message(new_anchor, str(e)))


def _validate_omit_for_anchor_or_fail(anchor_expr: str, anchor_file_expr: str, omit_expr: str, omit_file: str) -> None:
    try:
        _validate_shared_omit_on_modify(omit_expr)
        findings = core._import_sibling("hook_validation_pipeline").validate_recurrence_files(
            anchor_expr, anchor_file_expr, omit_expr, omit_file,
            load_anchor_file=_load_anchor_file_dates,
            load_omit_file=_load_omit_file_dates,
        )
    except Exception as exc:
        _fail_and_exit("Invalid omit", str(exc))
        return
    if findings:
        finding = findings[0]
        _fail_and_exit(f"Invalid {finding.field}", finding.reason)


def _semantic_diff_value(old_text: str, new_text: str) -> str:
    return f"[dim]{old_text}[/] [cyan]→[/] [bold]{new_text}[/]"


def _render_recurrence_updated_panel(changes: list[tuple[str, str, str]], new: dict) -> None:
    modify_feedback = _module("modify_feedback")
    modify_models = _module("modify_models")
    add_validation = core._import_sibling("add_validation")
    modify_feedback.render_recurrence_updated_panel(
        changes,
        modify_models.TaskView.from_mapping(new),
        parse_datetime=core.parse_dt_any,
        format_local=_fmtlocal,
        describe_native_until_carry=add_validation.describe_native_until_carry,
        to_local=core.to_local,
        coerce_int=core.coerce_int,
        describe_anchor=core.describe_anchor_expr,
        resolve_omit_presets=core.resolve_omit_presets,
        first_recurrence_target=_first_recurrence_target,
        panel_mode=getattr(core, "PANEL_MODE", "rich"),
        strip_markup=core.strip_rich_markup,
        panel=_panel,
    )


def _first_recurrence_target(new: dict, source: str):
    task_view = _module("modify_models").TaskView.from_mapping(new)
    return _module("modify_completion_compute").first_recurrence_target(
        task_view,
        source,
        parse_datetime=core.parse_dt_any,
        format_datetime=core.fmt_isoz,
        generation_service=_chain_generation_service,
    )


def _recurrence_enabled_rows(new: dict, source: str) -> list[tuple[str, str]]:
    task_view = _module("modify_models").TaskView.from_mapping(new)
    return _module("modify_feedback").recurrence_enabled_rows(
        task_view,
        source,
        describe_anchor=core.describe_anchor_expr,
        parse_cp_sequence_tokens=core.parse_cp_sequence_tokens,
        first_recurrence_target=_first_recurrence_target,
        format_local=_fmtlocal,
    )


def _render_cp_schedule_adjusted_panel(
    adjustment,
) -> None:
    _module("modify_feedback").render_cp_schedule_adjusted_panel(
        adjustment,
        format_local=_fmtlocal,
        semantic_diff_value=_semantic_diff_value,
        format_offset=_fmt_td_dd_hhmm,
        panel=_panel,
    )


def _render_explicit_timing_order_warning(new: dict, changed_fields: tuple[str, ...]) -> None:
    new = _module("modify_models").TaskView.from_mapping(new)
    _module("modify_feedback").render_explicit_timing_order_warning(
        new,
        changed_fields,
        format_offset=_fmt_td_dd_hhmm,
        panel=_panel,
    )


def _render_disabled_chain_summary(old: dict, new: dict, decision) -> None:
    """Show the normal finished-chain summary when an active chain is stopped."""
    reason = str(getattr(decision, "reason", decision))
    if not (old.get("chainID") or new.get("chainID")):
        return
    modify_models = _module("modify_models")
    old_view = modify_models.TaskView.from_mapping(old)
    new_view = modify_models.TaskView.from_mapping(new)
    now_utc = _workflow_now_utc()
    try:
        _end_chain_summary(old_view, reason, now_utc, current_task=new_view)
    except Exception as exc:
        _diag(f"removed recurrence chain summary failed: {exc}")
        _panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", reason),
                ("Root", _format_root_and_age(old_view, now_utc)),
                ("Task", _short(old_view.get("uuid")) or "–"),
            ],
            kind="summary",
        )


def _ensure_terminal_chain_off(task: dict, event: str | None = None) -> bool:
    """Validate and apply one idempotent terminal patch for hook-side stops."""
    if event:
        lifecycle_models = _module("lifecycle_models")
        lifecycle_planner = _module("lifecycle_planner")
        task_codec = _module("task_codec")
        lifecycle_planner.terminal_plan_for_snapshot(
            lifecycle_models.TaskSnapshot.from_observation(
                task_codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify terminal")
            ),
            lifecycle_models.LifecycleEvent(event),
        )
    return _module("modify_lifecycle").ensure_terminal_chain_off(task)


def main():
    _module("modify_composition").run_on_modify(
        _module("modify_composition").hook_host(globals(), __name__)
    )


def run_hook(
    *,
    raw_input: bytes,
    argv: tuple[str, ...],
    hook_dir: str,
    core_base: str,
    protocol=None,
    probe=_PROBE_UNSET,
    protocol_error=None,
) -> int:
    """Run the extracted implementation with context captured by the wrapper."""
    global HOOK_DIR, TW_DIR, _CORE_BASE, _EARLY_PROTOCOL_RESULT, _PROTOCOL
    global _TASKDATA_RAW, _USE_RC_DATA_LOCATION, TW_DATA_DIR

    HOOK_DIR = Path(hook_dir)
    TW_DIR = HOOK_DIR.parent
    _CORE_BASE = Path(core_base)
    sys.argv = [sys.argv[0], *argv]
    try:
        _initialize_integration_context()
    except _hook_runtime_module().HookIntegrationContextError as exc:
        globals()["core"] = exc.core
        title = "Invalid Nautical configuration" if exc.stage in {"configuration", "timezone"} else "Nautical integration unavailable"
        _fail_and_exit(title, exc.detail)

    if protocol is None:
        protocol, _protocol_path, protocol_error = hook_bootstrap.load_core_helper_module(
            _CORE_BASE,
            "hook_protocol.py",
            "_nautical_hook_protocol_modify_impl",
        )
    if protocol is None:
        if protocol_error is not None:
            raise RuntimeError(f"could not load on-modify protocol: {protocol_error}") from protocol_error
        raise RuntimeError("could not load on-modify protocol")
    _PROTOCOL = protocol
    if probe is _PROBE_UNSET or (probe is None and protocol_error is not None):
        probe = protocol.probe_on_modify(raw_input, max_bytes=_MAX_JSON_BYTES)
    _EARLY_PROTOCOL_RESULT = probe
    main()
    return 0


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                sys.stderr.write(f"[nautical] on-modify unexpected error: {e}\n")
            except Exception:
                pass
        _panic_passthrough()
        raise SystemExit(1)
