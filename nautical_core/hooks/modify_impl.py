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
# Test and diagnostic callers may request the extracted timeline adapter without
# making it part of the production route surface.
def __getattr__(name: str):
    if name == "_timeline_lines":
        host = _module("modify_composition").hook_host(globals(), __name__)
        def _timeline_lines(kind, task, child_due_utc, child_short, dnf, **kwargs):
            return _module("modify_presentation_effects").timeline_lines(
                host, kind, task, child_due_utc, child_short, dnf, **kwargs
            )
        return _timeline_lines
    raise AttributeError(name)


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
        parsed, error = _module("modify_datetime_effects").safe_parse_datetime(
            _module("modify_composition").hook_host(globals(), __name__), task.get(field)
        )
        if not error and parsed is not None:
            local = _to_local_cached(parsed)
            return local.hour, local.minute
    return default_local.hour, default_local.minute


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
        service = getattr(_modify_runtime_state(), "lifecycle_read_service", None)
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
    "modify_spawn_effects": (
        "_MODIFY_SPAWN_EFFECTS",
        "_MODIFY_SPAWN_EFFECTS_LOAD_FAILED",
        "modify_spawn_effects.py",
        "nautical_core.modify_spawn_effects",
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
    "modify_presentation_effects": (
        "_MODIFY_PRESENTATION_EFFECTS",
        "_MODIFY_PRESENTATION_EFFECTS_LOAD_FAILED",
        "modify_presentation_effects.py",
        "nautical_core.modify_presentation_effects",
    ),
    "modify_diagnostics_effects": (
        "_MODIFY_DIAGNOSTICS_EFFECTS",
        "_MODIFY_DIAGNOSTICS_EFFECTS_LOAD_FAILED",
        "modify_diagnostics_effects.py",
        "nautical_core.modify_diagnostics_effects",
    ),
    "modify_validation_effects": (
        "_MODIFY_VALIDATION_EFFECTS",
        "_MODIFY_VALIDATION_EFFECTS_LOAD_FAILED",
        "modify_validation_effects.py",
        "nautical_core.modify_validation_effects",
    ),
    "modify_schedule_effects": (
        "_MODIFY_SCHEDULE_EFFECTS",
        "_MODIFY_SCHEDULE_EFFECTS_LOAD_FAILED",
        "modify_schedule_effects.py",
        "nautical_core.modify_schedule_effects",
    ),
    "modify_datetime_effects": (
        "_MODIFY_DATETIME_EFFECTS",
        "_MODIFY_DATETIME_EFFECTS_LOAD_FAILED",
        "modify_datetime_effects.py",
        "nautical_core.modify_datetime_effects",
    ),
    "modify_anchor_effects": (
        "_MODIFY_ANCHOR_EFFECTS",
        "_MODIFY_ANCHOR_EFFECTS_LOAD_FAILED",
        "modify_anchor_effects.py",
        "nautical_core.modify_anchor_effects",
    ),
    "modify_task_fields": (
        "_MODIFY_TASK_FIELDS",
        "_MODIFY_TASK_FIELDS_LOAD_FAILED",
        "modify_task_fields.py",
        "nautical_core.modify_task_fields",
    ),
    "modify_time_effects": (
        "_MODIFY_TIME_EFFECTS",
        "_MODIFY_TIME_EFFECTS_LOAD_FAILED",
        "modify_time_effects.py",
        "nautical_core.modify_time_effects",
    ),
    "modify_read_effects": (
        "_MODIFY_READ_EFFECTS",
        "_MODIFY_READ_EFFECTS_LOAD_FAILED",
        "modify_read_effects.py",
        "nautical_core.modify_read_effects",
    ),
    "modify_format_effects": (
        "_MODIFY_FORMAT_EFFECTS",
        "_MODIFY_FORMAT_EFFECTS_LOAD_FAILED",
        "modify_format_effects.py",
        "nautical_core.modify_format_effects",
    ),
    "modify_generation_effects": (
        "_MODIFY_GENERATION_EFFECTS",
        "_MODIFY_GENERATION_EFFECTS_LOAD_FAILED",
        "modify_generation_effects.py",
        "nautical_core.modify_generation_effects",
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

# ------------------------------------------------------------------------------
# Taskwarrior integration
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# On modify-without-completion helpers
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# Multi-time occurrence helpers (hook-level)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Due calculators
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Child build (copy almost everything; override minimal set)
# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------
# Timeline (capped) — no dependency on core.next_anchor_after
# ------------------------------------------------------------------------------

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
