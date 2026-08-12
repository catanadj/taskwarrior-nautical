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
NAUTICAL_RECONCILE_PROTOCOL = 2

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
import sqlite3
import stat
import subprocess
import tempfile
import time as _time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta, timezone, time
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nautical_core.modify_models import CompletionLifecycleResult


# set config show_analytics=false to disable analytics panel entry.

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

_MAX_CHAIN_WALK = 500  # cap for chain summaries/analytics
_MAX_UUID_LOOKUPS = 50  # max individual UUID exports before giving up
_MAX_ITERATIONS = 2000  # prevent infinite loops in stepping functions
_MIN_FUTURE_WARN = 365 * 2  # warn if chain extends >2 years


_MAX_SPAWN_ATTEMPTS = 3
_SPAWN_RETRY_DELAY = 0.1  # seconds between retries
_STABLE_CHILD_UUID_NAMESPACE = uuid.UUID("1f4b2396-df58-5a32-a879-33f0d3fe711f")
# Spawn intent queue guards (override via env for heavy workloads).
# spawn_queue_max_bytes: warn when queue exceeds this size (on-exit drains).
_DEFAULT_SPAWN_QUEUE_MAX_BYTES = 524288
_DEFAULT_CHAIN_EXPORT_TIMEOUT_BASE = 1.5
_DEFAULT_CHAIN_EXPORT_TIMEOUT_PER_100 = 1.0
_DEFAULT_CHAIN_EXPORT_TIMEOUT_MAX = 12.0

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
_WARNED_SPAWN_QUEUE_GROWTH = False
_WARNED_CHAIN_EXPORT: set[str] = set()
_LAST_CHAIN_EXPORT_STATUS: tuple[bool, str] = (True, "")


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


def _is_lock_error(stderr: str) -> bool:
    try:
        _load_core()
    except Exception:
        pass
    if core is not None:
        return core.is_lock_error(stderr)
    s = (stderr or "").lower()
    return (
        "database is locked" in s or "unable to lock" in s
        or "resource temporarily unavailable" in s or "another task is running" in s
        or "lock file" in s or "lockfile" in s or "locked by" in s or "timeout" in s
    )


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


def _anchor_file_provider_for(
    anchor_file: str,
    *,
    fallback_hhmm: tuple[int, int],
    seed_base: str,
):
    """Return one anchor-file provider for this hook's projection session."""
    if not anchor_file:
        return None
    state = _modify_runtime_state()
    anchor_file_dir = getattr(core, "ANCHOR_FILE_DIR", "")
    key = (anchor_file, anchor_file_dir, fallback_hhmm, seed_base)
    provider = state.anchor_file_providers.get(key)
    if provider is None:
        provider = core._import_sibling("anchor_inclusion")._build_anchor_file_provider(
            anchor_file,
            anchor_file_dir=anchor_file_dir,
            fallback_hhmm=fallback_hhmm,
            seed_base=seed_base,
            core=core,
        )
        state.anchor_file_providers[key] = provider
    return provider


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
        if any(p.startswith("uuid:") for p in parts):
            if "rc.json.array=off" in parts:
                return "export_uuid_short"
            if "rc.json.array=1" in parts:
                return "export_uuid_full"
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
    for name in (
        "_export_uuid_short_cached",
    ):
        try:
            clear = getattr(globals().get(name), "cache_clear", None)
            if callable(clear):
                clear()
        except Exception:
            pass
    try:
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
    """Append Scheduled/Wait lines for the next link showing local time and Δ to the recurrence anchor."""
    if not (isinstance(nxt_due_utc, datetime) and nxt_due_utc):
        return

    dt_s = _dtparse(nxt.get("scheduled"))
    dt_w = _dtparse(nxt.get("wait"))
    anchor_label = "scheduled" if anchor_field == "scheduled" else "due"

    for fld, label, dt in (
        ("scheduled", "Scheduled", dt_s),
        ("wait", "Wait", dt_w),
    ):
        if fld == anchor_field:
            continue
        if not isinstance(dt, datetime):
            continue
        delta_s = _fmt_td_dd_hhmm(dt - nxt_due_utc)
        fb.append((label, f"{core.fmt_dt_local(dt)}  (Δ {delta_s})"))

    # Informative order validation: due > scheduled > wait
    # This can be violated when due is auto-assigned but scheduled/wait are user-specified.
    issues: list[str] = []
    if anchor_field != "scheduled" and isinstance(dt_s, datetime) and _compare_datetimes(dt_s, nxt_due_utc) > 0:
        issues.append(f"scheduled is after {anchor_label} by {_fmt_td_dd_hhmm(dt_s - nxt_due_utc)}")
    if isinstance(dt_w, datetime) and _compare_datetimes(dt_w, nxt_due_utc) > 0:
        issues.append(f"wait is after {anchor_label} by {_fmt_td_dd_hhmm(dt_w - nxt_due_utc)}")
    if anchor_field != "scheduled" and isinstance(dt_s, datetime) and isinstance(dt_w, datetime) and _compare_datetimes(dt_w, dt_s) > 0:
        issues.append(f"wait is after scheduled by {_fmt_td_dd_hhmm(dt_w - dt_s)}")

    if issues:
        expected = "scheduled > wait" if anchor_field == "scheduled" else "due > scheduled > wait"
        fb.append((
            "⚠ Wait/Sched",
            f"Expected order: {expected}. " + "; ".join(issues),
        ))
        fb.append((
            "⚠ Wait/Sched",
            "This can happen when due is auto-assigned; adjust scheduled/wait if undesired.",
        ))

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
_LIFECYCLE_EXECUTOR = None
_LIFECYCLE_EXECUTOR_LOAD_FAILED = False
_MODIFY_FEEDBACK = None
_MODIFY_FEEDBACK_LOAD_FAILED = False
_MODIFY_TIMELINE = None
_MODIFY_TIMELINE_LOAD_FAILED = False
_MODIFY_EXPIRATION = None
_MODIFY_EXPIRATION_LOAD_FAILED = False
_QUEUE_STORE = None
_QUEUE_STORE_LOAD_FAILED = False
_QUEUE_MODELS = None
_QUEUE_MODELS_LOAD_FAILED = False
_RECONCILE = None
_RECONCILE_LOAD_FAILED = False
_HOOK_RUNTIME = None
_HOOK_RUNTIME_LOAD_FAILED = False
_HOOK_MODULE_ACCESS = None
_RECURRENCE_EVALUATOR = None
_RECURRENCE_EVALUATOR_LOAD_FAILED = False
_ADD_ANCHOR_COMPUTE = None
_ADD_ANCHOR_COMPUTE_LOAD_FAILED = False
_MODULE_SPECS = {
    "hook_runtime": (
        "_HOOK_RUNTIME",
        "_HOOK_RUNTIME_LOAD_FAILED",
        "hook_runtime.py",
        "nautical_core.hook_runtime",
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
    "reconcile": (
        "_RECONCILE",
        "_RECONCILE_LOAD_FAILED",
        "reconcile.py",
        "nautical_core.reconcile",
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

_TASKDATA_RAW, _USE_RC_DATA_LOCATION = _resolve_task_data_context()
TW_DATA_DIR = Path(_TASKDATA_RAW).expanduser()


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


def _build_hook_runtime_context():
    hook_runtime = _hook_runtime_module()
    return hook_runtime.build_hook_runtime_context(
        module_access=_hook_module_access(),
        hook_name="on-modify",
        taskdata_dir=str(TW_DATA_DIR),
        use_rc_data_location=_USE_RC_DATA_LOCATION,
        tw_dir=str(TW_DIR),
        hook_dir=str(HOOK_DIR),
        import_ms=_IMPORT_MS,
    )


def _task_cmd_prefix() -> list[str]:
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.build_task_cmd_prefix(
            use_rc_data_location=_USE_RC_DATA_LOCATION,
            tw_data_dir=TW_DATA_DIR,
        )
    cmd = ["task"]
    if _USE_RC_DATA_LOCATION:
        cmd.append(f"rc.data.location={TW_DATA_DIR}")
    return cmd

# ------------------------------------------------------------------------------
# Deferred next-link spawn queue (used when nested `task import` times out due to TW lock)
# ------------------------------------------------------------------------------
# Keep import-time state path setup dependency-free.  The queue adapter is
# loaded by the lifecycle that actually needs it and run_hook refreshes these
# paths after resolving TASKDATA/rc.data.location.
_SPAWN_QUEUE_LOCK = TW_DATA_DIR / ".nautical-locks" / ".nautical_spawn_queue.lock"
_SPAWN_QUEUE_DB_PATH = TW_DATA_DIR / ".nautical-state" / ".nautical_queue.db"
_DEAD_LETTER_PATH = TW_DATA_DIR / ".nautical-state" / ".nautical_dead_letter.jsonl"
_DEAD_LETTER_LOCK = TW_DATA_DIR / ".nautical-locks" / ".nautical_dead_letter.lock"
_SPAWN_LOCK_RETRIES = 6
_SPAWN_LOCK_SLEEP_BASE = 0.03
_SPAWN_LOCK_STALE_AFTER = 30.0
_DEAD_LETTER_LOCK_RETRIES = _SPAWN_LOCK_RETRIES
_DEAD_LETTER_LOCK_SLEEP_BASE = _SPAWN_LOCK_SLEEP_BASE
_WARNED_SPAWN_QUEUE_LOCK = False
_DURABLE_QUEUE = os.environ.get("NAUTICAL_DURABLE_QUEUE") == "1"

def _migrate_legacy_nautical_state() -> None:
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        issues = queue_store.migrate_nautical_state(tw_data_dir=TW_DATA_DIR)
        for issue in issues:
            _diag(f"queue state migration failed: {issue.current} from {issue.legacy}: {issue.error}")
        globals()["_SPAWN_QUEUE_DB_PATH"] = queue_store.queue_db_path(TW_DATA_DIR)
        globals()["_DEAD_LETTER_PATH"] = queue_store.dead_letter_path(TW_DATA_DIR)
        globals()["_DEAD_LETTER_LOCK"] = queue_store.dead_letter_lock_path(TW_DATA_DIR)
        return

def _load_core() -> None:
    global core, _MAX_JSON_BYTES, _CORE_READY, _IMPORT_MS
    if core is not None and _CORE_READY:
        return
    if core is None:
        module, target, import_error = hook_bootstrap.import_core_package(_CORE_BASE)
        if target is not None:
            globals()["_CORE_IMPORT_TARGET"] = target
        if import_error is not None:
            globals()["_CORE_IMPORT_ERROR"] = import_error
        if module is not None:
            core = module
    if core is None:
        msg = (
            "nautical_core package not found. Expected nautical_core/__init__.py in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {_CORE_BASE})"
        )
        raise ModuleNotFoundError(msg)
    reload_config = getattr(core, "reload_taskdata_config", None)
    if callable(reload_config):
        if _USE_RC_DATA_LOCATION:
            reload_config(TW_DATA_DIR)
    elif getattr(core, "__file__", None):
        raise RuntimeError("nautical_core does not provide validated configuration reload")
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
_SPAWN_QUEUE_MAX_BYTES = _DEFAULT_SPAWN_QUEUE_MAX_BYTES
_MAX_CHAIN_WALK = _MAX_CHAIN_WALK

def _apply_core_config() -> None:
    global _CHAIN_COLOR_PER_CHAIN, _SHOW_TIMELINE_GAPS, _SHOW_ANALYTICS, _ANALYTICS_STYLE
    global _ANALYTICS_ONTIME_TOL_SECS, _CHECK_CHAIN_INTEGRITY
    global _DEBUG_WAIT_SCHED, _RECURRENCE_UPDATE_UDAS, _SPAWN_QUEUE_MAX_BYTES, _MAX_CHAIN_WALK
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
    _SPAWN_QUEUE_MAX_BYTES = core.SPAWN_QUEUE_MAX_BYTES if hasattr(core, "SPAWN_QUEUE_MAX_BYTES") else _DEFAULT_SPAWN_QUEUE_MAX_BYTES
    _MAX_CHAIN_WALK = core.MAX_CHAIN_WALK
_CHAIN_EXPORT_TIMEOUT_BASE = _env_float(
    "NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE",
    _DEFAULT_CHAIN_EXPORT_TIMEOUT_BASE,
    min_value=0.1,
    max_value=120.0,
)
_CHAIN_EXPORT_TIMEOUT_PER_100 = _env_float(
    "NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100",
    _DEFAULT_CHAIN_EXPORT_TIMEOUT_PER_100,
    min_value=0.0,
    max_value=120.0,
)
_CHAIN_EXPORT_TIMEOUT_MAX = _env_float(
    "NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX",
    _DEFAULT_CHAIN_EXPORT_TIMEOUT_MAX,
    min_value=0.1,
    max_value=300.0,
)
_CHAIN_EXPORT_TIMEOUT_MAX = max(_CHAIN_EXPORT_TIMEOUT_BASE, _CHAIN_EXPORT_TIMEOUT_MAX)
_CHAIN_EXPORT_TIMES: list[float] = []
_CHAIN_EXPORT_TIMES_MAX = 20
_CHAIN_EXPORT_TIMEOUT_FLOOR = _CHAIN_EXPORT_TIMEOUT_BASE


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


@lru_cache(maxsize=512)
def _export_uuid_short_cached(u_short: str):
    obj = _export_uuid_short(u_short, env=None)
    if isinstance(obj, dict) and obj.get("uuid"):
        return obj
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.LookupResult.unavailable("short UUID lookup unavailable")
    return None


def _export_uuid_short_lookup(u_short: str):
    """Return a tri-state UUID lookup without collapsing failures to None."""
    cached = _export_uuid_short_cached(u_short)
    hook_support = _module("hook_support", required=False)
    if hook_support is not None and isinstance(cached, hook_support.LookupResult):
        return cached
    if isinstance(cached, dict) and cached.get("uuid"):
        if hook_support is not None:
            return hook_support.LookupResult.found(cached)
    if hook_support is None:
        return None
    return hook_support.export_uuid_short_result(
        run_task=_run_task_result,
        task_cmd_prefix=_task_cmd_prefix(),
        uuid_short=u_short,
        env=os.environ.copy(),
        timeout=2.5,
        retries=2,
        diag=_diag,
    )


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
def _fail_and_exit(title: str, msg: str) -> None:
    _panel(f"❌ {title}", [("Message", msg)], kind="error")
    sys.exit(1)

_RAW_INPUT_TEXT = ""
_PARSED_NEW = None


def _fail_protocol_error(msg: str) -> None:
    _fail_and_exit("Protocol error", msg)


def _fail_invalid_input(msg: str) -> None:
    _fail_and_exit("Invalid input", msg)


def _task_uuid_or_empty(task: dict) -> str:
    if not isinstance(task, dict):
        return ""
    try:
        return str(task.get("uuid") or "").strip()
    except Exception:
        return ""


def _validate_modify_pair(old: dict, new: dict) -> tuple[dict, dict]:
    old_uuid = _task_uuid_or_empty(old)
    new_uuid = _task_uuid_or_empty(new)
    if not old_uuid or not new_uuid:
        _fail_protocol_error("Missing task UUID in on-modify input")
    if old_uuid != new_uuid:
        if not _task_has_nautical_fields(old, new):
            return old, new
        _fail_protocol_error("Old and new task UUIDs differ")
    return old, new


def _validate_single_modify_task(task: dict) -> tuple[dict, dict]:
    if not _task_uuid_or_empty(task):
        if not _task_has_nautical_fields(task, task):
            return task, task
        _fail_protocol_error("Missing task UUID in on-modify input")
    return task, task


def _decode_leading_json_objects(raw: str, max_objects: int = 2) -> tuple[list[object], int]:
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[object] = []
    n = len(raw)
    tries = 0
    loop_guard = 0
    max_loops = 10

    while idx < n and len(objs) < max_objects:
        loop_guard += 1
        if loop_guard > max_loops:
            _fail_protocol_error("Invalid JSON input: too many parse attempts")
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except Exception as e:
            _diag(f"json decode error: {e}")
            _fail_protocol_error("Invalid JSON input")
        objs.append(obj)
        if end <= idx:
            tries += 1
            if tries >= 2:
                _fail_protocol_error("Invalid JSON input: parser made no progress")
            idx += 1
            continue
        idx = end

    return objs, idx


def _read_two():
    global _RAW_INPUT_TEXT, _PARSED_NEW
    if _EARLY_PROTOCOL_RESULT is not None:
        _RAW_INPUT_TEXT = _EARLY_PROTOCOL_RESULT.raw_text
        _PARSED_NEW = _EARLY_PROTOCOL_RESULT.new
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
    aliases = core._import_sibling("description_aliases")
    try:
        aliases.apply_description_aliases(new, previous=old)
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
_TW_JISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_UNREC_ATTR_RE = re.compile(r"Unrecognized attribute '([^']+)'", re.I)



def _spawn_queue_db_size_bytes() -> int:
    total = 0
    paths = (
        _SPAWN_QUEUE_DB_PATH,
        Path(str(_SPAWN_QUEUE_DB_PATH) + "-wal"),
        Path(str(_SPAWN_QUEUE_DB_PATH) + "-shm"),
    )
    for p in paths:
        try:
            if p.exists():
                total += p.stat().st_size
        except Exception:
            continue
    return total


def _spawn_queue_total_bytes() -> int:
    return _spawn_queue_db_size_bytes()


def _spawn_queue_warn_growth(queue_path: Path, size: int) -> None:
    global _WARNED_SPAWN_QUEUE_GROWTH
    try:
        if _WARNED_SPAWN_QUEUE_GROWTH:
            return
        if _SPAWN_QUEUE_MAX_BYTES <= 0:
            return
        if size > _SPAWN_QUEUE_MAX_BYTES:
            _WARNED_SPAWN_QUEUE_GROWTH = True
            _panel(
                "⚠ Spawn queue growing",
                [
                    ("Queue", str(queue_path)),
                    ("Size", f"{size} bytes"),
                    ("Limit", f"{_SPAWN_QUEUE_MAX_BYTES} bytes"),
                    ("Hint", "Run the on-exit hook or reduce load."),
                ],
                kind="warning",
            )
    except Exception:
        pass


def _handle_enqueue_lock_busy(task_obj: dict) -> tuple[bool, str]:
    _write_dead_letter(task_obj, "queue lock busy")
    _diag("queue lock busy; intent dead-lettered")
    _diag_count("queue_lock_failures")
    global _WARNED_SPAWN_QUEUE_LOCK
    if not _WARNED_SPAWN_QUEUE_LOCK:
        _WARNED_SPAWN_QUEUE_LOCK = True
        _panel(
            "⚠ Spawn queue busy",
            [
                ("Queue", str(_SPAWN_QUEUE_DB_PATH)),
                ("Hint", "Queue lock busy; spawn intent not queued."),
            ],
            kind="warning",
        )
    return False, "queue lock busy"


def _spawn_queue_db_connect_result():
    queue_store = _module("queue_store")
    timeout_base = max(1.0, _SPAWN_LOCK_SLEEP_BASE * max(1, _SPAWN_LOCK_RETRIES) * 4.0)
    return queue_store.connect_queue_db_result(
        _SPAWN_QUEUE_DB_PATH,
        attempts=2,
        timeout_base=timeout_base,
        timeout_max=timeout_base,
        backoff_base=0.0,
        durable=_DURABLE_QUEUE,
        diag=_diag,
    )




def _spawn_queue_db_init(conn: sqlite3.Connection) -> None:
    queue_store = _module("queue_store")
    queue_store.init_queue_db(conn)


def _queue_close_silent(conn: sqlite3.Connection) -> None:
    queue_store = _module("queue_store")
    queue_store.close_silent(conn)


def _spawn_queue_db_open_ready() -> sqlite3.Connection | None:
    queue_store = _module("queue_store")
    return queue_store.open_ready_queue_db_result(
        _SPAWN_QUEUE_DB_PATH,
        connect_fn=_spawn_queue_db_connect_result,
        init_fn=_spawn_queue_db_init,
        close_fn=_queue_close_silent,
        diag=_diag,
    ).conn


def _spawn_queue_capacity_guard(task_obj: dict) -> tuple[bool, str] | None:
    if _SPAWN_QUEUE_MAX_BYTES <= 0:
        return None
    try:
        if _spawn_queue_total_bytes() > _SPAWN_QUEUE_MAX_BYTES:
            _write_dead_letter(task_obj, "spawn queue full")
            _diag("spawn queue full; intent dropped")
            return False, "spawn queue full"
    except Exception as exc:
        reason = f"spawn queue capacity check unavailable: {exc}"
        _write_dead_letter(task_obj, reason)
        _diag(reason)
        return False, reason
    return None

def _spawn_queue_write_failure(task_obj: dict, err: Exception) -> tuple[bool, str]:
    fail_reason = f"spawn queue write failed: {err}"
    _write_dead_letter(task_obj, fail_reason)
    _diag(fail_reason)
    return False, fail_reason


def _enqueue_deferred_spawn_sqlite(
    task_obj: dict,
    *,
    require_lifecycle_plan: bool = False,
) -> tuple[bool, str]:
    guard = _spawn_queue_capacity_guard(task_obj)
    if guard is not None:
        return guard

    conn = _spawn_queue_db_open_ready()
    if conn is None:
        return False, "spawn queue db unavailable"
    lock_busy = {"value": False}
    errors: list[str] = []
    try:
        queue_store = _module("queue_store")
        result = queue_store.enqueue_entries_sqlite_result(
            conn,
            [task_obj],
            now=_time.time(),
            diag=lambda msg: errors.append(str(msg)),
            on_lock_busy=lambda: lock_busy.__setitem__("value", True),
            require_lifecycle_plan=require_lifecycle_plan,
            max_payload_bytes=_SPAWN_QUEUE_MAX_BYTES,
        )
        if result.ok:
            queue_store.repair_sqlite_permissions(_SPAWN_QUEUE_DB_PATH)
            _spawn_queue_warn_growth(_SPAWN_QUEUE_DB_PATH, _spawn_queue_db_size_bytes())
            return True, ""
        if lock_busy["value"]:
            return _handle_enqueue_lock_busy(task_obj)
        err = result.err or (errors[-1] if errors else "spawn queue write failed")
        return _spawn_queue_write_failure(task_obj, RuntimeError(err))
    finally:
        _queue_close_silent(conn)


def _enqueue_deferred_spawn(
    task_obj: dict,
    *,
    require_lifecycle_plan: bool = False,
) -> tuple[bool, str]:
    return _enqueue_deferred_spawn_sqlite(task_obj, require_lifecycle_plan=require_lifecycle_plan)


def _write_dead_letter(entry: dict, reason: str) -> None:
    if not _require_core():
        return
    queue_store = _module("queue_store")
    payload = queue_store.build_dead_letter_payload(
        hook="on-modify",
        hook_version=NAUTICAL_HOOK_VERSION,
        entry=entry,
        reason=reason,
        now_fn=lambda: _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    )
    try:
        queue_store.append_dead_letter_jsonl(
            path=_DEAD_LETTER_PATH,
            payload=payload,
            durable=_DURABLE_QUEUE,
            acquire_lock=lambda: core.safe_lock(
                _DEAD_LETTER_LOCK,
                retries=_DEAD_LETTER_LOCK_RETRIES,
                sleep_base=_DEAD_LETTER_LOCK_SLEEP_BASE,
                jitter=_DEAD_LETTER_LOCK_SLEEP_BASE,
                mkdir=True,
                stale_after=_SPAWN_LOCK_STALE_AFTER,
            ),
            diag=_diag,
        )
    except Exception:
        pass



def _short(u):
    return (u or "")[:8]


def _run_task(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
) -> tuple[bool, str, str]:
    load_err = None
    try:
        _load_core()
    except Exception as e:
        load_err = e
    _diag_count("run_task_calls")
    t0 = _ptime.perf_counter()
    core_runner = getattr(core, "run_task", None) if core is not None else None
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        if load_err is not None and not callable(core_runner):
            _diag(f"core.run_task unavailable; falling back to subprocess: {load_err}")
        ok, out, err = hook_support.run_task(
            cmd,
            core_run_task=core_runner,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            use_tempfiles=use_tempfiles,
        )
    else:
        if callable(core_runner):
            ok, out, err = core.run_task(
                cmd,
                env=env,
                input_text=input_text,
                timeout=timeout,
                retries=retries,
                use_tempfiles=use_tempfiles,
            )
        else:
            if load_err is not None:
                _diag(f"core.run_task unavailable; falling back to subprocess: {load_err}")
            env = env or os.environ.copy()
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    close_fds=True,
                    env=env,
                )
                out, err = proc.communicate(input=input_text, timeout=timeout)
                ok, out, err = (proc.returncode == 0, out or "", err or "")
            except subprocess.TimeoutExpired:
                if proc is not None:
                    proc.kill()
                try:
                    out, err = proc.communicate(timeout=1.0) if proc is not None else ("", "")
                except Exception:
                    out, err = "", ""
                ok, out, err = (False, out or "", "timeout")
            except Exception as e:
                if proc is not None:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=1.0)
                    except Exception:
                        pass
                ok, out, err = (False, "", str(e))
    elapsed = _ptime.perf_counter() - t0
    _diag_count("run_task_seconds", elapsed)
    _diag_record_run_task(cmd, ok=ok, elapsed=elapsed)
    if not ok:
        _diag_count("run_task_failures")
    return ok, out, err


_DEFAULT_RUN_TASK = _run_task


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
    """Return typed command state while retaining the legacy tuple wrapper."""
    core_runner = (
        getattr(core, "run_task_result", None)
        if core is not None and _run_task is _DEFAULT_RUN_TASK
        else None
    )
    if _run_task is not _DEFAULT_RUN_TASK:
        from nautical_core.task_command import coerce_command_result
        return coerce_command_result(
            _run_task(
                cmd,
                env=env,
                input_text=input_text,
                timeout=timeout,
                retries=retries,
                use_tempfiles=use_tempfiles,
            ),
            cmd,
            timeout=timeout,
            attempts=retries,
        )
    if callable(core_runner):
        return core_runner(
            cmd,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            use_tempfiles=use_tempfiles,
        )
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.run_task_result(
            run_task=_run_task,
            cmd=cmd,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            use_tempfiles=use_tempfiles,
        )
    from nautical_core.task_command import coerce_command_result
    return coerce_command_result(
        _run_task(
            cmd,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        ),
        cmd,
        timeout=timeout,
        attempts=retries,
    )


def _export_uuid_short(u_short: str, env=None):
    if env is None:
        cached_read = _read_query_get("uuid", str(u_short or "").lower())
        if cached_read is not _READ_QUERY_MISSING:
            return cached_read
    cache_chain_id = ""
    if env is None and u_short:
        cached, cache_chain_id = _lifecycle_read_service().lookup_short(u_short)
        if isinstance(cached, dict):
            _diag_count("export_uuid_cache_hits")
            return dict(cached)
    if env is None:
        if cache_chain_id:
            _diag_count("unexpected_cache_misses")
            _diag(f"cache miss: short uuid {u_short} (chainID={cache_chain_id})")
        else:
            _diag_count("export_uuid_cache_misses")
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        obj = hook_support.export_uuid_short(
            run_task=_run_task_result,
            task_cmd_prefix=_task_cmd_prefix(),
            uuid_short=u_short,
            env=(env or os.environ.copy()),
            timeout=2.5,
            retries=2,
            diag=_diag,
        )
        if env is None and isinstance(obj, dict):
            _read_query_set("uuid", str(u_short or "").lower(), obj)
            return _seed_runtime_lookup_task(obj, lookup_short=u_short)
        return obj
    env = env or os.environ.copy()
    result = _run_task_result(
        _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{u_short}", "export"],
        env=env,
        timeout=2.5,
        retries=2,
    )
    if not result.ok:
        _diag(f"export uuid:{u_short} failed: {result.stderr.strip()}")
        return None
    try:
        obj = json.loads(result.stdout.strip() or "{}")
        if not obj.get("uuid"):
            return None
        if not str(obj.get("uuid") or "").lower().startswith((u_short or "").lower()):
            _diag(f"uuid prefix mismatch for {u_short}")
            return None
        if env is None:
            _read_query_set("uuid", str(u_short or "").lower(), obj)
            return _seed_runtime_lookup_task(obj, lookup_short=u_short)
        return obj
    except Exception:
        return None


def _task_lookup_by_uuid(u: str, env: dict | None):
    """Return a tri-state child verification result for mutation decisions."""
    hook_support = _module("hook_support", required=False)
    if hook_support is None:
        return None
    if env is None:
        cached_read = _read_query_get("uuid", str(u or "").lower())
        if cached_read is not _READ_QUERY_MISSING:
            # Empty legacy cache entries can represent an earlier parse or
            # command failure; only a task-bearing entry is authoritative.
            if isinstance(cached_read, dict) and cached_read.get("uuid"):
                return hook_support.LookupResult.found(cached_read)
    return hook_support.task_lookup_by_uuid_uncached(
        run_task=_run_task_result,
        task_cmd_prefix=_task_cmd_prefix(),
        uuid_str=u,
        env=env,
        timeout=2.5,
        retries=2,
        diag=_diag,
    )


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
    removed = set()
    for m in _UNREC_ATTR_RE.finditer(stderr or ""):
        bad = m.group(1)
        if bad in payload:
            payload.pop(bad, None)
            removed.add(bad)
    return removed


def _normalise_datetime_fields(obj: dict) -> None:
    def _to_tw_compact_isoz(s: str) -> str:
        if isinstance(s, str) and _TW_JISO.fullmatch(s):
            return s.replace("-", "").replace(":", "")
        return s

    for k in ("entry", "modified", "due", "end", "wait", "until", "scheduled"):
        if k in obj and obj[k]:
            obj[k] = _to_tw_compact_isoz(obj[k])
    if "annotations" in obj and isinstance(obj["annotations"], list):
        for ann in obj["annotations"]:
            if isinstance(ann, dict) and ann.get("entry"):
                ann["entry"] = _to_tw_compact_isoz(ann["entry"])


def _strip_none_and_cast(obj: dict):
    out = {}
    for k, v in obj.items():
        if v is None:
            continue
        if k in ("link", "chainMax"):
            try:
                v = int(v)
            except Exception:
                pass
        out[k] = v
    return out

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
        parse_datetime=_dtparse,
        on_time_delta=_fmt_on_time_delta,
        human_delta=_human_delta,
    )


# Helper to categorize subprocess failures
def _categorize_spawn_error(returncode: int, stderr: str) -> tuple[str, bool]:
    """
    Categorize spawn errors and return (category, is_retryable).
    category: "parse", "attribute", "validation", "taskwarrior", "unknown"
    is_retryable: whether we should retry this attempt
    """
    stderr_lower = (stderr or "").lower()

    if returncode == 0:
        return ("success", False)

    # Unrecognized attribute - NOT retryable, just strip and retry
    if "unrecognized attribute" in stderr_lower:
        return ("attribute", True)

    # JSON parsing errors - likely malformed task, NOT retryable
    if "json" in stderr_lower or "parse" in stderr_lower:
        return ("parse", False)

    # Validation errors (e.g., bad due date format) - NOT retryable
    if "invalid" in stderr_lower or "bad date" in stderr_lower:
        return ("validation", False)

    # Taskwarrior internal errors - possibly retryable
    if "error" in stderr_lower or "failed" in stderr_lower:
        return ("taskwarrior", True)

    return ("unknown", True)


def _spawn_intent_entry(
    parent_uuid: str,
    child_obj: dict,
    child_short: str,
    parent_nextlink: str | None = None,
    spawn_intent_id: str | None = None,
    parent_guard: dict | None = None,
    lifecycle_plan: dict | None = None,
) -> dict:
    intent_id = (spawn_intent_id or "").strip()
    if not intent_id:
        intent_id = f"si_{uuid.uuid4().hex[:12]}"
    queue_models = _module("queue_models")
    return queue_models.normalize_spawn_queue_entry(
        {
            "parent_uuid": parent_uuid,
            "parent_nextlink": (parent_nextlink or "").strip(),
            "child_short": child_short,
            "child": child_obj,
            "spawn_intent_id": intent_id,
            "parent_guard": parent_guard,
            "lifecycle_plan": lifecycle_plan,
        }
    )


def _enqueue_spawn_intent(entry: dict) -> tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, "invalid spawn intent"
    queue_models = _module("queue_models")
    try:
        normalized = queue_models.normalize_spawn_queue_entry(entry)
    except Exception as e:
        return False, str(e)
    return _enqueue_deferred_spawn(normalized, require_lifecycle_plan=True)


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
    child_task: dict,
    parent_task_with_nextlink: dict,
) -> tuple[str, set[str], bool, bool, str | None, str | None]:
    """
    Queue a child spawn intent for the on-exit hook.

    Important: The parent update is applied by Taskwarrior using this hook's stdout.
    We intentionally avoid importing the parent from inside the hook to reduce the
    risk of re-entering Taskwarrior while it is holding the datastore lock.

    We enqueue the child for the on-exit hook to import, then update the parent link.
    """
    env = os.environ.copy()
    modify_spawn_prep = _module("modify_spawn_prep")
    child_obj, child_uuid, child_short = modify_spawn_prep.prepare_spawn_child_payload(
        child_task,
        parent_task_with_nextlink,
        env,
        child_uuid_for_spawn=_child_uuid_for_spawn,
        fmt_isoz=core.fmt_isoz,
        now_utc=core.now_utc,
        strip_none_and_cast=_strip_none_and_cast,
        normalise_datetime_fields=_normalise_datetime_fields,
    )

    stripped_attrs: set[str] = set()
    last_stderr = ""
    last_category = "unknown"

    # Decision-only mode: enqueue for on-exit spawn and return unverified.
    lifecycle_models = _module("lifecycle_models")
    lifecycle_identity = _lifecycle_spawn_identity(parent_task_with_nextlink, child_obj)
    spawn_intent_id = lifecycle_identity.idempotency_key
    recurrence_guard = lifecycle_models.recurrence_fingerprint(
        parent_task_with_nextlink,
        parse_datetime=getattr(core, "parse_dt_any", None),
    )
    parent_guard = {
        "status": parent_task_with_nextlink.get("status") or "",
        "chain": parent_task_with_nextlink.get("chain") or "",
        "chainID": parent_task_with_nextlink.get("chainID") or "",
        "link": parent_task_with_nextlink.get("link") or "",
        "modified": parent_task_with_nextlink.get("modified") or "",
        "recurrence_fingerprint": recurrence_guard,
    }
    lifecycle_plan = lifecycle_models.LifecyclePlan.from_mappings(
        identity=lifecycle_identity,
        action=lifecycle_models.LifecycleAction.SPAWN_CHILD,
        parent_guard=lifecycle_models.ParentGuard.from_mapping(parent_guard),
        child_payload=child_obj,
        parent_patch={"nextLink": child_short},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    ).to_dict()
    entry = _spawn_intent_entry(
        parent_task_with_nextlink.get("uuid") or "",
        child_obj,
        child_short,
        parent_task_with_nextlink.get("nextLink") or "",
        spawn_intent_id,
        parent_guard=parent_guard,
        lifecycle_plan=lifecycle_plan,
    )
    queued, queue_reason = _enqueue_spawn_intent(entry)
    if not queued:
        return (
            child_short,
            stripped_attrs,
            False,
            False,
            f"Spawn intent queue failed: {queue_reason}",
            spawn_intent_id,
        )
    _diag_count("spawn_deferred")
    return (
        child_short,
        stripped_attrs,
        False,
        True,
        "Spawn intent queued for on-exit processing",
        spawn_intent_id,
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
    if env is None:
        rows = _lifecycle_read_service().get_chain_export(chain_id)
        if rows is None:
            raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
        return rows
    ok, rows, error = _tw_export_chain_checked(chain_id, env=env, limit=_MAX_CHAIN_WALK)
    if not ok:
        raise RuntimeError(error or f"Chain export unavailable for chainID {chain_id}")
    return rows
def _tw_get_cached(ref: str) -> str:
    """Return `task _get <ref>` stdout stripped. Cached within one hook run."""
    try:
        if ref.endswith(".entry"):
            short = ref[:-6].strip()
            cached, cache_chain_id = (
                _lifecycle_read_service().lookup_short(short) if short else (None, "")
            )
            if short and isinstance(cached, dict):
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





def _validate_anchor_on_modify(expr: str):
    """Mirror on-add strict checks for anchor; raise ValueError on problems."""
    if not expr or not expr.strip():
        raise ValueError("anchor is required if chaining by anchor")

    # Syntax first (for friendlier messages)
    try:
        dnf_raw = core.parse_anchor_expr_to_dnf(expr)
    except Exception as e:
        raise ValueError(f"anchor syntax error: {str(e)}")


    # NOTE: legacy weekday ':' syntax is accepted for backward compatibility.

    # Strict validation
    try:
        _validate_anchor_expr_cached(expr)  # calls core.validate_anchor_expr_strict
    except Exception as e:
        raise ValueError(f"anchor validation failed: {str(e)}")


def _validate_omit_on_modify(expr: str):
    if not expr or not expr.strip():
        return
    try:
        _validate_omit_expr_cached(expr)
    except Exception as e:
        raise ValueError(f"omit validation failed: {str(e)}")


def _validate_cp_on_modify(cp_str: str, chain_max_val, chain_until_val):
    """
    Mirror on-add CP checks for a plain modify:
      - cp must parse as a duration
      - optional chainMax must be a positive integer
      - optional chainUntil must parse as a datetime
    """
    if not cp_str or not cp_str.strip():
        return  # nothing to validate

    seq = core.parse_cp_sequence(cp_str)
    if not seq:
        reason = core.cp_sequence_parse_error(cp_str) or f"invalid duration format '{cp_str}'"
        raise ValueError(f"{reason} (expected: 3d, 2w, 1h, etc.)")

    # chainMax
    add_validation = core._import_sibling("add_validation")
    _cpmax, chain_max_err = add_validation.parse_chain_max(chain_max_val)
    if chain_max_err:
        raise ValueError(chain_max_err)

    # chainUntil
    cu = (chain_until_val or "").strip()
    if cu:
        dt = core.parse_dt_any(cu)
        if dt is None:
            raise ValueError(f"Invalid chainUntil '{cu}'")


def _validate_chain_limits_on_modify(task: dict) -> None:
    add_validation = core._import_sibling("add_validation")
    cpmax, chain_max_err = add_validation.parse_chain_max(task.get("chainMax"))
    if chain_max_err:
        _fail_and_exit("Invalid chainMax", chain_max_err)
    if cpmax is not None:
        task["chainMax"] = cpmax

    chain_until = str(task.get("chainUntil") or "").strip()
    if not chain_until:
        return
    until_dt = core.parse_dt_any(chain_until)
    if until_dt is None:
        _fail_and_exit("Invalid chainUntil", f"Unrecognized datetime format '{chain_until}'")
    is_valid, until_err = _validate_until_not_past(until_dt, core.now_utc())
    if not is_valid:
        _fail_and_exit("Invalid chainUntil", until_err or "chainUntil is in the past")


def _validate_native_until_after_target_or_fail(task: dict) -> None:
    until_raw = task.get("until")
    if not until_raw:
        return
    add_validation = core._import_sibling("add_validation")
    mode_is_valid, mode_reason = add_validation.validate_native_until_anchor_mode(
        until_raw,
        task.get("anchor"),
        task.get("anchor_file"),
        task.get("anchor_mode"),
    )
    if not mode_is_valid:
        mode = str(task.get("anchor_mode") or "skip").strip().lower()
        _panel(
            "❌ Invalid expiration mode",
            [
                ("Mode", mode),
                ("Conflict", mode_reason or "Native until conflicts with strict anchor backfill."),
                ("Action", "Remove until or use anchor_mode:skip."),
            ],
            kind="error",
        )
        sys.exit(1)
    target_field = "due" if task.get("due") else "scheduled" if task.get("scheduled") else ""
    if not target_field:
        return
    target_dt, target_err = _safe_parse_datetime(task.get(target_field))
    if target_err or target_dt is None:
        _fail_and_exit(f"Invalid {target_field}", target_err or f"{target_field} must be a valid datetime")
    until_dt, until_err = _safe_parse_datetime(until_raw)
    if until_err or until_dt is None:
        _fail_and_exit("Invalid until", until_err or "until must be a valid datetime")
    is_valid, reason = add_validation.validate_native_until_after_target(
        until_dt,
        target_dt,
        target_field,
    )
    if is_valid:
        return
    label = "Scheduled" if target_field == "scheduled" else "Due"
    _panel(
        "❌ Invalid expiration window",
        [
            (label, core.fmt_dt_local(target_dt)),
            ("Expires", core.fmt_dt_local(until_dt)),
            ("Required", reason or f"until must be later than {target_field}"),
        ],
        kind="error",
    )
    sys.exit(1)


def _validate_native_until_anchor_slots_or_fail(task: dict) -> None:
    until_raw = task.get("until")
    anchor_value = str(task.get("anchor") or "").strip()
    anchor_file_value = str(task.get("anchor_file") or "").strip()
    if not until_raw or not (anchor_value or anchor_file_value):
        return
    target_field = "due" if task.get("due") else "scheduled" if task.get("scheduled") else ""
    if not target_field:
        return
    target_dt, target_err = _safe_parse_datetime(task.get(target_field))
    until_dt, until_err = _safe_parse_datetime(until_raw)
    if target_err or until_err or target_dt is None or until_dt is None:
        return
    dnf = None
    if anchor_value:
        try:
            dnf = _validate_anchor_expr_cached(anchor_value)
        except Exception:
            return
    add_validation = core._import_sibling("add_validation")
    target_local = _tolocal(target_dt)
    try:
        slots = add_validation.collect_anchor_time_slots(
            dnf,
            anchor_file_value,
            (target_local.hour, target_local.minute),
            normalize_time_slots=_norm_hhmm_list,
            anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
            target_date=target_local.date(),
            resolve_time_slots=lambda value, target_date: _norm_hhmm_list(value, target_date),
            recurrence_context=core._import_sibling("recurrence_context").RecurrenceContext.from_task(task),
        )
    except Exception as exc:
        astronomy = core._import_sibling("astronomy")
        if astronomy.is_astronomy_error(exc):
            _panel(
                "❌ Invalid astronomy time",
                [("Required", astronomy.scheduling_error_message(exc))],
                kind="error",
            )
            sys.exit(1)
        return
    is_valid, reason = add_validation.validate_native_until_calendar_slots(
        until_dt,
        target_dt,
        slots,
        to_local=_tolocal,
    )
    if is_valid:
        return
    _panel(
        "❌ Invalid expiration window",
        [
            ("Expires", core.fmt_dt_local(until_dt)),
            ("Anchor slots", ", ".join(f"{hh:02d}:{mm:02d}" for hh, mm in slots) or "none"),
            ("Required", reason or "calendar expiration must be later than every anchor slot"),
        ],
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


def _collect_prev_two(current_task: dict, chain_by_link: dict[int, list[dict]] | None = None) -> list[dict]:
    return _lifecycle_read_service().collect_prev_two(
        current_task,
        get_chain_export=lambda chain_id: _lifecycle_read_service().get_chain_export(chain_id),
        panel_chain_by_link=_modify_chain_state().panel_chain_by_link,
        panel_chain_snapshot_loaded=_modify_chain_state().panel_chain_snapshot_loaded,
        chain_by_link=chain_by_link,
    )


def _chain_export_for_cache(
    chain_id: str,
    since: datetime | None,
    extra: str | None,
    limit: int,
) -> tuple[dict, ...]:
    """Validated exporter injected into the lifecycle read service cache."""
    global _LAST_CHAIN_EXPORT_STATUS
    _LAST_CHAIN_EXPORT_STATUS = (True, "")
    rows = tw_export_chain(chain_id, since=since, extra=extra, env=None, limit=limit)
    ok, error = _LAST_CHAIN_EXPORT_STATUS
    if not ok:
        raise RuntimeError(error or "chain export unavailable")
    return tuple(rows)

def _cached_chain_token_match(task: dict, token: str) -> bool:
    if not isinstance(task, dict) or not isinstance(token, str) or not token:
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
        return existing
    lifecycle_read_service = _module("lifecycle_read_service")
    if getattr(state, "chain_cache_store", None) is None:
        state.chain_cache_store = lifecycle_read_service.ChainCacheStore()

    service = lifecycle_read_service.LifecycleReadService(
        coerce_int=core.coerce_int,
        parse_extra_tokens=_parse_extra_tokens,
        token_matcher=_cached_chain_token_match,
        read_query_get=_read_query_get,
        read_query_set=_read_query_set,
        read_query_delete=_read_query_delete,
        chain_cache_get=lambda _chain_id: None,
        export_chain_cached=_chain_export_for_cache,
        max_chain_walk=_MAX_CHAIN_WALK,
        diag=_diag,
        record_stat=_record_chain_snapshot_stat,
        cache_store=state.chain_cache_store,
        task_cmd_prefix=_task_cmd_prefix,
        read_query_missing=_READ_QUERY_MISSING,
    )
    state.lifecycle_read_service = service
    return service


def _existing_next_lookup(parent_task: dict, next_no: int, chain_snapshot=None):
    hook_support = _module("hook_support", required=False)
    if getattr(chain_snapshot, "is_unavailable", False) and hook_support is not None:
        return hook_support.LookupResult.unavailable(
            getattr(chain_snapshot, "error", "completion chain snapshot unavailable")
        )
    return _lifecycle_read_service().existing_next_lookup(
        parent_task,
        next_no,
        export_uuid_short_cached=_export_uuid_short_lookup,
        get_chain_export=lambda chain_id, **kwargs: _lifecycle_read_service().get_chain_export(
            chain_id, **kwargs
        ),
        snapshot_rows=getattr(chain_snapshot, "rows", None),
        snapshot_loaded=bool(getattr(chain_snapshot, "loaded", False)),
    )


def _seed_runtime_lookup_task(task: dict | None, *, lookup_short: str | None = None) -> dict | None:
    if not isinstance(task, dict):
        return None
    uuid_str = str(task.get("uuid") or "").strip()
    if not uuid_str:
        return None
    short = uuid_str[:8]
    service = _lifecycle_read_service()
    task_obj = service.seed_lookup_task(dict(task), short_uuid=short)
    requested_short = str(lookup_short or "").strip()
    if requested_short and requested_short != short:
        task_obj = service.seed_lookup_task(task_obj, short_uuid=requested_short)
    entry = task_obj.get("entry")
    if short and entry:
        _query_ctx_set("tw_get", f"{short}.entry", str(entry).strip())
    return task_obj


def _seed_runtime_lookup_tasks(*tasks: dict | None) -> None:
    for task in tasks:
        _seed_runtime_lookup_task(task)


# ------------------------------------------------------------------------------
# Multi-time occurrence helpers (hook-level)
# ------------------------------------------------------------------------------

def _recurrence_seed_base(task: dict) -> str:
    """Resolve the task recurrence identity once for preview-like paths."""
    context = core._import_sibling("recurrence_context").RecurrenceContext.from_task(
        task,
        fallback_chain_id="preview",
    )
    return context.seed_base

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
    state = _modify_runtime_state()
    identity = str(task.get("uuid") or task.get("chainID") or "").strip()
    if identity:
        cache_key = (
            "task",
            identity,
            str(task.get("modified") or ""),
            str(task.get("anchor") or ""),
            str(task.get("anchor_file") or ""),
            str(task.get("omit") or ""),
            str(task.get("omit_file") or ""),
            str(task.get("cp") or ""),
            str(task.get("anchor_mode") or ""),
            str(task.get("chainMax") or ""),
            str(task.get("chainUntil") or ""),
            str(task.get("bc") or ""),
        )
    else:
        cache_key = ("object", id(task))
    cached = state.evaluator_sessions.get(cache_key)
    if cached is not None:
        _diag_count("evaluator_session_hits")
        return cached[1]
    _diag_count("evaluator_session_misses")
    evaluator_module = _module("recurrence_evaluator")
    evaluator = evaluator_module.RecurrenceEvaluator.from_task(
        task,
        fallback_chain_id=_recurrence_seed_base(task),
        timezone=core._LOCAL_TZ,
        business_calendar=core.business_calendar_for_task(task),
        astronomy_config=getattr(core, "ASTRONOMY_CONFIG", None),
        anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
    )
    state.evaluator_sessions[cache_key] = (task, evaluator)
    return evaluator


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
    anchor_inclusion = core._import_sibling("anchor_inclusion")
    occurrence_provider = core._import_sibling("occurrence_provider")
    evaluator = _recurrence_evaluator_for_task(parent)
    anchor_file = (parent.get("anchor_file") or "").strip()
    if anchor_file_provider is None:
        anchor_file_provider = (
            anchor_inclusion._build_anchor_file_provider(
                anchor_file,
                anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
                fallback_hhmm=fallback_hhmm,
                seed_base=seed_base,
                core=core,
            )
            if anchor_file
            else None
        )
    provider = occurrence_provider.AnchorOccurrenceProvider(
        lambda value: anchor_inclusion.next_included_occurrence_local(
            dnf=dnf,
            anchor_file_str=anchor_file,
            after_local_dt=value,
            inclusive=False,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=evaluator._default_next_occurrence_after_local_dt,
            anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
            anchor_file_provider=anchor_file_provider,
        ),
    )
    return [
        occurrence.local_datetime
        for occurrence in occurrence_provider.collect_after(
            provider,
            after_local_dt,
            limit=limit,
            inclusive=inclusive,
            build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
            to_local=lambda value: value,
        )
        if occurrence.local_datetime is not None
    ]


def _estimate_cp_final_by_max(task: dict, next_due_utc):
    """
    Estimate the final due date when chainMax cap is reached.
    Returns the due datetime of link #chainMax.
    """
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if not cpmax:
        return None

    cur_no = core.coerce_int(task.get("link"), 1)
    if cur_no >= cpmax:
        return None

    cp_str = task.get("cp") or ""
    tokens = core.parse_cp_sequence_tokens(cp_str)
    if not tokens:
        return None

    fut_dt = next_due_utc
    fut_no = cur_no + 1
    iterations = 0

    # Step forward from next due until we reach cap_no
    while fut_no < cpmax:
        iterations += 1
        if iterations > _MAX_ITERATIONS:
            _diag(
                f"chainMax forecast stopped after {_MAX_ITERATIONS} occurrences; "
                "final date is unavailable"
            )
            return None
        td = _cp_sequence_period_for_link(
            tokens,
            cp_str,
            fut_no,
            str(task.get("chainID") or "").strip(),
        )
        fut_no += 1
        fut_dt = _cp_add_td(fut_dt, td)

    return fut_dt


def _estimate_anchor_final_by_max(task: dict, next_due_utc, dnf):
    """
    Estimate the final due date when chainMax cap is reached for anchors.
    Returns the due datetime of link #chainMax.
    """
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if not cpmax:
        return None

    cur_no = core.coerce_int(task.get("link"), 1)
    if cur_no >= cpmax:
        return None

    seed_base = _recurrence_seed_base(task)
    nxt_local = _to_local_cached(next_due_utc)

    # Use a stable default seed (prefer the original due date).
    due0, _ = _safe_parse_datetime(task.get("due"))
    default_seed = _to_local_cached(due0 or next_due_utc).date()

    fallback_hhmm = _anchor_file_fallback_hhmm(task, nxt_local)
    _omit_expr, omit_dnf = _omit_dnf_from_parent(task)
    scheduler = _recurrence_evaluator_for_task(task)._default_next_occurrence_after_local_dt
    anchor_file = (task.get("anchor_file") or "").strip()
    anchor_file_provider = None
    if anchor_file:
        anchor_file_provider = _anchor_file_provider_for(
            anchor_file, fallback_hhmm=fallback_hhmm, seed_base=seed_base
        )
    fut_no = cur_no + 1
    fut_local = nxt_local
    iterations = 0
    while fut_no < cpmax:
        iterations += 1
        if iterations > _MAX_ITERATIONS:
            _diag(
                f"chainMax forecast stopped after {_MAX_ITERATIONS} occurrences; "
                "final date is unavailable"
            )
            return None
        if anchor_file:
            future = _anchor_included_occurrences(
                task,
                after_local_dt=fut_local,
                inclusive=False,
                limit=2,
                fallback_hhmm=fallback_hhmm,
                omit_dnf=omit_dnf,
                seed_base=seed_base,
                default_seed_date=default_seed,
                dnf=dnf,
                anchor_file_provider=anchor_file_provider,
            )
            fut_local = future[0] if future else None
        else:
            fut_local = scheduler(
                dnf,
                fut_local,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                fallback_hhmm=fallback_hhmm,
            )
        if fut_local is None:
            return None
        fut_no += 1

    return fut_local.astimezone(timezone.utc)


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


# Helper to warn if chain extends too far into future
def _validate_chain_duration_reasonable(
    child_due: datetime, until_dt: datetime, now_utc: datetime
) -> tuple[bool, str | None]:
    """
    Warn if chain will extend unreasonably far into the future.
    Returns (is_reasonable, warning_msg).
    """
    if not until_dt:
        return (True, None)

    span = until_dt - now_utc
    days = span.days

    if days > _MIN_FUTURE_WARN:
        years = days / 365.25
        return (
            True,
            f"Chain extends {years:.1f} years into future (until {core.fmt_dt_local(until_dt)})",
        )

    return (True, None)


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
def _median(nums: list[float]) -> float | None:
    if not nums:
        return None
    s = sorted(nums)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _lateness_stats(chain: list[dict], tol_secs: int = 60) -> dict:
    early = on = late = 0
    deltas = []
    best = None
    worst = None
    for obj in chain:
        due = _dtparse(obj.get("due"))
        end = _dtparse(obj.get("end"))
        if not (due and end):
            continue
        diff = (end - due).total_seconds()
        deltas.append(diff)
        if diff > tol_secs:
            late += 1
            worst = diff if (worst is None or diff > worst) else worst
        elif diff < -tol_secs:
            early += 1
            best = diff if (best is None or diff < best) else best
        else:
            on += 1
    avg = (sum(deltas) / len(deltas)) if deltas else None
    med = _median(deltas) if deltas else None
    return {
        "early": early,
        "on_time": on,
        "late": late,
        "avg": avg,
        "median": med,
        "best_early": best,
        "worst_late": worst,
        "count": len(deltas),
    }


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
    if not chain:
        return []

    # Get link number for sorting - handle tasks without link numbers
    def get_link(obj):
        link = obj.get("link")
        if link is None or link == "":
            return -1  # Put tasks without links at the beginning
        return core.coerce_int(link, 999999)

    # Sort by link number descending (most recent first)
    # But tasks without links (link=-1) should go at the end (oldest)
    chain_sorted = sorted(chain, key=get_link, reverse=True)

    # Filter out tasks without link numbers for display (they're usually root tasks)
    chain_with_links = [t for t in chain_sorted if get_link(t) > 0]

    # Determine max link number for formatting (only from tasks with links)
    if chain_with_links:
        max_link = max(get_link(obj) for obj in chain_with_links)
        label_width = len(str(max_link)) + 1  # +1 for the # symbol
    else:
        label_width = 4  # default width

    def history_line(obj: dict, no: int) -> str:
        end = _dtparse(obj.get("end"))
        due = _dtparse(obj.get("due"))
        is_deleted = str(obj.get("status") or "").strip().lower() == "deleted"
        if is_deleted and not end:
            end_s = "deleted"
            delta = ""
            marker = "×"
        else:
            end_s = _fmtlocal(end) if end else "(no end)"
            delta = _fmt_on_time_delta(due, end)
            marker = "✓"
        short = _short(obj.get("uuid"))
        lab = f"[bold]#{no:<{label_width}}[/]"
        return f"{lab} {marker:<2} {end_s} {delta} [dim]{short}[/]"

    # If chain has more than 10 tasks, show top 3 (most recent) and bottom 3 (oldest)
    if len(chain_with_links) > 10:
        # Top 3: most recent tasks (highest link numbers)
        top_tasks = chain_with_links[:3]

        # Bottom 3: oldest tasks (lowest link numbers)
        bottom_tasks = chain_with_links[-3:]  # Already in descending order (e.g., [3, 2, 1])

        # Create lines for top tasks (most recent)
        top_lines = []
        for obj in top_tasks:
            no = get_link(obj)
            line = history_line(obj, no)
            # Highlight the most recent task
            if no == get_link(chain_with_links[0]):
                line = f"[green]{line}[/]"
            top_lines.append(line)

        # Add ellipsis
        ellipsis_line = f"[dim]{' ' * (label_width + 4)}... ({len(chain_with_links) - 6} more tasks) ...[/dim]"

        # Create lines for bottom tasks (oldest) - also in descending order
        bottom_lines = []
        for obj in bottom_tasks:  # Already in descending order (e.g., 3, 2, 1)
            no = get_link(obj)
            bottom_lines.append(history_line(obj, no))

        return top_lines + [ellipsis_line] + bottom_lines

    # For chains with <= 10 tasks, show all in reverse order (most recent at top)
    lines = []
    for obj in chain_with_links[:n]:
        no = get_link(obj)
        line = history_line(obj, no)
        # Highlight the most recent task
        if no == get_link(chain_with_links[0]):
            line = f"[green]{line}[/]"
        lines.append(line)

    return lines

def _end_summary_current(current: dict, current_task: dict | None) -> dict:
    return current_task if current_task else current


def _end_summary_chain_id_row(actual_current: dict) -> str:
    return (actual_current.get("chainID") or "").strip()


def _end_summary_sorted_chain(chain_id: str, actual_current: dict) -> list[dict]:
    chain = tw_export_chain_required(actual_current)
    if actual_current and chain:
        for i, task in enumerate(chain):
            if task.get("uuid") == actual_current.get("uuid"):
                chain[i] = actual_current
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
    first_task = chain[0] if chain else None
    last_task = chain[-1] if chain else None
    if not first_task and chain_id:
        first_task = _export_chain_endpoint(chain_id, "first")
    if not last_task and chain_id:
        last_task = _export_chain_endpoint(chain_id, "last")
    first = _dtparse((first_task or {}).get("due")) if first_task else None
    last = _dtparse((last_task or {}).get("end")) if last_task else None
    span = "–"
    if first and last:
        span = (
            _human_delta(first, last, prefer_months=True)
            .replace("in ", "")
            .replace("overdue by ", "")
        )
    elif first and stop_at and stopped_by_delete:
        active = (
            _human_delta(first, stop_at, prefer_months=True)
            .replace("in ", "")
            .replace("overdue by ", "")
        )
        span = f"Active for {active} before deletion"
    return first, last, span


def _end_summary_kind_rows(rows: list[tuple[str, str]], kind: str, current: dict) -> None:
    if kind == "anchor":
        expr = (current.get("anchor") or "").strip()
        mode = (current.get("anchor_mode") or "skip").lower()
        tag = {
            "skip": "[cyan]SKIP[/]",
            "all": "[yellow]ALL[/]",
            "flex": "[magenta]FLEX[/]",
        }.get(mode, "[cyan]SKIP[/]")
        try:
            preset_display = core.anchor_preset_display(expr)
        except Exception:
            preset_display = None
        if preset_display:
            label, text = preset_display
            rows.append((label, f"{text}  {tag}"))
        else:
            rows.append(("Pattern", f"{expr}  {tag}"))
        try:
            dnf = _validate_anchor_expr_cached(expr)
            rows.append(("Natural", core.describe_anchor_dnf(dnf, current)))
        except Exception:
            pass
        return
    if kind == "anchor_file":
        expr = (current.get("anchor_file") or "").strip()
        mode = (current.get("anchor_mode") or "skip").lower()
        tag = {
            "skip": "[cyan]SKIP[/]",
            "all": "[yellow]ALL[/]",
            "flex": "[magenta]FLEX[/]",
        }.get(mode, "[cyan]SKIP[/]")
        rows.append(("Anchor file", f"{expr}  {tag}"))
        rows.append(("Natural", f"Dates from {expr.split('@', 1)[0]}"))
        return
    rows.append(("Period", current.get("cp") or "–"))


def _end_summary_stats_rows(rows: list[tuple[str, str]], chain: list[dict], now_utc) -> None:
    stats = _lateness_stats(chain)
    rows.append(
        (
            "Performance",
            f"early {stats['early']}, on-time {stats['on_time']}, late {stats['late']}",
        )
    )
    rows.append(("Avg lateness", _fmt_secs_delta(now_utc, stats["avg"])))
    rows.append(("Median lateness", _fmt_secs_delta(now_utc, stats["median"])))
    rows.append(("Best early", _fmt_secs_delta(now_utc, stats["best_early"])))
    rows.append(("Worst late", _fmt_secs_delta(now_utc, stats["worst_late"])))


def _end_summary_limits_row(rows: list[tuple[str, str]], current: dict) -> None:
    cpmax = core.coerce_int(current.get("chainMax"), 0)
    until = _dtparse(current.get("chainUntil"))
    if cpmax:
        rows.append(("Chain cap", f"#{cpmax}"))
    if until:
        rows.append(("Chain end point", core.fmt_dt_local(until)))
    if not cpmax and not until:
        rows.append(("Chain limits", "None"))


def _end_chain_summary(current: dict, reason: str, now_utc, current_task: dict = None) -> None:
    actual_current = _end_summary_current(current, current_task)
    kind_anchor = bool((actual_current.get("anchor") or "").strip())
    kind_anchor_file = bool((actual_current.get("anchor_file") or "").strip())
    kind = "anchor" if kind_anchor else ("anchor_file" if kind_anchor_file else "cp")

    chain_id = _end_summary_chain_id_row(actual_current)
    if not chain_id:
        _panel(
            "⚠ Chain summary skipped",
            [
                ("Reason", "ChainID is required in v3+ and legacy link-walk is removed."),
                ("Fix", "Run dev_tools/nautical_backfill_chainid.py."),
            ],
            kind="warning",
        )
        return

    chain_read_error = ""
    try:
        chain = _end_summary_sorted_chain(chain_id, actual_current)
    except Exception as exc:
        chain = []
        chain_read_error = str(exc) or "chain export unavailable"
        _diag(f"chain summary export unavailable (chainID={chain_id}): {chain_read_error}")

    L = core.coerce_int(current.get("link"), len(chain))
    root = _short(_root_uuid_from(current))
    cur_s = _short(current.get("uuid"))
    stopped_by_delete = str(reason or "").strip().lower().startswith("pending task deleted")
    first, last, span = _end_summary_span_fields(
        chain_id,
        chain,
        stop_at=now_utc,
        stopped_by_delete=stopped_by_delete,
    )

    rows = []
    rows.append(("Reason", reason))
    rows.append(("Root", _format_root_and_age(current, now_utc)))

    chain_display = f"{root} … {cur_s}  [dim](#{L}, {len(chain)} tasks"
    if len(chain) >= _MAX_CHAIN_WALK:
        chain_display += f", truncated at {_MAX_CHAIN_WALK})"
    else:
        chain_display += ")"
    rows.append(("Chain", chain_display))
    if chain_read_error:
        rows.append(("Chain read", f"Unavailable: {chain_read_error}"))

    _end_summary_kind_rows(rows, kind, current)

    if first:
        rows.append(("First due", core.fmt_dt_local(first)))
    if last:
        rows.append(("Last end", core.fmt_dt_local(last)))
    if stopped_by_delete:
        rows.append(("Stopped at", core.fmt_dt_local(now_utc)))
    rows.append(("Span", span))

    _end_summary_stats_rows(rows, chain, now_utc)
    _end_summary_limits_row(rows, current)

    tail = _last_n_timeline(chain, n=6)
    if tail:
        rows.append(("History", "\n".join(tail)))

    rows = _format_chain_summary_rows(rows)
    title = "⛔ Chain stopped – summary" if stopped_by_delete else "⛔ Chain finished – summary"
    _panel(title, rows, kind="summary")



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
    """Compact timeline with inline gaps."""
    if not _require_core():
        return []
    if kind == "anchor_file" or (kind == "anchor" and (task.get("anchor_file") or "").strip()):
        modify_timeline = _module("modify_timeline")
        _omit_expr, omit_dnf = _omit_dnf_from_parent(task)
        anchor_omit = _module("anchor_omit") if omit_dnf else None
        seed_base = _recurrence_seed_base(task)
        child_local = _to_local_cached(child_due_utc)
        fallback_hhmm = _anchor_file_fallback_hhmm(task, child_local)
        default_seed = child_local.date()
        dnf_for_merge = dnf if kind == "anchor" else None
        anchor_inclusion = core._import_sibling("anchor_inclusion")
        evaluator = _recurrence_evaluator_for_task(task)
        AnchorEventOccurrenceProvider = core._import_sibling("occurrence_provider").AnchorEventOccurrenceProvider
        collect_after = core._import_sibling("occurrence_provider").collect_after
        anchor_file_str = (task.get("anchor_file") or "").strip()
        anchor_file_provider = (
            _anchor_file_provider_for(
                anchor_file_str, fallback_hhmm=fallback_hhmm, seed_base=seed_base
            )
            if anchor_file_str
            else None
        )

        event_provider = AnchorEventOccurrenceProvider(
            lambda value: anchor_inclusion.next_occurrence_event_local(
                dnf=dnf_for_merge,
                anchor_file_str=anchor_file_str,
                after_local_dt=value,
                inclusive=False,
                fallback_hhmm=fallback_hhmm,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                core=core,
                next_occurrence_after_local_dt=evaluator._default_next_occurrence_after_local_dt,
                scheduler_omit_dnf=None,
                anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
                anchor_file_provider=anchor_file_provider,
            ),
            source="anchor+anchor_file" if anchor_file_str and dnf else ("anchor_file" if anchor_file_str else "anchor"),
        )
        projection_warning = None
        try:
            events = [
                occurrence
                for occurrence in collect_after(
                    event_provider,
                    child_local,
                    limit=max(8, next_count + 6),
                    inclusive=True,
                    max_iterations=_MAX_ITERATIONS,
                    build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                    to_local=lambda value: value,
                )
                if occurrence.local_datetime is not None
            ]
        except Exception as exc:
            events = []
            projection_warning = modify_timeline._timeline_warning(
                f"Projection unavailable: {type(exc).__name__}: {exc}"
            )
        cur_no = core.coerce_int(task.get("link") if cur_no is None else cur_no, 1)
        nxt_no = cur_no + 1
        allowed_future = next_count if cap_no is None else max(0, min(next_count, cap_no - nxt_no))
        prev_style, cur_style, next_style, future_style = modify_timeline._timeline_styles(
            task,
            "anchor",
            future_style_for_chain=_future_style_for_chain,
        )
        items = modify_timeline._timeline_initial_items(
            task,
            cur_no,
            nxt_no,
            child_due_utc,
            child_short,
            core=core,
            collect_prev_two=_collect_prev_two,
            dtparse=_dtparse,
        )
        if projection_warning is not None:
            items.append(projection_warning)
        fut_no = nxt_no
        actual_future = 0
        for occurrence in events:
            item_local = occurrence.local_datetime
            is_omitted = occurrence.omitted
            if item_local is None:
                continue
            item_utc = item_local.astimezone(timezone.utc)
            if _compare_datetimes(item_utc, child_due_utc) <= 0:
                continue
            if is_omitted:
                items.append(
                    (
                        "··",
                        item_utc,
                        {
                            "is_omit": True,
                            "omit_label": (
                                modify_timeline._timeline_omit_label(
                                    omit_dnf,
                                    item_local.date(),
                                    omit_description_for_date=(
                                        anchor_omit.omit_description_for_date if anchor_omit is not None else None
                                    ),
                                )
                                if omit_dnf else None
                            ),
                        },
                        "omitted",
                    )
                )
                continue
            fut_no += 1
            if cap_no is not None and fut_no > cap_no:
                break
            items.append((fut_no, item_utc, {"is_future": True}, "future"))
            actual_future += 1
            if actual_future >= allowed_future:
                break
        lines: list[str] = []
        for i, (no, dt, obj, item_type) in enumerate(items):
            base_line = modify_timeline._timeline_base_line(
                no,
                dt,
                obj,
                item_type,
                task=task,
                cap_no=cap_no,
                prev_style=prev_style,
                cur_style=cur_style,
                next_style=next_style,
                future_style=future_style,
                core=core,
                dtparse=_dtparse,
                fmt_on_time_delta=_fmt_on_time_delta,
                fmtlocal=_fmtlocal,
                short=_short,
            )
            lines.append(
                modify_timeline._timeline_with_gap(
                    base_line,
                    idx=i,
                    items=items,
                    show_gaps=show_gaps,
                    kind="anchor",
                    round_anchor_gaps=round_anchor_gaps,
                    format_gap=_module("modify_timeline").format_gap,
                )
            )
        return lines
    modify_timeline = _module("modify_timeline")
    _omit_expr, omit_dnf = _omit_dnf_from_parent(task) if kind == "anchor" else ("", None)
    anchor_omit = _module("anchor_omit") if kind == "anchor" else None
    evaluator = _recurrence_evaluator_for_task(task) if kind in {"anchor", "cp"} else None
    timeline_scheduler = _next_occurrence_after_local_dt
    if evaluator is not None and kind == "anchor":
        timeline_scheduler = evaluator._default_next_occurrence_after_local_dt
    return modify_timeline.timeline_lines(
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
        next_occurrence_after_local_dt=timeline_scheduler,
        to_local_cached=_to_local_cached,
        safe_parse_datetime=_safe_parse_datetime,
        format_gap=_module("modify_timeline").format_gap,
        omit_dnf=omit_dnf,
        omit_expr_fires_on_date=(
            (lambda dnf_, d, default_seed, seed_base: anchor_omit.omit_expr_fires_on_date(dnf_, d, default_seed, seed_base, core=core))
            if anchor_omit is not None else None
        ),
        omit_description_for_date=(anchor_omit.omit_description_for_date if anchor_omit is not None else None),
        evaluator=evaluator,
    )

def _got_anchor_invalid(msg: str) -> None:
    _fail_and_exit("Invalid anchor", msg)


# chainUntil -> numeric cap and final permitted occurrence
def _cap_from_until_cp(task, next_due_utc):
    until = _dtparse(task.get("chainUntil"))
    if not until:
        return (None, None)
    cp_str = task.get("cp") or ""
    tokens = core.parse_cp_sequence_tokens(cp_str)
    if not tokens:
        return (None, None)
    cur = core.coerce_int(task.get("link"), 1)
    nno = cur + 1
    ndt = next_due_utc
    last_no = None
    last_dt = None
    iterations = 0

    while ndt and _compare_datetimes(ndt, until) <= 0 and iterations < _MAX_ITERATIONS:
        iterations += 1
        last_no, last_dt = nno, ndt
        td = _cp_sequence_period_for_link(
            tokens,
            cp_str,
            nno,
            str(task.get("chainID") or "").strip(),
        )
        ndt = _cp_add_td(ndt, td)
        nno += 1

    return (last_no, last_dt)


def _cap_from_until_anchor(task, next_due_utc, dnf):
    """
    Return (final_no, final_dt) for anchors limited by chainUntil.
    WITH iteration guard to prevent infinite loops.
    """
    until_utc = _dtparse(task.get("chainUntil"))
    if not until_utc:
        return (None, None)

    cur_no = core.coerce_int(task.get("link"), 1)
    seed_base = _recurrence_seed_base(task)

    nxt_local = _to_local_cached(next_due_utc)
    until_local = _to_local_cached(until_utc)
    due0, _ = _safe_parse_datetime(task.get("due"))
    default_seed = _to_local_cached(due0 or next_due_utc).date()
    fallback_hhmm = _anchor_file_fallback_hhmm(task, nxt_local)
    _omit_expr, omit_dnf = _omit_dnf_from_parent(task)
    scheduler = _recurrence_evaluator_for_task(task)._default_next_occurrence_after_local_dt
    anchor_file = (task.get("anchor_file") or "").strip()
    anchor_file_provider = None
    if anchor_file:
        anchor_file_provider = _anchor_file_provider_for(
            anchor_file, fallback_hhmm=fallback_hhmm, seed_base=seed_base
        )

    count = 0
    last_hit = None
    cursor = nxt_local
    iterations = 0

    # Count occurrences starting with the already-computed next due.
    while iterations < _MAX_ITERATIONS and _compare_datetimes(cursor, until_local) <= 0:
        iterations += 1
        count += 1
        last_hit = cursor
        if anchor_file:
            future = _anchor_included_occurrences(
                task,
                after_local_dt=cursor,
                inclusive=False,
                limit=2,
                fallback_hhmm=fallback_hhmm,
                omit_dnf=omit_dnf,
                seed_base=seed_base,
                default_seed_date=default_seed,
                dnf=dnf,
                anchor_file_provider=anchor_file_provider,
            )
            cursor = future[0] if future else None
        else:
            cursor = scheduler(
                dnf,
                cursor,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                fallback_hhmm=fallback_hhmm,
            )
        if cursor is None:
            break

    if cursor is not None and _compare_datetimes(cursor, until_local) <= 0 and iterations >= _MAX_ITERATIONS:
        raise ValueError(
            f"Anchor chainUntil projection exceeded {_MAX_ITERATIONS} occurrences; "
            "narrow chainUntil or use a larger recurrence interval."
        )

    if count == 0 or last_hit is None:
        return (None, None)

    final_no = cur_no + count
    final_dt = last_hit.astimezone(timezone.utc)
    return (final_no, final_dt)

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

def _chain_export_timeout(chain_id: str) -> float:
    global _CHAIN_EXPORT_TIMEOUT_FLOOR
    base = float(_CHAIN_EXPORT_TIMEOUT_BASE)
    per_100 = float(_CHAIN_EXPORT_TIMEOUT_PER_100)
    max_t = float(_CHAIN_EXPORT_TIMEOUT_MAX)
    est = base
    cache_len = _lifecycle_read_service().cache_size(chain_id) if chain_id else 0
    cache_match = bool(cache_len)
    if cache_match:
        extra = max(0, cache_len // 100)
        est = base + (extra * per_100)
    adaptive = 0.0
    if _CHAIN_EXPORT_TIMES:
        try:
            times = sorted(t for t in _CHAIN_EXPORT_TIMES if t > 0)
        except Exception:
            times = []
        if times:
            idx = int(0.95 * (len(times) - 1))
            p95 = times[max(0, min(idx, len(times) - 1))]
            adaptive = p95 * 2.0
    floor = _CHAIN_EXPORT_TIMEOUT_FLOOR
    if floor < base:
        floor = base
        _CHAIN_EXPORT_TIMEOUT_FLOOR = base
    if est < base:
        est = base
    timeout = max(est, adaptive, floor)
    if timeout > max_t:
        timeout = max_t
    return timeout

def _tw_export_chain_success(elapsed: float) -> None:
    global _CHAIN_EXPORT_TIMEOUT_FLOOR
    if elapsed > 0:
        _CHAIN_EXPORT_TIMES.append(elapsed)
        if len(_CHAIN_EXPORT_TIMES) > _CHAIN_EXPORT_TIMES_MAX:
            del _CHAIN_EXPORT_TIMES[:len(_CHAIN_EXPORT_TIMES) - _CHAIN_EXPORT_TIMES_MAX]
    if _CHAIN_EXPORT_TIMEOUT_FLOOR > _CHAIN_EXPORT_TIMEOUT_BASE:
        _CHAIN_EXPORT_TIMEOUT_FLOOR = max(_CHAIN_EXPORT_TIMEOUT_BASE, _CHAIN_EXPORT_TIMEOUT_FLOOR * 0.9)


def _tw_export_chain_failure(chain_id: str, err: str, timeout: float) -> None:
    global _CHAIN_EXPORT_TIMEOUT_FLOOR
    if "timeout" in (err or "").lower():
        _CHAIN_EXPORT_TIMEOUT_FLOOR = min(
            _CHAIN_EXPORT_TIMEOUT_MAX,
            max(_CHAIN_EXPORT_TIMEOUT_FLOOR, timeout * 1.5),
        )
    _diag(f"tw_export_chain failed (chainID={chain_id}): {err.strip()}")
    if chain_id and chain_id in _WARNED_CHAIN_EXPORT:
        return
    if chain_id:
        _WARNED_CHAIN_EXPORT.add(chain_id)
    if _is_lock_error(err):
        reason = "Taskwarrior lock active"
    else:
        reason = (err or "").strip() or "task export failed"
    _panel("⚠ Chain export failed", [("ChainID", chain_id), ("Reason", reason)], kind="warning")


def _tw_export_chain_parse(out: str) -> list[dict]:
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.parse_export_array(out, diag=_diag)
    try:
        data = json.loads(out.strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        _diag(f"tw_export_chain JSON parse failed: {e}")
        return []


def _tw_export_chain_checked(
    chain_id: str,
    since: datetime | None = None,
    extra: str | None = None,
    env=None,
    limit: int | None = None,
) -> tuple[bool, list[dict], str]:
    """Compatibility facade for the typed lifecycle chain-read service."""
    service = _lifecycle_read_service()
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        def run_task_result(command, **kwargs):
            return hook_support.run_task_result(run_task=_run_task_result, cmd=command, **kwargs)

        def parse_result(result):
            return hook_support.parse_export_array_result(result, diag=_diag)
    else:
        run_task_result = _run_task_result

        def parse_result(result):
            if not result.ok:
                return False, [], result.stderr or "task export failed"
            try:
                parsed = json.loads((result.stdout or "").strip())
                if not isinstance(parsed, list) or any(not isinstance(row, dict) for row in parsed):
                    raise ValueError("expected an array of task objects")
                return True, parsed, ""
            except Exception as exc:
                return False, [], f"Taskwarrior export returned invalid JSON: {exc}"

    def run_export(args, run_env, timeout):
        return service.run_checked_export(
            chain_id,
            args,
            env=run_env,
            timeout=timeout,
            run_task_result=run_task_result,
            parse_result=parse_result,
            on_failure=lambda error, export_timeout: _tw_export_chain_failure(
                chain_id, error, export_timeout
            ),
            on_success=_tw_export_chain_success,
        )

    result = service.checked_export(
        chain_id,
        since=since,
        extra=extra,
        env=env,
        limit=limit,
        build_args=service.build_export_args,
        run_export=run_export,
        timeout_for_chain=_chain_export_timeout,
        read_query_missing=_READ_QUERY_MISSING,
    )
    return result.ok, result.rows, result.error


def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None, env=None, limit: int | None = None) -> list[dict]:
    """Return rows for compatibility; internal mutation reads use checked results."""
    global _LAST_CHAIN_EXPORT_STATUS
    _ok, rows, _error = _tw_export_chain_checked(
        chain_id,
        since=since,
        extra=extra,
        env=env,
        limit=limit,
    )
    _LAST_CHAIN_EXPORT_STATUS = (_ok, _error)
    return rows


def _export_chain_endpoint(chain_id: str, direction: str) -> dict | None:
    """Return the first/last chain task using a minimal export."""
    modify_queries = _module("modify_queries")
    hook_support = _module("hook_support", required=False)
    parser = (hook_support.parse_export_array if hook_support is not None else _tw_export_chain_parse)
    return modify_queries.export_chain_endpoint(
        chain_id,
        direction,
        run_task=_run_task_result,
        task_cmd_prefix=_task_cmd_prefix(),
        parse_export_array=parser,
        diag=_diag,
        timeout=3.0,
        retries=1,
    )

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def _is_non_completion_modify(old: dict, new: dict) -> bool:
    return (old.get("status") == new.get("status")) or (new.get("status") != "completed")


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
    if lifecycle_result is None:
        lifecycle_result = _module("modify_models").CompletionLifecycleResult(
            state="queued" if deferred_spawn else "applied",
            child_short=child_short,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
        )
    calendar_feedback = importlib.import_module("nautical_core.calendar_feedback")
    calendar_feedback.render_business_calendar_displacement(
        new,
        child_due,
        core=core,
        panel=_panel,
    )
    diagnostics = _module("panel_diagnostics")
    panel_warnings = diagnostics.panel_warnings(core, new)
    if panel_warnings:
        integrity_warnings = list(integrity_warnings or [])
        integrity_warnings.extend(panel_warnings)
    modify_feedback = _module("modify_feedback")
    modify_models = _module("modify_models")
    modify_runtime = _module("modify_runtime")
    feedback = modify_models.AnchorCompletionFeedbackModel(
        new=new,
        child=child,
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
    )
    runtime = _modify_runtime_services()
    services = modify_runtime.build_anchor_feedback_services(runtime)
    modify_feedback.render_anchor_completion_feedback(
        feedback=feedback,
        services=services,
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
    if lifecycle_result is None:
        lifecycle_result = _module("modify_models").CompletionLifecycleResult(
            state="queued" if deferred_spawn else "applied",
            child_short=child_short,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
        )
    diagnostics = _module("panel_diagnostics")
    panel_warnings = diagnostics.panel_warnings(core, new, include_files=False)
    if panel_warnings:
        integrity_warnings = list(integrity_warnings or [])
        integrity_warnings.extend(panel_warnings)
    modify_feedback = _module("modify_feedback")
    modify_models = _module("modify_models")
    modify_runtime = _module("modify_runtime")
    feedback = modify_models.CpCompletionFeedbackModel(
        new=new,
        child=child,
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
    )
    runtime = _modify_runtime_services()
    services = modify_runtime.build_cp_feedback_services(runtime)
    modify_feedback.render_cp_completion_feedback(
        feedback=feedback,
        services=services,
    )


def _render_lifecycle_result(result, task: dict) -> None:
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
    _panel(title, rows, kind="warning" if state == "manual_review" else "error", task=task)


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
    if omit_expr and not (anchor_expr or anchor_file_expr):
        _fail_and_exit("Invalid omit", "omit requires anchor or anchor_file")
    if omit_file and not (anchor_expr or anchor_file_expr):
        _fail_and_exit("Invalid omit_file", "omit_file requires anchor or anchor_file")
    if omit_expr:
        try:
            _validate_omit_on_modify(omit_expr)
        except Exception as e:
            _fail_and_exit("Invalid omit", str(e))
    if anchor_file_expr:
        try:
            _load_anchor_file_dates(anchor_file_expr)
        except Exception as e:
            _fail_and_exit("Invalid anchor_file", str(e))
    if omit_file:
        try:
            _load_omit_file_dates(omit_file)
        except Exception as e:
            _fail_and_exit("Invalid omit_file", str(e))


def _non_completion_reject_conflicting_types(new_anchor: str, new_anchor_file: str, new_cp: str) -> None:
    if new_anchor and new_cp:
        _fail_and_exit("Invalid chain config", "anchor and cp cannot both be set; clear one")
    if new_anchor_file and new_cp:
        _fail_and_exit("Invalid chain config", "anchor_file and cp cannot both be set; clear one")


def _recurrence_update_label(field: str) -> str:
    return {
        "anchor": "Anchor",
        "anchor_file": "Anchor file",
        "omit": "Omit",
        "omit_file": "Omit file",
        "anchor_mode": "Mode",
        "bc": "Business calendar",
        "cp": "Period",
        "until": "Expiration",
        "chainMax": "Max links",
        "chainUntil": "Chain end point",
    }.get(field, field)


def _semantic_diff_value(old_text: str, new_text: str) -> str:
    return f"[dim]{old_text}[/] [cyan]→[/] [bold]{new_text}[/]"


def _recurrence_display_value(field: str, value: str) -> str:
    if not value:
        return "-"
    if field in {"until", "chainUntil"}:
        parsed = core.parse_dt_any(value)
        if parsed:
            return _fmtlocal(parsed)
    return value


def _recurrence_update_value(field: str, old_value: str, new_value: str) -> str:
    old_text = _recurrence_display_value(field, old_value)
    new_text = _recurrence_display_value(field, new_value)
    return _semantic_diff_value(old_text, new_text)


def _recurrence_change_row(field: str, old_value: str, new_value: str) -> tuple[str, str]:
    label = _recurrence_update_label(field)
    if old_value and new_value:
        return "Changed", f"{label}: {_recurrence_update_value(field, old_value, new_value)}"
    if new_value:
        return "Added", f"{label}: [bold]{_recurrence_display_value(field, new_value)}[/]"
    return "Removed", f"{label}: [dim]{_recurrence_display_value(field, old_value)}[/]"


def _recurrence_update_panel_rows(changes: list[tuple[str, str, str]], rows: list[tuple[str | None, str]]) -> list[tuple[str | None, str]]:
    """Keep multi-field updates scannable and preserve all changes in one-line modes."""
    if len(changes) > 1:
        recurrence_fields = {"anchor", "anchor_file", "cp", "anchor_mode", "omit", "omit_file", "bc"}
        limit_fields = {"chainMax", "chainUntil"}
        first_limit = next((idx for idx, (field, _old, _new) in enumerate(changes) if field in limit_fields), None)
        if first_limit is not None and any(field in recurrence_fields for field, _old, _new in changes):
            rows = list(rows)
            rows.insert(first_limit, (None, ""))

    mode = str(getattr(core, "PANEL_MODE", "rich") or "rich").strip().lower()
    if mode == "quiet":
        mode = "text"
    if mode == "minimal":
        mode = "line"
    if mode in {"line", "text"}:
        change_rows = [(label, value) for label, value in rows if label in {"Added", "Changed", "Removed"}]
        if len(change_rows) > 1:
            summary = " · ".join(
                f"{label}: {core.strip_rich_markup(str(value))}" for label, value in change_rows
            )
            rows = [("Changes", summary)] + [
                (label, value)
                for label, value in rows
                if label not in {"Added", "Changed", "Removed"}
            ]
    return rows


def _render_recurrence_updated_panel(changes: list[tuple[str, str, str]], new: dict) -> None:
    if not changes:
        return
    rows: list[tuple[str, str]] = [
        _recurrence_change_row(field, old_value, new_value)
        for field, old_value, new_value in changes
    ]

    if any(field == "until" for field, _old, _new in changes):
        try:
            target_field = "due" if new.get("due") else "scheduled" if new.get("scheduled") else ""
            until_dt = core.parse_dt_any(new.get("until"))
            target_dt = core.parse_dt_any(new.get(target_field)) if target_field else None
            add_validation = core._import_sibling("add_validation")
            carry = add_validation.describe_native_until_carry(
                until_dt,
                target_dt,
                to_local=core.to_local,
            )
            if carry:
                rows.append(("Carry", carry))
        except Exception:
            pass

    if any(field in {"chainMax", "chainUntil"} for field, _old, _new in changes):
        max_link = core.coerce_int(new.get("chainMax"), 0)
        deadline = core.parse_dt_any(new.get("chainUntil"))
        if max_link:
            rows.append(("Final link", f"#{max_link}"))
        if deadline and not any(field == "chainUntil" for field, _old, _new in changes):
            rows.append(("Chain end point", _fmtlocal(deadline)))
        if max_link and deadline:
            rows.append(("Effective", "Whichever boundary is reached first"))
        elif not max_link and not deadline:
            rows.append(("Chain limits", "None"))

    anchor_expr = str(new.get("anchor") or "").strip()
    if anchor_expr and any(field == "anchor" for field, _old, _new in changes):
        try:
            rows.append(("Natural", core.describe_anchor_expr(anchor_expr)))
        except Exception:
            pass

    omit_expr = str(new.get("omit") or "").strip()
    if omit_expr and any(field == "omit" for field, _old, _new in changes):
        try:
            rows.append(("Except", core.describe_anchor_expr(core.resolve_omit_presets(omit_expr))))
        except Exception:
            pass

    recurrence_fields = {"anchor", "anchor_file", "cp", "anchor_mode", "omit", "omit_file", "bc"}
    if any(field in recurrence_fields for field, _old, _new in changes):
        source = "anchor" if anchor_expr else "anchor_file" if str(new.get("anchor_file") or "").strip() else "cp"
        first = _first_recurrence_target(new, source)
        if first:
            rows.append(("First next", _fmtlocal(first)))

    rows = _recurrence_update_panel_rows(changes, rows)
    _panel("⚓ Nautical recurrence updated", rows, kind="note")


def _first_recurrence_target(new: dict, source: str):
    target_field = "due" if new.get("due") else "scheduled" if new.get("scheduled") else ""
    if not target_field:
        return None
    target = core.parse_dt_any(new.get(target_field))
    if not target:
        return None
    parent = dict(new)
    parent["end"] = core.fmt_isoz(target)
    try:
        generation = _chain_generation_service()
        if source in {"anchor", "anchor_file"}:
            result = generation.compute_anchor_child_due(parent)
            return result[0] if result else None
        result = generation.compute_cp_child_due(parent)
        return result[0] if result else None
    except Exception:
        return None


def _recurrence_enabled_rows(new: dict, source: str) -> list[tuple[str, str]]:
    """Describe the recurrence that was added during a plain-task upgrade."""
    if source == "anchor":
        value = str(new.get("anchor") or "").strip()
        rows = [("Anchor", value)]
        try:
            natural = core.describe_anchor_expr(value)
        except Exception:
            natural = None
        if natural:
            rows.append(("Natural", natural))
        mode = (new.get("anchor_mode") or "skip").strip().lower()
        mode_explanations = {
            "skip": "Skip missed anchors; use the next anchor after completion",
            "all": "Backfill every missed anchor in order",
            "flex": "Skip missed anchors and continue from the next available anchor",
        }
        rows.append(("Mode", f"{mode.upper()} — {mode_explanations.get(mode, mode)}"))
        first = _first_recurrence_target(new, source)
        if first:
            rows.append(("First next", _fmtlocal(first)))
        return rows

    if source == "anchor_file":
        value = str(new.get("anchor_file") or "").strip()
        rows = [("Anchor file", value), ("Natural", f"Dates from {value.split('@', 1)[0]}")]
        mode = (new.get("anchor_mode") or "skip").strip().lower()
        rows.append(("Mode", f"{mode.upper()}"))
        first = _first_recurrence_target(new, source)
        if first:
            rows.append(("First next", _fmtlocal(first)))
        return rows

    value = str(new.get("cp") or "").strip()
    rows = [("Period", value)]
    natural = None
    try:
        def _duration_label(duration) -> str:
            seconds = int(duration.total_seconds())
            if seconds % 86400 == 0:
                return f"{seconds // 86400}d"
            if seconds % 3600 == 0:
                return f"{seconds // 3600}h"
            if seconds % 60 == 0:
                return f"{seconds // 60}m"
            return f"{seconds}s"

        tokens = core.parse_cp_sequence_tokens(value) or []
        descriptions = []
        for token in tokens:
            if token.get("kind") == "rand":
                descriptions.append(f"random interval {token.get('raw') or value}")
            else:
                duration = token.get("duration")
                descriptions.append(_duration_label(duration) if duration else str(token.get("raw") or value))
        if len(descriptions) == 1:
            natural = f"Every {descriptions[0]}"
        elif descriptions:
            natural = "Cycle through " + ", then ".join(descriptions)
    except Exception:
        natural = None
    if natural:
        rows.append(("Natural", natural))
    first = _first_recurrence_target(new, source)
    if first:
        rows.append(("First next", _fmtlocal(first)))
    return rows


def _render_cp_schedule_adjusted_panel(
    adjustment: tuple[
        datetime,
        datetime,
        list[tuple[str, datetime, datetime, timedelta]],
    ],
) -> None:
    old_due, new_due, field_adjustments = adjustment
    rows = [("Due", _semantic_diff_value(_fmtlocal(old_due), _fmtlocal(new_due)))]
    rows.extend(
        (field.capitalize(), _semantic_diff_value(_fmtlocal(old_value), _fmtlocal(new_value)))
        for field, old_value, new_value, _offset in field_adjustments
    )
    offset_text = "; ".join(
        f"{field.capitalize()} {_fmt_td_dd_hhmm(offset)}"
        for field, _old_value, _new_value, offset in field_adjustments
    )
    rows.append(("Offset" if len(field_adjustments) == 1 else "Offsets", offset_text))
    _panel(
        "⚓ Nautical schedule adjusted",
        rows,
        kind="note",
    )


def _render_explicit_timing_order_warning(new: dict, changed_fields: tuple[str, ...]) -> None:
    if not changed_fields:
        return

    def parsed(field: str) -> datetime | None:
        value = new.get(field)
        if not value:
            return None
        try:
            return core.parse_dt_any(value)
        except Exception:
            return None

    due = parsed("due")
    scheduled = parsed("scheduled")
    wait = parsed("wait")
    issues: list[str] = []
    if due and scheduled and scheduled > due:
        issues.append(f"Scheduled is after Due by {_fmt_td_dd_hhmm(scheduled - due)}")
    if due and wait and wait > due:
        issues.append(f"Wait is after Due by {_fmt_td_dd_hhmm(wait - due)}")
    if scheduled and wait and wait > scheduled:
        issues.append(f"Wait is after Scheduled by {_fmt_td_dd_hhmm(wait - scheduled)}")
    if not issues:
        return

    if due:
        expected = "Due >= Scheduled >= Wait"
        action = "Keep Scheduled at/before Due and Wait at/before Scheduled."
    else:
        expected = "Scheduled >= Wait"
        action = "Keep Wait at or before Scheduled."
    rows = [
        ("Changed", ", ".join(field.capitalize() for field in changed_fields)),
        ("Expected", expected),
    ]
    rows.extend(("Problem", issue) for issue in issues)
    rows.append(("Action", action))
    _panel("⚠ Nautical timing order", rows, kind="warning")


def _render_disabled_chain_summary(old: dict, new: dict, reason: str) -> None:
    """Show the normal finished-chain summary when an active chain is stopped."""
    if not (old.get("chainID") or new.get("chainID")):
        return
    now_utc = core.now_utc()
    try:
        _end_chain_summary(old, reason, now_utc, current_task=old)
    except Exception as exc:
        _diag(f"removed recurrence chain summary failed: {exc}")
        _panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", reason),
                ("Root", _format_root_and_age(old, now_utc)),
                ("Task", _short(old.get("uuid")) or "–"),
            ],
            kind="summary",
        )


def _ensure_terminal_chain_off(task: dict, event: str | None = None) -> bool:
    """Validate and apply one idempotent terminal patch for hook-side stops."""
    if event:
        lifecycle_models = _module("lifecycle_models")
        lifecycle_planner = _module("lifecycle_planner")
        lifecycle_planner.terminal_plan_for_snapshot(
            lifecycle_models.TaskSnapshot.from_mapping(task),
            lifecycle_models.LifecycleEvent(event),
        )
    return _module("modify_lifecycle").ensure_terminal_chain_off(task)


def _preserve_cp_relative_offsets_on_due_change(
    old: dict,
    new: dict,
    new_cp: str,
) -> tuple[
    datetime,
    datetime,
    list[tuple[str, datetime, datetime, timedelta]],
] | None:
    """Keep scheduled and wait relative to due when an existing cp task's due moves."""
    if not new_cp or not str(old.get("cp") or "").strip():
        return None
    if not _field_changed(old, new, "due"):
        return None
    if not (old.get("due") and new.get("due")):
        return None

    try:
        old_due = core.parse_dt_any(old.get("due"))
        new_due = core.parse_dt_any(new.get("due"))
        if not (old_due and new_due):
            raise ValueError("due timestamp is missing or invalid")
    except Exception as exc:
        carry_error = _module("chain_generation").CarryFieldError
        raise carry_error("due", str(exc) or "timestamp conversion failed") from exc

    adjustments: list[tuple[str, datetime, datetime, timedelta]] = []
    for field in ("scheduled", "wait"):
        if _field_changed(old, new, field) or not old.get(field):
            continue
        try:
            old_value = core.parse_dt_any(old.get(field))
            if not old_value:
                raise ValueError("timestamp is missing or invalid")
            local_offset = _utc_to_local_naive(old_value) - _utc_to_local_naive(old_due)
            new_value_local = _utc_to_local_naive(new_due) + local_offset
            new_value = _local_naive_to_utc(new_value_local)
            new[field] = core.fmt_isoz(new_value)
            adjustments.append((field, old_value, new_value, local_offset))
        except Exception as exc:
            carry_error = _module("chain_generation").CarryFieldError
            raise carry_error(field, str(exc) or "timezone conversion failed") from exc
    return (old_due, new_due, adjustments) if adjustments else None


def _reject_native_until_carry(
    old: dict,
    new: dict,
    new_target: datetime | None,
    old_target_field: str,
    exc: Exception,
) -> None:
    """Reject a target edit when its native expiration cannot be carried."""
    carry = None
    try:
        add_validation = core._import_sibling("add_validation")
        carry = add_validation.describe_native_until_carry(
            core.parse_dt_any(old.get("until")),
            core.parse_dt_any(old.get(old_target_field)),
            to_local=core.to_local,
        )
    except Exception:
        pass
    target_label = (
        core.fmt_dt_local(new_target)
        if isinstance(new_target, datetime)
        else str(new.get(_recurrence_anchor_field(new)) or "–")
    )
    rows = [("Target", target_label), ("Required", str(exc))]
    if carry:
        rows.insert(1, ("Carry", carry))
    _panel("❌ Invalid expiration window", rows, kind="error")
    sys.exit(1)


def _preserve_native_until_on_target_change(old: dict, new: dict, kind: str) -> bool:
    """Carry an untouched native until when an existing recurrence target moves."""
    if _field_changed(old, new, "until") or not old.get("until"):
        return False
    old_target_field = _recurrence_anchor_field(old)
    new_target_field = _recurrence_anchor_field(new)
    target_changed = (
        old_target_field != new_target_field
        or _field_changed(old, new, old_target_field)
    )
    if not target_changed:
        return False
    native_until = core._import_sibling("native_until")
    new_target: datetime | None = None
    try:
        new_target = core.parse_dt_any(new.get(new_target_field))
        if not new_target:
            raise native_until.NativeUntilCarryError(
                native_until.CARRY_INVALID,
                f"{new_target_field} timestamp is missing or invalid",
            )
        candidate = dict(new)
        _chain_generation_service().carry_native_until(
            old,
            candidate,
            new_target,
            kind,
            parent_anchor_field=old_target_field,
            child_anchor_field=new_target_field,
        )
        carried = candidate.get("until")
        if not carried:
            raise native_until.NativeUntilCarryError(
                native_until.CARRY_FAILED,
                "native until carry produced no expiration value",
            )
        new["until"] = carried
        return True
    except native_until.NativeUntilCarryError as exc:
        _reject_native_until_carry(old, new, new_target, old_target_field, exc)
    except Exception as exc:
        _diag(f"native until target carry failed: {exc}")
        typed_error = native_until.NativeUntilCarryError(
            native_until.CARRY_FAILED,
            f"native until target carry failed: {type(exc).__name__}: {exc}",
        )
        _reject_native_until_carry(old, new, new_target, old_target_field, typed_error)
    return False


def _handle_non_completion_modify(old: dict, new: dict) -> None:
    modify_ordinary = _module("modify_ordinary")
    modify_lifecycle = _module("modify_lifecycle")
    services = modify_ordinary.OrdinaryModifyServices(
        field_changed=_field_changed,
        strip_quotes=_strip_quotes,
        validate_anchor=_non_completion_validate_anchor,
        validate_omit=_validate_omit_for_anchor_or_fail,
        reject_conflicting_types=_non_completion_reject_conflicting_types,
        validate_chain_limits=_validate_chain_limits_on_modify,
        preserve_cp_offsets=_preserve_cp_relative_offsets_on_due_change,
        task_has_recurrence=modify_lifecycle.task_has_nautical_recurrence_fields,
        preserve_native_until=_preserve_native_until_on_target_change,
        validate_native_until=_validate_native_until_after_target_or_fail,
        validate_native_until_slots=_validate_native_until_anchor_slots_or_fail,
        render_cp_adjustment=_render_cp_schedule_adjusted_panel,
        render_timing_warning=_render_explicit_timing_order_warning,
        apply_transition=lambda old_task, new_task: modify_lifecycle.apply_nautical_transition(
            old_task,
            new_task,
            short_uuid=core.short_uuid,
        ),
        short_uuid=core.short_uuid,
        recurrence_enabled_rows=_recurrence_enabled_rows,
        panel=_panel,
        render_disabled_summary=_render_disabled_chain_summary,
        semantic_diff_value=_semantic_diff_value,
        first_recurrence_target=_first_recurrence_target,
        fmtlocal=_fmtlocal,
        render_recurrence_updated=_render_recurrence_updated_panel,
        print_task=_print_task,
    )
    try:
        modify_ordinary.handle_non_completion_modify(
            old,
            new,
            services=services,
            lifecycle=modify_lifecycle,
        )
    except _module("chain_generation").CarryFieldError as exc:
        _fail_and_exit("Nautical carry failed", str(exc))
    except modify_ordinary.RecurrenceActivationError as exc:
        _fail_and_exit("Nautical recurrence activation failed", str(exc))


def _completion_validate_cp_and_anchor(old: dict, new: dict) -> tuple[str, str, str]:
    # If we reach here, the task is being completed
    # Now we should validate CP (in addition to anchor which was already validated on modify)
    cp_raw = (new.get("cp") or "").strip()
    new_cp = _strip_quotes(cp_raw)
    anchor_raw = (new.get("anchor") or "").strip()
    new_anchor = _strip_quotes(anchor_raw)
    anchor_file_raw = (new.get("anchor_file") or "").strip()
    new_anchor_file = _strip_quotes(anchor_file_raw)
    if new_anchor_file:
        new["anchor_file"] = new_anchor_file
    omit_raw = (new.get("omit") or "").strip()
    new_omit = _strip_quotes(omit_raw)
    if new_omit:
        new["omit"] = new_omit
    omit_file_raw = (new.get("omit_file") or "").strip()
    new_omit_file = _strip_quotes(omit_file_raw)
    if new_omit_file:
        new["omit_file"] = new_omit_file
    _non_completion_reject_conflicting_types(new_anchor, new_anchor_file, new_cp)
    _validate_omit_for_anchor_or_fail(new_anchor, new_anchor_file, new_omit, new_omit_file)
    if new_cp or new_anchor or new_anchor_file:
        _validate_chain_limits_on_modify(new)

    if new_cp:
        # Validate CP on completion
        try:
            seq = core.parse_cp_sequence(new_cp)
            if not seq:
                reason = core.cp_sequence_parse_error(new_cp) or f"invalid duration format '{new_cp}'"
                raise ValueError(reason)
        except ValueError as e:
            _fail_and_exit("Invalid CP", str(e))
        except Exception as e:
            _diag(f"cp parse unexpected error: {e}")
            _fail_and_exit("CP parsing error", "Unexpected error while parsing cp")

        # Deep checks only if fields changed
        if _field_changed(old, new, "anchor") or _field_changed(old, new, "anchor_mode") or _field_changed(old, new, "anchor_file"):
            if new_anchor:
                _validate_anchor_on_modify(new_anchor)

        if (
            _field_changed(old, new, "cp")
            or _field_changed(old, new, "chainMax")
            or _field_changed(old, new, "chainUntil")
        ) and new_cp:
            _validate_cp_on_modify(new_cp, new.get("chainMax"), new.get("chainUntil"))

        modify_lifecycle = _module("modify_lifecycle")
        try:
            modify_lifecycle.apply_nautical_transition(old, new, short_uuid=core.short_uuid)
        except Exception as exc:
            _fail_and_exit(
                "Nautical recurrence activation failed",
                f"Nautical recurrence transition failed: {type(exc).__name__}: {exc}",
            )

    return new_cp, new_anchor, new_anchor_file


def _completion_link_numbers_or_fail(new: dict) -> tuple[int, int] | None:
    modify_completion_preflight = _module("modify_completion_preflight")
    return modify_completion_preflight.completion_link_numbers_or_fail(
        new,
        coerce_int=core.coerce_int,
        max_link_number=core.MAX_LINK_NUMBER,
        panel=_panel,
        print_task=_print_task,
    )


def _completion_kind_or_stop(new: dict, now_utc: datetime) -> str | None:
    modify_completion_preflight = _module("modify_completion_preflight")
    return modify_completion_preflight.completion_kind_or_stop(
        new,
        now_utc,
        panel=_panel,
        print_task=_print_task,
        end_chain_summary=_end_chain_summary,
    )


def _completion_chain_id_or_fail(new: dict) -> str | None:
    modify_completion_preflight = _module("modify_completion_preflight")
    return modify_completion_preflight.completion_chain_id_or_fail(
        new,
        panel=_panel,
        print_task=_print_task,
    )


def _completion_existing_next_or_fail(new: dict, next_no: int, chain_snapshot=None) -> bool:
    modify_completion_preflight = _module("modify_completion_preflight")
    return modify_completion_preflight.completion_existing_next_or_fail(
        new,
        next_no,
        existing_next_lookup=lambda task, link: _existing_next_lookup(task, link, chain_snapshot),
        short=_short,
        panel=_panel,
        print_task=_print_task,
    )


def _completion_chain_snapshot_mode() -> str:
    if _SHOW_ANALYTICS or _CHECK_CHAIN_INTEGRITY:
        return "full"
    mode = str(getattr(core, "PANEL_MODE", "rich") or "rich").strip().lower()
    if mode in {"line", "minimal", "quiet", "text"}:
        return "next"
    return "recent"


def _completion_chain_snapshot(chain_id: str, base_no: int, next_no: int):
    modify_models = _module("modify_models")
    modify_queries = _module("modify_queries")
    hook_support = _module("hook_support", required=False)
    parser = hook_support.parse_export_array if hook_support is not None else _tw_export_chain_parse
    mode = _completion_chain_snapshot_mode()
    links = None if mode == "full" else ([next_no] if mode == "next" else [base_no - 2, base_no - 1, next_no])
    if links is not None:
        links = sorted({link for link in links if link > 0})

    def _load_snapshot(snapshot_chain_id: str, snapshot_links: list[int] | None):
        snapshot_result = modify_queries.export_completion_chain_snapshot(
            snapshot_chain_id,
            snapshot_links,
            run_task=_run_task_result,
            task_cmd_prefix=_task_cmd_prefix(),
            parse_export_array=parser,
            diag=_diag,
            timeout=_chain_export_timeout(snapshot_chain_id),
        )
        lifecycle_read_service = _module("lifecycle_read_service")
        if not snapshot_result.loaded:
            return lifecycle_read_service.ChainReadResult.failure(snapshot_result.error)
        return lifecycle_read_service.ChainReadResult.success(snapshot_result.rows)

    snapshot = _lifecycle_read_service().completion_snapshot(
        chain_id,
        mode=mode,
        links=links,
        load_snapshot=_load_snapshot,
        read_query_missing=_READ_QUERY_MISSING,
    )
    return modify_models.CompletionChainSnapshot(
        mode=snapshot.mode,
        rows=snapshot.rows,
        loaded=snapshot.loaded,
        chain_id=snapshot.chain_id,
        error=snapshot.error,
    )


def _completion_preflight_context(new: dict, now_utc: datetime):
    modify_completion_preflight = _module("modify_completion_preflight")
    modify_runtime = _module("modify_runtime")
    services = modify_runtime.build_preflight_services(
        short=_short,
        completion_link_numbers_or_fail=_completion_link_numbers_or_fail,
        completion_kind_or_stop=_completion_kind_or_stop,
        completion_chain_id_or_fail=_completion_chain_id_or_fail,
        completion_chain_snapshot=_completion_chain_snapshot,
        completion_existing_next_or_fail=_completion_existing_next_or_fail,
    )
    return modify_completion_preflight.completion_preflight_context(
        new,
        now_utc,
        services=services,
    )


def _completion_compute_child_due(new: dict, kind: str):
    modify_completion_compute = _module("modify_completion_compute")
    generation = _chain_generation_service()

    def handle_terminal(exc) -> None:
        message = core._import_sibling("scheduler_models").occurrence_exhaustion_message(exc)
        if exc.is_date_limit:
            _ensure_terminal_chain_off(new, "complete")
            try:
                _end_chain_summary(new, message, core.now_utc(), current_task=new)
            except Exception as summary_exc:
                _diag(f"terminal chain summary failed: {summary_exc}")
                _panel(
                    "⛔ Nautical chain stopped",
                    [("Reason", message), ("Task", _short(new.get("uuid")) or "–")],
                    kind="summary",
                )
            _print_task(new)
            return
        _panel(
            "⛔ Chain error",
            [
                ("Scheduler", message),
                ("Fix", "Use a less sparse rule or adjust its search limits."),
            ],
            kind="error",
        )
        _print_task(new)

    return modify_completion_compute.completion_compute_child_due(
        new,
        kind,
        compute_anchor_child_due=generation.compute_anchor_child_due,
        compute_cp_child_due=generation.compute_cp_child_due,
        panel=_panel,
        print_task=_print_task,
        diag=_diag,
        on_terminal=handle_terminal,
    )


def _completion_until_or_fail(new: dict, now_utc: datetime) -> datetime | None | object:
    modify_completion_compute = _module("modify_completion_compute")
    return modify_completion_compute.completion_until_or_fail(
        new,
        now_utc,
        safe_parse_datetime=_safe_parse_datetime,
        validate_until_not_past=_validate_until_not_past,
        panel=_panel,
        print_task=_print_task,
    )


def _completion_until_guard_or_stop(new: dict, child_due, until_dt, now_utc: datetime) -> bool:
    modify_completion_compute = _module("modify_completion_compute")
    return modify_completion_compute.completion_until_guard_or_stop(
        new,
        child_due,
        until_dt,
        now_utc,
        end_chain_summary=_end_chain_summary,
        print_task=_print_task,
    )


def _completion_require_child_due_or_fail(new: dict, child_due) -> bool:
    modify_completion_compute = _module("modify_completion_compute")
    return modify_completion_compute.completion_require_child_due_or_fail(
        new,
        child_due,
        panel=_panel,
        print_task=_print_task,
    )


def _completion_warn_unreasonable_duration(new: dict, child_due, until_dt, now_utc: datetime) -> None:
    modify_completion_compute = _module("modify_completion_compute")
    modify_completion_compute.completion_warn_unreasonable_duration(
        new,
        child_due,
        until_dt,
        now_utc,
        validate_chain_duration_reasonable=_validate_chain_duration_reasonable,
        panel=_panel,
    )


def _completion_caps(kind: str, new: dict, child_due, dnf):
    modify_completion_compute = _module("modify_completion_compute")
    return modify_completion_compute.completion_caps(
        kind,
        new,
        child_due,
        dnf,
        coerce_int=core.coerce_int,
        dtparse=_dtparse,
        estimate_cp_final_by_max=_estimate_cp_final_by_max,
        estimate_anchor_final_by_max=_estimate_anchor_final_by_max,
        cap_from_until_cp=_cap_from_until_cp,
        cap_from_until_anchor=_cap_from_until_anchor,
    )


def _completion_cap_guard_or_stop(new: dict, next_no: int, cap_no: int | None, now_utc: datetime) -> bool:
    modify_completion_compute = _module("modify_completion_compute")
    return modify_completion_compute.completion_cap_guard_or_stop(
        new,
        next_no,
        cap_no,
        now_utc,
        end_chain_summary=_end_chain_summary,
        print_task=_print_task,
    )


def _completion_compute_next_and_limits(
    new: dict,
    kind: str,
    next_no: int,
    now_utc: datetime,
    *,
    preflight=None,
):
    modify_completion_compute = _module("modify_completion_compute")
    modify_runtime = _module("modify_runtime")
    services = modify_runtime.build_compute_services(
        completion_compute_child_due=_completion_compute_child_due,
        completion_until_or_fail=_completion_until_or_fail,
        completion_until_guard_or_stop=_completion_until_guard_or_stop,
        completion_require_child_due_or_fail=_completion_require_child_due_or_fail,
        completion_warn_unreasonable_duration=_completion_warn_unreasonable_duration,
        completion_caps=_completion_caps,
        completion_cap_guard_or_stop=_completion_cap_guard_or_stop,
    )
    computed = modify_completion_compute.completion_compute_next_and_limits(
        new,
        kind,
        next_no,
        now_utc,
        services=services,
    )
    if computed is None:
        return None
    if isinstance(computed, _module("modify_models").CompletionLifecycleResult):
        return computed

    # Direct helper callers may exercise computation without a full
    # Taskwarrior identity.  Completion preflight rejects that shape before a
    # live hook reaches this point, so retain the helper's result for those
    # characterization calls instead of weakening the planner contract.
    if not str(new.get("uuid") or "").strip() or not str(new.get("chainID") or "").strip():
        return computed

    # The existing preflight still owns user-facing validation and terminal
    # panels.  The planner now owns the pure successor payload so the spawn
    # executor does not rebuild it through a second lifecycle path.
    try:
        lifecycle_planner = _module("lifecycle_planner")
        lifecycle_models = _module("lifecycle_models")
        generation = _chain_generation_service()
        candidate = lifecycle_planner.RecurrenceCandidate(
            child_due=computed.child_due,
            metadata=tuple(sorted(dict(computed.meta or {}).items())),
            dnf=computed.dnf,
            until=computed.until_dt,
        )
        fingerprint_fn = getattr(core, "scheduler_config_fingerprint", None)
        fingerprint = fingerprint_fn() if callable(fingerprint_fn) else ""
        plan = lifecycle_planner.plan_candidate_successor(
            lifecycle_models.TaskSnapshot.from_mapping(new),
            lifecycle_models.LifecycleEvent.COMPLETE,
            candidate,
            generation=generation,
            validated_configuration={"scheduler_fingerprint": fingerprint},
            compare_datetimes=_compare_datetimes,
            preflight=(
                lifecycle_planner.LifecyclePreflight.from_context(
                    base_link=preflight.base_no,
                    next_link=preflight.next_no,
                    kind=preflight.kind,
                    chain_id=preflight.chain_id,
                )
                if preflight is not None
                else None
            ),
        )
        if plan.action is lifecycle_models.LifecycleAction.FINALIZE_CHAIN:
            _end_chain_summary(new, "Reached lifecycle successor limit", now_utc)
            _ensure_terminal_chain_off(new, "complete")
            _print_task(new)
            models = _module("modify_models")
            return models.CompletionLifecycleResult(
                state="terminal",
                reason="successor limit reached",
                diagnostic=models.CompletionLifecycleDiagnostic(
                    transition_id=f"{str(new.get('chainID') or '').strip()}:{new.get('link')}->{next_no}",
                    chain_id=str(new.get("chainID") or "").strip(),
                    parent_link=int(new.get("link")) if str(new.get("link") or "").isdigit() else None,
                    child_link=next_no,
                    stage="plan",
                    failure_kind="successor_limit",
                ),
            )
        computed.lifecycle_plan = plan
        computed.planned_child = plan.child_dict()
    except Exception as exc:
        _diag(f"lifecycle planner failed: {type(exc).__name__}: {exc}")
        _panel("⛓ Chain error", [("Reason", str(exc) or "Could not construct a lifecycle successor plan")], kind="error")
        _print_task(new)
        models = _module("modify_models")
        return models.CompletionLifecycleResult(
            state="retryable",
            reason=str(exc).strip() or "Could not construct a lifecycle successor plan",
            diagnostic=models.CompletionLifecycleDiagnostic(
                transition_id=f"{str(new.get('chainID') or '').strip()}:{new.get('link')}->{next_no}",
                chain_id=str(new.get("chainID") or "").strip(),
                parent_link=int(new.get("link")) if str(new.get("link") or "").isdigit() else None,
                child_link=next_no,
                stage="plan",
                failure_kind="planner_error",
            ),
        )
    return computed


def _completion_build_and_spawn_child(
    new: dict,
    *,
    child_due,
    child_field: str,
    next_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt,
    planned_child: dict | None = None,
):
    modify_completion_spawn = _module("modify_completion_spawn")
    modify_runtime = _module("modify_runtime")
    generation = _chain_generation_service()
    services = modify_runtime.build_spawn_services(
        build_child_from_parent=generation.build_child_from_parent,
        spawn_child_atomic=_spawn_child_atomic,
        panel=_panel,
        print_task=_print_task,
        diag=_diag,
    )
    return modify_completion_spawn.completion_build_and_spawn_child(
        new,
        child_due=child_due,
        child_field=child_field,
        next_no=next_no,
        parent_short=parent_short,
        kind=kind,
        cpmax=cpmax,
        until_dt=until_dt,
        planned_child=planned_child,
        services=services,
    )


def _handle_completion_modify(old: dict, new: dict) -> "CompletionLifecycleResult | None":
    new_cp, new_anchor, new_anchor_file = _completion_validate_cp_and_anchor(old, new)
    _preserve_cp_relative_offsets_on_due_change(old, new, new_cp)
    if any(str(old.get(field) or "").strip() for field in ("cp", "anchor", "anchor_file")):
        recurrence_kind = "cp" if new_cp else "anchor_file" if new_anchor_file else "anchor"
        _preserve_native_until_on_target_change(old, new, recurrence_kind)
    _validate_native_until_after_target_or_fail(new)
    _validate_native_until_anchor_slots_or_fail(new)
    now_utc = core.now_utc()
    ctx = _completion_preflight_context(new, now_utc)
    if ctx is None:
        return
    parent_short = ctx.parent_short
    base_no = ctx.base_no
    next_no = ctx.next_no
    kind = ctx.kind
    chain_id = ctx.chain_id

    computed = _completion_compute_next_and_limits(
        new,
        kind,
        next_no,
        now_utc,
        preflight=ctx,
    )
    if computed is None:
        return
    if isinstance(computed, _module("modify_models").CompletionLifecycleResult):
        _diag_lifecycle_result(computed)
        return computed
    snapshot = ctx.chain_snapshot
    preloaded_chain = list(snapshot.rows)
    indexes = _lifecycle_read_service().build_indexes(preloaded_chain)
    preloaded_chain_by_link, preloaded_chain_by_short = indexes.by_link, indexes.by_short
    if snapshot.mode == "full" and snapshot.loaded:
        _lifecycle_read_service().replace_chain_cache(chain_id, preloaded_chain)
        _diag_count("chain_cache_seeded")
        _export_uuid_short_cached.cache_clear()
    modify_completion_flow = importlib.import_module("nautical_core.modify_completion_flow")
    services = modify_completion_flow.CompletionFinalizeServices(
        build_and_spawn_child=_completion_build_and_spawn_child,
        seed_runtime_lookup_tasks=_seed_runtime_lookup_tasks,
        modify_chain_state=_modify_chain_state,
        export_uuid_short_cached=_export_uuid_short_cached,
        lifecycle_read_service=_lifecycle_read_service(),
        chain_health_advice=_chain_health_advice,
        chain_integrity_warnings=_chain_integrity_warnings,
        render_anchor_completion_feedback=_render_anchor_completion_feedback,
        render_cp_completion_feedback=_render_cp_completion_feedback,
        render_lifecycle_result=_render_lifecycle_result,
        print_task=_print_task,
        diag_summary=_diag_summary,
        show_analytics=_SHOW_ANALYTICS,
        check_integrity=_CHECK_CHAIN_INTEGRITY,
        analytics_style=_ANALYTICS_STYLE,
    )
    result = modify_completion_flow.finalize_completion_modify(
        new=new,
        ctx=ctx,
        computed=computed,
        now_utc=now_utc,
        need_chain=snapshot.mode == "full",
        chain_snapshot_loaded=snapshot.loaded,
        preloaded_chain=preloaded_chain,
        preloaded_chain_by_link=preloaded_chain_by_link,
        preloaded_chain_by_short=preloaded_chain_by_short,
        chain_id=chain_id,
        services=services,
    )
    _diag_lifecycle_result(result)
    return result


def _expiration_services():
    modify_expiration = _module("modify_expiration")
    generation = _chain_generation_service()
    return modify_expiration.ExpirationServices(
        core=core,
        reconcile=_module("reconcile"),
        safe_parse_datetime=_safe_parse_datetime,
        compute_anchor_child_due=generation.compute_anchor_child_due,
        compute_cp_child_due=generation.compute_cp_child_due,
        build_child_from_parent=generation.build_child_from_parent,
        spawn_child_atomic=_spawn_child_atomic,
        panel=_panel,
        short=_short,
        diag=_diag,
    )


def _expiration_recovery_warning(new: dict, reason: str) -> None:
    modify_expiration = _module("modify_expiration", required=False)
    if modify_expiration is not None:
        try:
            modify_expiration.render_recovery_warning(new, reason, services=_expiration_services())
            return
        except Exception as exc:
            _diag(f"expiration recovery warning render failed: {exc}")
    _panel(
        "⚠ Nautical expiration recovery deferred",
        [
            ("Task", _short(new.get("uuid")) or "–"),
            ("Reason", reason or "The next occurrence could not be prepared."),
            ("Action", "Run nautical reconcile --apply."),
        ],
        kind="warning",
    )


def _handle_expired_deleted_modify(new: dict) -> bool:
    modify_expiration = _module("modify_expiration")
    return modify_expiration.handle_expired_deleted_modify(new, services=_expiration_services())


def _handle_deleted_modify(old: dict, new: dict) -> None:
    if str(old.get("status") or "").strip().lower() != "pending":
        return
    if not ((old.get("chainID") or new.get("chainID") or "").strip()):
        return
    modify_expiration = _module("modify_expiration", required=False)
    if modify_expiration is None:
        _expiration_recovery_warning(new, "Expiration recovery module is unavailable; deletion was not classified.")
        return
    try:
        deletion_evidence = modify_expiration.classify_deleted_task(
            new,
            services=_expiration_services(),
        )
        disposition = deletion_evidence.disposition.value
        disposition_reason = deletion_evidence.reason
    except Exception as exc:
        _diag(f"deleted-task disposition failed: {exc}")
        _expiration_recovery_warning(new, "Deletion evidence could not be classified safely.")
        return
    if disposition == "ambiguous":
        _expiration_recovery_warning(
            new,
            disposition_reason or "Deletion evidence is unavailable or malformed.",
        )
        return
    if disposition == "expiration":
        try:
            if _handle_expired_deleted_modify(new):
                return
        except Exception as exc:
            _diag(f"expiration recovery failed: {exc}")
        _expiration_recovery_warning(
            new,
            "Expiration recovery could not be initialized; the chain remains active.",
        )
        return
    if disposition == "manual":
        _diag("deleted Nautical task classified as manual stop")

    _ensure_terminal_chain_off(new, "manual_delete")
    now_utc = core.now_utc()
    try:
        _end_chain_summary(new, "Pending task deleted.", now_utc, current_task=old)
    except Exception as exc:
        _diag(f"delete chain summary failed: {exc}")
        _panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", "Pending Nautical task was deleted."),
                ("Root", _format_root_and_age(old, now_utc)),
                ("Task", _short(old.get("uuid")) or "–"),
            ],
            kind="summary",
        )


def main():
    # Keep module import cheap while preserving the existing full-hook
    # contract: all mutation decisions run with the validated core loaded.
    _load_core()
    _migrate_legacy_nautical_state()
    _reset_modify_runtime_state()
    state = _modify_runtime_state()
    startup_t0 = _ptime.perf_counter()
    module_t0 = _ptime.perf_counter()
    hook_context = _module("hook_context")
    hook_results = _module("hook_results")
    hook_engine = _module("hook_engine")
    state.diag_stats["startup_module_ms"] = round((_ptime.perf_counter() - module_t0) * 1000.0, 3)
    read_t0 = _ptime.perf_counter()
    old, new = _read_two()
    _apply_description_uda_aliases(old, new)
    config_error = str(getattr(core, "scheduling_configuration_error", lambda: "")() or "")
    if config_error and _task_has_nautical_fields(old, new):
        _fail_and_exit(
            "Invalid Nautical configuration",
            f"{config_error}. Fix Nautical configuration before modifying a recurring task.",
        )
    state.diag_stats["startup_read_input_ms"] = round((_ptime.perf_counter() - read_t0) * 1000.0, 3)
    try:
        calendar_context = core.use_task_business_calendar(new)
    except Exception as exc:
        _fail_and_exit("Invalid business calendar", str(exc))
        return
    request_t0 = _ptime.perf_counter()
    _seed_runtime_lookup_tasks(old, new)
    request = hook_context.build_on_modify_request(
        runtime=_build_hook_runtime_context(),
        old=old,
        new=new,
    )
    if _IMPORT_MS is not None:
        state.diag_stats["startup_import_ms"] = round(float(_IMPORT_MS), 3)
    state.diag_stats["startup_request_ms"] = round((_ptime.perf_counter() - request_t0) * 1000.0, 3)
    state.diag_stats["startup_total_ms"] = round((_ptime.perf_counter() - startup_t0) * 1000.0, 3)
    displacement_context = (
        core.capture_business_calendar_displacements()
        if str(new.get("bc") or "").strip()
        else nullcontext()
    )
    with calendar_context, displacement_context:
        result = hook_engine.handle_on_modify(
            request,
            json_result_cls=hook_results.TaskHookResponse,
            task_has_nautical_fields=_task_has_nautical_fields,
            load_core=_load_core,
            diag=_diag,
            fail_and_exit=_fail_and_exit,
            is_non_completion_modify=_is_non_completion_modify,
            handle_non_completion_modify=_handle_non_completion_modify,
            handle_completion_modify=_handle_completion_modify,
            handle_deleted_modify=_handle_deleted_modify,
        )
    if result is not None:
        hook_results.emit_json_result(result, core=core)


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
    global _SPAWN_QUEUE_LOCK, _SPAWN_QUEUE_DB_PATH, _DEAD_LETTER_PATH, _DEAD_LETTER_LOCK

    HOOK_DIR = Path(hook_dir)
    TW_DIR = HOOK_DIR.parent
    _CORE_BASE = Path(core_base)
    sys.argv = [sys.argv[0], *argv]
    _TASKDATA_RAW, _USE_RC_DATA_LOCATION = _resolve_task_data_context()
    TW_DATA_DIR = Path(_TASKDATA_RAW).expanduser()

    state_dir = TW_DATA_DIR / ".nautical-state"
    lock_dir = TW_DATA_DIR / ".nautical-locks"
    _SPAWN_QUEUE_LOCK = lock_dir / ".nautical_spawn_queue.lock"
    _SPAWN_QUEUE_DB_PATH = state_dir / ".nautical_queue.db"
    _DEAD_LETTER_PATH = state_dir / ".nautical_dead_letter.jsonl"
    _DEAD_LETTER_LOCK = lock_dir / ".nautical_dead_letter.lock"

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
