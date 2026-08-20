#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heavy on-exit implementation loaded lazily by the executable wrapper.

Drains typed Nautical lifecycle outbox intents after Taskwarrior releases its lock.
"""

from __future__ import annotations

import sys
import os
import time
import importlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Mapping

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
from contextlib import contextmanager
from typing import Any

core = None
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
_HOOK_SUPPORT = None
_HOOK_SUPPORT_LOAD_FAILED = False
_EXIT_RUNTIME = None
_EXIT_RUNTIME_LOAD_FAILED = False
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
_LIFECYCLE_APPLICATION = None
_LIFECYCLE_APPLICATION_LOAD_FAILED = False
_LIFECYCLE_OUTBOX = None
_LIFECYCLE_OUTBOX_LOAD_FAILED = False
_INTEGRATION_CONTEXT = None
_CORE_READY = False
_HOOK_MODULE_ACCESS = None
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
    "hook_support": (
        "_HOOK_SUPPORT",
        "_HOOK_SUPPORT_LOAD_FAILED",
        "hook_support.py",
        "nautical_core.hook_support",
    ),
    "exit_runtime": (
        "_EXIT_RUNTIME",
        "_EXIT_RUNTIME_LOAD_FAILED",
        "exit_runtime.py",
        "nautical_core.exit_runtime",
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
    globals()["_OUTBOX_BATCH_MAX_ITEMS"] = _env_int(
        "NAUTICAL_OUTBOX_DRAIN_MAX_ITEMS",
        int(getattr(core, "OUTBOX_DRAIN_MAX_ITEMS", 200)),
        min_value=1,
        max_value=100000,
    )
    _CORE_READY = True


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

HOOK_IMPL_API = 1
NAUTICAL_HOOK_VERSION = "updateG-20260328"
# When set, exit 1 if any lifecycle plans require manual review or errored.
_EXIT_STRICT = (os.environ.get("NAUTICAL_EXIT_STRICT") or "").strip().lower() in ("1", "true", "yes", "on")

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


_OUTBOX_BATCH_MAX_ITEMS = _env_int(
    "NAUTICAL_OUTBOX_DRAIN_MAX_ITEMS",
    200,
    min_value=1,
    max_value=100000,
)
_OUTBOX_RETRY_MAX = _env_int("NAUTICAL_OUTBOX_RETRY_MAX", 6, min_value=0, max_value=100)
_TASK_TIMEOUT_EXPORT = _env_float("NAUTICAL_TASK_TIMEOUT_EXPORT", 3.0, min_value=0.1, max_value=300.0)
_TASK_TIMEOUT_IMPORT = _env_float("NAUTICAL_TASK_TIMEOUT_IMPORT", 8.0, min_value=0.1, max_value=300.0)
_TASK_TIMEOUT_MODIFY = _env_float("NAUTICAL_TASK_TIMEOUT_MODIFY", 4.0, min_value=0.1, max_value=300.0)
_TASK_RETRIES_EXPORT = _env_int("NAUTICAL_TASK_RETRIES_EXPORT", 2, min_value=0, max_value=20)
_TASK_RETRIES_MODIFY = _env_int("NAUTICAL_TASK_RETRIES_MODIFY", 2, min_value=0, max_value=20)
_TASK_RETRY_DELAY = _env_float("NAUTICAL_TASK_RETRY_DELAY", 0.2, min_value=0.0, max_value=10.0)
_PARENT_LOCK_RETRIES = _env_int("NAUTICAL_PARENT_LOCK_RETRIES", 6, min_value=0, max_value=100)
_PARENT_LOCK_SLEEP_BASE = _env_float("NAUTICAL_PARENT_LOCK_SLEEP_BASE", 0.03, min_value=0.0, max_value=10.0)
_PARENT_LOCK_STALE_AFTER = _env_float("NAUTICAL_PARENT_LOCK_STALE_AFTER", 30.0, min_value=0.0, max_value=86400.0)
_LOCK_STORM_THRESHOLD = _env_int("NAUTICAL_LOCK_STORM_THRESHOLD", 8, min_value=0, max_value=1000)
_LOCK_BACKOFF_BASE = _env_float("NAUTICAL_LOCK_BACKOFF_BASE", 0.05, min_value=0.0, max_value=60.0)
_LOCK_BACKOFF_MAX = _env_float("NAUTICAL_LOCK_BACKOFF_MAX", 1.0, min_value=0.0, max_value=300.0)
_OUTBOX_LEASE_SECONDS = _env_float(
    "NAUTICAL_OUTBOX_LEASE_SECONDS",
    300.0,
    min_value=0.0,
    max_value=7 * 86400.0,
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
def _drain_outbox_result(unit_of_work) -> dict[str, Any]:
    """Claim and execute one bounded batch of lifecycle intents.

    All staging, mutation, verification, and recovery logic lives in
    ``lifecycle_application.LifecycleApplicationService``; this hook only
    constructs the service from the invocation's unit of work, drains one
    bounded batch, and turns the typed outcomes into a diagnostics dict.
    """
    _reset_exit_runtime_state()
    state = _exit_runtime_state()
    state.unit_of_work = unit_of_work
    state.repository = unit_of_work.repository
    _reset_exit_diag_stats()
    unit_of_work.repository.configure_commands(
        timeout=_TASK_TIMEOUT_EXPORT,
        attempts=_TASK_RETRIES_EXPORT,
        retry_delay=_TASK_RETRY_DELAY,
    )

    lifecycle_application = _module("lifecycle_application")
    taskwarrior_mutations = _module("taskwarrior_mutations")
    lifecycle_outbox = _module("lifecycle_outbox")

    mutations = taskwarrior_mutations.TaskwarriorMutationService(unit_of_work)
    outbox = lifecycle_outbox.LifecycleOutboxRepository(unit_of_work.outbox.taskdata)
    owner = f"exit-{os.getpid()}-{os.urandom(8).hex()}"
    service = lifecycle_application.LifecycleApplicationService(
        unit_of_work=unit_of_work,
        mutations=mutations,
        outbox=outbox,
        owner=owner,
        lease_seconds=_OUTBOX_LEASE_SECONDS,
    )

    configuration = _INTEGRATION_CONTEXT.configuration
    drain_t0 = time.perf_counter()
    result = service.drain(
        limit=_OUTBOX_BATCH_MAX_ITEMS,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    drain_ms = round((time.perf_counter() - drain_t0) * 1000.0, 3)

    outcome_kind = lifecycle_application.LifecycleApplicationOutcomeKind
    outcomes = result.outcomes
    processed = sum(1 for o in outcomes if o.ok)
    retry_released = sum(1 for o in outcomes if o.kind is outcome_kind.RETRYABLE)
    manual_reviewed = sum(1 for o in outcomes if o.kind is outcome_kind.MANUAL_REVIEW)
    quarantined = sum(1 for o in outcomes if o.kind is outcome_kind.QUARANTINED)
    conflicted = sum(1 for o in outcomes if o.kind is outcome_kind.CONFLICT)
    errors = len(outcomes) - processed

    outbox_lock_failures = 0
    if not result.claim.ok:
        outbox_lock_failures = 1
        _diag(f"lifecycle outbox claim failed: {result.claim.reason or 'unknown error'}")

    for outcome in outcomes:
        if outcome.reason:
            _diag(
                f"lifecycle intent {outcome.intent_id or '(unstaged)'}: "
                f"{outcome.kind.value}: {outcome.reason}"
            )

    commands = getattr(unit_of_work, "commands", None)
    if commands is not None:
        state.diag_stats.update(
            run_task_calls=max(0, int(getattr(commands, "calls", 0) or 0)),
            run_task_failures=max(0, int(getattr(commands, "failures", 0) or 0)),
            run_task_seconds=max(0.0, float(getattr(commands, "duration", 0.0) or 0.0)),
        )

    return {
        "entries_total": len(outcomes),
        "processed": processed,
        "errors": errors,
        "retry_released": retry_released,
        "manual_reviewed": manual_reviewed,
        "quarantined": quarantined,
        "conflicted": conflicted,
        "outbox_lock_failures": outbox_lock_failures,
        "drain_ms": drain_ms,
    }


def _drain_outbox(unit_of_work) -> dict:
    return _drain_outbox_result(unit_of_work)


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
        ("processed", stats.get("processed", 0)),
        ("errors", stats.get("errors", 0)),
        ("retry_released", stats.get("retry_released", 0)),
        ("manual_reviewed", stats.get("manual_reviewed", 0)),
        ("quarantined", stats.get("quarantined", 0)),
        ("conflicted", stats.get("conflicted", 0)),
        ("outbox_lock_failures", stats.get("outbox_lock_failures", 0)),
        ("drain_ms", stats.get("drain_ms", 0)),
    ]
    _diag_block("on-exit drain", drain_items, columns=3)
    diag_stats = _exit_runtime_state().diag_stats
    task_stats = {
        "run_task_calls": diag_stats.get("run_task_calls", 0),
        "run_task_failures": diag_stats.get("run_task_failures", 0),
        "run_task_seconds": round(float(diag_stats.get("run_task_seconds", 0.0)), 4),
    }
    _diag_block("on-exit task stats", task_stats.items(), columns=3)


def _strict_exit_feedback_message(stats: dict) -> str | None:
    errors = stats.get("errors", 0)
    manual_reviewed = stats.get("manual_reviewed", 0)
    outbox_lock_failures = stats.get("outbox_lock_failures", 0)
    if not (_EXIT_STRICT and (errors > 0 or manual_reviewed > 0 or outbox_lock_failures > 0)):
        return None
    return (
        f"[nautical] on-exit: {manual_reviewed} manual-review intents, {errors} errors, "
        f"{outbox_lock_failures} outbox lock failures. Check nautical queue-status "
        "(set NAUTICAL_EXIT_STRICT=0 to disable)"
    )


class _OnExitServices:
    """Concrete adapter passed to the shared hook router."""

    def __init__(self, result_cls):
        self._result_cls = result_cls

    def redirect_stdout(self):
        _redirect_stdout_to_devnull()

    def drain_outbox(self, unit_of_work):
        return _drain_outbox(unit_of_work)

    def strict_feedback(self, stats):
        return _strict_exit_feedback_message(stats)

    def result(self, *, exit_code: int, feedback_message: str | None, stats):
        return self._result_cls(
            exit_code=exit_code,
            feedback_message=feedback_message,
            stats=stats,
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
    manual_reviewed = count("manual_reviewed")
    quarantined = count("quarantined")
    if not (errors or manual_reviewed or quarantined):
        return

    problems = []
    if manual_reviewed:
        problems.append(f"{manual_reviewed} manual-review intents")
    if quarantined:
        suffix = "" if quarantined == 1 else "s"
        problems.append(f"{quarantined} quarantined intent{suffix}")
    other_errors = max(0, errors - manual_reviewed - quarantined)
    if other_errors:
        suffix = "" if other_errors == 1 else "s"
        problems.append(f"{other_errors} other drain error{suffix}")

    rows = [
        ("Action", "Run nautical queue-status"),
        ("Problems", "; ".join(problems) or f"{errors} drain errors"),
    ]
    if manual_reviewed or quarantined:
        rows.append(("Review", "Run nautical queue-status"))
    retry_released = count("retry_released")
    if retry_released:
        rows.append(("Retrying", str(retry_released)))
    outbox_lock_failures = count("outbox_lock_failures")
    if outbox_lock_failures:
        rows.append(("Lock events", str(outbox_lock_failures)))

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
        services=_OnExitServices(hook_results.ExitHookResponse),
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
        raise SystemExit(1)
