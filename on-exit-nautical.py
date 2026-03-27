#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
on-exit-nautical.py

Drains Nautical spawn intents after Taskwarrior releases its lock.
Reads JSONL queue entries, imports child tasks, and updates parent nextLink.
"""

from __future__ import annotations

import sys
import os
import json
import time
import subprocess
import random
import sqlite3
import importlib.util
from pathlib import Path
from contextlib import contextmanager
from typing import Any
try:
    import fcntl  # POSIX advisory lock
except Exception:
    fcntl = None


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

def _trusted_core_base(default_base: Path) -> Path:
    raw = (os.environ.get("NAUTICAL_CORE_PATH") or "").strip()
    if not raw:
        return default_base
    try:
        cand = Path(raw).expanduser().resolve()
    except Exception:
        return default_base
    if (os.environ.get("NAUTICAL_TRUST_CORE_PATH") or "").strip().lower() in ("1", "true", "yes", "on"):
        return cand
    try:
        st = os.stat(cand)
        uid_fn = getattr(os, "getuid", None)
        if callable(uid_fn) and st.st_uid != uid_fn():
            raise PermissionError("owner mismatch")
        if (st.st_mode & 0o002) != 0:
            raise PermissionError("path is world-writable")
        return cand
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe NAUTICAL_CORE_PATH '{raw}': {e}\n")
            except Exception:
                pass
        return default_base

_CORE_BASE = _trusted_core_base(TW_DIR)


def _core_target_from_base(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "nautical_core.py":
                return base
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                return base
            return None
    except Exception:
        return None
    pyfile = base / "nautical_core.py"
    pkgini = base / "nautical_core" / "__init__.py"
    return pkgini if pkgini.is_file() else pyfile if pyfile.is_file() else None

core = None
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
_HOOK_LOADER = None
_HOOK_LOADER_LOAD_FAILED = False
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
_HOOK_CONTEXT = None
_HOOK_CONTEXT_LOAD_FAILED = False
_HOOK_ENGINE = None
_HOOK_ENGINE_LOAD_FAILED = False
_HOOK_RESULTS = None
_HOOK_RESULTS_LOAD_FAILED = False
_MODULE_SPECS = {
    "hook_support": (
        "_HOOK_SUPPORT",
        "_HOOK_SUPPORT_LOAD_FAILED",
        "hook_support.py",
        "nautical_hook_support",
    ),
    "exit_queries": (
        "_EXIT_QUERIES",
        "_EXIT_QUERIES_LOAD_FAILED",
        "exit_queries.py",
        "nautical_exit_queries",
    ),
    "exit_side_effects": (
        "_EXIT_SIDE_EFFECTS",
        "_EXIT_SIDE_EFFECTS_LOAD_FAILED",
        "exit_side_effects.py",
        "nautical_exit_side_effects",
    ),
    "exit_entry_flow": (
        "_EXIT_ENTRY_FLOW",
        "_EXIT_ENTRY_FLOW_LOAD_FAILED",
        "exit_entry_flow.py",
        "nautical_exit_entry_flow",
    ),
    "queue_store": (
        "_QUEUE_STORE",
        "_QUEUE_STORE_LOAD_FAILED",
        "queue_store.py",
        "nautical_queue_store",
    ),
    "queue_models": (
        "_QUEUE_MODELS",
        "_QUEUE_MODELS_LOAD_FAILED",
        "queue_models.py",
        "nautical_queue_models",
    ),
    "exit_models": (
        "_EXIT_MODELS",
        "_EXIT_MODELS_LOAD_FAILED",
        "exit_models.py",
        "nautical_exit_models",
    ),
    "hook_context": (
        "_HOOK_CONTEXT",
        "_HOOK_CONTEXT_LOAD_FAILED",
        "hook_context.py",
        "nautical_hook_context",
    ),
    "hook_engine": (
        "_HOOK_ENGINE",
        "_HOOK_ENGINE_LOAD_FAILED",
        "hook_engine.py",
        "nautical_hook_engine",
    ),
    "hook_results": (
        "_HOOK_RESULTS",
        "_HOOK_RESULTS_LOAD_FAILED",
        "hook_results.py",
        "nautical_hook_results",
    ),
}
try:
    target = _core_target_from_base(_CORE_BASE)
    _CORE_IMPORT_TARGET = target
    if target:
        spec = importlib.util.spec_from_file_location("nautical_core", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_core"] = module
            spec.loader.exec_module(module)
            core = module
            try:
                core._warn_once_per_day_any("core_path", f"[nautical] core loaded: {getattr(core, '__file__', 'unknown')}")
            except Exception:
                pass
except Exception as e:
    core = None
    _CORE_IMPORT_ERROR = e

def _resolve_task_data_context() -> tuple[str, bool]:
    resolver = getattr(core, "resolve_task_data_context", None) if core is not None else None
    if not callable(resolver):
        if core is not None:
            raise RuntimeError("nautical_core.resolve_task_data_context is required")
        if _CORE_IMPORT_ERROR is not None:
            target = str(_CORE_IMPORT_TARGET or (_CORE_BASE / "nautical_core.py"))
            raise RuntimeError(
                f"Failed to import nautical_core from {target}: "
                f"{type(_CORE_IMPORT_ERROR).__name__}: {_CORE_IMPORT_ERROR}"
            ) from _CORE_IMPORT_ERROR
        raise ModuleNotFoundError(
            f"nautical_core.py not found. Expected in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {_CORE_BASE})"
        )
    data_dir, use_rc, _source = resolver(
        argv=sys.argv[1:],
        env=os.environ,
        tw_dir=str(TW_DIR),
    )
    return str(data_dir), bool(use_rc)


_TASKDATA_RAW, _USE_RC_DATA_LOCATION = _resolve_task_data_context()
TW_DATA_DIR = Path(_TASKDATA_RAW).expanduser()


def _tw_data_dir_path() -> Path:
    td = TW_DATA_DIR
    if isinstance(td, Path):
        return td
    try:
        return Path(str(td)).expanduser()
    except Exception:
        return Path(".")


def _hook_loader_target(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                target = base.parent / "hook_loader.py"
                return target if target.is_file() else None
            return None
    except Exception:
        return None
    target = base / "nautical_core" / "hook_loader.py"
    return target if target.is_file() else None


def _load_hook_loader():
    global _HOOK_LOADER, _HOOK_LOADER_LOAD_FAILED
    if _HOOK_LOADER is not None:
        return _HOOK_LOADER
    if _HOOK_LOADER_LOAD_FAILED:
        return None
    base = _CORE_IMPORT_TARGET or _CORE_BASE
    target = _hook_loader_target(base)
    if not target:
        _HOOK_LOADER_LOAD_FAILED = True
        return None
    try:
        spec = importlib.util.spec_from_file_location("nautical_hook_loader", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_hook_loader"] = module
            spec.loader.exec_module(module)
            _HOOK_LOADER = module
            return module
    except Exception:
        pass
    _HOOK_LOADER_LOAD_FAILED = True
    return None


def _load_named_module(name: str):
    hook_loader = _load_hook_loader()
    if hook_loader is None:
        return None
    return hook_loader.load_named_module(
        name,
        _MODULE_SPECS,
        globals(),
        globals(),
        base=_CORE_IMPORT_TARGET or _CORE_BASE,
    )


def _require_loaded_module(module, rel_name: str):
    hook_loader = _load_hook_loader()
    if hook_loader is not None:
        return hook_loader.require_loaded_module(module, rel_name)
    if module is None:
        raise RuntimeError(f"nautical_core/{rel_name} is required")
    return module


def _module(name: str, *, required: bool = True):
    module = _load_named_module(name)
    if not required:
        return module
    rel_name = _MODULE_SPECS[name][2]
    return _require_loaded_module(module, rel_name)

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





def _build_hook_runtime_context():
    hook_context = _module("hook_context")
    return hook_context.build_hook_runtime_context(
        hook_name="on-exit",
        taskdata_dir=str(TW_DATA_DIR),
        use_rc_data_location=_USE_RC_DATA_LOCATION,
        tw_dir=str(TW_DIR),
        hook_dir=str(HOOK_DIR),
    )

def _nautical_state_dir_path() -> Path:
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        return queue_store.nautical_state_dir_path(TW_DATA_DIR)
    return TW_DATA_DIR / ".nautical-state"

def _nautical_lock_dir_path() -> Path:
    queue_store = _module("queue_store", required=False)
    if queue_store is not None:
        return queue_store.nautical_lock_dir_path(TW_DATA_DIR)
    return TW_DATA_DIR / ".nautical-locks"

_QUEUE_PATH = _nautical_state_dir_path() / ".nautical_spawn_queue.jsonl"
_QUEUE_PROCESSING_PATH = _nautical_state_dir_path() / ".nautical_spawn_queue.processing.jsonl"
_QUEUE_LOCK = _nautical_lock_dir_path() / ".nautical_spawn_queue.lock"
_QUEUE_DB_PATH = _nautical_state_dir_path() / ".nautical_queue.db"
_DEAD_LETTER_PATH = _nautical_state_dir_path() / ".nautical_dead_letter.jsonl"
_DEAD_LETTER_LOCK = _nautical_lock_dir_path() / ".nautical_dead_letter.lock"
_DEAD_LETTER_RETENTION_DAYS = int(os.environ.get("NAUTICAL_DEAD_LETTER_RETENTION_DAYS") or 30)
_QUEUE_MAX_BYTES = int(os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_BYTES") or 524288)
_QUEUE_MAX_LINES = int(os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_LINES") or 10000)
_DEAD_LETTER_MAX_BYTES = int(os.environ.get("NAUTICAL_DEAD_LETTER_MAX_BYTES") or 524288)
_QUEUE_QUARANTINE_PATH = _nautical_state_dir_path() / ".nautical_spawn_queue.bad.jsonl"
_QUEUE_QUARANTINE_MAX_BYTES = int(os.environ.get("NAUTICAL_QUEUE_BAD_MAX_BYTES") or 262144)
NAUTICAL_HOOK_VERSION = "updateE-20260126"
_QUEUE_RETRY_MAX = int(os.environ.get("NAUTICAL_QUEUE_RETRY_MAX") or 6)
_QUEUE_LOCK_FAIL_MARKER = _nautical_lock_dir_path() / ".nautical_spawn_queue.lock_failed"
_QUEUE_LOCK_FAIL_COUNT = _nautical_lock_dir_path() / ".nautical_spawn_queue.lock_failed.count"
_DURABLE_QUEUE = os.environ.get("NAUTICAL_DURABLE_QUEUE") == "1"
# When set, exit 1 if any spawns were dead-lettered or errored (for scripting/monitoring).
_EXIT_STRICT = (os.environ.get("NAUTICAL_EXIT_STRICT") or "").strip().lower() in ("1", "true", "yes", "on")

def _migrate_legacy_nautical_state() -> None:
    queue_store = _module("queue_store", required=False)
    file_pairs = (
        (_QUEUE_PATH, TW_DATA_DIR / ".nautical_spawn_queue.jsonl"),
        (_QUEUE_PROCESSING_PATH, TW_DATA_DIR / ".nautical_spawn_queue.processing.jsonl"),
        (_QUEUE_DB_PATH, TW_DATA_DIR / ".nautical_queue.db"),
        (_DEAD_LETTER_PATH, TW_DATA_DIR / ".nautical_dead_letter.jsonl"),
        (_QUEUE_QUARANTINE_PATH, TW_DATA_DIR / ".nautical_spawn_queue.bad.jsonl"),
        (_intent_log_path(), TW_DATA_DIR / ".nautical_spawn_intents.jsonl"),
    )
    db_sidecars = (
        (_QUEUE_DB_PATH, TW_DATA_DIR / ".nautical_queue.db"),
    )
    if queue_store is not None:
        queue_store.migrate_legacy_state(file_pairs=file_pairs, db_sidecars=db_sidecars)
        return
    for current, legacy in file_pairs:
        try:
            if current.exists() or not legacy.exists():
                continue
            current.parent.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, current)
        except Exception:
            pass
    for current, legacy in db_sidecars:
        for suffix in ("-wal", "-shm"):
            try:
                sidecar_current = Path(str(current) + suffix)
                sidecar_legacy = Path(str(legacy) + suffix)
                if sidecar_current.exists() or not sidecar_legacy.exists():
                    continue
                sidecar_current.parent.mkdir(parents=True, exist_ok=True)
                os.replace(sidecar_legacy, sidecar_current)
            except Exception:
                pass

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return float(default)

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return int(default)

_TASK_TIMEOUT_EXPORT = _env_float("NAUTICAL_TASK_TIMEOUT_EXPORT", 3.0)
_TASK_TIMEOUT_IMPORT = _env_float("NAUTICAL_TASK_TIMEOUT_IMPORT", 8.0)
_TASK_TIMEOUT_MODIFY = _env_float("NAUTICAL_TASK_TIMEOUT_MODIFY", 4.0)
_TASK_RETRIES_EXPORT = _env_int("NAUTICAL_TASK_RETRIES_EXPORT", 2)
_TASK_RETRIES_MODIFY = _env_int("NAUTICAL_TASK_RETRIES_MODIFY", 2)
_TASK_RETRY_DELAY = _env_float("NAUTICAL_TASK_RETRY_DELAY", 0.2)
_QUEUE_LOCK_RETRIES = _env_int("NAUTICAL_QUEUE_LOCK_RETRIES", 6)
_QUEUE_LOCK_SLEEP_BASE = _env_float("NAUTICAL_QUEUE_LOCK_SLEEP_BASE", 0.03)
_QUEUE_LOCK_STALE_AFTER = _env_float("NAUTICAL_QUEUE_LOCK_STALE_AFTER", 30.0)
_INTENT_LOG_MAX_BYTES = _env_int("NAUTICAL_INTENT_LOG_MAX_BYTES", 524288)
_INTENT_LOG_MAX_ENTRIES = _env_int("NAUTICAL_INTENT_LOG_MAX_ENTRIES", 20000)
_LOCK_STORM_THRESHOLD = _env_int("NAUTICAL_LOCK_STORM_THRESHOLD", 8)
_LOCK_BACKOFF_BASE = _env_float("NAUTICAL_LOCK_BACKOFF_BASE", 0.05)
_LOCK_BACKOFF_MAX = _env_float("NAUTICAL_LOCK_BACKOFF_MAX", 1.0)
_QUEUE_PROCESSING_STALE_AFTER = _env_float("NAUTICAL_QUEUE_PROCESSING_STALE_AFTER", 300.0)
_QUEUE_DB_CONNECT_RETRIES = _env_int("NAUTICAL_QUEUE_DB_CONNECT_RETRIES", 3)
_QUEUE_DB_CONNECT_TIMEOUT_MAX = _env_float("NAUTICAL_QUEUE_DB_CONNECT_TIMEOUT_MAX", 60.0)
_QUEUE_DB_CONNECT_BACKOFF_BASE = _env_float("NAUTICAL_QUEUE_DB_CONNECT_BACKOFF_BASE", 0.05)

_RUN_QUEUE_DB_CONN: sqlite3.Connection | None = None
_RUN_QUEUE_DB_ACTIVE = False
_QUEUE_DB_OPEN_COUNT = 0
_QUEUE_DB_REUSE_COUNT = 0


def _queue_db_begin_run() -> None:
    global _RUN_QUEUE_DB_ACTIVE, _QUEUE_DB_OPEN_COUNT, _QUEUE_DB_REUSE_COUNT
    _RUN_QUEUE_DB_ACTIVE = True
    _QUEUE_DB_OPEN_COUNT = 0
    _QUEUE_DB_REUSE_COUNT = 0


def _queue_db_end_run() -> None:
    global _RUN_QUEUE_DB_ACTIVE, _RUN_QUEUE_DB_CONN
    _RUN_QUEUE_DB_ACTIVE = False
    conn = _RUN_QUEUE_DB_CONN
    _RUN_QUEUE_DB_CONN = None
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
        core.diag(safe_msg, "on-exit", str(TW_DATA_DIR))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {safe_msg}\n")
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


def _run_task_retry_or_stop(attempt: int, attempts: int, delay: float) -> bool:
    if attempt >= attempts or delay <= 0:
        return False
    jitter = random.uniform(0.0, delay)
    _sleep(delay * (2 ** (attempt - 1)) + jitter)
    return True


def _run_task_terminate(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass


def _run_task_attempt(
    cmd: list[str],
    *,
    env: dict[str, str],
    input_text: str | None,
    timeout: float,
) -> tuple[bool, str, str]:
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
        try:
            out, err = proc.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
            try:
                out, err = proc.communicate(timeout=1.0) if proc is not None else ("", "")
            except Exception:
                out, err = "", ""
            return False, out or "", "timeout"
        out = out or ""
        err = err or ""
        if proc.returncode == 0:
            return True, out, err
        return False, out, err
    except subprocess.TimeoutExpired:
        _run_task_terminate(proc)
        return False, "", "timeout"
    except Exception as e:
        _run_task_terminate(proc)
        return False, "", str(e)


def _run_task(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 6.0,
    retries: int = 1,
    retry_delay: float = 0.0,
) -> tuple[bool, str, str]:
    env_map = env or os.environ.copy()
    run_fn = core.run_task if core is not None else None
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.run_task(
            cmd,
            core_run_task=run_fn,
            env=env_map,
            input_text=input_text,
            timeout=timeout,
            retries=max(1, int(retries)),
            retry_delay=max(0.0, float(retry_delay)),
        )
    if run_fn is not None:
        return run_fn(
            cmd,
            env=env_map,
            input_text=input_text,
            timeout=timeout,
            retries=max(1, int(retries)),
            retry_delay=max(0.0, float(retry_delay)),
        )
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay))
    last_out = ""
    last_err = ""
    for attempt in range(1, attempts + 1):
        ok, out, err = _run_task_attempt(
            cmd,
            env=env_map,
            input_text=input_text,
            timeout=timeout,
        )
        last_out = out or ""
        last_err = err or ""
        if ok:
            return True, last_out, last_err
        if _run_task_retry_or_stop(attempt, attempts, delay):
            continue
        return (False, last_out, last_err)
    return (False, last_out, last_err)


def _is_lock_error(err: str) -> bool:
    if core is not None:
        return core.is_lock_error(err)
    e = (err or "").lower()
    return (
        "database is locked" in e or "unable to lock" in e
        or "resource temporarily unavailable" in e or "another task is running" in e
        or "lock file" in e or "lockfile" in e or "locked by" in e or "timeout" in e
    )

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

_LAST_QUEUE_LOCK_DIAG_TS = 0.0
_QUEUE_LOCK_FAILURES_THIS_RUN = 0

def _record_queue_lock_failure() -> None:
    global _LAST_QUEUE_LOCK_DIAG_TS
    global _QUEUE_LOCK_FAILURES_THIS_RUN
    _QUEUE_LOCK_FAILURES_THIS_RUN += 1
    now = time.time()
    if now - _LAST_QUEUE_LOCK_DIAG_TS >= 60.0:
        _diag("queue lock not acquired; drain deferred")
        _LAST_QUEUE_LOCK_DIAG_TS = now
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
def _lock_queue():
    lock_fn = core.safe_lock if core is not None and hasattr(core, "safe_lock") else _local_safe_lock
    with lock_fn(
        _QUEUE_LOCK,
        retries=_QUEUE_LOCK_RETRIES,
        sleep_base=_QUEUE_LOCK_SLEEP_BASE,
        stale_after=_QUEUE_LOCK_STALE_AFTER,
    ) as acquired:
        yield acquired


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


def _intent_log_path() -> Path:
    return _nautical_state_dir_path() / ".nautical_spawn_intents.jsonl"


def _intent_log_lock_path() -> Path:
    return _nautical_lock_dir_path() / ".nautical_spawn_intents.lock"


_migrate_legacy_nautical_state()


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


def _write_dead_letter(entry: dict, reason: str) -> None:
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hook": "on-exit",
        "hook_version": NAUTICAL_HOOK_VERSION,
        "reason": reason,
        "spawn_intent_id": (entry.get("spawn_intent_id") or "").strip(),
        "parent_uuid": (entry.get("parent_uuid") or "").strip(),
        "child_short": (entry.get("child_short") or "").strip(),
        "child_uuid": ((entry.get("child") or {}).get("uuid") or "").strip(),
        "entry": entry,
    }
    with _lock_dead_letter() as locked:
        if not locked:
            _diag("dead-letter lock busy; entry not recorded")
            return
        try:
            if _DEAD_LETTER_MAX_BYTES > 0 and _DEAD_LETTER_PATH.exists():
                try:
                    st = _DEAD_LETTER_PATH.stat()
                    if st.st_size > _DEAD_LETTER_MAX_BYTES:
                        ts = int(time.time())
                        overflow = _DEAD_LETTER_PATH.with_suffix(f".overflow.{ts}.jsonl")
                        os.replace(_DEAD_LETTER_PATH, overflow)
                        _diag(f"dead-letter rotated: {overflow}")
                        if _DEAD_LETTER_RETENTION_DAYS > 0:
                            try:
                                cutoff = time.time() - (_DEAD_LETTER_RETENTION_DAYS * 86400)
                                candidates = sorted(_DEAD_LETTER_PATH.parent.glob(f"{_DEAD_LETTER_PATH.stem}.overflow.*.jsonl"))
                                for old in candidates:
                                    try:
                                        if old.stat().st_mtime < cutoff:
                                            old.unlink()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
            fd = os.open(str(_DEAD_LETTER_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
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
                        _fsync_dir(_DEAD_LETTER_PATH.parent)
                    except Exception:
                        pass
        except Exception:
            pass

def _quarantine_queue_line(raw_line: str, reason: str) -> None:
    if not raw_line:
        return
    try:
        _QUEUE_QUARANTINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        if _QUEUE_QUARANTINE_MAX_BYTES > 0 and _QUEUE_QUARANTINE_PATH.exists():
            try:
                st = _QUEUE_QUARANTINE_PATH.stat()
                if st.st_size > _QUEUE_QUARANTINE_MAX_BYTES:
                    ts = int(time.time())
                    overflow = _QUEUE_QUARANTINE_PATH.with_suffix(f".overflow.{ts}.jsonl")
                    os.replace(_QUEUE_QUARANTINE_PATH, overflow)
                    _diag(f"queue quarantine rotated: {overflow}")
            except Exception:
                pass
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
            "raw": raw_line,
        }
        fd = os.open(str(_QUEUE_QUARANTINE_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass

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

def _queue_recover_processing_file() -> bool:
    queue_store = _module("queue_store")
    return queue_store.recover_processing_file(
        queue_processing_path=_QUEUE_PROCESSING_PATH,
        queue_path=_QUEUE_PATH,
        durable_queue=_DURABLE_QUEUE,
        fsync_dir_fn=_fsync_dir,
    )


def _queue_source_path_with_overflow() -> tuple[Path, Path | None]:
    queue_store = _module("queue_store")
    return queue_store.source_path_with_overflow(
        queue_path=_QUEUE_PATH,
        queue_max_bytes=_QUEUE_MAX_BYTES,
        diag=_diag,
    )


def _queue_split_source_to_staging(src_path: Path, tmp_path: Path, tmp_processing: Path) -> list[dict]:
    queue_store = _module("queue_store")
    return queue_store.split_source_to_staging(
        src_path=src_path,
        tmp_path=tmp_path,
        tmp_processing=tmp_processing,
        queue_max_lines=_QUEUE_MAX_LINES,
        durable_queue=_DURABLE_QUEUE,
        quarantine_line=_quarantine_queue_line,
        write_dead_letter=_write_dead_letter,
    )


def _queue_commit_staging(
    tmp_path: Path,
    tmp_processing: Path,
    entries: list[dict],
    overflow_path: Path | None,
) -> None:
    queue_store = _module("queue_store")
    queue_store.commit_staging(
        queue_path=_QUEUE_PATH,
        queue_processing_path=_QUEUE_PROCESSING_PATH,
        tmp_path=tmp_path,
        tmp_processing=tmp_processing,
        entries=entries,
        overflow_path=overflow_path,
        durable_queue=_DURABLE_QUEUE,
        fsync_dir_fn=_fsync_dir,
    )


def _queue_cleanup_staging(tmp_path: Path, tmp_processing: Path) -> None:
    queue_store = _module("queue_store")
    queue_store.cleanup_staging(tmp_path, tmp_processing)


def _take_queue_entries_jsonl_batch():
    queue_store = _module("queue_store")
    exit_models = _module("exit_models")
    return exit_models.ExitQueueBatch(
        entries=queue_store.take_queue_entries_jsonl_result(
            queue_path=_QUEUE_PATH,
            queue_processing_path=_QUEUE_PROCESSING_PATH,
            queue_max_bytes=_QUEUE_MAX_BYTES,
            queue_max_lines=_QUEUE_MAX_LINES,
            durable_queue=_DURABLE_QUEUE,
            lock_queue=_lock_queue,
            record_lock_failure=_record_queue_lock_failure,
            quarantine_line=_quarantine_queue_line,
            write_dead_letter=_write_dead_letter,
            fsync_dir_fn=_fsync_dir,
            diag=_diag,
        ).entries
    )


def _take_queue_entries_jsonl() -> list[dict]:
    return _take_queue_entries_jsonl_batch().entries


def _requeue_entries_jsonl_result(entries: list[dict]):
    queue_store = _module("queue_store")
    exit_models = _module("exit_models")
    items = [e for e in (entries or []) if isinstance(e, dict)]
    result = queue_store.requeue_entries_jsonl_result(
        queue_path=_QUEUE_PATH,
        entries=items,
        durable_queue=_DURABLE_QUEUE,
        lock_queue=_lock_queue,
        record_lock_failure=_record_queue_lock_failure,
        fsync_dir_fn=_fsync_dir,
        diag=_diag,
    )
    return exit_models.ExitQueueWriteResult(ok=result.ok, count=result.count)


def _requeue_entries_jsonl(entries: list[dict]) -> bool:
    return _requeue_entries_jsonl_result(entries).ok


def _queue_jsonl_has_data() -> bool:
    queue_store = _module("queue_store")
    return queue_store.queue_jsonl_has_data(_QUEUE_PATH, _QUEUE_PROCESSING_PATH)


def _queue_db_connect() -> sqlite3.Connection | None:
    global _QUEUE_DB_OPEN_COUNT
    queue_store = _module("queue_store")
    db_path = queue_store.resolve_queue_db_path(_QUEUE_PATH, _QUEUE_DB_PATH)
    timeout_base = max(1.0, _QUEUE_LOCK_SLEEP_BASE * max(1, _QUEUE_LOCK_RETRIES) * 4.0)
    timeout_max = max(timeout_base, float(_QUEUE_DB_CONNECT_TIMEOUT_MAX or timeout_base))
    conn = queue_store.connect_queue_db(
        db_path,
        attempts=max(1, int(_QUEUE_DB_CONNECT_RETRIES or 1)),
        timeout_base=timeout_base,
        timeout_max=timeout_max,
        backoff_base=float(_QUEUE_DB_CONNECT_BACKOFF_BASE or 0.0),
        row_factory=sqlite3.Row,
        diag=_diag,
        sleep_fn=_sleep,
    )
    if conn is not None:
        _QUEUE_DB_OPEN_COUNT += 1
    return conn


def _sqlite_error_looks_corrupt(exc: Exception) -> bool:
    queue_store = _module("queue_store")
    return queue_store.sqlite_error_looks_corrupt(exc)


def _queue_db_quarantine_current(path: Path, reason: Exception | str) -> bool:
    queue_store = _module("queue_store")
    return queue_store.quarantine_sqlite_db(path, reason, diag=_diag)


def _queue_db_init(conn: sqlite3.Connection) -> None:
    queue_store = _module("queue_store")
    queue_store.init_queue_db(conn)


def _queue_db_open_ready() -> sqlite3.Connection | None:
    global _RUN_QUEUE_DB_CONN, _QUEUE_DB_REUSE_COUNT
    if _RUN_QUEUE_DB_ACTIVE and _RUN_QUEUE_DB_CONN is not None:
        try:
            _RUN_QUEUE_DB_CONN.execute("SELECT 1")
            _QUEUE_DB_REUSE_COUNT += 1
            return _RUN_QUEUE_DB_CONN
        except Exception:
            _queue_close_silent(_RUN_QUEUE_DB_CONN)
            _RUN_QUEUE_DB_CONN = None
    queue_store = _module("queue_store")
    db_path = queue_store.resolve_queue_db_path(_QUEUE_PATH, _QUEUE_DB_PATH)
    conn = queue_store.open_ready_queue_db(
        db_path,
        connect_fn=_queue_db_connect,
        init_fn=_queue_db_init,
        close_fn=_queue_close_silent,
        diag=_diag,
    )
    if _RUN_QUEUE_DB_ACTIVE and conn is not None:
        _RUN_QUEUE_DB_CONN = conn
    return conn


def _queue_select_queued_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    queue_store = _module("queue_store")
    return queue_store.select_queued_rows(conn, max_lines=_QUEUE_MAX_LINES)


def _queue_row_ids(rows: list[sqlite3.Row]) -> list[int]:
    queue_store = _module("queue_store")
    return queue_store.row_ids(rows)


def _queue_claim_rows_sqlite(conn: sqlite3.Connection, token: str, now: float):
    queue_store = _module("queue_store")
    return queue_store.claim_rows_sqlite_result(
        conn,
        token=token,
        now=now,
        processing_stale_after=_QUEUE_PROCESSING_STALE_AFTER,
        max_lines=_QUEUE_MAX_LINES,
        diag=_diag,
        on_lock_busy=_record_queue_lock_failure,
    )


def _queue_rows_to_entries(rows: list[sqlite3.Row]) -> list[dict]:
    queue_store = _module("queue_store")
    return queue_store.rows_to_entries_result(rows).entries


def _queue_close_silent(conn: sqlite3.Connection) -> None:
    if conn is None:
        return
    if _RUN_QUEUE_DB_ACTIVE and conn is _RUN_QUEUE_DB_CONN:
        return
    queue_store = _module("queue_store")
    queue_store.close_silent(conn)


def _take_queue_entries_sqlite_batch():
    conn = _queue_db_open_ready()
    exit_models = _module("exit_models")
    if conn is None:
        return exit_models.ExitQueueBatch(entries=[])
    token = f"drain-{os.getpid()}-{int(time.time() * 1000)}"
    now = time.time()
    claim = _queue_claim_rows_sqlite(conn, token, now)
    entries = _queue_rows_to_entries(claim.rows)
    _queue_close_silent(conn)
    return exit_models.ExitQueueBatch(entries=entries)


def _take_queue_entries_sqlite() -> list[dict]:
    return _take_queue_entries_sqlite_batch().entries


def _ack_queue_entries_sqlite_result(entry_ids: list[int]):
    conn = _queue_db_open_ready()
    exit_models = _module("exit_models")
    ids = [int(raw) for raw in (entry_ids or []) if str(raw).isdigit() and int(raw) > 0]
    if conn is None:
        return exit_models.ExitQueueWriteResult(ok=False, count=len(ids))
    try:
        queue_store = _module("queue_store")
        result = queue_store.ack_entry_ids_sqlite_result(
            conn,
            ids,
            diag=_diag,
            on_lock_busy=_record_queue_lock_failure,
        )
        return exit_models.ExitQueueWriteResult(ok=result.ok, count=result.count)
    finally:
        _queue_close_silent(conn)


def _ack_queue_entries_sqlite(entry_ids: list[int]) -> bool:
    return _ack_queue_entries_sqlite_result(entry_ids).ok


def _requeue_entries_sqlite_result(entries: list[dict]):
    conn = _queue_db_open_ready()
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


def _requeue_entries_sqlite(entries: list[dict]) -> bool:
    return _requeue_entries_sqlite_result(entries).ok


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


def _enqueue_entries_sqlite(entries: list[dict]) -> bool:
    return _enqueue_entries_sqlite_result(entries).ok


def _migrate_legacy_jsonl_to_sqlite() -> None:
    if not _queue_jsonl_has_data():
        return
    legacy_batch = _take_queue_entries_jsonl_batch()
    legacy_entries = legacy_batch.entries
    if not legacy_entries:
        return
    enqueue_result = _enqueue_entries_sqlite_result(legacy_entries)
    if enqueue_result.ok:
        try:
            if _QUEUE_PROCESSING_PATH.exists():
                _QUEUE_PROCESSING_PATH.unlink()
                if _DURABLE_QUEUE:
                    _fsync_dir(_QUEUE_PROCESSING_PATH.parent)
        except Exception:
            pass
        _diag(f"migrated {legacy_batch.entries_total} legacy queue entries to sqlite")
        return
    _diag(f"legacy queue migration failed; restoring {legacy_batch.entries_total} entries to JSONL")
    _requeue_entries_jsonl(legacy_entries)


def _take_queue_entries() -> list[dict]:
    _migrate_legacy_jsonl_to_sqlite()
    return _take_queue_entries_sqlite_batch().entries


def _requeue_entries(entries: list[dict]) -> bool:
    items = [e for e in (entries or []) if isinstance(e, dict)]
    if not items:
        return True
    claimed: list[dict] = []
    fresh: list[dict] = []
    for e in items:
        if (e.get("__queue_backend") == "sqlite") and e.get("__queue_id"):
            claimed.append(e)
        else:
            fresh.append(e)
    claimed_result = _requeue_entries_sqlite_result(claimed) if claimed else _module("exit_models").ExitQueueWriteResult(ok=True, count=0)
    fresh_result = _enqueue_entries_sqlite_result(fresh) if fresh else _module("exit_models").ExitQueueWriteResult(ok=True, count=0)
    return claimed_result.ok and fresh_result.ok


def _normalize_queue_entry(entry: dict) -> dict:
    queue_models = _module("queue_models")
    return queue_models.normalize_spawn_queue_entry(entry)


def _validate_queue_entry(entry: dict) -> tuple[bool, str]:
    try:
        _normalize_queue_entry(entry)
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
    exit_queries = _module("exit_queries")
    return exit_queries.short_uuid(value, core=core)


def _coerce_exit_export_result(value):
    exit_models = _module("exit_models")
    if hasattr(value, "exists") and hasattr(value, "retryable") and hasattr(value, "err") and hasattr(value, "obj"):
        return exit_models.ExitExportResult(
            bool(getattr(value, "exists")),
            bool(getattr(value, "retryable")),
            str(getattr(value, "err") or ""),
            getattr(value, "obj") if isinstance(getattr(value, "obj"), dict) else None,
        )
    if isinstance(value, dict):
        obj = value.get("obj") if isinstance(value.get("obj"), dict) else None
        return exit_models.ExitExportResult(
            bool(value.get("exists")),
            bool(value.get("retryable")),
            str(value.get("err") or ""),
            obj,
        )
    return exit_models.ExitExportResult(False, False, "invalid export result", None)


def _coerce_exit_equivalent_child_result(value):
    exit_models = _module("exit_models")
    if hasattr(value, "exists") and hasattr(value, "retryable") and hasattr(value, "err") and hasattr(value, "obj"):
        return exit_models.ExitEquivalentChildResult(
            bool(getattr(value, "exists")),
            bool(getattr(value, "retryable")),
            str(getattr(value, "err") or ""),
            getattr(value, "obj") if isinstance(getattr(value, "obj"), dict) else None,
        )
    if isinstance(value, dict):
        obj = value.get("obj") if isinstance(value.get("obj"), dict) else None
        return exit_models.ExitEquivalentChildResult(
            bool(value.get("exists")),
            bool(value.get("retryable")),
            str(value.get("err") or ""),
            obj,
        )
    return exit_models.ExitEquivalentChildResult(False, False, "invalid equivalent result", None)


def _coerce_exit_import_result(value):
    exit_models = _module("exit_models")
    if hasattr(value, "ok") and hasattr(value, "err"):
        return exit_models.ExitImportResult(bool(getattr(value, "ok")), str(getattr(value, "err") or ""))
    if isinstance(value, tuple) and len(value) >= 2:
        return exit_models.ExitImportResult(bool(value[0]), str(value[1] or ""))
    return exit_models.ExitImportResult(False, "invalid import result")


def _coerce_exit_parent_nextlink_state_result(value):
    exit_models = _module("exit_models")
    if hasattr(value, "state") and hasattr(value, "err"):
        return exit_models.ExitParentNextlinkStateResult(str(getattr(value, "state") or ""), str(getattr(value, "err") or ""))
    if isinstance(value, tuple) and len(value) >= 2:
        return exit_models.ExitParentNextlinkStateResult(str(value[0] or ""), str(value[1] or ""))
    return exit_models.ExitParentNextlinkStateResult("invalid", "invalid parent nextLink state result")


def _coerce_exit_parent_update_result(value):
    exit_models = _module("exit_models")
    if hasattr(value, "ok") and hasattr(value, "err"):
        return exit_models.ExitParentUpdateResult(bool(getattr(value, "ok")), str(getattr(value, "err") or ""))
    if isinstance(value, tuple) and len(value) >= 2:
        return exit_models.ExitParentUpdateResult(bool(value[0]), str(value[1] or ""))
    return exit_models.ExitParentUpdateResult(False, "invalid parent update result")


def _export_uuid(uuid_str: str) -> dict:
    exit_queries = _module("exit_queries")
    hook_support = _module("hook_support", required=False)
    res = _coerce_exit_export_result(
        exit_queries.export_uuid(
            uuid_str,
            hook_support=hook_support,
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            timeout=_TASK_TIMEOUT_EXPORT,
            retries=_TASK_RETRIES_EXPORT,
            retry_delay=_TASK_RETRY_DELAY,
            is_lock_error=_is_lock_error,
        )
    )
    return {"exists": res.exists, "retryable": res.retryable, "err": res.err, "obj": res.obj}


def _existing_equivalent_child(child: dict, parent_uuid: str = "") -> dict:
    exit_queries = _module("exit_queries")
    res = _coerce_exit_equivalent_child_result(
        exit_queries.existing_equivalent_child(
            child,
            parent_uuid=parent_uuid,
            task_cmd_prefix=_task_cmd_prefix(),
            run_task=_run_task,
            timeout=_TASK_TIMEOUT_EXPORT,
            retries=_TASK_RETRIES_EXPORT,
            retry_delay=_TASK_RETRY_DELAY,
            is_lock_error=_is_lock_error,
            short_uuid_fn=_short_uuid,
        )
    )
    return {"exists": res.exists, "retryable": res.retryable, "err": res.err, "obj": res.obj}


def _import_child(obj: dict) -> tuple[bool, str]:
    exit_side_effects = _module("exit_side_effects")
    res = _coerce_exit_import_result(
        exit_side_effects.import_child(
            obj,
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            timeout_import=_TASK_TIMEOUT_IMPORT,
            is_lock_error=_is_lock_error,
            sleep=_sleep,
            random_uniform=random.uniform,
        )
    )
    return res.ok, res.err


def _update_parent_nextlink(parent_uuid: str, child_short: str, expected_prev: str | None = None) -> tuple[bool, str]:
    exit_side_effects = _module("exit_side_effects")
    res = _coerce_exit_parent_update_result(
        exit_side_effects.update_parent_nextlink(
            parent_uuid,
            child_short,
            expected_prev=expected_prev,
            lock_parent_nextlink=_lock_parent_nextlink,
            parent_nextlink_state_fn=lambda parent_uuid, child_short, expected_prev: _coerce_exit_parent_nextlink_state_result(
                _parent_nextlink_state(parent_uuid, child_short, expected_prev)
            ),
            run_task=_run_task,
            task_cmd_prefix=_task_cmd_prefix(),
            timeout_modify=_TASK_TIMEOUT_MODIFY,
            retries_modify=_TASK_RETRIES_MODIFY,
            retry_delay=_TASK_RETRY_DELAY,
        )
    )
    return res.ok, res.err


def _parent_nextlink_state(parent_uuid: str, child_short: str, expected_prev: str | None = None) -> tuple[str, str]:
    exit_side_effects = _module("exit_side_effects")
    res = _coerce_exit_parent_nextlink_state_result(
        exit_side_effects.parent_nextlink_state(
            parent_uuid,
            child_short,
            expected_prev=expected_prev,
            export_uuid=lambda uuid_str: _coerce_exit_export_result(_export_uuid(uuid_str)),
        )
    )
    return res.state, res.err


def _cleanup_orphan_child(child_uuid: str, spawn_intent_id: str = "") -> None:
    exit_side_effects = _module("exit_side_effects")
    exit_side_effects.cleanup_orphan_child(
        child_uuid,
        spawn_intent_id=spawn_intent_id,
        run_task=_run_task,
        task_cmd_prefix=_task_cmd_prefix(),
        timeout_modify=_TASK_TIMEOUT_MODIFY,
        retries_modify=_TASK_RETRIES_MODIFY,
        retry_delay=_TASK_RETRY_DELAY,
        diag=_diag,
    )


def _take_queue_batch():
    exit_models = _module("exit_models")
    return exit_models.ExitQueueBatch(entries=_take_queue_entries())


def _requeue_entries_result(entries: list[dict]):
    exit_models = _module("exit_models")
    ok = _requeue_entries(entries)
    failed = 0 if ok else len(entries)
    return exit_models.ExitRequeueResult(ok=ok, failed=failed)


class _DrainState:
    def __init__(
        self,
        entries: list[dict],
        entries_total: int,
        finalized_intents: set[str],
        intent_log_ready: bool,
        intent_log_load_ms: float,
    ) -> None:
        self.entries = entries
        self.entries_total = entries_total
        self.finalized_intents = finalized_intents
        self.intent_log_ready = intent_log_ready
        self.intent_log_load_ms = intent_log_load_ms
        self.processed = 0
        self.errors = 0
        self.requeue: list[dict] = []
        self.dead_lettered = 0
        self.skipped_idempotent = 0
        self.lock_events = 0
        self.lock_streak = 0
        self.lock_streak_max = 0
        self.circuit_breaks = 0
        self.intent_mark_ok = 0
        self.intent_mark_fail = 0
        self.sqlite_acked_ids: set[int] = set()

    def mark_final(self, entry: dict, status: str, reason: str) -> None:
        sid = (entry.get("spawn_intent_id") or "").strip() if isinstance(entry, dict) else ""
        if not sid:
            return
        if _mark_intent_status(sid, status, reason):
            self.finalized_intents.add(sid)
            self.intent_mark_ok += 1
        else:
            self.intent_mark_fail += 1

    def queue_backend(self, entry: dict) -> str:
        if not isinstance(entry, dict):
            return "sqlite"
        b = (entry.get("__queue_backend") or "").strip().lower()
        return b if b else "sqlite"

    def queue_id(self, entry: dict) -> int:
        if not isinstance(entry, dict):
            return 0
        try:
            return int(entry.get("__queue_id") or 0)
        except Exception:
            return 0

    def entry_clean(self, entry: dict) -> dict:
        if not isinstance(entry, dict):
            return {}
        out = dict(entry)
        out.pop("__queue_backend", None)
        out.pop("__queue_id", None)
        return out

    def ack_sqlite(self, entry: dict) -> None:
        if self.queue_backend(entry) != "sqlite":
            return
        rid = self.queue_id(entry)
        if rid > 0:
            self.sqlite_acked_ids.add(rid)

    def dead_letter(self, entry: dict, reason: str) -> None:
        _write_dead_letter(self.entry_clean(entry), reason)
        self.dead_lettered += 1
        self.errors += 1
        self.mark_final(entry, "dead", reason)
        self.ack_sqlite(entry)

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
            queue_lock_failures=_QUEUE_LOCK_FAILURES_THIS_RUN,
            entries_total=self.entries_total,
            entries_skipped_idempotent=self.skipped_idempotent,
            lock_events=self.lock_events,
            lock_streak_max=self.lock_streak_max,
            circuit_breaks=self.circuit_breaks,
            intent_log_ready=1 if self.intent_log_ready else 0,
            intent_log_size=len(self.finalized_intents),
            intent_log_load_ms=round(self.intent_log_load_ms, 3),
            intent_mark_ok=self.intent_mark_ok,
            intent_mark_fail=self.intent_mark_fail,
            queue_db_opens=_QUEUE_DB_OPEN_COUNT,
            queue_db_reuses=_QUEUE_DB_REUSE_COUNT,
            drain_ms=round((time.perf_counter() - drain_t0) * 1000.0, 3),
        )


def _requeue_or_dead_letter_for_lock(entry: dict, idx: int, state: _DrainState) -> bool:
    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
        state.dead_letter(entry, "exceeded retry budget")
    else:
        state.requeue.append(entry)
    return state.record_lock_event(idx)


def _handle_entry_gate(entry: dict, state: _DrainState) -> bool:
    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip() if isinstance(entry, dict) else ""
    if spawn_intent_id and spawn_intent_id in state.finalized_intents:
        state.skipped_idempotent += 1
        state.processed += 1
        state.ack_sqlite(entry)
        state.reset_lock_streak()
        return True

    valid, reason = _validate_queue_entry(entry)
    if not valid:
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
        child=child,
        child_uuid=child_uuid,
        spawn_intent_id=spawn_intent_id,
    )


def _precheck_parent_link_state(ctx) -> tuple[str, bool]:
    exit_entry_flow = _module("exit_entry_flow")
    exit_models = _module("exit_models")
    services = exit_models.ExitPrecheckServices(
        parent_nextlink_state=lambda parent_uuid, child_short, expected_prev: _coerce_exit_parent_nextlink_state_result(
            _parent_nextlink_state(parent_uuid, child_short, expected_prev)
        ),
        requeue_or_dead_letter_for_lock=_requeue_or_dead_letter_for_lock,
    )
    return exit_entry_flow.precheck_parent_link_state(ctx, services=services)


def _ensure_child_exists_for_entry(ctx) -> tuple[str, bool]:
    exit_entry_flow = _module("exit_entry_flow")
    exit_models = _module("exit_models")
    services = exit_models.ExitEnsureChildServices(
        export_uuid=lambda uuid_str: _coerce_exit_export_result(_export_uuid(uuid_str)),
        import_child=lambda obj: _coerce_exit_import_result(_import_child(obj)),
        is_lock_error=_is_lock_error,
        diag=_diag,
        requeue_or_dead_letter_for_lock=_requeue_or_dead_letter_for_lock,
    )
    return exit_entry_flow.ensure_child_exists_for_entry(ctx, services=services)


def _apply_parent_update_for_entry(
    ctx,
    *,
    parent_linked_already: bool,
    imported: bool,
) -> str:
    exit_entry_flow = _module("exit_entry_flow")
    exit_models = _module("exit_models")
    services = exit_models.ExitApplyParentUpdateServices(
        update_parent_nextlink=lambda parent_uuid, child_short, expected_prev: _coerce_exit_parent_update_result(
            _update_parent_nextlink(parent_uuid, child_short, expected_prev)
        ),
        is_lock_error=_is_lock_error,
        cleanup_orphan_child=_cleanup_orphan_child,
        diag=_diag,
        requeue_or_dead_letter_for_lock=_requeue_or_dead_letter_for_lock,
    )
    return exit_entry_flow.apply_parent_update_for_entry(
        ctx,
        parent_linked_already=parent_linked_already,
        imported=imported,
        services=services,
    )


def _process_queue_entry(idx: int, entry: dict, state: _DrainState) -> bool:
    if _handle_entry_gate(entry, state):
        return False
    entry = _normalize_queue_entry(entry)

    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip()
    parent_uuid = (entry.get("parent_uuid") or "").strip()
    expected_parent_nextlink = (entry.get("parent_nextlink") or "").strip()
    child = entry.get("child") or {}
    child_short = (entry.get("child_short") or "").strip()
    child_uuid = (child.get("uuid") or "").strip()
    exact_child = _coerce_exit_export_result(_export_uuid(child_uuid))
    if exact_child.retryable:
        return _requeue_or_dead_letter_for_lock(entry, idx, state)
    if not exact_child.exists:
        equivalent = _coerce_exit_equivalent_child_result(_existing_equivalent_child(child, parent_uuid))
        if equivalent.retryable:
            return _requeue_or_dead_letter_for_lock(entry, idx, state)
        existing_obj = equivalent.obj
        if isinstance(existing_obj, dict):
            child_uuid = (existing_obj.get("uuid") or "").strip()
            child_short = _short_uuid(child_uuid)
            if child_short:
                if spawn_intent_id:
                    _diag(
                        f"equivalent child already exists; binding intent {spawn_intent_id} "
                        f"to child {child_short}"
                    )
                else:
                    _diag(f"equivalent child already exists; binding to child {child_short}")

    ctx = _build_exit_entry_context(
        entry,
        idx,
        state,
        parent_uuid=parent_uuid,
        child_short=child_short,
        expected_parent_nextlink=expected_parent_nextlink or None,
        child=child,
        child_uuid=child_uuid,
        spawn_intent_id=spawn_intent_id,
    )

    link_action, parent_linked_already = _precheck_parent_link_state(ctx)
    if link_action == "break":
        return True
    if link_action == "continue":
        return False

    child_action, imported = _ensure_child_exists_for_entry(ctx)
    if child_action == "break":
        return True
    if child_action == "continue":
        return False

    parent_action = _apply_parent_update_for_entry(
        ctx,
        parent_linked_already=parent_linked_already,
        imported=imported,
    )
    if parent_action == "break":
        return True
    if parent_action == "continue":
        return False

    state.processed += 1
    state.mark_final(entry, "done", "processed")
    state.ack_sqlite(entry)
    state.reset_lock_streak()
    return False


def _drain_queue_result():
    _queue_db_begin_run()
    try:
        drain_t0 = time.perf_counter()
        batch = _take_queue_batch()
        entries = batch.entries
        intent_t0 = time.perf_counter()
        finalized_intents, intent_log_ready = _load_finalized_intents()
        intent_log_load_ms = (time.perf_counter() - intent_t0) * 1000.0
        state = _DrainState(
            entries=entries,
            entries_total=batch.entries_total,
            finalized_intents=finalized_intents,
            intent_log_ready=bool(intent_log_ready),
            intent_log_load_ms=float(intent_log_load_ms),
        )

        for idx, entry in enumerate(entries):
            if _process_queue_entry(idx, entry, state):
                break

        requeue_result = _requeue_entries_result(state.requeue) if state.requeue else _module("exit_models").ExitRequeueResult(ok=True, failed=0)
        if not requeue_result.ok:
            state.errors += requeue_result.failed
            _diag(f"requeue failed for {requeue_result.failed} entries")

        if state.sqlite_acked_ids:
            if not _ack_queue_entries_sqlite(sorted(state.sqlite_acked_ids)):
                state.errors += len(state.sqlite_acked_ids)
                _diag(f"queue db ack failed for {len(state.sqlite_acked_ids)} entries")

        return state.to_stats_model(drain_t0, requeue_result.ok, requeue_result.failed)
    finally:
        _queue_db_end_run()


def _drain_queue() -> dict:
    return _drain_queue_result().to_dict()


def _redirect_stdout_to_devnull() -> None:
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass


def _emit_drain_stats_diag(stats: dict) -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    _diag(
        "on-exit drain: "
        f"entries_total={stats.get('entries_total', 0)} "
        f"idempotent_skipped={stats.get('entries_skipped_idempotent', 0)} "
        f"processed={stats.get('processed', 0)} "
        f"errors={stats.get('errors', 0)} "
        f"requeued={stats.get('requeued', 0)} "
        f"requeue_failed={stats.get('requeue_failed', 0)} "
        f"dead_lettered={stats.get('dead_lettered', 0)} "
        f"queue_lock_failures={stats.get('queue_lock_failures', 0)} "
        f"lock_events={stats.get('lock_events', 0)} "
        f"lock_streak_max={stats.get('lock_streak_max', 0)} "
        f"circuit_breaks={stats.get('circuit_breaks', 0)} "
        f"intent_log_ready={stats.get('intent_log_ready', 0)} "
        f"intent_log_size={stats.get('intent_log_size', 0)} "
        f"intent_mark_ok={stats.get('intent_mark_ok', 0)} "
        f"intent_mark_fail={stats.get('intent_mark_fail', 0)} "
        f"intent_log_load_ms={stats.get('intent_log_load_ms', 0)} "
        f"queue_db_opens={stats.get('queue_db_opens', 0)} "
        f"queue_db_reuses={stats.get('queue_db_reuses', 0)} "
        f"drain_ms={stats.get('drain_ms', 0)}"
    )


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


def main() -> int:
    hook_context = _module("hook_context")
    hook_results = _module("hook_results")
    hook_engine = _module("hook_engine")
    request = hook_context.build_on_exit_request(runtime=_build_hook_runtime_context())
    result = hook_engine.handle_on_exit(
        request,
        exit_result_cls=hook_results.HookExitResult,
        redirect_stdout_to_devnull=_redirect_stdout_to_devnull,
        drain_queue=_drain_queue,
        strict_exit_result=_strict_exit_feedback_message,
    )
    return hook_results.emit_exit_result(
        result,
        emit_exit_feedback=_emit_exit_feedback,
        emit_stats_diag=_emit_drain_stats_diag,
    )


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
