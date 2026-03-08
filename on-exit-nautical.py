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
    return pyfile if pyfile.is_file() else pkgini if pkgini.is_file() else None

core = None
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
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


def _task_cmd_prefix() -> list[str]:
    cmd = ["task"]
    if _USE_RC_DATA_LOCATION:
        cmd.append(f"rc.data.location={TW_DATA_DIR}")
    return cmd

_QUEUE_PATH = TW_DATA_DIR / ".nautical_spawn_queue.jsonl"
_QUEUE_PROCESSING_PATH = TW_DATA_DIR / ".nautical_spawn_queue.processing.jsonl"
_QUEUE_LOCK = TW_DATA_DIR / ".nautical_spawn_queue.lock"
_QUEUE_DB_PATH = TW_DATA_DIR / ".nautical_queue.db"
_DEAD_LETTER_PATH = TW_DATA_DIR / ".nautical_dead_letter.jsonl"
_DEAD_LETTER_LOCK = TW_DATA_DIR / ".nautical_dead_letter.lock"
_DEAD_LETTER_RETENTION_DAYS = int(os.environ.get("NAUTICAL_DEAD_LETTER_RETENTION_DAYS") or 30)
_QUEUE_MAX_BYTES = int(os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_BYTES") or 524288)
_QUEUE_MAX_LINES = int(os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_LINES") or 10000)
_DEAD_LETTER_MAX_BYTES = int(os.environ.get("NAUTICAL_DEAD_LETTER_MAX_BYTES") or 524288)
_QUEUE_QUARANTINE_PATH = TW_DATA_DIR / ".nautical_spawn_queue.bad.jsonl"
_QUEUE_QUARANTINE_MAX_BYTES = int(os.environ.get("NAUTICAL_QUEUE_BAD_MAX_BYTES") or 262144)
NAUTICAL_HOOK_VERSION = "updateE-20260126"
_QUEUE_RETRY_MAX = int(os.environ.get("NAUTICAL_QUEUE_RETRY_MAX") or 6)
_QUEUE_LOCK_FAIL_MARKER = TW_DATA_DIR / ".nautical_spawn_queue.lock_failed"
_QUEUE_LOCK_FAIL_COUNT = TW_DATA_DIR / ".nautical_spawn_queue.lock_failed.count"
_DURABLE_QUEUE = os.environ.get("NAUTICAL_DURABLE_QUEUE") == "1"
# When set, exit 1 if any spawns were dead-lettered or errored (for scripting/monitoring).
_EXIT_STRICT = (os.environ.get("NAUTICAL_EXIT_STRICT") or "").strip().lower() in ("1", "true", "yes", "on")

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

@contextmanager
def _local_safe_lock(path: Path, *, retries: int = 6, sleep_base: float = 0.05, stale_after: float | None = 60.0):
    path_str = str(path) if path else ""
    if not path_str:
        yield False
        return
    def _sleep_once():
        try:
            delay = float(sleep_base or 0.0)
        except Exception:
            delay = 0.0
        if delay > 0:
            time.sleep(delay)

    tries = max(1, int(retries or 0))
    if fcntl is not None:
        lf = None
        acquired = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
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
                    _sleep_once()
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
        return

    fd = None
    acquired = False
    def _lock_age(path_str: str) -> float | None:
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

    def _lock_stale_pid(path_str: str) -> bool:
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

    for _ in range(tries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
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
            pid_stale = _lock_stale_pid(path_str)
            age_stale = False
            if stale_after is not None:
                age = _lock_age(path_str)
                if age is not None and age >= float(stale_after):
                    age_stale = True
            if pid_stale and age_stale:
                try:
                    os.unlink(path_str)
                except Exception:
                    pass
            else:
                _sleep_once()
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


def _diag(msg: str) -> None:
    if core is not None:
        core.diag(msg, "on-exit", str(TW_DATA_DIR))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass


def _emit_exit_feedback(msg: str) -> None:
    """Write required feedback to stderr when exiting non-zero (Taskwarrior hook contract)."""
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass

def _run_task(
    cmd: list[str],
    *,
    input_text: str | None = None,
    timeout: float = 6.0,
    retries: int = 1,
    retry_delay: float = 0.0,
) -> tuple[bool, str, str]:
    run_fn = core.run_task if core is not None else None
    if run_fn is not None:
        return run_fn(
            cmd,
            env=os.environ.copy(),
            input_text=input_text,
            timeout=timeout,
            retries=max(1, int(retries)),
            retry_delay=max(0.0, float(retry_delay)),
        )
    env = os.environ.copy()
    attempts = max(1, int(retries))
    delay = max(0.0, float(retry_delay))
    last_out = ""
    last_err = ""
    for attempt in range(1, attempts + 1):
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
                last_out = out or ""
                last_err = "timeout"
                if attempt < attempts and delay > 0:
                    jitter = random.uniform(0.0, delay)
                    _sleep(delay * (2 ** (attempt - 1)) + jitter)
                    continue
                return (False, last_out, last_err)
            last_out = out or ""
            last_err = err or ""
            if proc.returncode == 0:
                return (True, last_out, last_err)
            if attempt < attempts and delay > 0:
                jitter = random.uniform(0.0, delay)
                _sleep(delay * (2 ** (attempt - 1)) + jitter)
                continue
            return (False, last_out, last_err)
        except subprocess.TimeoutExpired:
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            last_err = "timeout"
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
            last_err = str(e)
        if attempt < attempts and delay > 0:
            jitter = random.uniform(0.0, delay)
            _sleep(delay * (2 ** (attempt - 1)) + jitter)
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
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": "queue lock busy",
        }
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
    return _tw_data_dir_path() / ".nautical_spawn_intents.jsonl"


def _intent_log_lock_path() -> Path:
    return _tw_data_dir_path() / ".nautical_spawn_intents.lock"


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
    return _tw_data_dir_path() / f".nautical_parent_nextlink.{safe}.lock"


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


def _load_finalized_intents() -> tuple[set[str], bool]:
    """Return finalized spawn_intent_id set, with best-effort compaction."""
    final_states: dict[str, str] = {}
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
        try:
            with open(p, "r", encoding="utf-8") as f:
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
        except Exception as e:
            _diag(f"intent log read failed: {e}")
            return set(), False

        try:
            st_size = p.stat().st_size
        except Exception:
            st_size = 0
        needs_compact = bool(
            (_INTENT_LOG_MAX_BYTES > 0 and st_size > _INTENT_LOG_MAX_BYTES)
            or (_INTENT_LOG_MAX_ENTRIES > 0 and len(final_states) > _INTENT_LOG_MAX_ENTRIES)
        )
        if _INTENT_LOG_MAX_ENTRIES > 0 and len(final_states) > _INTENT_LOG_MAX_ENTRIES:
            drop_n = len(final_states) - _INTENT_LOG_MAX_ENTRIES
            for sid in list(final_states.keys())[:drop_n]:
                final_states.pop(sid, None)
        if needs_compact:
            tmp_path = p.with_suffix(".staging")
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
                os.replace(tmp_path, p)
                if _DURABLE_QUEUE:
                    _fsync_dir(p.parent)
            except Exception as e:
                _diag(f"intent log compaction failed: {e}")
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
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

def _take_queue_entries_jsonl() -> list[dict]:
    entries: list[dict] = []
    with _lock_queue() as locked:
        if not locked:
            _record_queue_lock_failure()
            return entries
        try:
            if _QUEUE_PROCESSING_PATH.exists():
                try:
                    if not _QUEUE_PATH.exists():
                        os.replace(_QUEUE_PROCESSING_PATH, _QUEUE_PATH)
                    else:
                        with open(_QUEUE_PROCESSING_PATH, "r", encoding="utf-8") as f_in:
                            fd = os.open(str(_QUEUE_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
                            try:
                                os.fchmod(fd, 0o600)
                            except Exception:
                                pass
                            with os.fdopen(fd, "a", encoding="utf-8") as f_out:
                                for line in f_in:
                                    f_out.write(line)
                                if _DURABLE_QUEUE:
                                    try:
                                        f_out.flush()
                                        os.fsync(f_out.fileno())
                                    except Exception:
                                        pass
                        os.unlink(_QUEUE_PROCESSING_PATH)
                        if _DURABLE_QUEUE:
                            _fsync_dir(_QUEUE_PATH.parent)
                except Exception:
                    return entries
        except Exception:
            return entries
        try:
            if not _QUEUE_PATH.exists():
                return entries
        except Exception:
            return entries
        try:
            overflow_path = None
            st = _QUEUE_PATH.stat()
            if _QUEUE_MAX_BYTES > 0 and st.st_size > _QUEUE_MAX_BYTES:
                try:
                    ts = int(time.time())
                    overflow_path = _QUEUE_PATH.with_suffix(f".overflow.{ts}.jsonl")
                    os.replace(_QUEUE_PATH, overflow_path)
                    _diag(f"queue rotated: {overflow_path}")
                except Exception:
                    pass
        except Exception:
            pass

        src_path = overflow_path or _QUEUE_PATH
        tmp_path = _QUEUE_PATH.with_suffix(".staging")
        tmp_processing = _QUEUE_PROCESSING_PATH.with_suffix(".staging")
        try:
            fd_out = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            fd_proc = os.open(str(tmp_processing), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with open(src_path, "r", encoding="utf-8") as f_in, os.fdopen(fd_out, "w", encoding="utf-8") as f_out, os.fdopen(fd_proc, "w", encoding="utf-8") as f_proc:
                for line in f_in:
                    ln = line.strip()
                    if not ln:
                        continue
                    if _QUEUE_MAX_LINES > 0 and len(entries) >= _QUEUE_MAX_LINES:
                        f_out.write(line)
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        _quarantine_queue_line(line, "queue json parse")
                        _write_dead_letter({"raw": line}, "queue json parse")
                        continue
                    if isinstance(obj, dict):
                        entries.append(obj)
                        f_proc.write(line)
                    else:
                        _quarantine_queue_line(line, "queue json not object")
                        _write_dead_letter({"raw": line}, "queue json not object")
                        continue
                if _DURABLE_QUEUE:
                    try:
                        f_out.flush()
                        os.fsync(f_out.fileno())
                        f_proc.flush()
                        os.fsync(f_proc.fileno())
                    except Exception:
                        pass
            os.replace(tmp_path, _QUEUE_PATH)
            if entries:
                os.replace(tmp_processing, _QUEUE_PROCESSING_PATH)
            else:
                try:
                    tmp_processing.unlink()
                except Exception:
                    pass
            if _DURABLE_QUEUE:
                _fsync_dir(_QUEUE_PATH.parent)
            if overflow_path:
                try:
                    overflow_path.unlink()
                except Exception:
                    pass
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            try:
                if tmp_processing.exists():
                    tmp_processing.unlink()
            except Exception:
                pass
    return entries


def _requeue_entries_jsonl(entries: list[dict]) -> bool:
    if not entries:
        return True
    with _lock_queue() as locked:
        if not locked:
            _record_queue_lock_failure()
            return False
        try:
            fd = os.open(str(_QUEUE_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                for obj in entries:
                    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
                if _DURABLE_QUEUE:
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                        _fsync_dir(_QUEUE_PATH.parent)
                    except Exception:
                        pass
            return True
        except Exception as e:
            _diag(f"requeue write failed: {e}")
            return False


def _queue_jsonl_has_data() -> bool:
    for p in (_QUEUE_PATH, _QUEUE_PROCESSING_PATH):
        try:
            if not p.exists():
                continue
            try:
                if p.stat().st_size > 0:
                    return True
            except Exception:
                # Be conservative when metadata is unavailable; migration will
                # validate content and dead-letter malformed entries.
                return True
        except Exception:
            continue
    return False


def _queue_db_connect() -> sqlite3.Connection | None:
    db_path = _QUEUE_DB_PATH
    try:
        # Follow an overridden queue JSONL path in tests/mixed setups when DB path wasn't updated.
        if isinstance(_QUEUE_PATH, Path) and isinstance(_QUEUE_DB_PATH, Path):
            if _QUEUE_DB_PATH.parent != _QUEUE_PATH.parent:
                db_path = _QUEUE_PATH.parent / _QUEUE_DB_PATH.name
    except Exception:
        db_path = _QUEUE_DB_PATH
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        conn = sqlite3.connect(str(db_path), timeout=max(1.0, _QUEUE_LOCK_SLEEP_BASE * max(1, _QUEUE_LOCK_RETRIES) * 4.0))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=1500")
        try:
            if db_path.exists():
                os.chmod(db_path, 0o600)
        except Exception:
            pass
        return conn
    except Exception as e:
        _diag(f"queue db connect failed: {e}")
        return None


def _queue_db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            spawn_intent_id TEXT,
            payload TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'queued',
            claim_token TEXT,
            claimed_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_entries_spawn_intent
        ON queue_entries (spawn_intent_id)
        WHERE spawn_intent_id IS NOT NULL AND spawn_intent_id <> ''
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_entries_state_id ON queue_entries (state, id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_entries_claimed_at ON queue_entries (claimed_at)")
    conn.commit()


def _take_queue_entries_sqlite() -> list[dict]:
    conn = _queue_db_connect()
    if conn is None:
        return []
    try:
        _queue_db_init(conn)
    except Exception as e:
        _diag(f"queue db init failed: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return []
    token = f"drain-{os.getpid()}-{int(time.time() * 1000)}"
    now = time.time()
    rows: list[sqlite3.Row] = []
    try:
        if _QUEUE_PROCESSING_STALE_AFTER > 0:
            cutoff = now - _QUEUE_PROCESSING_STALE_AFTER
            conn.execute(
                "UPDATE queue_entries SET state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                "WHERE state='processing' AND claimed_at IS NOT NULL AND claimed_at < ?",
                (now, cutoff),
            )
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        q = "SELECT id, payload, attempts FROM queue_entries WHERE state='queued' ORDER BY id"
        params: tuple = ()
        if _QUEUE_MAX_LINES > 0:
            q += " LIMIT ?"
            params = (_QUEUE_MAX_LINES,)
        rows = list(conn.execute(q, params))
        if rows:
            ids: list[int] = []
            for r in rows:
                try:
                    rid = int(r["id"])
                except Exception:
                    continue
                if rid > 0:
                    ids.append(rid)
            if ids:
                conn.executemany(
                    "UPDATE queue_entries SET state='processing', claim_token=?, claimed_at=?, updated_at=? "
                    "WHERE id=?",
                    [(token, now, now, rid) for rid in ids],
                )
        conn.commit()
    except sqlite3.OperationalError as e:
        try:
            conn.rollback()
        except Exception:
            pass
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            _record_queue_lock_failure()
        else:
            _diag(f"queue db claim failed: {e}")
        rows = []
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        _diag(f"queue db claim failed: {e}")
        rows = []

    entries: list[dict] = []
    for r in rows:
        rid = int(r["id"])
        payload = (r["payload"] or "").strip()
        attempts_db = int(r["attempts"] or 0)
        try:
            obj = json.loads(payload) if payload else {}
        except Exception:
            obj = {"raw": payload}
        if not isinstance(obj, dict):
            obj = {"raw": payload}
        if "attempts" not in obj:
            obj["attempts"] = attempts_db
        obj["__queue_backend"] = "sqlite"
        obj["__queue_id"] = rid
        entries.append(obj)
    try:
        conn.close()
    except Exception:
        pass
    return entries


def _ack_queue_entries_sqlite(entry_ids: list[int]) -> bool:
    ids: list[int] = []
    for raw in (entry_ids or []):
        try:
            rid = int(raw)
        except Exception:
            continue
        if rid > 0:
            ids.append(rid)
    if not ids:
        return True
    conn = _queue_db_connect()
    if conn is None:
        return False
    try:
        _queue_db_init(conn)
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "DELETE FROM queue_entries WHERE id=?",
            [(rid,) for rid in ids],
        )
        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        try:
            conn.rollback()
        except Exception:
            pass
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            _record_queue_lock_failure()
        else:
            _diag(f"queue db ack failed: {e}")
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        _diag(f"queue db ack failed: {e}")
    try:
        conn.close()
    except Exception:
        pass
    return False


def _requeue_entries_sqlite(entries: list[dict]) -> bool:
    items = [e for e in (entries or []) if isinstance(e, dict) and e.get("__queue_backend") == "sqlite" and e.get("__queue_id")]
    if not items:
        return True
    conn = _queue_db_connect()
    if conn is None:
        return False
    now = time.time()
    try:
        _queue_db_init(conn)
        conn.execute("BEGIN IMMEDIATE")
        for e in items:
            rid = int(e.get("__queue_id") or 0)
            if rid <= 0:
                continue
            out = dict(e)
            out.pop("__queue_backend", None)
            out.pop("__queue_id", None)
            try:
                attempts = int(out.get("attempts") or 0)
            except Exception:
                attempts = 0
            payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
            conn.execute(
                "UPDATE queue_entries SET state='queued', claim_token=NULL, claimed_at=NULL, attempts=?, payload=?, updated_at=? "
                "WHERE id=?",
                (attempts, payload, now, rid),
            )
        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        try:
            conn.rollback()
        except Exception:
            pass
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            _record_queue_lock_failure()
        else:
            _diag(f"queue db requeue failed: {e}")
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        _diag(f"queue db requeue failed: {e}")
    try:
        conn.close()
    except Exception:
        pass
    return False


def _enqueue_entries_sqlite(entries: list[dict]) -> bool:
    items = [e for e in (entries or []) if isinstance(e, dict)]
    if not items:
        return True
    conn = _queue_db_connect()
    if conn is None:
        return False
    now = time.time()
    try:
        _queue_db_init(conn)
        conn.execute("BEGIN IMMEDIATE")
        for e in items:
            out = dict(e)
            out.pop("__queue_backend", None)
            out.pop("__queue_id", None)
            try:
                attempts = int(out.get("attempts") or 0)
            except Exception:
                attempts = 0
            payload = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
            sid = (out.get("spawn_intent_id") or "").strip()
            if sid:
                cur = conn.execute(
                    "UPDATE queue_entries SET payload=?, attempts=?, state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                    "WHERE spawn_intent_id=?",
                    (payload, attempts, now, sid),
                )
                if int(getattr(cur, "rowcount", 0) or 0) <= 0:
                    try:
                        conn.execute(
                            "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) "
                            "VALUES (?, ?, ?, 'queued', NULL, NULL, ?, ?)",
                            (sid, payload, attempts, now, now),
                        )
                    except sqlite3.IntegrityError:
                        conn.execute(
                            "UPDATE queue_entries SET payload=?, attempts=?, state='queued', claim_token=NULL, claimed_at=NULL, updated_at=? "
                            "WHERE spawn_intent_id=?",
                            (payload, attempts, now, sid),
                        )
            else:
                conn.execute(
                    "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) "
                    "VALUES (NULL, ?, ?, 'queued', NULL, NULL, ?, ?)",
                    (payload, attempts, now, now),
                )
        conn.commit()
        conn.close()
        return True
    except sqlite3.OperationalError as e:
        try:
            conn.rollback()
        except Exception:
            pass
        msg = str(e).lower()
        if "locked" in msg or "busy" in msg:
            _record_queue_lock_failure()
        else:
            _diag(f"queue db migrate enqueue failed: {e}")
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        _diag(f"queue db migrate enqueue failed: {e}")
    try:
        conn.close()
    except Exception:
        pass
    return False


def _migrate_legacy_jsonl_to_sqlite() -> None:
    if not _queue_jsonl_has_data():
        return
    legacy_entries = _take_queue_entries_jsonl()
    if not legacy_entries:
        return
    if _enqueue_entries_sqlite(legacy_entries):
        try:
            if _QUEUE_PROCESSING_PATH.exists():
                _QUEUE_PROCESSING_PATH.unlink()
                if _DURABLE_QUEUE:
                    _fsync_dir(_QUEUE_PROCESSING_PATH.parent)
        except Exception:
            pass
        _diag(f"migrated {len(legacy_entries)} legacy queue entries to sqlite")
        return
    _diag(f"legacy queue migration failed; restoring {len(legacy_entries)} entries to JSONL")
    _requeue_entries_jsonl(legacy_entries)


def _take_queue_entries() -> list[dict]:
    _migrate_legacy_jsonl_to_sqlite()
    return _take_queue_entries_sqlite()


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
    ok_claimed = _requeue_entries_sqlite(claimed) if claimed else True
    ok_fresh = _enqueue_entries_sqlite(fresh) if fresh else True
    return ok_claimed and ok_fresh


def _validate_queue_entry(entry: dict) -> tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, "entry not object"
    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip()
    if not spawn_intent_id:
        return False, "missing spawn_intent_id"
    child = entry.get("child")
    if not isinstance(child, dict):
        return False, "missing child object"
    child_uuid = (child.get("uuid") or "").strip()
    if not child_uuid:
        return False, "missing child uuid"
    return True, ""

def _bump_attempts(entry: dict) -> int:
    try:
        attempts = int(entry.get("attempts") or 0)
    except Exception:
        attempts = 0
    attempts += 1
    entry["attempts"] = attempts
    return attempts

def _export_uuid(uuid_str: str) -> dict:
    if not uuid_str:
        return {"exists": False, "retryable": False, "err": "missing uuid", "obj": None}
    ok, out, err = _run_task(
        _task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", "rc.verbose=nothing", "rc.color=off", f"uuid:{uuid_str}", "export"],
        timeout=_TASK_TIMEOUT_EXPORT,
        retries=_TASK_RETRIES_EXPORT,
        retry_delay=_TASK_RETRY_DELAY,
    )
    if not ok:
        return {"exists": False, "retryable": _is_lock_error(err), "err": err or "", "obj": None}
    try:
        obj = json.loads(out.strip() or "{}")
        if obj.get("uuid"):
            return {"exists": True, "retryable": False, "err": "", "obj": obj}
        return {"exists": False, "retryable": False, "err": "not found", "obj": None}
    except Exception:
        if uuid_str in out:
            return {"exists": True, "retryable": False, "err": "", "obj": {"uuid": uuid_str}}
        return {"exists": False, "retryable": False, "err": "parse error", "obj": None}


def _import_child(obj: dict) -> tuple[bool, str]:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    max_retries = 4
    last_err = ""
    for attempt in range(max_retries):
        ok, _out, err = _run_task(
            _task_cmd_prefix() + ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            input_text=payload,
            timeout=_TASK_TIMEOUT_IMPORT,
        )
        if ok:
            return True, ""
        last_err = err or ""
        if not _is_lock_error(last_err):
            return False, last_err
        if attempt < max_retries - 1:
            base = 0.2 * (2 ** attempt)
            jitter = random.uniform(0.0, 0.1)
            _sleep(base + jitter)
    return False, last_err

def _update_parent_nextlink(parent_uuid: str, child_short: str, expected_prev: str | None = None) -> tuple[bool, str]:
    if not parent_uuid or not child_short:
        return False, "missing parent or child"
    with _lock_parent_nextlink(parent_uuid) as locked:
        if not locked:
            return False, "parent lock busy"
        state, msg = _parent_nextlink_state(parent_uuid, child_short, expected_prev)
        if state == "ok":
            ok, _out, err = _run_task(
                _task_cmd_prefix() + ["rc.hooks=off", "rc.verbose=nothing", f"uuid:{parent_uuid}", "modify", f"nextLink:{child_short}"],
                timeout=_TASK_TIMEOUT_MODIFY,
                retries=_TASK_RETRIES_MODIFY,
                retry_delay=_TASK_RETRY_DELAY,
            )
            return ok, err or ""
        if state == "already":
            return True, ""
        return False, msg


def _parent_nextlink_state(parent_uuid: str, child_short: str, expected_prev: str | None = None) -> tuple[str, str]:
    if not parent_uuid or not child_short:
        return "invalid", "missing parent or child"
    res = _export_uuid(parent_uuid)
    if res.get("retryable"):
        return "locked", "parent export locked"
    parent = res.get("obj") if isinstance(res, dict) else None
    if not parent:
        return "missing", "parent missing"
    current = (parent.get("nextLink") or "").strip()
    expected = (expected_prev or "").strip()
    if current == child_short:
        return "already", ""
    if expected:
        if current != expected:
            return "conflict", "parent nextLink changed"
    else:
        if current:
            return "conflict", "parent nextLink already set"
    return "ok", ""


def _cleanup_orphan_child(child_uuid: str, spawn_intent_id: str = "") -> None:
    if not child_uuid:
        return
    ok, _out, err = _run_task(
        _task_cmd_prefix() + ["rc.hooks=off", "rc.verbose=nothing", f"uuid:{child_uuid}", "modify", "status:deleted"],
        timeout=_TASK_TIMEOUT_MODIFY,
        retries=_TASK_RETRIES_MODIFY,
        retry_delay=_TASK_RETRY_DELAY,
    )
    if not ok:
        if spawn_intent_id:
            _diag(f"orphan cleanup failed (intent={spawn_intent_id} child={child_uuid[:8]}): {err}")
        else:
            _diag(f"orphan cleanup failed (child={child_uuid[:8]}): {err}")


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

    def to_stats(self, drain_t0: float, requeue_ok: bool, requeue_failed: int) -> dict:
        return {
            "processed": self.processed,
            "errors": self.errors,
            "requeued": len(self.requeue) if requeue_ok else 0,
            "requeue_failed": requeue_failed,
            "dead_lettered": self.dead_lettered,
            "queue_lock_failures": _QUEUE_LOCK_FAILURES_THIS_RUN,
            "entries_total": self.entries_total,
            "entries_skipped_idempotent": self.skipped_idempotent,
            "lock_events": self.lock_events,
            "lock_streak_max": self.lock_streak_max,
            "circuit_breaks": self.circuit_breaks,
            "intent_log_ready": 1 if self.intent_log_ready else 0,
            "intent_log_size": len(self.finalized_intents),
            "intent_log_load_ms": round(self.intent_log_load_ms, 3),
            "intent_mark_ok": self.intent_mark_ok,
            "intent_mark_fail": self.intent_mark_fail,
            "drain_ms": round((time.perf_counter() - drain_t0) * 1000.0, 3),
        }


def _requeue_or_dead_letter_for_lock(entry: dict, idx: int, state: _DrainState) -> bool:
    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
        state.dead_letter(entry, "exceeded retry budget")
    else:
        state.requeue.append(entry)
    return state.record_lock_event(idx)


def _process_queue_entry(idx: int, entry: dict, state: _DrainState) -> bool:
    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip() if isinstance(entry, dict) else ""
    if spawn_intent_id and spawn_intent_id in state.finalized_intents:
        state.skipped_idempotent += 1
        state.processed += 1
        state.ack_sqlite(entry)
        state.reset_lock_streak()
        return False

    valid, reason = _validate_queue_entry(entry)
    if not valid:
        state.dead_letter(entry, reason)
        state.reset_lock_streak()
        return False

    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip()
    parent_uuid = (entry.get("parent_uuid") or "").strip()
    expected_parent_nextlink = (entry.get("parent_nextlink") or "").strip()
    child = entry.get("child") or {}
    child_short = (entry.get("child_short") or "").strip()
    child_uuid = (child.get("uuid") or "").strip()

    parent_linked_already = False
    if parent_uuid and child_short:
        link_state, link_err = _parent_nextlink_state(parent_uuid, child_short, expected_parent_nextlink)
        if link_state == "locked":
            return _requeue_or_dead_letter_for_lock(entry, idx, state)
        if link_state in {"conflict", "missing", "invalid"}:
            state.dead_letter(entry, f"parent update failed: {link_err}")
            state.reset_lock_streak()
            return False
        if link_state == "already":
            parent_linked_already = True

    export_res = _export_uuid(child_uuid)
    imported = False
    if not export_res.get("exists"):
        if export_res.get("retryable"):
            if spawn_intent_id:
                _diag(f"task lock active; requeue (intent={spawn_intent_id})")
            return _requeue_or_dead_letter_for_lock(entry, idx, state)
        ok, err = _import_child(child)
        if not ok:
            if _is_lock_error(err):
                return _requeue_or_dead_letter_for_lock(entry, idx, state)
            # If import reported failure but the child exists, continue.
            if _export_uuid(child_uuid).get("exists"):
                export_res = {"exists": True}
            else:
                if spawn_intent_id:
                    _diag(f"child import failed (intent={spawn_intent_id}): {err}")
                else:
                    _diag(f"child import failed: {err}")
                state.dead_letter(entry, f"child import failed: {err}")
                state.reset_lock_streak()
                return False
        imported = True

    if imported:
        # Confirm child exists before touching parent only when we just imported.
        confirm_res = _export_uuid(child_uuid)
        if not confirm_res.get("exists"):
            if confirm_res.get("retryable"):
                return _requeue_or_dead_letter_for_lock(entry, idx, state)
            if spawn_intent_id:
                _diag(f"child missing after import (intent={spawn_intent_id})")
            state.dead_letter(entry, "child missing after import")
            state.reset_lock_streak()
            return False

    if parent_uuid and child_short and not parent_linked_already:
        ok, err = _update_parent_nextlink(parent_uuid, child_short, expected_parent_nextlink)
        if not ok:
            if spawn_intent_id:
                _diag(f"parent update failed (intent={spawn_intent_id}): {parent_uuid}")
            else:
                _diag(f"parent update failed: {parent_uuid}")
            if err in {"parent export locked", "parent lock busy"} or _is_lock_error(err):
                return _requeue_or_dead_letter_for_lock(entry, idx, state)
            if imported:
                _cleanup_orphan_child(child_uuid, spawn_intent_id)
            state.dead_letter(entry, f"parent update failed: {err}")
            state.reset_lock_streak()
            return False

    state.processed += 1
    state.mark_final(entry, "done", "processed")
    state.ack_sqlite(entry)
    state.reset_lock_streak()
    return False


def _drain_queue() -> dict:
    drain_t0 = time.perf_counter()
    entries = _take_queue_entries()
    entries_total = len(entries)
    intent_t0 = time.perf_counter()
    finalized_intents, intent_log_ready = _load_finalized_intents()
    intent_log_load_ms = (time.perf_counter() - intent_t0) * 1000.0
    state = _DrainState(
        entries=entries,
        entries_total=entries_total,
        finalized_intents=finalized_intents,
        intent_log_ready=bool(intent_log_ready),
        intent_log_load_ms=float(intent_log_load_ms),
    )

    for idx, entry in enumerate(entries):
        if _process_queue_entry(idx, entry, state):
            break

    requeue_failed = 0
    requeue_ok = True
    if state.requeue:
        requeue_ok = _requeue_entries(state.requeue)
        if not requeue_ok:
            requeue_failed = len(state.requeue)
            state.errors += requeue_failed
            _diag(f"requeue failed for {requeue_failed} entries")

    if state.sqlite_acked_ids:
        if not _ack_queue_entries_sqlite(sorted(state.sqlite_acked_ids)):
            state.errors += len(state.sqlite_acked_ids)
            _diag(f"queue db ack failed for {len(state.sqlite_acked_ids)} entries")

    return state.to_stats(drain_t0, requeue_ok, requeue_failed)


def main() -> int:
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass
    stats = _drain_queue()
    if os.environ.get("NAUTICAL_DIAG") == "1":
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
            f"drain_ms={stats.get('drain_ms', 0)}"
        )
    errors = stats.get("errors", 0)
    dead_lettered = stats.get("dead_lettered", 0)
    queue_lock_failures = stats.get("queue_lock_failures", 0)
    if _EXIT_STRICT and (errors > 0 or dead_lettered > 0 or queue_lock_failures > 0):
        _emit_exit_feedback(
            f"[nautical] on-exit: {dead_lettered} dead-lettered, {errors} errors, {queue_lock_failures} queue lock failures. "
            "Check .nautical_dead_letter.jsonl (set NAUTICAL_EXIT_STRICT=0 to disable)"
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        _diag(f"on-exit unexpected error: {e}")
        _emit_exit_feedback("[nautical] on-exit: unexpected error")
        try:
            _write_dead_letter({"error": "unexpected_error"}, "on-exit exception")
        except Exception:
            pass
        raise SystemExit(1)
