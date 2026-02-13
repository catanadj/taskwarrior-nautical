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
_CORE_BASE = Path(os.environ.get("NAUTICAL_CORE_PATH") or str(TW_DIR)).expanduser().resolve()
core = None
try:
    pyfile = _CORE_BASE / "nautical_core.py"
    pkgini = _CORE_BASE / "nautical_core" / "__init__.py"
    target = pyfile if pyfile.is_file() else pkgini if pkgini.is_file() else None
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
except Exception:
    core = None
TW_DATA_DIR = Path(os.environ.get("TASKDATA") or str(TW_DIR)).expanduser()

_QUEUE_PATH = TW_DATA_DIR / ".nautical_spawn_queue.jsonl"
_QUEUE_PROCESSING_PATH = TW_DATA_DIR / ".nautical_spawn_queue.processing.jsonl"
_QUEUE_LOCK = TW_DATA_DIR / ".nautical_spawn_queue.lock"
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
    return TW_DATA_DIR / "lock"

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

def _record_queue_lock_failure() -> None:
    global _LAST_QUEUE_LOCK_DIAG_TS
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

def _take_queue_entries() -> list[dict]:
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
            st = _QUEUE_PATH.stat()
            overflow_path = None
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


def _requeue_entries(entries: list[dict]) -> bool:
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
        [
            "task",
            f"rc.data.location={TW_DATA_DIR}",
            "rc.hooks=off",
            "rc.json.array=off",
            "rc.verbose=nothing",
            "rc.color=off",
            f"uuid:{uuid_str}",
            "export",
        ],
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
            ["task", f"rc.data.location={TW_DATA_DIR}", "rc.hooks=off", "rc.verbose=nothing", "import", "-"],
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
    res = _export_uuid(parent_uuid)
    if res.get("retryable"):
        return False, "parent export locked"
    parent = res.get("obj") if isinstance(res, dict) else None
    if not parent:
        return False, "parent missing"
    current = (parent.get("nextLink") or "").strip()
    expected = (expected_prev or "").strip()
    if current == child_short:
        return True, ""
    if expected:
        if current != expected:
            return False, "parent nextLink changed"
    else:
        if current:
            return False, "parent nextLink already set"
    ok, _out, err = _run_task(
        [
            "task",
            f"rc.data.location={TW_DATA_DIR}",
            "rc.hooks=off",
            "rc.verbose=nothing",
            f"uuid:{parent_uuid}",
            "modify",
            f"nextLink:{child_short}",
        ],
        timeout=_TASK_TIMEOUT_MODIFY,
        retries=_TASK_RETRIES_MODIFY,
        retry_delay=_TASK_RETRY_DELAY,
    )
    return ok, err or ""


def _drain_queue() -> dict:
    processed = 0
    errors = 0
    requeue: list[dict] = []
    dead_lettered = 0

    for entry in _take_queue_entries():
        valid, reason = _validate_queue_entry(entry)
        if not valid:
            _write_dead_letter(entry, reason)
            dead_lettered += 1
            errors += 1
            continue
        spawn_intent_id = (entry.get("spawn_intent_id") or "").strip()
        parent_uuid = (entry.get("parent_uuid") or "").strip()
        expected_parent_nextlink = (entry.get("parent_nextlink") or "").strip()
        child = entry.get("child") or {}
        child_short = (entry.get("child_short") or "").strip()
        child_uuid = (child.get("uuid") or "").strip()
        export_res = _export_uuid(child_uuid)
        imported = False
        if not export_res.get("exists"):
            if export_res.get("retryable"):
                if spawn_intent_id:
                    _diag(f"task lock active; requeue (intent={spawn_intent_id})")
                if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
                    _write_dead_letter(entry, "exceeded retry budget")
                    dead_lettered += 1
                    errors += 1
                else:
                    requeue.append(entry)
                continue
            ok, err = _import_child(child)
            if not ok:
                if _is_lock_error(err):
                    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
                        _write_dead_letter(entry, "exceeded retry budget")
                        dead_lettered += 1
                        errors += 1
                    else:
                        requeue.append(entry)
                    continue
                # If import reported failure but the child exists, continue.
                if _export_uuid(child_uuid).get("exists"):
                    export_res = {"exists": True}
                else:
                    if spawn_intent_id:
                        _diag(f"child import failed (intent={spawn_intent_id}): {err}")
                    else:
                        _diag(f"child import failed: {err}")
                    _write_dead_letter(entry, f"child import failed: {err}")
                    dead_lettered += 1
                    errors += 1
                    continue
            imported = True

        if imported:
            # Confirm child exists before touching parent only when we just imported.
            confirm_res = _export_uuid(child_uuid)
            if not confirm_res.get("exists"):
                if confirm_res.get("retryable"):
                    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
                        _write_dead_letter(entry, "exceeded retry budget")
                        errors += 1
                    else:
                        requeue.append(entry)
                    continue
                if spawn_intent_id:
                    _diag(f"child missing after import (intent={spawn_intent_id})")
                _write_dead_letter(entry, "child missing after import")
                dead_lettered += 1
                errors += 1
                continue

        if parent_uuid and child_short:
            ok, err = _update_parent_nextlink(parent_uuid, child_short, expected_parent_nextlink)
            if not ok:
                if spawn_intent_id:
                    _diag(f"parent update failed (intent={spawn_intent_id}): {parent_uuid}")
                else:
                    _diag(f"parent update failed: {parent_uuid}")
                if err == "parent export locked" or _is_lock_error(err):
                    if _bump_attempts(entry) > _QUEUE_RETRY_MAX:
                        _write_dead_letter(entry, "exceeded retry budget")
                        dead_lettered += 1
                        errors += 1
                    else:
                        requeue.append(entry)
                else:
                    _write_dead_letter(entry, f"parent update failed: {err}")
                    dead_lettered += 1
                    errors += 1
                continue

        processed += 1

    requeue_failed = 0
    requeue_ok = True
    if requeue:
        requeue_ok = _requeue_entries(requeue)
        if not requeue_ok:
            requeue_failed = len(requeue)
            errors += requeue_failed
            _diag(f"requeue failed for {requeue_failed} entries; keeping processing file")
    if requeue_ok:
        try:
            if _QUEUE_PROCESSING_PATH.exists():
                _QUEUE_PROCESSING_PATH.unlink()
                if _DURABLE_QUEUE:
                    _fsync_dir(_QUEUE_PROCESSING_PATH.parent)
        except Exception:
            pass

    return {
        "processed": processed,
        "errors": errors,
        "requeued": len(requeue) if requeue_ok else 0,
        "requeue_failed": requeue_failed,
        "dead_lettered": dead_lettered,
    }


def main() -> int:
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass
    stats = _drain_queue()
    if os.environ.get("NAUTICAL_DIAG") == "1":
        _diag(
            "on-exit drain: "
            f"processed={stats.get('processed', 0)} "
            f"errors={stats.get('errors', 0)} "
            f"requeued={stats.get('requeued', 0)} "
            f"requeue_failed={stats.get('requeue_failed', 0)} "
            f"dead_lettered={stats.get('dead_lettered', 0)}"
        )
    errors = stats.get("errors", 0)
    dead_lettered = stats.get("dead_lettered", 0)
    if _EXIT_STRICT and (errors > 0 or dead_lettered > 0):
        _emit_exit_feedback(
            f"[nautical] on-exit: {dead_lettered} dead-lettered, {errors} errors. "
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
        _emit_exit_feedback(f"[nautical] on-exit: unexpected error: {e}")
        try:
            _write_dead_letter({"error": str(e)}, "on-exit exception")
        except Exception:
            pass
        raise SystemExit(1)
