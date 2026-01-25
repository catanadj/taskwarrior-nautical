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
TW_DATA_DIR = Path(os.environ.get("TASKDATA") or str(TW_DIR)).expanduser()

_QUEUE_PATH = TW_DATA_DIR / ".nautical_spawn_queue.jsonl"
_QUEUE_LOCK = TW_DATA_DIR / ".nautical_spawn_queue.lock"
_DEAD_LETTER_PATH = TW_DATA_DIR / ".nautical_dead_letter.jsonl"
_DEAD_LETTER_LOCK = TW_DATA_DIR / ".nautical_dead_letter.lock"
_QUEUE_MAX_BYTES = int(os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_BYTES") or 524288)
_QUEUE_MAX_LINES = int(os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_LINES") or 10000)
_DEAD_LETTER_MAX_BYTES = int(os.environ.get("NAUTICAL_DEAD_LETTER_MAX_BYTES") or 524288)


def _diag(msg: str) -> None:
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass


def _run_task(cmd: list[str], *, input_text: str | None = None, timeout: float = 6.0) -> tuple[bool, str, str]:
    try:
        r = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return (r.returncode == 0, r.stdout or "", r.stderr or "")
    except subprocess.TimeoutExpired:
        return (False, "", "timeout")
    except Exception as e:
        return (False, "", str(e))

def _is_lock_error(err: str) -> bool:
    e = (err or "").lower()
    return "lock" in e or "locked" in e or "database is locked" in e

def _sleep(secs: float) -> None:
    time.sleep(secs)


@contextmanager
def _lock_queue():
    if fcntl is not None:
        lf = None
        acquired = False
        try:
            _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            fd = os.open(str(_QUEUE_LOCK), os.O_CREAT | os.O_RDWR, 0o600)
            lf = os.fdopen(fd, "a", encoding="utf-8")
            for _ in range(6):
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except Exception:
                    time.sleep(0.05)
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
    for _ in range(6):
        try:
            _QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            fd = os.open(str(_QUEUE_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
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
                os.unlink(_QUEUE_LOCK)
        except Exception:
            pass


@contextmanager
def _lock_dead_letter():
    if fcntl is not None:
        lf = None
        acquired = False
        try:
            _DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            fd = os.open(str(_DEAD_LETTER_LOCK), os.O_CREAT | os.O_RDWR, 0o600)
            lf = os.fdopen(fd, "a", encoding="utf-8")
            for _ in range(6):
                try:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except Exception:
                    time.sleep(0.05)
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
    for _ in range(6):
        try:
            _DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            fd = os.open(str(_DEAD_LETTER_LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            acquired = True
            break
        except FileExistsError:
            time.sleep(0.05)
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
                os.unlink(_DEAD_LETTER_LOCK)
        except Exception:
            pass


def _write_dead_letter(entry: dict, reason: str) -> None:
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": reason,
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
                except Exception:
                    pass
            fd = os.open(str(_DEAD_LETTER_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            pass

def _take_queue_entries() -> list[dict]:
    entries: list[dict] = []
    with _lock_queue() as locked:
        if not locked:
            _diag("queue lock not acquired; drain deferred")
            return entries
        try:
            if not _QUEUE_PATH.exists():
                return entries
        except Exception:
            return entries
        try:
            st = _QUEUE_PATH.stat()
            if _QUEUE_MAX_BYTES > 0 and st.st_size > _QUEUE_MAX_BYTES:
                try:
                    ts = int(time.time())
                    overflow = _QUEUE_PATH.with_suffix(f".overflow.{ts}.jsonl")
                    os.replace(_QUEUE_PATH, overflow)
                    _diag(f"queue rotated: {overflow}")
                except Exception:
                    pass
                return entries
        except Exception:
            pass

        tmp_path = _QUEUE_PATH.with_suffix(".staging")
        try:
            fd_out = os.open(str(tmp_path), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            with open(_QUEUE_PATH, "r", encoding="utf-8") as f_in, os.fdopen(fd_out, "w", encoding="utf-8") as f_out:
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
                        f_out.write(line)
                        continue
                    if isinstance(obj, dict):
                        entries.append(obj)
                    else:
                        f_out.write(line)
            os.replace(tmp_path, _QUEUE_PATH)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
    return entries


def _requeue_entries(entries: list[dict]) -> None:
    if not entries:
        return
    with _lock_queue() as locked:
        if not locked:
            return
        try:
            fd = os.open(str(_QUEUE_PATH), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as f:
                for obj in entries:
                    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        except Exception:
            pass


def _export_uuid(uuid_str: str) -> dict | None:
    if not uuid_str:
        return None
    ok, out, _err = _run_task(
        ["task", "rc.hooks=off", "rc.json.array=off", f"uuid:{uuid_str}", "export"],
        timeout=3.0,
    )
    if not ok:
        return None
    try:
        obj = json.loads(out.strip() or "{}")
        return obj if obj.get("uuid") else None
    except Exception:
        return None


def _import_child(obj: dict) -> tuple[bool, str]:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    max_retries = 4
    last_err = ""
    for attempt in range(max_retries):
        ok, _out, err = _run_task(
            ["task", f"rc.data.location={TW_DATA_DIR}", "rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            input_text=payload,
            timeout=8.0,
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


def _update_parent_nextlink(parent_uuid: str, child_short: str) -> tuple[bool, str]:
    if not parent_uuid or not child_short:
        return False, "missing parent or child"
    ok, _out, err = _run_task(
        ["task", "rc.hooks=off", "rc.verbose=nothing", f"uuid:{parent_uuid}", "modify", f"nextLink:{child_short}"],
        timeout=4.0,
    )
    return ok, err or ""


def _drain_queue() -> dict:
    processed = 0
    errors = 0
    requeue: list[dict] = []

    for entry in _take_queue_entries():
        parent_uuid = (entry.get("parent_uuid") or "").strip()
        child = entry.get("child") or {}
        child_short = (entry.get("child_short") or "").strip()
        child_uuid = (child.get("uuid") or "").strip()

        if not child_uuid:
            _write_dead_letter(entry, "missing child uuid")
            errors += 1
            continue

        exists = _export_uuid(child_uuid)
        if not exists:
            ok, err = _import_child(child)
            if not ok:
                # If import reported failure but the child exists, continue.
                if _export_uuid(child_uuid):
                    exists = True
                else:
                    _diag(f"child import failed: {err}")
                    _write_dead_letter(entry, f"child import failed: {err}")
                    errors += 1
                    continue

        # Confirm child exists before touching parent.
        if not _export_uuid(child_uuid):
            _write_dead_letter(entry, "child missing after import")
            errors += 1
            continue

        if parent_uuid and child_short:
            ok, err = _update_parent_nextlink(parent_uuid, child_short)
            if not ok:
                _diag(f"parent update failed: {parent_uuid}")
                if _is_lock_error(err):
                    requeue.append(entry)
                else:
                    _write_dead_letter(entry, f"parent update failed: {err}")
                errors += 1
                continue

        processed += 1

    if requeue:
        _requeue_entries(requeue)

    return {"processed": processed, "errors": errors, "requeued": len(requeue)}


def main() -> int:
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass
    stats = _drain_queue()
    if os.environ.get("NAUTICAL_DIAG") == "1":
        _diag(f"on-exit drain: processed={stats.get('processed', 0)} errors={stats.get('errors', 0)} requeued={stats.get('requeued', 0)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            _diag(f"on-exit unexpected error: {e}")
        try:
            _write_dead_letter({"error": str(e)}, "on-exit exception")
        except Exception:
            pass
        raise SystemExit(0)
