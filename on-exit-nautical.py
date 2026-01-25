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
            lf = open(_QUEUE_LOCK, "a", encoding="utf-8")
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


def _take_queue_entries() -> list[dict]:
    entries: list[dict] = []
    with _lock_queue() as locked:
        if not locked:
            return entries
        try:
            if not _QUEUE_PATH.exists():
                return entries
        except Exception:
            return entries
        try:
            raw = _QUEUE_PATH.read_text(encoding="utf-8")
        except Exception:
            return entries
        if not (raw or "").strip():
            try:
                _QUEUE_PATH.unlink()
            except Exception:
                pass
            return entries

        remainder: list[str] = []
        for line in raw.splitlines():
            ln = line.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                remainder.append(line)
                continue
            if isinstance(obj, dict):
                entries.append(obj)
            else:
                remainder.append(line)

        try:
            if remainder:
                _QUEUE_PATH.write_text("\n".join(remainder) + "\n", encoding="utf-8")
            else:
                _QUEUE_PATH.write_text("", encoding="utf-8")
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
            with open(_QUEUE_PATH, "a", encoding="utf-8") as f:
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
    ok, _out, err = _run_task(
        ["task", f"rc.data.location={TW_DATA_DIR}", "rc.hooks=off", "rc.verbose=nothing", "import", "-"],
        input_text=payload,
        timeout=8.0,
    )
    return ok, err or ""


def _update_parent_nextlink(parent_uuid: str, child_short: str) -> bool:
    if not parent_uuid or not child_short:
        return False
    ok, _out, _err = _run_task(
        ["task", "rc.hooks=off", "rc.verbose=nothing", f"uuid:{parent_uuid}", "modify", f"nextLink:{child_short}"],
        timeout=4.0,
    )
    return ok


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
            errors += 1
            continue

        exists = _export_uuid(child_uuid)
        if not exists:
            ok, err = _import_child(child)
            if not ok:
                # Keep for retry unless it is a hard error.
                _diag(f"child import failed: {err}")
                requeue.append(entry)
                errors += 1
                continue

        # Confirm child exists before touching parent.
        if not _export_uuid(child_uuid):
            requeue.append(entry)
            errors += 1
            continue

        if parent_uuid and child_short:
            if not _update_parent_nextlink(parent_uuid, child_short):
                _diag(f"parent update failed: {parent_uuid}")
                requeue.append(entry)
                errors += 1
                continue

        processed += 1

    if requeue:
        _requeue_entries(requeue)

    return {"processed": processed, "errors": errors, "requeued": len(requeue)}


def main() -> int:
    stats = _drain_queue()
    ok = stats.get("errors", 0) == 0
    print(json.dumps({"ok": ok, **stats}, ensure_ascii=False), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
