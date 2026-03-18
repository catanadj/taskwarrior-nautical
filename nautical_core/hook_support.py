from __future__ import annotations

import os
import random
import subprocess
import time
import json
import re


def build_task_cmd_prefix(*, use_rc_data_location: bool, tw_data_dir) -> list[str]:
    cmd = ["task"]
    if use_rc_data_location:
        cmd.append(f"rc.data.location={tw_data_dir}")
    return cmd


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
        return (proc.returncode == 0, out, err)
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
        return False, "", "timeout"
    except Exception as exc:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass
        return False, "", str(exc)


def run_task(
    cmd: list[str],
    *,
    core_run_task=None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 6.0,
    retries: int = 1,
    retry_delay: float = 0.0,
    use_tempfiles: bool = False,
) -> tuple[bool, str, str]:
    if callable(core_run_task):
        return core_run_task(
            cmd,
            env=(env or os.environ.copy()),
            input_text=input_text,
            timeout=timeout,
            retries=max(1, int(retries)),
            retry_delay=max(0.0, float(retry_delay)),
            use_tempfiles=use_tempfiles,
        )

    env_map = env or os.environ.copy()
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
        if attempt >= attempts or delay <= 0:
            break
        jitter = random.uniform(0.0, delay)
        time.sleep(delay * (2 ** (attempt - 1)) + jitter)
    return False, last_out, last_err


def export_uuid_short(
    *,
    run_task,
    task_cmd_prefix,
    uuid_short: str,
    env=None,
    timeout: float = 2.5,
    retries: int = 2,
    diag=None,
):
    ok, out, err = run_task(
        list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=off", f"uuid:{uuid_short}", "export"],
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not ok:
        if callable(diag):
            diag(f"export uuid:{uuid_short} failed: {(err or '').strip()}")
        return None
    try:
        obj = json.loads((out or "").strip() or "{}")
        if not obj.get("uuid"):
            return None
        if not str(obj.get("uuid") or "").lower().startswith((uuid_short or "").lower()):
            if callable(diag):
                diag(f"uuid prefix mismatch for {uuid_short}")
            return None
        return obj
    except Exception:
        return None


def task_exists_by_uuid_uncached(
    *,
    run_task,
    task_cmd_prefix,
    uuid_str: str,
    env=None,
    timeout: float = 2.5,
    retries: int = 2,
    diag=None,
) -> bool:
    ok, out, err = run_task(
        list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=off", f"uuid:{uuid_str}", "export"],
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not ok:
        if callable(diag):
            diag(f"task exists check failed (uuid={uuid_str[:8]}): {(err or '').strip()}")
        return False
    try:
        data = json.loads((out or "").strip() or "{}")
    except Exception:
        data = {}
    return bool(data.get("uuid"))


def export_uuid_full(
    *,
    run_task,
    task_cmd_prefix,
    uuid_str: str,
    env=None,
    timeout: float = 3.0,
    retries: int = 2,
    diag=None,
):
    ok, out, err = run_task(
        list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=1", f"export uuid:{uuid_str}"],
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not ok:
        if callable(diag):
            diag(f"task export uuid:{uuid_str} failed: {(err or '').strip()}")
        return None
    try:
        arr = json.loads(out) if out and out.strip().startswith("[") else []
        return arr[0] if arr else None
    except Exception:
        return None


def export_uuid_status(
    *,
    run_task,
    task_cmd_prefix,
    uuid_str: str,
    timeout: float,
    retries: int,
    retry_delay: float = 0.0,
    env=None,
    is_lock_error=None,
    tolerate_noisy_stdout: bool = False,
):
    if not uuid_str:
        return {"exists": False, "retryable": False, "err": "missing uuid", "obj": None}
    cmd = list(task_cmd_prefix) + [
        "rc.hooks=off",
        "rc.json.array=off",
        "rc.verbose=nothing",
        "rc.color=off",
        f"uuid:{uuid_str}",
        "export",
    ]
    ok, out, err = run_task(
        cmd,
        env=env,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    if not ok:
        retryable = bool(is_lock_error(err)) if callable(is_lock_error) else False
        return {"exists": False, "retryable": retryable, "err": err or "", "obj": None}
    try:
        obj = json.loads((out or "").strip() or "{}")
        if obj.get("uuid"):
            return {"exists": True, "retryable": False, "err": "", "obj": obj}
        return {"exists": False, "retryable": False, "err": "not found", "obj": None}
    except Exception:
        if tolerate_noisy_stdout and uuid_str in (out or ""):
            return {"exists": True, "retryable": False, "err": "", "obj": {"uuid": uuid_str}}
        return {"exists": False, "retryable": False, "err": "parse error", "obj": None}


def parse_extra_tokens(extra: str | None) -> list[str] | None:
    """Parse extra Taskwarrior filters in strict token form: key:value."""
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
