from __future__ import annotations

import os
import random
import subprocess
import time


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
