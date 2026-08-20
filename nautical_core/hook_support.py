from __future__ import annotations

import os
import re
from typing import Callable, Mapping, Sequence, TypeAlias

from .integration_models import TaskCommandResult


CommandRunner: TypeAlias = Callable[..., TaskCommandResult]


def build_task_cmd_prefix(*, use_rc_data_location: bool, tw_data_dir: object) -> list[str]:
    # Test-only binary injection keeps failure-path benchmarks deterministic
    # without changing the production Taskwarrior command contract.
    cmd = [os.environ.get("NAUTICAL_BENCH_TASK_BIN", "task")]
    if use_rc_data_location:
        cmd.append(f"rc.data.location={tw_data_dir}")
    return cmd


def run_task_result(
    *,
    run_task: CommandRunner,
    cmd: Sequence[str],
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 6.0,
    retries: int = 1,
    retry_delay: float = 0.0,
    use_tempfiles: bool = False,
) -> TaskCommandResult:
    """Invoke a typed hook runner without tuple or failure coercion."""
    runner_kwargs = {
        "env": env,
        "input_text": input_text,
        "timeout": timeout,
        "retries": retries,
        "retry_delay": retry_delay,
    }
    if use_tempfiles:
        runner_kwargs["use_tempfiles"] = True
    result = run_task(cmd, **runner_kwargs)
    if not isinstance(result, TaskCommandResult):
        raise TypeError("Taskwarrior command runner returned an untyped result")
    return result


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
