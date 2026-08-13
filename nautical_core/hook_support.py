from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Mapping, Sequence, TypeAlias

from .integration_models import CommandFailureKind, TaskCommandResult
from .task_command import failure_message


TaskRow: TypeAlias = dict[str, Any]
CommandRunner: TypeAlias = Callable[..., TaskCommandResult]
DiagnosticCallback: TypeAlias = Callable[[str], object]


@dataclass(frozen=True, slots=True)
class LookupResult:
    """Outcome of a mutation-sensitive Taskwarrior read.

    ``absent`` is a confirmed empty result.  ``unavailable`` means the
    command or its output could not be trusted and must not be used to make
    a spawn decision.
    """

    state: str
    task: dict | None = None
    reason: str = ""

    @classmethod
    def found(cls, task: dict) -> "LookupResult":
        return cls("found", task=task)

    @classmethod
    def absent(cls, reason: str = "") -> "LookupResult":
        return cls("absent", reason=reason)

    @classmethod
    def unavailable(cls, reason: str) -> "LookupResult":
        return cls("unavailable", reason=reason or "lookup unavailable")

    @property
    def is_found(self) -> bool:
        return self.state == "found" and isinstance(self.task, dict)

    @property
    def is_absent(self) -> bool:
        return self.state == "absent"

    @property
    def is_unavailable(self) -> bool:
        return self.state == "unavailable"


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


def parse_export_array_result(
    result: TaskCommandResult,
    *,
    diag: DiagnosticCallback | None = None,
) -> tuple[bool, list[TaskRow], str]:
    """Strictly parse a successful array export; never turn failures into empty data."""
    if not result.ok:
        return False, [], failure_message(result, "Taskwarrior export")
    raw = (result.stdout or "").strip()
    if not raw:
        return False, [], "Taskwarrior export returned empty output"
    try:
        data = json.loads(raw)
    except (TypeError, ValueError) as exc:
        message = f"Taskwarrior export returned invalid JSON: {exc}"
        if callable(diag):
            diag(message)
        return False, [], message
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        message = "Taskwarrior export returned a non-array or invalid row payload"
        if callable(diag):
            diag(message)
        return False, [], message
    return True, data, ""


def export_uuid_short(
    *,
    run_task: CommandRunner,
    task_cmd_prefix: Sequence[str],
    uuid_short: str,
    env: Mapping[str, str] | None = None,
    timeout: float = 2.5,
    retries: int = 2,
    diag: DiagnosticCallback | None = None,
) -> TaskRow | None:
    lookup = export_uuid_short_result(
        run_task=run_task,
        task_cmd_prefix=task_cmd_prefix,
        uuid_short=uuid_short,
        env=env,
        timeout=timeout,
        retries=retries,
        diag=diag,
    )
    return lookup.task if lookup.is_found else None


def export_uuid_short_result(
    *,
    run_task: CommandRunner,
    task_cmd_prefix: Sequence[str],
    uuid_short: str,
    env: Mapping[str, str] | None = None,
    timeout: float = 2.5,
    retries: int = 2,
    diag: DiagnosticCallback | None = None,
) -> LookupResult:
    result = run_task_result(
        run_task=run_task,
        cmd=list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=off", f"uuid:{uuid_short}", "export"],
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not result.ok:
        reason = f"{failure_message(result, 'UUID export')}"
        if callable(diag):
            diag(f"export uuid:{uuid_short} failed: {reason}")
        return LookupResult.unavailable(reason)
    raw = (result.stdout or "").strip()
    if not raw:
        return LookupResult.absent("Taskwarrior returned no matching task")
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return LookupResult.unavailable("UUID export returned a non-object payload")
        if not obj.get("uuid"):
            return LookupResult.absent("Taskwarrior returned no matching task")
        if not str(obj.get("uuid") or "").lower().startswith((uuid_short or "").lower()):
            if callable(diag):
                diag(f"uuid prefix mismatch for {uuid_short}")
            return LookupResult.unavailable(f"UUID export returned a different task for {uuid_short}")
        return LookupResult.found(obj)
    except Exception as exc:
        return LookupResult.unavailable(f"UUID export returned invalid JSON: {exc}")


def task_lookup_by_uuid_uncached(
    *,
    run_task: CommandRunner,
    task_cmd_prefix: Sequence[str],
    uuid_str: str,
    env: Mapping[str, str] | None = None,
    timeout: float = 2.5,
    retries: int = 2,
    diag: DiagnosticCallback | None = None,
) -> LookupResult:
    result = run_task_result(
        run_task=run_task,
        cmd=list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=off", f"uuid:{uuid_str}", "export"],
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not result.ok:
        if callable(diag):
            diag(f"task exists check failed (uuid={uuid_str[:8]}): {failure_message(result, 'UUID existence check')}")
        return LookupResult.unavailable(failure_message(result, "UUID existence check"))
    raw = (result.stdout or "").strip()
    if not raw:
        return LookupResult.absent("Taskwarrior returned no matching task")
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return LookupResult.unavailable("UUID existence check returned a non-object payload")
    except Exception as exc:
        return LookupResult.unavailable(f"UUID existence check returned invalid JSON: {exc}")
    if not data.get("uuid"):
        return LookupResult.absent("Taskwarrior returned no matching task")
    if not str(data.get("uuid") or "").lower().startswith((uuid_str or "").lower()):
        return LookupResult.unavailable(f"UUID existence check returned a different task for {uuid_str}")
    return LookupResult.found(data)


def export_uuid_full(
    *,
    run_task: CommandRunner,
    task_cmd_prefix: Sequence[str],
    uuid_str: str,
    env: Mapping[str, str] | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    diag: DiagnosticCallback | None = None,
) -> TaskRow | None:
    result = run_task_result(
        run_task=run_task,
        cmd=list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=1", f"export uuid:{uuid_str}"],
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not result.ok:
        if callable(diag):
            diag(f"task export uuid:{uuid_str} failed: {failure_message(result, 'UUID export')}")
        return None
    parsed, rows, error = parse_export_array_result(result, diag=diag)
    if not parsed:
        return None
    return rows[0] if rows else None


def export_uuid_status(
    *,
    run_task: CommandRunner,
    task_cmd_prefix: Sequence[str],
    uuid_str: str,
    timeout: float,
    retries: int,
    retry_delay: float = 0.0,
    env: Mapping[str, str] | None = None,
    tolerate_noisy_stdout: bool = False,
) -> dict[str, Any]:
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
    result = run_task_result(
        run_task=run_task,
        cmd=cmd,
        env=env,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    if not result.ok:
        retryable = result.kind in {CommandFailureKind.BUSY, CommandFailureKind.TIMEOUT}
        return {"exists": False, "retryable": retryable, "err": failure_message(result, "UUID status lookup"), "obj": None}
    raw_stdout = (result.stdout or "").strip()
    if not raw_stdout:
        # Taskwarrior uses a successful empty export for a valid query with
        # no matching UUID.  This is an authoritative absence, not malformed
        # protocol data; preserve fail-closed behavior for non-empty junk.
        return {"exists": False, "retryable": False, "err": "not found", "obj": None}
    try:
        obj = json.loads(raw_stdout)
        if isinstance(obj, dict) and obj.get("uuid"):
            return {"exists": True, "retryable": False, "err": "", "obj": obj}
        return {"exists": False, "retryable": False, "err": "not found", "obj": None}
    except Exception:
        # A successful process exit does not make malformed output safe to
        # interpret. Treat it as temporarily unavailable so mutation callers
        # requeue instead of importing a duplicate child.
        return {"exists": False, "retryable": True, "err": "parse error", "obj": None}


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


def build_chain_export_args(
    *,
    task_cmd_prefix: Sequence[str],
    chain_id: str,
    since: datetime | None = None,
    extra: str | None = None,
    limit: int | None = None,
    parse_extra_tokens: Callable[[str | None], list[str] | None],
    diag: DiagnosticCallback | None = None,
) -> list[str] | None:
    args = list(task_cmd_prefix) + ["rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
    if since:
        args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
    if limit and isinstance(limit, int) and limit > 0:
        args.append(f"limit:{limit}")
    if extra:
        extra_tokens = parse_extra_tokens(extra)
        if extra_tokens is None:
            if callable(diag):
                diag(f"tw_export_chain rejected extra: {extra!r}")
            return None
        args += extra_tokens
    args.append("export")
    return args


def parse_export_array(
    out: str,
    *,
    diag: DiagnosticCallback | None = None,
) -> list[TaskRow]:
    try:
        data = json.loads((out or "").strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception as exc:
        if callable(diag):
            diag(f"tw_export_chain JSON parse failed: {exc}")
        return []
