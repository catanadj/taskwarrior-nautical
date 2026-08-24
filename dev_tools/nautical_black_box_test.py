#!/usr/bin/env python3
"""Black-box Nautical lifecycle test using a disposable Taskwarrior database."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], env: dict[str, str], *, timeout: float = 30.0) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
        )
    return proc


def _task(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return _run(["task", *args], env)


def _write_taskrc(path: Path, data_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"data.location={data_dir}",
                f"hooks.location={data_dir / 'hooks'}",
                "hooks=on",
                "confirmation=off",
                "verbose=nothing",
                f"include {ROOT / 'uda.conf'}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _install_runtime(data_dir: Path) -> None:
    hooks_dir = data_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    for source, target in (
        ("on-add.nautical", "on-add"),
        ("on-modify.nautical", "on-modify"),
        ("on-exit.nautical", "on-exit"),
    ):
        shutil.copy2(ROOT / source, hooks_dir / target)
        (hooks_dir / target).chmod(0o755)
    shutil.copytree(ROOT / "nautical_core", data_dir / "nautical_core", dirs_exist_ok=True)


def _write_config(path: Path, anchor_dir: Path, omit_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                'tz = "UTC"',
                f'anchor_file_dir = "{anchor_dir}"',
                f'omit_file_dir = "{omit_dir}"',
                "show_analytics = false",
                "show_timeline_gaps = false",
                'panel_mode = "text"',
                "",
                "[anchor_presets]",
                'blackbox_weekdays = "w:mon,tue,wed,thu,fri"',
                "",
                "[omit_presets]",
                'blackbox_monday = "w:mon"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _export(env: dict[str, str], *filters: str) -> list[dict]:
    proc = _task(["rc.hooks=off", "rc.json.array=1", *filters, "export"], env)
    raw = (proc.stdout or "").strip()
    if not raw:
        return []
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise AssertionError(f"unexpected export payload: {payload!r}")
    return [row for row in payload if isinstance(row, dict)]


def _one(env: dict[str, str], *filters: str) -> dict:
    rows = _export(env, *filters)
    if len(rows) != 1:
        raise AssertionError(f"expected one task for {filters!r}, got {len(rows)}")
    return rows[0]


def _one_eventually(env: dict[str, str], *filters: str, timeout: float = 5.0) -> dict:
    """Allow on-exit's synchronous queue drain to settle before asserting visibility."""
    deadline = time.monotonic() + timeout
    last_count = 0
    while True:
        rows = _export(env, *filters)
        last_count = len(rows)
        if last_count == 1:
            return rows[0]
        if last_count > 1 or time.monotonic() >= deadline:
            raise AssertionError(f"expected one task for {filters!r}, got {last_count}")
        time.sleep(0.1)


def _parse_tw_datetime(value: object) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise AssertionError("task datetime is missing")
    return datetime.strptime(raw, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _assert_root(task: dict, recurrence_field: str) -> None:
    if task.get("chain") != "on":
        raise AssertionError(f"{recurrence_field}: chain was not enabled")
    if not str(task.get("chainID") or "").strip():
        raise AssertionError(f"{recurrence_field}: chainID was not assigned")
    if int(task.get("link") or 0) != 1:
        raise AssertionError(f"{recurrence_field}: root link is not 1")


def _complete_and_child(env: dict[str, str], root: dict) -> dict:
    root_uuid = str(root.get("uuid") or "").strip()
    chain_id = str(root.get("chainID") or "").strip()
    _task([f"uuid:{root_uuid}", "done"], env)
    child = _one_eventually(env, f"chainID:{chain_id}", "status:pending", "link:2")
    if child.get("prevLink") != root_uuid[:8]:
        raise AssertionError("spawned child does not point to its parent")
    parent = _one(env, f"uuid:{root_uuid}")
    if parent.get("nextLink") != str(child.get("uuid") or "")[:8]:
        raise AssertionError("completed parent does not point to its child")
    return child


def _scenario_cp(env: dict[str, str]) -> dict:
    description = "blackbox cp"
    _task(["add", description, "cp:1d", "due:today"], env)
    root = _one(env, f"description:{description}", "status:pending")
    _assert_root(root, "cp")
    child = _complete_and_child(env, root)
    delta = _parse_tw_datetime(child.get("due")) - _parse_tw_datetime(root.get("due"))
    if delta != timedelta(days=1):
        raise AssertionError(f"cp child delta was {delta}, expected 1 day")
    return {
        "root": root["uuid"],
        "root_id": root["id"],
        "child": child["uuid"],
        "chainID": root["chainID"],
    }


def _scenario_navigator(env: dict[str, str], cp_result: dict) -> dict:
    """Exercise Navigator's real Taskwarrior export contract."""
    proc = subprocess.run(
        [sys.executable, str(ROOT / "nautical_navigator.py"), "--mode", "chain", "--id", str(cp_result["root_id"]), "--count", "1"],
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Navigator failed against Taskwarrior: stdout={proc.stdout!r} stderr={proc.stderr!r}")
    if "Total chain length" not in proc.stdout:
        raise AssertionError(f"Navigator did not render the selected chain: {proc.stdout!r}")
    return {"root_id": cp_result["root_id"], "rendered": True}


def _scenario_anchor_preset(env: dict[str, str]) -> dict:
    description = "blackbox preset"
    # Keep the smoke test independent of the day it runs.  ``today`` can be
    # the omitted Monday and resolves to a past midnight timestamp, which
    # exercises a different recovery edge than preset/omit selection.
    preset_day = datetime.now(timezone.utc).date() + timedelta(days=2)
    while preset_day.weekday() >= 5 or preset_day.weekday() == 0:
        preset_day += timedelta(days=1)
    preset_due = preset_day.strftime("%Y%m%dT090000Z")
    _task(
        [
            "add",
            description,
            "anchor:@blackbox_weekdays",
            "omit:@blackbox_monday",
            f"due:{preset_due}",
        ],
        env,
    )
    root = _one(env, f"description:{description}", "status:pending")
    _assert_root(root, "anchor preset")
    child = _complete_and_child(env, root)
    child_due = _parse_tw_datetime(child.get("due"))
    if child_due.weekday() == 0 or child_due.weekday() > 4:
        raise AssertionError(f"preset/omit produced invalid weekday: {child_due.date()}")
    return {"root": root["uuid"], "child": child["uuid"]}


def _scenario_files(env: dict[str, str], anchor_dir: Path, omit_dir: Path) -> dict:
    today = datetime.now(timezone.utc).date()
    first = today + timedelta(days=5)
    second = today + timedelta(days=8)
    (anchor_dir / "blackbox.csv").write_text(
        "description,date,ignored\n"
        f"omitted,{first.isoformat()},x\n"
        f"accepted,{second.isoformat()},y\n",
        encoding="utf-8",
    )
    (omit_dir / "blackbox-omit.csv").write_text(
        "date,description\n"
        f"{first.isoformat()},excluded\n",
        encoding="utf-8",
    )

    description = "blackbox files"
    _task(
        [
            "add",
            description,
            "anchor_file:blackbox.csv@t=12:00",
            "omit_file:blackbox-omit.csv",
            "due:today",
        ],
        env,
    )
    root = _one(env, f"description:{description}", "status:pending")
    _assert_root(root, "anchor_file")
    child = _complete_and_child(env, root)
    if _parse_tw_datetime(child.get("due")).date() != second:
        raise AssertionError("anchor_file/omit_file did not select the non-omitted date")
    return {"root": root["uuid"], "child": child["uuid"]}


def _scenario_duplicate_guard(env: dict[str, str], cp_result: dict) -> dict:
    root_uuid = cp_result["root"]
    chain_id = str(_one(env, f"uuid:{root_uuid}").get("chainID") or "")
    before = _export(env, f"chainID:{chain_id}", "link:2", "status.not:deleted")
    _task([f"uuid:{root_uuid}", "modify", "status:pending"], env)
    _task([f"uuid:{root_uuid}", "done"], env)
    after = _export(env, f"chainID:{chain_id}", "link:2", "status.not:deleted")
    if len(before) != 1 or len(after) != 1:
        raise AssertionError(f"duplicate completion changed child count: {len(before)} -> {len(after)}")
    if before[0].get("uuid") != after[0].get("uuid"):
        raise AssertionError("duplicate completion replaced the existing child")
    return {"child": after[0]["uuid"], "count": len(after)}


def _scenario_modify(env: dict[str, str]) -> dict:
    """Exercise the real on-modify hook across separate Taskwarrior processes."""
    description = "blackbox modify"
    _task(["add", description, "cp:1d", "due:today"], env)
    root = _one(env, f"description:{description}", "status:pending")
    root_uuid = str(root.get("uuid") or "").strip()
    target = datetime.now(timezone.utc) + timedelta(days=2)
    target_token = target.strftime("%Y%m%dT090000Z")
    _task([f"uuid:{root_uuid}", "modify", f"due:{target_token}"], env)
    updated = _one(env, f"uuid:{root_uuid}")
    if updated.get("chain") != "on":
        raise AssertionError("on-modify disabled the chain during a due-date edit")
    if _parse_tw_datetime(updated.get("due")) != target.replace(hour=9, minute=0, second=0, microsecond=0):
        raise AssertionError("on-modify did not persist the requested due date")
    children = _export(env, f"chainID:{updated.get('chainID')}", "link:2", "status.not:deleted")
    if children:
        raise AssertionError("editing an uncompleted root unexpectedly spawned a child")
    return {"root": root_uuid, "due": updated.get("due")}


def _install_hook_counter(hooks_dir: Path, hook_name: str, counter_dir: Path) -> None:
    """Wrap one installed hook and count real process launches."""
    hook = hooks_dir / hook_name
    real_hook = hooks_dir / f"{hook_name}.real"
    hook.rename(real_hook)
    counter = counter_dir / f"{hook_name}.count"
    wrapper = (
        "#!/usr/bin/env python3\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"counter = Path({str(counter)!r})\n"
        "counter.parent.mkdir(parents=True, exist_ok=True)\n"
        "with counter.open('a', encoding='ascii') as stream:\n"
        "    stream.write('1\\n')\n"
        f"real_hook = {str(real_hook)!r}\n"
        "os.execv(real_hook, [real_hook, *sys.argv[1:]])\n"
    )
    hook.write_text(wrapper, encoding="utf-8")
    hook.chmod(0o755)


def _install_task_command_shim(root: Path, env: dict[str, str]) -> Path:
    """Log every Taskwarrior invocation while delegating to the real binary."""
    real_task = shutil.which("task")
    if not real_task:
        raise AssertionError("Taskwarrior binary is required for the hook process harness")
    shim_dir = root / "task-shim"
    shim_dir.mkdir()
    shim = shim_dir / "task"
    log_path = root / "task-command-log.jsonl"
    fail_marker = root / "task-shim-failed-once"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"log_path = {str(log_path)!r}\n"
        "with open(log_path, 'a', encoding='utf-8') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:], ensure_ascii=False) + '\\n')\n"
        f"fail_marker = Path({str(fail_marker)!r})\n"
        "fail_match = os.environ.get('NAUTICAL_TASK_SHIM_FAIL_MATCH', '')\n"
        "if fail_match and fail_match in ' '.join(sys.argv[1:]) and not fail_marker.exists():\n"
        "    fail_marker.touch()\n"
        "    print('injected Taskwarrior failure', file=sys.stderr)\n"
        "    raise SystemExit(42)\n"
        f"real_task = {real_task!r}\n"
        "os.execv(real_task, [real_task, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    env["PATH"] = str(shim_dir) + os.pathsep + str(env.get("PATH") or "")
    env["NAUTICAL_TASK_SHIM_LOG"] = str(log_path)
    return log_path


def _read_hook_count(counter_dir: Path, hook_name: str) -> int:
    path = counter_dir / f"{hook_name}.count"
    if not path.is_file():
        return 0
    return len(path.read_text(encoding="ascii").splitlines())


def _read_task_command_log(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    commands: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if isinstance(value, list):
            commands.append([str(item) for item in value])
    return commands


def _scenario_no_nested_hooks(env: dict[str, str], data_dir: Path) -> dict:
    """Real completion must not recursively launch lifecycle hooks."""
    hooks_dir = data_dir / "hooks"
    counter_dir = data_dir.parent / "hook-counters"
    counter_dir.mkdir()
    _install_hook_counter(hooks_dir, "on-modify", counter_dir)
    _install_hook_counter(hooks_dir, "on-exit", counter_dir)
    command_log = _install_task_command_shim(data_dir.parent, env)

    description = "blackbox hook recursion"
    future_due = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y%m%dT090000Z")
    _task(["add", description, "cp:1d", f"due:{future_due}"], env)
    root = _one(env, f"description:{description}", "status:pending")
    root_uuid = str(root.get("uuid") or "").strip()
    chain_id = str(root.get("chainID") or "").strip()
    if not chain_id:
        raise AssertionError("hook recursion fixture did not receive a chainID")

    # The fixture setup itself is outside the measured lifecycle operation.
    for hook_name in ("on-modify", "on-exit"):
        count_path = counter_dir / f"{hook_name}.count"
        if count_path.exists():
            count_path.unlink()
    if command_log.exists():
        command_log.unlink()

    _task([f"uuid:{root_uuid}", "done"], env)
    deadline = time.monotonic() + 5.0
    rows: list[dict] = []
    while time.monotonic() < deadline:
        rows = _export(env, f"chainID:{chain_id}")
        by_link = {int(float(row.get("link"))): row for row in rows if row.get("link") is not None}
        if set(by_link) == {1, 2}:
            break
        time.sleep(0.1)
    by_link = {int(float(row.get("link"))): row for row in rows if row.get("link") is not None}
    if set(by_link) != {1, 2}:
        raise AssertionError(f"hook recursion fixture did not produce one child: {rows!r}")
    child = by_link[2]
    if by_link[1].get("nextLink") != str(child.get("uuid") or "")[:8]:
        raise AssertionError("hook recursion fixture did not link the parent")

    modify_count = _read_hook_count(counter_dir, "on-modify")
    exit_count = _read_hook_count(counter_dir, "on-exit")
    if modify_count != 1 or exit_count != 1:
        raise AssertionError(
            f"lifecycle completion recursively launched hooks: on-modify={modify_count}, on-exit={exit_count}"
        )
    modify_commands = [args for args in _read_task_command_log(command_log) if "modify" in args]
    if not modify_commands:
        raise AssertionError("hook recursion fixture recorded no internal modify command")
    if any("rc.hooks=off" not in args for args in modify_commands):
        raise AssertionError(f"internal lifecycle modify omitted rc.hooks=off: {modify_commands!r}")
    return {
        "on_modify": modify_count,
        "on_exit": exit_count,
        "internal_modify_commands": len(modify_commands),
    }


def _assert_clean_state(data_dir: Path) -> None:
    outbox_db = data_dir / ".nautical-state" / ".nautical_lifecycle_outbox.db"
    if not outbox_db.exists():
        raise AssertionError("lifecycle outbox database was not created")
    with sqlite3.connect(str(outbox_db)) as conn:
        active = conn.execute(
            "SELECT COUNT(*) FROM lifecycle_outbox "
            "WHERE processing_state IN ('ready', 'claimed', 'retry', 'manual_review', 'quarantined')"
        ).fetchone()
    if active is None or int(active[0]) != 0:
        raise AssertionError(f"lifecycle outbox did not drain: {active}")


def _scenario_operator_cutover(env: dict[str, str], data_dir: Path, cp_result: dict) -> dict:
    """Exercise the installed operator surfaces against the disposable database."""
    task_bin = shutil.which("task") or "task"
    tools = {
        "doctor": ROOT / "nautical_core" / "tools" / "nautical_doctor.py",
        "query": ROOT / "nautical_core" / "tools" / "nautical_query.py",
        "reconcile": ROOT / "nautical_core" / "tools" / "nautical_reconcile.py",
        "queue": ROOT / "nautical_core" / "tools" / "nautical_queue_status.py",
    }

    def invoke(name: str, *args: str, allowed: tuple[int, ...] = (0,)) -> dict:
        proc = subprocess.run(
            [sys.executable, str(tools[name]), *args],
            text=True,
            capture_output=True,
            env=env,
            timeout=30.0,
        )
        if proc.returncode not in allowed:
            raise AssertionError(
                f"{name} failed ({proc.returncode}): stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )
        try:
            payload = json.loads((proc.stdout or "").strip() or "{}")
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{name} did not emit JSON: {proc.stdout!r}") from exc
        if not isinstance(payload, dict):
            raise AssertionError(f"{name} emitted a non-object payload: {payload!r}")
        return payload

    installation = invoke(
        "doctor", "--taskdata", str(data_dir), "--task-bin", task_bin, "--json", "--installation-only"
    )
    if installation.get("status") != "ok":
        raise AssertionError(f"installation Doctor was not healthy: {installation!r}")
    full_doctor = invoke("doctor", "--taskdata", str(data_dir), "--task-bin", task_bin, "--json", allowed=(0, 1))
    integrity = invoke("query", "integrity", "--all", allowed=(0, 3))
    if integrity.get("status") not in {"ok", "found", "empty", "manual_review"}:
        raise AssertionError(f"integrity query was unavailable: {integrity!r}")
    chain_id = str(cp_result.get("chainID") or "")
    if not chain_id:
        raise AssertionError("cutover scenario lost the CP chain identity")
    dry_run = invoke("reconcile", "--json", "--chain-id", chain_id, allowed=(0, 1, 2))
    applied = invoke("reconcile", "--json", "--apply", "--chain-id", chain_id, allowed=(0, 1, 2))
    queue = invoke("queue", "--taskdata", str(data_dir), "--json", allowed=(0, 1))
    return {
        "installation": installation.get("status"),
        "doctor": full_doctor.get("status"),
        "integrity": integrity.get("status"),
        "reconcile_dry_run": dry_run.get("status"),
        "reconcile_apply": applied.get("status"),
        "queue": queue.get("status"),
    }


def _scenario_invalid_chain_isolation(env: dict[str, str], safe_result: dict) -> dict:
    """An invalid unrelated chain must not suppress a safe scoped query."""
    _task(
        [
            "rc.hooks=off",
            "add",
            "blackbox invalid chain",
            "anchor:w:mon",
            "chain:on",
        ],
        env,
    )
    safe_uuid = str(safe_result.get("child") or "")
    if not safe_uuid:
        raise AssertionError("safe scenario did not provide a child UUID")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "nautical_core" / "tools" / "nautical_query.py"),
            "occurrences",
            "--uuid",
            safe_uuid,
            "--from",
            datetime.now(timezone.utc).date().isoformat(),
            "--count",
            "1",
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise AssertionError(f"safe scoped query failed: stdout={proc.stdout!r} stderr={proc.stderr!r}")
    payload = json.loads((proc.stdout or "{}").strip() or "{}")
    if payload.get("status") not in {"found", "empty"}:
        raise AssertionError(f"invalid unrelated chain hid safe query: {payload!r}")
    return {"invalid_added": True, "safe_status": payload.get("status")}


def _scenario_recovery_paths(env: dict[str, str]) -> dict:
    """Exercise deletion, hookless completion recovery, and review reporting."""
    delete_description = "blackbox deletion"
    _task(["add", delete_description, "anchor:w:mon", "due:today"], env)
    deleted = _one(env, f"description:{delete_description}", "status:pending")
    deleted_uuid = str(deleted.get("uuid") or "")
    _task([f"uuid:{deleted_uuid}", "delete"], env)
    deleted_row = _one(env, f"uuid:{deleted_uuid}")
    if deleted_row.get("status") != "deleted" or deleted_row.get("chain") != "off":
        raise AssertionError(f"manual deletion did not complete the chain: {deleted_row!r}")

    recovery_description = "blackbox hookless completion"
    _task(["add", recovery_description, "cp:1d", "due:today"], env)
    recovery_root = _one(env, f"description:{recovery_description}", "status:pending")
    recovery_uuid = str(recovery_root.get("uuid") or "")
    recovery_chain = str(recovery_root.get("chainID") or "")
    _task(["rc.hooks=off", f"uuid:{recovery_uuid}", "done"], env)
    before = _export(env, f"chainID:{recovery_chain}", "link:2", "status.not:deleted")
    if before:
        raise AssertionError("hookless completion unexpectedly spawned a child before reconcile")
    env["NAUTICAL_TASK_SHIM_FAIL_MATCH"] = "import"
    failed_reconcile = subprocess.run(
        [
            sys.executable,
            str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"),
            "--json",
            "--apply",
            "--chain-id",
            recovery_chain,
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    env.pop("NAUTICAL_TASK_SHIM_FAIL_MATCH", None)
    if failed_reconcile.returncode == 0:
        raise AssertionError("injected child-import failure was not reported")
    still_missing = _export(env, f"chainID:{recovery_chain}", "link:2", "status.not:deleted")
    if still_missing:
        raise AssertionError("partial child import left a visible successor before retry")
    reconcile = subprocess.run(
        [
            sys.executable,
            str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"),
            "--json",
            "--apply",
            "--chain-id",
            recovery_chain,
        ],
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if reconcile.returncode not in (0, 1, 2):
        raise AssertionError(f"hookless completion reconcile retry failed: {reconcile.stderr!r}")
    after = _export(env, f"chainID:{recovery_chain}", "link:2", "status.not:deleted")
    if len(after) != 1:
        raise AssertionError(f"reconcile did not recover the hookless completion: {after!r}")
    return {
        "deleted_chain_off": True,
        "partial_failure_exit": failed_reconcile.returncode,
        "recovered_child": after[0].get("uuid"),
        "reconcile_exit": reconcile.returncode,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a JSON result")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()

    if shutil.which("task") is None:
        print("task not found in PATH", file=sys.stderr)
        return 2

    temp_dir = Path(tempfile.mkdtemp(prefix="nautical-black-box-"))
    data_dir = temp_dir / "taskdata"
    anchor_dir = temp_dir / "anchor-files"
    omit_dir = temp_dir / "omit-files"
    data_dir.mkdir()
    anchor_dir.mkdir()
    omit_dir.mkdir()
    _install_runtime(data_dir)
    taskrc = temp_dir / "taskrc"
    config = temp_dir / "config-nautical.toml"
    _write_taskrc(taskrc, data_dir)
    _write_config(config, anchor_dir, omit_dir)

    env = os.environ.copy()
    env.update(
        {
            "TASKRC": str(taskrc),
            "TASKDATA": str(data_dir),
            "NAUTICAL_CONFIG": str(config),
            "NAUTICAL_TRUST_CONFIG_PATH": "1",
            "TZ": "UTC",
        }
    )
    env.pop("NAUTICAL_DIAG", None)

    result: dict[str, object] = {"ok": False, "taskdata": str(data_dir), "scenarios": {}}
    try:
        scenarios = result["scenarios"]
        assert isinstance(scenarios, dict)
        scenarios["cp"] = _scenario_cp(env)
        scenarios["navigator"] = _scenario_navigator(env, scenarios["cp"])
        scenarios["preset"] = _scenario_anchor_preset(env)
        scenarios["files"] = _scenario_files(env, anchor_dir, omit_dir)
        scenarios["modify"] = _scenario_modify(env)
        scenarios["duplicate_guard"] = _scenario_duplicate_guard(env, scenarios["cp"])
        scenarios["no_nested_hooks"] = _scenario_no_nested_hooks(env, data_dir)
        scenarios["invalid_chain_isolation"] = _scenario_invalid_chain_isolation(env, scenarios["preset"])
        scenarios["recovery_paths"] = _scenario_recovery_paths(env)
        scenarios["operator_cutover"] = _scenario_operator_cutover(env, data_dir, scenarios["cp"])
        _assert_clean_state(data_dir)
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        if args.json:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            print(f"status={'ok' if result['ok'] else 'fail'} taskdata={data_dir}")
            for name, detail in result["scenarios"].items():
                print(f"- {name}: ok {detail}")
            if result.get("error"):
                print(f"- error: {result['error']}")
        if not args.keep:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
