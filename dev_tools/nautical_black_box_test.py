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
        ("on-add-nautical.py", "on-add"),
        ("on-modify-nautical.py", "on-modify"),
        ("on-exit-nautical.py", "on-exit"),
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
    child = _one(env, f"chainID:{chain_id}", "status:pending", "link:2")
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
    return {"root": root["uuid"], "child": child["uuid"]}


def _scenario_anchor_preset(env: dict[str, str]) -> dict:
    description = "blackbox preset"
    _task(
        [
            "add",
            description,
            "anchor:@blackbox_weekdays",
            "omit:@blackbox_monday",
            "due:today",
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


def _assert_clean_state(data_dir: Path) -> None:
    for dead_letter in (
        data_dir / ".nautical-state" / ".nautical_dead_letter.jsonl",
        data_dir / ".nautical_dead_letter.jsonl",
    ):
        if dead_letter.exists() and dead_letter.stat().st_size:
            raise AssertionError(f"dead-letter queue is not empty: {dead_letter}")

    queue_db = data_dir / ".nautical-state" / ".nautical_queue.db"
    if not queue_db.exists():
        raise AssertionError("durable spawn queue database was not created")
    with sqlite3.connect(str(queue_db)) as conn:
        active = conn.execute(
            "SELECT COUNT(*) FROM queue_entries WHERE state IN ('queued', 'processing')"
        ).fetchone()
    if active is None or int(active[0]) != 0:
        raise AssertionError(f"spawn queue did not drain: {active}")


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
        scenarios["preset"] = _scenario_anchor_preset(env)
        scenarios["files"] = _scenario_files(env, anchor_dir, omit_dir)
        scenarios["duplicate_guard"] = _scenario_duplicate_guard(env, scenarios["cp"])
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
