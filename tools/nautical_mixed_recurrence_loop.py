#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mixed recurrence loop test for Nautical.

This runner creates a disposable Taskwarrior data dir, installs the local
hooks/core, seeds anchor/anchor_file/omit/omit_file cases, then loops through
completions to exercise the merged recurrence paths repeatedly.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
def _which_task() -> str | None:
    return shutil.which("task")


def _run(cmd: list[str], env: dict, timeout: float = 20.0) -> tuple[bool, str, str, float]:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd,
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        return p.returncode == 0, p.stdout or "", p.stderr or "", time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        return False, "", "timeout", time.perf_counter() - t0
    except Exception as exc:
        return False, "", str(exc), time.perf_counter() - t0


def _write_taskrc(path: Path, data_dir: Path) -> None:
    hooks_dir = data_dir / "hooks"
    lines = [
        f"data.location={data_dir}",
        "hooks=on",
        "confirmation=off",
        "verbose=nothing",
        f"include {ROOT / 'uda.conf'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_nautical_config(path: Path, anchor_dir: Path, omit_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'anchor_file_dir = "{anchor_dir}"',
                f'omit_file_dir = "{omit_dir}"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _install_hooks(data_dir: Path) -> None:
    hooks_dir = data_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "on-add-nautical.py", hooks_dir / "on-add")
    shutil.copy2(ROOT / "on-modify-nautical.py", hooks_dir / "on-modify")
    shutil.copy2(ROOT / "on-exit-nautical.py", hooks_dir / "on-exit")
    shutil.copytree(ROOT / "nautical_core", data_dir / "nautical_core", dirs_exist_ok=True)
    for p in (hooks_dir / "on-add", hooks_dir / "on-modify", hooks_dir / "on-exit"):
        try:
            p.chmod(0o755)
        except Exception:
            pass


def _task(cmd: list[str], env: dict, timeout: float = 20.0) -> tuple[bool, str, str, float]:
    return _run(["task"] + cmd, env=env, timeout=timeout)


def _export_pending(env: dict) -> list[dict]:
    ok, out, _err, _dt = _task(["rc.hooks=off", "rc.json.array=1", "status:pending", "export"], env=env, timeout=30.0)
    if not ok or not out.strip():
        return []
    try:
        obj = json.loads(out)
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    return [row for row in obj if isinstance(row, dict)]


def _find_pending_by_description(env: dict, description: str) -> dict | None:
    for row in _export_pending(env):
        if str(row.get("description") or "") == description:
            return row
    return None


def _health_snapshot(env: dict, taskdata: Path) -> dict:
    ok, out, err, dt = _run(
        [
            sys.executable,
            str(ROOT / "tools" / "nautical_health_check.py"),
            "--taskdata",
            str(taskdata),
            "--json",
            "--queue-db-warn-rows",
            "10",
            "--queue-db-crit-rows",
            "1000",
            "--processing-warn-lines",
            "10",
            "--processing-crit-lines",
            "1000",
            "--dead-letter-warn-lines",
            "1",
            "--dead-letter-crit-lines",
            "10",
        ],
        env=env,
        timeout=20.0,
    )
    if not ok:
        return {"ok": False, "status": "crit", "error": err or out or "health check failed", "duration_s": dt}
    try:
        obj = json.loads(out or "{}")
    except Exception as exc:
        return {"ok": False, "status": "crit", "error": f"health JSON parse failed: {exc}", "duration_s": dt}
    obj["ok"] = True
    obj["duration_s"] = dt
    return obj


@dataclass
class SeedTask:
    description: str
    args: list[str]


def _add_seed_tasks(env: dict, seeds: list[SeedTask]) -> list[dict]:
    created: list[dict] = []
    for seed in seeds:
        ok, out, err, _dt = _task(["add", seed.description] + seed.args, env=env, timeout=30.0)
        if not ok:
            raise RuntimeError(
                f"failed to add seed task {seed.description!r}: rc failed; stdout={out!r}; stderr={err!r}"
            )
        row = _find_pending_by_description(env, seed.description)
        if not row:
            raise RuntimeError(
                f"could not find added task for {seed.description!r}: stdout={out!r}; stderr={err!r}"
            )
        created.append(
            {
                "description": seed.description,
                "uuid": str(row.get("uuid") or "").strip(),
                "id": row.get("id"),
            }
        )
    return created


def main() -> int:
    ap = argparse.ArgumentParser(description="Mixed recurrence loop test for Nautical")
    ap.add_argument("--cycles", type=int, default=8, help="completion cycles to run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--enforce", action="store_true", help="exit non-zero on failures")
    ap.add_argument("--keep", action="store_true", help="keep temp data dir")
    args = ap.parse_args()

    if _which_task() is None:
        payload = {"ok": False, "error": "task not found in PATH"}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return 2

    cycles = max(1, int(args.cycles))
    temp_dir = Path(tempfile.mkdtemp(prefix="nautical-mixed-loop-"))
    data_dir = temp_dir / "taskdata"
    anchor_dir = temp_dir / "anchor-files"
    omit_dir = temp_dir / "omit-files"
    data_dir.mkdir(parents=True, exist_ok=True)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    omit_dir.mkdir(parents=True, exist_ok=True)
    _install_hooks(data_dir)
    taskrc = temp_dir / "taskrc"
    config = temp_dir / "config-nautical.toml"
    _write_taskrc(taskrc, data_dir)
    _write_nautical_config(config, anchor_dir, omit_dir)

    # Small, deterministic fixtures.
    (anchor_dir / "2026.csv").write_text(
        "date,description\n"
        "2026-04-17,Friday anchor\n"
        "2026-04-25,Saturday file\n"
        "2026-05-05,Anniversary anchor\n",
        encoding="utf-8",
    )
    (omit_dir / "holidays.csv").write_text(
        "2026-04-28\n2026-05-05\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["TASKRC"] = str(taskrc)
    env["TASKDATA"] = str(data_dir)
    env["NAUTICAL_CONFIG"] = str(config)
    env["NAUTICAL_TRUST_CONFIG_PATH"] = "1"
    env["TZ"] = "UTC"

    seeds = [
        SeedTask(
            "mixed expr anchor",
            ["anchor:w:mon,wed,fri@t=09:00", "omit:w:wed", "anchor_mode:skip", "chain:on", "due:today"],
        ),
        SeedTask(
            "mixed anchor file",
            ["anchor_file:2026.csv@t=12:00", "anchor_mode:skip", "chain:on", "due:today"],
        ),
        SeedTask(
            "mixed merged recurrence",
            [
                "anchor:w:tue,fri | y:05-05",
                "anchor_file:2026.csv@-1d@t=12:00,18:00",
                "omit:y:04-28..05-05",
                "omit_file:holidays.csv",
                "anchor_mode:skip",
                "chain:on",
                "due:today",
            ],
        ),
        SeedTask(
            "mixed cp control",
            ["cp:1d", "chain:on", "due:today"],
        ),
    ]

    created = _add_seed_tasks(env, seeds)
    health: list[dict] = []
    violations: list[str] = []
    loop_results: list[dict] = []
    started = time.perf_counter()

    for i in range(cycles):
        pending = _export_pending(env)
        target_desc = seeds[i % len(seeds)].description
        target = None
        for row in pending:
            if str(row.get("description") or "") == target_desc:
                target = row
                break
        if target is None and pending:
            target = pending[0]
        if target is None:
            violations.append(f"cycle {i + 1}: no pending task found")
            break

        uuid = str(target.get("uuid") or "").strip()
        desc = str(target.get("description") or "").strip()
        if not uuid:
            violations.append(f"cycle {i + 1}: pending task missing uuid")
            break

        ok, out, err, dt = _task([f"uuid:{uuid}", "done"], env=env, timeout=30.0)
        loop_results.append(
            {
                "cycle": i + 1,
                "description": desc,
                "uuid": uuid,
                "ok": ok,
                "duration_s": round(dt, 3),
                "stderr": err.strip(),
            }
        )
        if not ok:
            violations.append(f"cycle {i + 1}: task done failed: {err or out}")
            break

        snap = _health_snapshot(env, data_dir)
        health.append(snap)
        if not snap.get("ok", False) or str(snap.get("status") or "") == "crit":
            violations.append(f"cycle {i + 1}: health check bad: {snap}")
            break

    final_health = _health_snapshot(env, data_dir)
    health.append(final_health)
    if not final_health.get("ok", False) or str(final_health.get("status") or "") == "crit":
        violations.append(f"final health bad: {final_health}")

    summary = {
        "ok": len(violations) == 0,
        "cycles_requested": cycles,
        "cycles_completed": len(loop_results),
        "duration_s": round(time.perf_counter() - started, 3),
        "taskdata": str(data_dir),
        "created": created,
        "loops": loop_results,
        "health": health[-1] if health else {},
        "violations": violations,
        "kept_temp_dir": bool(args.keep),
    }

    if not args.keep:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"ok={summary['ok']} cycles={summary['cycles_completed']}/{summary['cycles_requested']} duration_s={summary['duration_s']}")
        for item in loop_results:
            print(f"- cycle {item['cycle']}: {item['description']} rc={'ok' if item['ok'] else 'fail'} {item['duration_s']:.3f}s")
        if violations:
            print("violations=" + ",".join(violations))
        if args.keep:
            print(f"kept temp dir: {temp_dir}")

    return 1 if args.enforce and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
