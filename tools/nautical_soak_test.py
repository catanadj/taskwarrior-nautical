#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long-running soak test for Nautical hooks using a disposable Taskwarrior data dir.

The goal is not peak throughput. It is stability over time:
- repeated add/done churn
- periodic queue/dead-letter health checks
- threshold enforcement for latency, failures, and stale queue state

Usage:
  python3 tools/nautical_soak_test.py --minutes 10
  python3 tools/nautical_soak_test.py --seconds 120 --json --enforce
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _which_task() -> str | None:
    return shutil.which("task")


def _write_taskrc(path: Path, data_dir: Path) -> None:
    hooks_dir = data_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"data.location={data_dir}",
        "hooks=on",
        "confirmation=off",
        "verbose=nothing",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_hooks(data_dir: Path) -> None:
    hooks_dir = data_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "on-add-nautical.py", hooks_dir / "on-add")
    shutil.copy2(REPO_ROOT / "on-modify-nautical.py", hooks_dir / "on-modify")
    shutil.copy2(REPO_ROOT / "on-exit-nautical.py", hooks_dir / "on-exit")
    shutil.copytree(REPO_ROOT / "nautical_core", data_dir / "nautical_core", dirs_exist_ok=True)
    for path in (hooks_dir / "on-add", hooks_dir / "on-modify", hooks_dir / "on-exit"):
        try:
            path.chmod(0o755)
        except Exception:
            pass


def _run(cmd: list[str], *, env: dict, timeout: float = 30.0) -> tuple[bool, str, str, float]:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        dt = time.perf_counter() - t0
        return proc.returncode == 0, proc.stdout or "", proc.stderr or "", dt
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return False, "", "timeout", dt
    except Exception as exc:
        dt = time.perf_counter() - t0
        return False, "", str(exc), dt


def _parse_created_id(stdout: str) -> int | None:
    match = re.search(r"Created task (\d+)", stdout or "")
    return int(match.group(1)) if match else None


def _export_pending_uuids(env: dict) -> set[str]:
    ok, out, err, dt = _run(
        ["task", "rc.hooks=off", "rc.json.array=1", "status:pending", "export"],
        env=env,
        timeout=30.0,
    )
    if not ok or not out:
        return set()
    try:
        payload = json.loads(out)
    except Exception:
        return set()
    uuids = set()
    for obj in payload or []:
        if isinstance(obj, dict):
            uuid_str = str(obj.get("uuid") or "").strip()
            if uuid_str:
                uuids.add(uuid_str)
    return uuids


def _export_pending_tasks(env: dict) -> list[dict]:
    ok, out, err, dt = _run(
        ["task", "rc.hooks=off", "rc.json.array=1", "status:pending", "export"],
        env=env,
        timeout=30.0,
    )
    if not ok or not out:
        return []
    try:
        payload = json.loads(out)
    except Exception:
        return []
    tasks: list[dict] = []
    for obj in payload or []:
        if isinstance(obj, dict):
            tasks.append(obj)
    return tasks


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = int(round((pct / 100.0) * (len(vals) - 1)))
    idx = max(0, min(len(vals) - 1, idx))
    return vals[idx]


def _health_snapshot(env: dict, data_dir: Path) -> dict:
    ok, out, err, dt = _run(
        [sys.executable, str(REPO_ROOT / "tools" / "nautical_health_check.py"), "--taskdata", str(data_dir), "--json"],
        env=env,
        timeout=15.0,
    )
    if not ok:
        return {"ok": False, "status": "crit", "error": err or out or "health check failed", "duration_s": dt}
    try:
        payload = json.loads(out or "{}")
    except Exception as exc:
        return {"ok": False, "status": "crit", "error": f"health check JSON parse failed: {exc}", "duration_s": dt}
    payload["ok"] = True
    payload["duration_s"] = dt
    return payload


def _stats(vals: list[float], failures: int) -> dict:
    if not vals:
        return {
            "samples": 0,
            "failures": int(failures),
            "fail_rate": 0.0,
            "avg_s": 0.0,
            "p50_s": 0.0,
            "p95_s": 0.0,
        }
    return {
        "samples": len(vals),
        "failures": int(failures),
        "fail_rate": float(failures) / float(len(vals)),
        "avg_s": sum(vals) / len(vals),
        "p50_s": _percentile(vals, 50),
        "p95_s": _percentile(vals, 95),
    }


def _summarize_health_checks(health_checks: list[dict], health_rank: dict[str, int]) -> dict:
    worst_health = "ok"
    max_queue_bytes = 0
    max_dead_letters = 0
    last_snapshot: dict = {}
    for snap in health_checks:
        if isinstance(snap, dict):
            last_snapshot = snap
        status = str(snap.get("status") or "crit")
        if health_rank.get(status, 2) > health_rank[worst_health]:
            worst_health = status
        metrics = snap.get("metrics") if isinstance(snap.get("metrics"), dict) else {}
        max_queue_bytes = max(max_queue_bytes, int(metrics.get("queue_bytes") or 0))
        max_dead_letters = max(max_dead_letters, int(metrics.get("dead_letter_lines") or 0))
    last_metrics = last_snapshot.get("metrics") if isinstance(last_snapshot.get("metrics"), dict) else {}
    return {
        "checks": len(health_checks),
        "worst_status": worst_health,
        "last_status": str(last_snapshot.get("status") or worst_health),
        "max_queue_bytes": max_queue_bytes,
        "max_dead_letter_lines": max_dead_letters,
        "last_queue_bytes": int(last_metrics.get("queue_bytes") or 0),
        "last_dead_letter_lines": int(last_metrics.get("dead_letter_lines") or 0),
    }


def _emit_progress(*, started: float, duration_s: float, cycles: int, add_lat: list[float], add_fail: int, done_lat: list[float], done_fail: int, last_health: dict | None, progress_format: str) -> None:
    add_summary = _stats(add_lat, add_fail)
    done_summary = _stats(done_lat, done_fail)
    elapsed = max(0.0, time.perf_counter() - started)
    remaining = max(0.0, duration_s - elapsed)
    health_status = "n/a"
    queue_bytes = 0
    dead_letters = 0
    if isinstance(last_health, dict):
        health_status = str(last_health.get("status") or "n/a")
        metrics = last_health.get("metrics") if isinstance(last_health.get("metrics"), dict) else {}
        queue_bytes = int(metrics.get("queue_bytes") or 0)
        dead_letters = int(metrics.get("dead_letter_lines") or 0)
    payload = {
        "elapsed_s": round(elapsed, 3),
        "remaining_s": round(remaining, 3),
        "cycles": cycles,
        "add_n": add_summary["samples"],
        "add_p95_s": round(add_summary["p95_s"], 6),
        "add_fail_rate": round(add_summary["fail_rate"], 6),
        "done_n": done_summary["samples"],
        "done_p95_s": round(done_summary["p95_s"], 6),
        "done_fail_rate": round(done_summary["fail_rate"], 6),
        "health": health_status,
        "queue_bytes": queue_bytes,
        "dead_letters": dead_letters,
    }
    try:
        if progress_format == "jsonl":
            sys.stderr.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            sys.stderr.write(
                "[soak] "
                f"elapsed={elapsed:.1f}s remaining={remaining:.1f}s cycles={cycles} "
                f"add_n={add_summary['samples']} add_p95={add_summary['p95_s']:.3f}s add_fail_rate={add_summary['fail_rate']:.3f} "
                f"done_n={done_summary['samples']} done_p95={done_summary['p95_s']:.3f}s done_fail_rate={done_summary['fail_rate']:.3f} "
                f"health={health_status} queue_bytes={queue_bytes} dead_letters={dead_letters}\n"
            )
        sys.stderr.flush()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=0.0, help="soak duration in minutes")
    ap.add_argument("--seconds", type=float, default=0.0, help="soak duration in seconds")
    ap.add_argument("--batch-size", type=int, default=20, help="adds per cycle")
    ap.add_argument("--done-rate", type=float, default=0.6, help="fraction of created tasks to mark done")
    ap.add_argument("--anchor-rate", type=float, default=0.35, help="fraction of adds using anchor")
    ap.add_argument("--cp-rate", type=float, default=0.35, help="fraction of adds using cp")
    ap.add_argument("--health-every-cycles", type=int, default=2, help="run health check every N cycles")
    ap.add_argument("--limit-add-p95", type=float, default=1.0, help="fail when add p95 exceeds this")
    ap.add_argument("--limit-done-p95", type=float, default=1.0, help="fail when done p95 exceeds this")
    ap.add_argument("--limit-fail-rate", type=float, default=0.05, help="fail when add/done fail rate exceeds this")
    ap.add_argument("--limit-queue-bytes", type=int, default=262144, help="fail when health queue bytes exceeds this")
    ap.add_argument("--limit-dead-letters", type=int, default=0, help="fail when dead-letter lines exceed this")
    ap.add_argument("--max-health-status", choices=["ok", "warn", "crit"], default="warn", help="highest allowed health status")
    ap.add_argument("--progress-every-seconds", type=float, default=5.0, help="emit periodic live stats to stderr; 0 disables")
    ap.add_argument("--progress-format", choices=["line", "jsonl"], default="line", help="stderr format for live progress output")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--enforce", action="store_true", help="exit non-zero on threshold violations")
    ap.add_argument("--keep", action="store_true", help="keep temp data dir")
    args = ap.parse_args()

    duration_s = float(args.seconds or 0.0)
    if duration_s <= 0.0:
        duration_s = max(30.0, float(args.minutes or 0.0) * 60.0)
    rng = random.Random(args.seed)

    if _which_task() is None:
        payload = {"ok": False, "error": "task not found in PATH"}
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), indent=2) if args.json else payload["error"])
        return 2

    temp_dir = Path(tempfile.mkdtemp(prefix="nautical-soak-"))
    data_dir = temp_dir / "taskdata"
    taskrc = temp_dir / "taskrc"
    data_dir.mkdir(parents=True, exist_ok=True)
    _install_hooks(data_dir)
    _write_taskrc(taskrc, data_dir)

    env = os.environ.copy()
    env["TASKRC"] = str(taskrc)
    env["TASKDATA"] = str(data_dir)

    add_lat: list[float] = []
    done_lat: list[float] = []
    add_fail = 0
    done_fail = 0
    cycles = 0
    health_checks: list[dict] = []
    violations: list[str] = []
    health_rank = {"ok": 0, "warn": 1, "crit": 2}
    max_allowed_health = health_rank[str(args.max_health_status)]
    next_progress_at = max(0.0, float(args.progress_every_seconds or 0.0))
    last_health: dict | None = None

    def _make_add_cmd(i: int) -> list[str]:
        desc = f"soak task {cycles}-{i}"
        cmd = ["task", "add", desc]
        roll = rng.random()
        if roll < max(0.0, min(1.0, float(args.anchor_rate))):
            return cmd + ["anchor:w:mon..fri@t=09:00", "anchor_mode:skip", "chain:on"]
        if roll < max(0.0, min(1.0, float(args.anchor_rate) + float(args.cp_rate))):
            return cmd + ["cp:1d", "chain:on"]
        return cmd

    def _mutate_pending_task(task: dict) -> None:
        uuid = str(task.get("uuid") or "").strip()
        if not uuid:
            return
        roll = rng.random()
        if roll < 0.25:
            _run(["task", uuid, "modify", "chain:off"], env=env, timeout=20.0)
            return
        if roll < 0.5:
            _run(["task", uuid, "modify", "-anchor", "-anchor_file", "-cp", "-omit", "-omit_file"], env=env, timeout=20.0)
            return
        if roll < 0.75:
            _run(["task", uuid, "modify", "anchor:w:mon", "chain:on"], env=env, timeout=20.0)
            return
        _run(["task", uuid, "modify", "anchor_file:2026.csv@-1d@t=12:00,18:00"], env=env, timeout=20.0)

    started = time.perf_counter()
    while time.perf_counter() - started < duration_s:
        cycles += 1
        created_refs: list[str] = []
        before_uuids = _export_pending_uuids(env)
        for i in range(max(1, int(args.batch_size))):
            ok, out, err, dt = _run(_make_add_cmd(i), env=env, timeout=20.0)
            add_lat.append(dt)
            if not ok:
                add_fail += 1
                continue
            task_id = _parse_created_id(out)
            if task_id is not None:
                created_refs.append(str(task_id))

        after_uuids = _export_pending_uuids(env)
        new_uuids = sorted(after_uuids - before_uuids)
        if new_uuids:
            created_refs = new_uuids

        if created_refs:
            pending_tasks = _export_pending_tasks(env)
            pending_by_uuid = {str(t.get("uuid") or "").strip(): t for t in pending_tasks if str(t.get("uuid") or "").strip()}
            mutate_refs = created_refs[: max(1, len(created_refs) // 4)]
            for ref in mutate_refs:
                task = pending_by_uuid.get(ref)
                if task:
                    _mutate_pending_task(task)

        rng.shuffle(created_refs)
        done_count = int(round(len(created_refs) * max(0.0, min(1.0, float(args.done_rate)))))
        for task_ref in created_refs[:done_count]:
            ok, out, err, dt = _run(["task", str(task_ref), "done"], env=env, timeout=20.0)
            done_lat.append(dt)
            if not ok:
                done_fail += 1

        if cycles % max(1, int(args.health_every_cycles)) == 0:
            snap = _health_snapshot(env, data_dir)
            health_checks.append(snap)
            last_health = snap

        if next_progress_at > 0.0:
            elapsed = time.perf_counter() - started
            if elapsed >= next_progress_at:
                _emit_progress(
                    started=started,
                    duration_s=duration_s,
                    cycles=cycles,
                    add_lat=add_lat,
                    add_fail=add_fail,
                    done_lat=done_lat,
                    done_fail=done_fail,
                    last_health=last_health,
                    progress_format=str(args.progress_format),
                )
                next_progress_at += max(0.1, float(args.progress_every_seconds))

    if not health_checks or (cycles % max(1, int(args.health_every_cycles)) != 0):
        last_health = _health_snapshot(env, data_dir)
        health_checks.append(last_health)

    add_summary = _stats(add_lat, add_fail)
    done_summary = _stats(done_lat, done_fail)

    health_summary = _summarize_health_checks(health_checks, health_rank)
    worst_health = str(health_summary["worst_status"])
    max_queue_bytes = int(health_summary["max_queue_bytes"])
    max_dead_letters = int(health_summary["max_dead_letter_lines"])

    if add_summary["p95_s"] > float(args.limit_add_p95):
        violations.append(f"add.p95_s>{args.limit_add_p95}")
    if done_summary["samples"] > 0 and done_summary["p95_s"] > float(args.limit_done_p95):
        violations.append(f"done.p95_s>{args.limit_done_p95}")
    if add_summary["fail_rate"] > float(args.limit_fail_rate):
        violations.append(f"add.fail_rate>{args.limit_fail_rate}")
    if done_summary["fail_rate"] > float(args.limit_fail_rate):
        violations.append(f"done.fail_rate>{args.limit_fail_rate}")
    if max_queue_bytes > int(args.limit_queue_bytes):
        violations.append(f"queue_bytes>{args.limit_queue_bytes}")
    if max_dead_letters > int(args.limit_dead_letters):
        violations.append(f"dead_letter_lines>{args.limit_dead_letters}")
    if health_rank.get(worst_health, 2) > max_allowed_health:
        violations.append(f"health_status>{args.max_health_status}")

    summary = {
        "ok": len(violations) == 0,
        "duration_s": round(time.perf_counter() - started, 3),
        "cycles": cycles,
        "batch_size": int(args.batch_size),
        "taskdata": str(data_dir),
        "add": add_summary,
        "done": done_summary,
        "health": health_summary,
        "violations": violations,
        "kept_temp_dir": bool(args.keep),
    }

    if not args.keep:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), indent=2))
    else:
        print(f"ok={summary['ok']} duration_s={summary['duration_s']} cycles={cycles}")
        print(
            f"add_p95={add_summary['p95_s']:.3f}s done_p95={done_summary['p95_s']:.3f}s "
            f"add_fail_rate={add_summary['fail_rate']:.3f} done_fail_rate={done_summary['fail_rate']:.3f}"
        )
        print(
            f"health_worst={worst_health} queue_bytes={max_queue_bytes} "
            f"dead_letter_lines={max_dead_letters}"
        )
        if violations:
            print("violations=" + ",".join(violations))
        if args.keep:
            print(f"kept temp dir: {temp_dir}")

    return 1 if args.enforce and violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
