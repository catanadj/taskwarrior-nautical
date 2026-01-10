#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load test for Nautical hooks using a disposable Taskwarrior data directory.

Usage:
  python3 tools/load_test_nautical.py --tasks 500 --concurrency 4
  python3 tools/load_test_nautical.py --tasks 2000 --concurrency 8 --keep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def _which_task() -> str | None:
    return shutil.which("task")


def _write_taskrc(path: Path, data_dir: Path, hooks_dir: Path) -> None:
    lines = [
        f"data.location={data_dir}",
        "hooks=on",
        "confirmation=off",
        "verbose=nothing",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_hooks(hooks_dir: Path) -> None:
    hooks_dir.mkdir(parents=True, exist_ok=True)
    src_on_add = REPO_ROOT / "on-add-nautical.py"
    src_on_modify = REPO_ROOT / "on-modify-nautical.py"
    src_core = REPO_ROOT / "nautical_core.py"
    dst_on_add = hooks_dir / "on-add"
    dst_on_modify = hooks_dir / "on-modify"
    shutil.copy2(src_on_add, dst_on_add)
    shutil.copy2(src_on_modify, dst_on_modify)
    shutil.copy2(src_core, hooks_dir / "nautical_core.py")
    for p in (dst_on_add, dst_on_modify):
        try:
            p.chmod(0o755)
        except Exception:
            pass


def _run_task(cmd: list[str], env: dict, timeout: float = 15.0) -> tuple[bool, str, str, float]:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd,
            text=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        dt = time.perf_counter() - t0
        return (r.returncode == 0, r.stdout or "", r.stderr or "", dt)
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return (False, "", "timeout", dt)
    except Exception as e:
        dt = time.perf_counter() - t0
        return (False, "", str(e), dt)


def _parse_created_id(stdout: str) -> int | None:
    m = re.search(r"Created task (\d+)", stdout or "")
    return int(m.group(1)) if m else None


def _export_pending_uuids(env: dict) -> set[str]:
    ok, out, _err, _dt = _run_task(
        ["task", "rc.hooks=off", "rc.json.array=1", "status:pending", "export"],
        env,
        timeout=30.0,
    )
    if not ok or not out:
        return set()
    try:
        arr = json.loads(out)
    except Exception:
        return set()
    uuids = set()
    for obj in arr or []:
        u = obj.get("uuid") if isinstance(obj, dict) else None
        if u:
            uuids.add(str(u))
    return uuids


def _percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = int(round((pct / 100.0) * (len(vals) - 1)))
    k = max(0, min(k, len(vals) - 1))
    return vals[k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=200, help="number of tasks to add")
    ap.add_argument("--done-rate", type=float, default=0.5, help="fraction of tasks to mark done")
    ap.add_argument("--anchor-rate", type=float, default=0.3, help="fraction of tasks using anchor")
    ap.add_argument("--cp-rate", type=float, default=0.3, help="fraction of tasks using cp")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel worker count")
    ap.add_argument("--ramp", action="store_true", help="ramp up task volume until a limit is hit")
    ap.add_argument("--done-only", action="store_true", help="measure only done latency (adds are warmup)")
    ap.add_argument("--debug", action="store_true", help="print per-stage debug counts")
    ap.add_argument("--rate", action="store_true", help="run a fixed-duration rate test")
    ap.add_argument("--rate-ramp", action="store_true", help="ramp target ops/sec until a limit is hit")
    ap.add_argument("--rate-secs", type=int, default=60, help="duration for rate test")
    ap.add_argument("--rate-target", type=float, default=50.0, help="target ops/sec for add or done")
    ap.add_argument("--rate-start", type=float, default=10.0, help="starting ops/sec for rate ramp")
    ap.add_argument("--rate-step", type=float, default=10.0, help="ops/sec increment per rate ramp stage")
    ap.add_argument("--rate-max", type=float, default=200.0, help="max ops/sec for rate ramp")
    ap.add_argument("--ramp-start", type=int, default=200, help="starting tasks per stage")
    ap.add_argument("--ramp-step", type=int, default=200, help="increment per stage")
    ap.add_argument("--ramp-max", type=int, default=3000, help="max tasks per stage")
    ap.add_argument("--limit-p95", type=float, default=1.0, help="stop when p95 add latency exceeds (s)")
    ap.add_argument("--limit-fail-rate", type=float, default=0.01, help="stop when add fail rate exceeds")
    ap.add_argument("--limit-queue-bytes", type=int, default=1, help="stop when queue bytes exceeds")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--keep", action="store_true", help="keep temp data dir")
    args = ap.parse_args()

    if _which_task() is None:
        print("task not found in PATH. Install Taskwarrior to run this test.")
        return 2

    random.seed(args.seed)
    tasks_total = max(1, int(args.tasks))
    done_total = int(round(tasks_total * max(0.0, min(1.0, args.done_rate))))

    anchor_rate = max(0.0, min(1.0, args.anchor_rate))
    cp_rate = max(0.0, min(1.0, args.cp_rate))

    temp_dir = Path(tempfile.mkdtemp(prefix="nautical-loadtest-"))
    data_dir = temp_dir / "taskdata"
    hooks_dir = data_dir / "hooks"
    taskrc_path = temp_dir / "taskrc"

    data_dir.mkdir(parents=True, exist_ok=True)
    _install_hooks(hooks_dir)
    _write_taskrc(taskrc_path, data_dir, hooks_dir)

    env = os.environ.copy()
    env["TASKRC"] = str(taskrc_path)
    env["TASKDATA"] = str(data_dir)

    def _make_add_cmd(i: int) -> list[str]:
        desc = f"loadtest task {i}"
        cmd = ["task", "add", desc]
        r = random.random()
        if r < anchor_rate:
            cmd += ["anchor:w:mon..fri@t=09:00", "anchor_mode:skip", "chain:on"]
        elif r < anchor_rate + cp_rate:
            cmd += ["cp:1d", "chain:on"]
        return cmd

    def _summary(name: str, vals: list[float], fails: int):
        if not vals:
            print(f"{name}: no samples")
            return
        p50 = _percentile(vals, 50)
        p95 = _percentile(vals, 95)
        avg = sum(vals) / len(vals)
        print(f"{name}: n={len(vals)} fail={fails} avg={avg:.3f}s p50={p50:.3f}s p95={p95:.3f}s")

    def _run_batch(batch_tasks: int, batch_done_rate: float, measure_add: bool) -> dict:
        add_lat = []
        done_lat = []
        add_fail = 0
        done_fail = 0
        task_ids = []
        before_uuids = set()

        if args.done_only:
            before_uuids = _export_pending_uuids(env)

        if args.concurrency <= 1:
            for i in range(batch_tasks):
                ok, out, err, dt = _run_task(_make_add_cmd(i), env)
                if measure_add:
                    add_lat.append(dt)
                if not ok:
                    if measure_add:
                        add_fail += 1
                else:
                    tid = _parse_created_id(out)
                    if tid is not None:
                        task_ids.append(tid)
        else:
            from concurrent.futures import ThreadPoolExecutor

            def _add_one(i: int):
                return _run_task(_make_add_cmd(i), env)

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for ok, out, err, dt in ex.map(_add_one, range(batch_tasks)):
                    if measure_add:
                        add_lat.append(dt)
                    if not ok:
                        if measure_add:
                            add_fail += 1
                    else:
                        tid = _parse_created_id(out)
                        if tid is not None:
                            task_ids.append(tid)

        if args.done_only or not task_ids:
            after_uuids = _export_pending_uuids(env)
            new_uuids = list(after_uuids - before_uuids)
            task_ids = new_uuids

        random.shuffle(task_ids)
        to_done = task_ids[:int(round(batch_tasks * batch_done_rate))]

        def _done_cmd(tid) -> list[str]:
            return ["task", str(tid), "done"]

        if args.concurrency <= 1:
            for tid in to_done:
                ok, out, err, dt = _run_task(_done_cmd(tid), env)
                done_lat.append(dt)
                if not ok:
                    done_fail += 1
        else:
            from concurrent.futures import ThreadPoolExecutor

            def _done_one(tid: int):
                return _run_task(_done_cmd(tid), env)

            with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
                for ok, out, err, dt in ex.map(_done_one, to_done):
                    done_lat.append(dt)
                    if not ok:
                        done_fail += 1

        queue_path = data_dir / ".nautical_spawn_queue.jsonl"
        queue_bytes = queue_path.stat().st_size if queue_path.exists() else 0

        return {
            "add_lat": add_lat,
            "done_lat": done_lat,
            "add_fail": add_fail,
            "done_fail": done_fail,
            "queue_bytes": queue_bytes,
            "done_targets": len(to_done),
            "tasks_created": len(task_ids),
        }

    def _run_rate(duration_s: int, target_ops: float) -> dict:
        add_lat = []
        done_lat = []
        add_fail = 0
        done_fail = 0
        task_ids = []

        start = time.perf_counter()
        next_tick = start
        idx = 0

        while True:
            now = time.perf_counter()
            if now - start >= duration_s:
                break
            # pacing
            next_tick += 1.0 / max(0.1, target_ops)
            sleep_for = max(0.0, next_tick - now)
            if sleep_for > 0:
                time.sleep(sleep_for)

            ok, out, err, dt = _run_task(_make_add_cmd(idx), env)
            idx += 1
            if not args.done_only:
                add_lat.append(dt)
            if not ok:
                if not args.done_only:
                    add_fail += 1
                continue
            tid = _parse_created_id(out)
            if tid is not None:
                task_ids.append(tid)

            if args.done_only and tid is not None:
                okd, outd, errd, dtd = _run_task(["task", str(tid), "done"], env)
                done_lat.append(dtd)
                if not okd:
                    done_fail += 1

        queue_path = data_dir / ".nautical_spawn_queue.jsonl"
        queue_bytes = queue_path.stat().st_size if queue_path.exists() else 0

        return {
            "add_lat": add_lat,
            "done_lat": done_lat,
            "add_fail": add_fail,
            "done_fail": done_fail,
            "queue_bytes": queue_bytes,
        }

    def _thresholds_summary() -> str:
        return (
            f"thresholds: p95>{args.limit_p95:.3f}s "
            f"fail_rate>{args.limit_fail_rate:.3f} "
            f"queue_bytes>{args.limit_queue_bytes}"
        )

    if args.rate_ramp:
        print("\nRate Ramp Results")
        print("-----------------")
        stages = []
        cur = max(0.1, float(args.rate_start))
        step = max(0.1, float(args.rate_step))
        limit = max(cur, float(args.rate_max))
        while cur <= limit:
            res = _run_rate(args.rate_secs, cur)
            add_lat = res["add_lat"]
            add_fail = res["add_fail"]
            done_lat = res["done_lat"]
            done_fail = res["done_fail"]
            fail_rate = (add_fail / len(add_lat)) if add_lat else 0.0
            done_fail_rate = (done_fail / len(done_lat)) if done_lat else 0.0
            p95 = _percentile(add_lat, 95) if add_lat else 0.0
            done_p95 = _percentile(done_lat, 95) if done_lat else 0.0
            stages.append({
                "rate": cur,
                "p95": p95,
                "fail_rate": fail_rate,
                "queue_bytes": res["queue_bytes"],
                "done_p95": done_p95,
                "done_fail_rate": done_fail_rate,
            })
            if args.done_only:
                print(
                    f"rate={cur:.1f}/s done_p95={done_p95:.3f}s "
                    f"done_fail_rate={done_fail_rate:.3f} queue_bytes={res['queue_bytes']}"
                )
            else:
                print(
                    f"rate={cur:.1f}/s p95={p95:.3f}s "
                    f"fail_rate={fail_rate:.3f} queue_bytes={res['queue_bytes']}"
                )
            if args.done_only:
                if done_p95 > args.limit_p95 or done_fail_rate > args.limit_fail_rate or res["queue_bytes"] > args.limit_queue_bytes:
                    break
            elif p95 > args.limit_p95 or fail_rate > args.limit_fail_rate or res["queue_bytes"] > args.limit_queue_bytes:
                break
            cur += step

        print("\nRate Ramp Summary")
        print("-----------------")
        if stages:
            last = stages[-1]
            if args.done_only:
                print(
                    f"last_rate={last['rate']:.1f}/s done_p95={last['done_p95']:.3f}s "
                    f"done_fail_rate={last['done_fail_rate']:.3f} queue_bytes={last['queue_bytes']}"
                )
            else:
                print(
                    f"last_rate={last['rate']:.1f}/s p95={last['p95']:.3f}s "
                    f"fail_rate={last['fail_rate']:.3f} queue_bytes={last['queue_bytes']}"
                )
        print(_thresholds_summary())
        print(f"data_dir={data_dir}")
    elif args.rate:
        res = _run_rate(args.rate_secs, args.rate_target)
        print("\nRate Results")
        print("------------")
        if not args.done_only:
            _summary("add", res["add_lat"], res["add_fail"])
        _summary("done", res["done_lat"], res["done_fail"])
        print(f"queue_bytes={res['queue_bytes']}")
        print(_thresholds_summary())
        print(f"data_dir={data_dir}")
    elif not args.ramp:
        res = _run_batch(tasks_total, args.done_rate, measure_add=not args.done_only)
        print("\nResults")
        print("-------")
        if not args.done_only:
            _summary("add", res["add_lat"], res["add_fail"])
        _summary("done", res["done_lat"], res["done_fail"])
        print(f"queue_bytes={res['queue_bytes']}")
        print(_thresholds_summary())
        print(f"data_dir={data_dir}")
    else:
        print("\nRamp Results")
        print("------------")
        stages = []
        cur = max(1, int(args.ramp_start))
        step = max(1, int(args.ramp_step))
        limit = max(cur, int(args.ramp_max))
        while cur <= limit:
            res = _run_batch(cur, args.done_rate, measure_add=not args.done_only)
            add_lat = res["add_lat"]
            add_fail = res["add_fail"]
            done_lat = res["done_lat"]
            done_fail = res["done_fail"]
            fail_rate = (add_fail / len(add_lat)) if add_lat else 0.0
            done_fail_rate = (done_fail / len(done_lat)) if done_lat else 0.0
            p95 = _percentile(add_lat, 95) if add_lat else 0.0
            done_p95 = _percentile(done_lat, 95) if done_lat else 0.0
            stages.append({
                "tasks": cur,
                "p95": p95,
                "fail_rate": fail_rate,
                "queue_bytes": res["queue_bytes"],
                "done_p95": done_p95,
                "done_fail_rate": done_fail_rate,
            })
            if args.done_only:
                print(
                    f"tasks={cur} done_p95={done_p95:.3f}s done_fail_rate={done_fail_rate:.3f} "
                    f"queue_bytes={res['queue_bytes']}"
                )
            else:
                print(f"tasks={cur} p95={p95:.3f}s fail_rate={fail_rate:.3f} queue_bytes={res['queue_bytes']}")
            if args.debug:
                print(
                    f"debug: created={res['tasks_created']} done_targets={res['done_targets']} "
                    f"done_samples={len(done_lat)}"
                )
            if args.done_only:
                if done_p95 > args.limit_p95 or done_fail_rate > args.limit_fail_rate or res["queue_bytes"] > args.limit_queue_bytes:
                    break
            elif p95 > args.limit_p95 or fail_rate > args.limit_fail_rate or res["queue_bytes"] > args.limit_queue_bytes:
                break
            cur += step

        print("\nRamp Summary")
        print("------------")
        if stages:
            last = stages[-1]
            if args.done_only:
                print(
                    f"last_stage tasks={last['tasks']} done_p95={last['done_p95']:.3f}s "
                    f"done_fail_rate={last['done_fail_rate']:.3f} queue_bytes={last['queue_bytes']}"
                )
            else:
                print(
                    f"last_stage tasks={last['tasks']} p95={last['p95']:.3f}s "
                    f"fail_rate={last['fail_rate']:.3f} queue_bytes={last['queue_bytes']}"
                )
        print(_thresholds_summary())
        print(f"data_dir={data_dir}")

    if not args.keep:
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        print(f"kept temp dir: {temp_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
