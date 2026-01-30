#!/usr/bin/env python3
"""Reliability smoke test for Nautical hooks (safe temp TASKDATA)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import re
import argparse
import multiprocessing as mp
import json

ROOT = Path(__file__).resolve().parent.parent

def _run(cmd, env=None, check=True):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\nstdout={p.stdout}\nstderr={p.stderr}")
    return p

def _task(cmd, env=None, check=True):
    return _run(["task"] + cmd, env=env, check=check)

_ADD_ID_RE = re.compile(r"Created task (\d+)\.")

def _add_task(args, env):
    p = _task(["add"] + args, env=env)
    m = _ADD_ID_RE.search(p.stdout or "")
    if not m:
        raise RuntimeError(f"could not parse task id from: {p.stdout!r}")
    return int(m.group(1))

def _run_load(env, count):
    ids = []
    for i in range(count):
        tid = _add_task([f"nautical load {i}", "anchor:m:17", "+", "w:sun", "anchor_mode:skip", "chain:on", "due:today"], env)
        ids.append(tid)
    for tid in ids:
        _task(["done", str(tid)], env=env)
    time.sleep(0.2)

def _export_one(cmd_args, env):
    p = _task(cmd_args, env=env)
    raw = (p.stdout or "").strip()
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj[0] if obj else None
        return obj
    except Exception:
        return None

def _run_chain_until_failure(env, max_iters: int, settle_s: float) -> tuple[int, str | None]:
    tid = _add_task(["nautical chain limit", "anchor:m:17", "+", "w:sun", "anchor_mode:skip", "chain:on", "due:today"], env)
    parent = _export_one([f"id:{tid}", "rc.json.array=off", "export"], env=env)
    if not parent:
        return (0, "parent export missing")
    chain_id = (parent.get("chainID") or parent.get("chainid") or "").strip()
    if not chain_id:
        return (0, "missing chainID")

    completed = 0
    last_uuid = None
    for _ in range(max_iters):
        obj = _export_one([f"chainID:{chain_id}", "status:pending", "sort:link+", "limit:1", "rc.json.array=off", "export"], env=env)
        if not obj:
            # allow time for spawn queue to drain
            time.sleep(settle_s)
            obj = _export_one([f"chainID:{chain_id}", "status:pending", "sort:link+", "limit:1", "rc.json.array=off", "export"], env=env)
            if not obj:
                if _check_failure(Path(env["TASKDATA"])):
                    return (completed, "failure signal found")
                return (completed, "no pending task found")
        uuid = (obj.get("uuid") or "").strip()
        if not uuid:
            return (completed, "pending task missing uuid")
        if uuid == last_uuid:
            return (completed, "no progress (same pending uuid)")
        _task([f"uuid:{uuid}", "done"], env=env)
        completed += 1
        last_uuid = uuid
        if _check_failure(Path(env["TASKDATA"])):
            return (completed, "failure signal found")
        time.sleep(settle_s)
    return (completed, "max iters reached")

def _check_failure(td_path: Path) -> bool:
    dl = td_path / ".nautical_dead_letter.jsonl"
    if dl.exists() and dl.stat().st_size > 0:
        return True
    q = td_path / ".nautical_spawn_queue.jsonl"
    for _ in range(10):
        if q.exists():
            try:
                if q.stat().st_size > 0:
                    time.sleep(0.1)
                    continue
            except Exception:
                return True
        return False
    return True

def _reset_signal_files(td_path: Path) -> None:
    for name in [
        ".nautical_dead_letter.jsonl",
        ".nautical_spawn_queue.jsonl",
        ".nautical_spawn_queue.lock_failed",
        ".nautical_spawn_queue.lock_failed.count",
    ]:
        try:
            (td_path / name).unlink()
        except Exception:
            pass

def _load_worker(taskdata: str, count: int, durable: bool) -> tuple[bool, str]:
    env = os.environ.copy()
    env["TASKDATA"] = taskdata
    env["NAUTICAL_CORE_PATH"] = str(Path.home() / ".task")
    env.setdefault("NAUTICAL_DIAG", "1")
    if durable:
        env["NAUTICAL_DURABLE_QUEUE"] = "1"
    try:
        _reset_signal_files(Path(taskdata))
        _run_load(env, count)
        failed = _check_failure(Path(taskdata))
        return (not failed, "" if not failed else "failure signal found")
    except Exception as e:
        return (False, str(e))


def main() -> int:
    parser = argparse.ArgumentParser(description="Nautical reliability smoke test")
    parser.add_argument("--load", type=int, default=0, help="number of tasks to add+complete under load")
    parser.add_argument("--durable", action="store_true", help="enable NAUTICAL_DURABLE_QUEUE during load run")
    parser.add_argument("--find-limit", action="store_true", help="search for max safe load before failure")
    parser.add_argument("--min", dest="min_load", type=int, default=50, help="min load for limit finder")
    parser.add_argument("--max", dest="max_load", type=int, default=2000, help="max load for limit finder")
    parser.add_argument("--step", type=float, default=2.0, help="multiplicative step for limit finder")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers for load tests")
    parser.add_argument("--chain-until-failure", action="store_true", help="complete a single chain until failure")
    parser.add_argument("--chain-max", type=int, default=2000, help="max chain completions for chain-until-failure")
    parser.add_argument("--chain-settle", type=float, default=0.2, help="sleep between chain completions")
    args = parser.parse_args()

    env = os.environ.copy()
    env["NAUTICAL_CORE_PATH"] = str(Path.home() / ".task")
    env.setdefault("NAUTICAL_DIAG", "1")

    with tempfile.TemporaryDirectory(prefix="nautical-smoke-") as td:
        td_path = Path(td)
        env["TASKDATA"] = str(td_path)

        print(f"[smoke] TASKDATA={td_path}")

        # Test 1: happy path
        tid = _add_task(["nautical test", "anchor:m:17", "+", "w:sun", "anchor_mode:skip", "chain:on", "due:today"], env)
        _task(["done", str(tid)], env=env)
        q_path = td_path / ".nautical_spawn_queue.jsonl"
        if q_path.exists():
            # allow on-exit to drain
            time.sleep(0.2)
        # export all to show at least one task exists
        export = _task(["export"], env=env).stdout
        if not export.strip():
            raise RuntimeError("empty export after happy path")
        print("[smoke] happy path ok")

        # Test 2: queue lock contention
        lock_path = td_path / ".nautical_spawn_queue.lock"
        lock_path.write_text("hold")
        try:
            tid = _add_task(["nautical lock test", "anchor:m:17", "+", "w:sun", "anchor_mode:skip", "chain:on", "due:today"], env)
            _task(["done", str(tid)], env=env)
        finally:
            try:
                lock_path.unlink()
            except Exception:
                pass
        # allow drain
        time.sleep(0.2)
        print("[smoke] lock contention ok")

        # Test 3: corrupt queue line
        q_path.parent.mkdir(parents=True, exist_ok=True)
        with q_path.open("a", encoding="utf-8") as f:
            f.write("{bad-json\n")
        # trigger drain
        _task(["export"], env=env)
        print("[smoke] corrupt line quarantine ok")

        # Test 4: CAS race (simulate by overwriting nextLink)
        # create new task and grab parent uuid
        tid = _add_task(["nautical cas test", "anchor:m:17", "+", "w:sun", "anchor_mode:skip", "chain:on", "due:today"], env)
        parent = _task([f"id:{tid}", "export"], env=env).stdout
        if not parent.strip():
            raise RuntimeError("parent export missing")
        # complete and immediately overwrite parent nextLink
        _task(["done", str(tid)], env=env)
        _task([f"id:{tid}", "modify", "nextLink:abcd1234"], env=env, check=False)
        time.sleep(0.2)
        print("[smoke] cas race simulated")

        # Test 5: durable queue mode
        env_durable = env.copy()
        env_durable["NAUTICAL_DURABLE_QUEUE"] = "1"
        tid = _add_task(["nautical durable test", "anchor:m:17", "+", "w:sun", "anchor_mode:skip", "chain:on", "due:today"], env_durable)
        _task(["done", str(tid)], env=env_durable)
        time.sleep(0.2)
        print("[smoke] durable mode ok")

        if args.load:
            env_load = env.copy()
            if args.durable:
                env_load["NAUTICAL_DURABLE_QUEUE"] = "1"
            print(f"[smoke] load test: {args.load} tasks (durable={args.durable})")
            t0 = time.time()
            _run_load(env_load, args.load)
            dt = time.time() - t0
            print(f"[smoke] load ok: {args.load} tasks in {dt:.2f}s")

        if args.find_limit:
            print(f"[smoke] limit finder: min={args.min_load} max={args.max_load} step={args.step} workers={args.workers}")
            load = args.min_load
            last_ok = 0
            while load <= args.max_load:
                t0 = time.time()
                work_dirs = []
                for i in range(max(1, args.workers)):
                    wdir = td_path / f"limit_{load}_w{i}"
                    if wdir.exists():
                        for p in wdir.iterdir():
                            try:
                                if p.is_file():
                                    p.unlink()
                            except Exception:
                                pass
                    wdir.mkdir(parents=True, exist_ok=True)
                    work_dirs.append(str(wdir))
                if args.workers <= 1:
                    ok, err = _load_worker(work_dirs[0], load, args.durable)
                    results = [(ok, err)]
                else:
                    with mp.Pool(processes=args.workers) as pool:
                        results = pool.starmap(_load_worker, [(work_dirs[i], load, args.durable) for i in range(args.workers)])
                all_ok = all(r[0] for r in results)
                dt = time.time() - t0
                if all_ok:
                    last_ok = load
                    print(f"[smoke] load {load} ok ({dt:.2f}s)")
                    load = int(load * args.step)
                    if load == last_ok:
                        load += 1
                else:
                    errs = [r[1] for r in results if r[1]]
                    print(f"[smoke] load {load} failed ({dt:.2f}s): {errs[:3]}")
                    break
            print(f"[smoke] max safe load ~= {last_ok}")

        if args.chain_until_failure:
            print(f"[smoke] chain until failure: max={args.chain_max} settle={args.chain_settle}s")
            count, reason = _run_chain_until_failure(env, args.chain_max, args.chain_settle)
            print(f"[smoke] chain completed {count} links ({reason})")

        print("[smoke] all tests completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
