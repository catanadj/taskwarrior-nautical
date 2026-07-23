#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic performance budget checks for core anchor paths.

Usage:
  python3 dev_tools/nautical_perf_budget.py
  python3 dev_tools/nautical_perf_budget.py --enforce
  python3 dev_tools/nautical_perf_budget.py --json --enforce
  python3 dev_tools/nautical_perf_budget.py --budget-file dev_tools/perf_budget.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

core = importlib.import_module("nautical_core")
install_runtime = importlib.import_module("nautical_core.install_runtime")
queue_store = importlib.import_module("nautical_core.queue_store")


def _load_budget_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read budget file '{path}': {e}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Budget file '{path}' must contain a JSON object.")
    workload = data.get("workload")
    budgets = data.get("budgets_seconds")
    if not isinstance(workload, dict) or not isinstance(budgets, dict):
        raise RuntimeError("Budget file requires 'workload' and 'budgets_seconds' objects.")
    return data


def _clear_caches() -> None:
    try:
        core._clear_all_caches()
    except Exception:
        pass


def _bench_parse_validate(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.validate_anchor_expr_strict(expr)
    return time.perf_counter() - t0


def _bench_describe_expr(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.describe_anchor_expr(expr)
    return time.perf_counter() - t0


def _bench_next_after(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    dnfs = [core.validate_anchor_expr_strict(expr) for expr in exprs]
    ref = date(2026, 1, 1)
    t0 = time.perf_counter()
    for _ in range(rounds):
        for dnf in dnfs:
            core.next_after_expr(dnf, ref)
    return time.perf_counter() - t0


def _bench_build_hints(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.build_and_cache_hints(expr, "skip")
    return time.perf_counter() - t0


def _bench_cache_key_hot(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.cache_key_for_task(expr, "skip")
    return time.perf_counter() - t0


@contextmanager
def _perf_cache_context():
    """Isolate cache I/O benchmarks from user cache directories."""
    saved_enable = bool(getattr(core, "ENABLE_ANCHOR_CACHE", False))
    saved_override = str(getattr(core, "ANCHOR_CACHE_DIR_OVERRIDE", "") or "")
    saved_cache_dir = getattr(core, "_CACHE_DIR", None)
    saved_ttl = int(getattr(core, "ANCHOR_CACHE_TTL", 0) or 0)
    with tempfile.TemporaryDirectory(prefix="nautical-perf-cache-") as td:
        try:
            core.ENABLE_ANCHOR_CACHE = True
            core.ANCHOR_CACHE_DIR_OVERRIDE = td
            core.ANCHOR_CACHE_TTL = 0
            core._CACHE_DIR = None
            _clear_caches()
            yield td
        finally:
            core.ENABLE_ANCHOR_CACHE = saved_enable
            core.ANCHOR_CACHE_DIR_OVERRIDE = saved_override
            core.ANCHOR_CACHE_TTL = saved_ttl
            core._CACHE_DIR = saved_cache_dir
            _clear_caches()


def _cache_payload(expr: str, idx: int) -> dict:
    return {
        "natural": expr,
        "next_dates": ["2026-01-01", "2026-01-08", "2026-01-15"],
        "meta": {"i": idx},
        # Keep payload shape aligned with cache schema checks.
        "dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]],
    }


def _bench_cache_save(exprs: list[str], rounds: int) -> float:
    with _perf_cache_context():
        keys = [f"perf-save-{i}" for i in range(max(1, len(exprs)))]
        t0 = time.perf_counter()
        idx = 0
        for _ in range(rounds):
            for i, expr in enumerate(exprs):
                payload = _cache_payload(expr, idx)
                if not core.cache_save(keys[i], payload):
                    raise RuntimeError("cache_save benchmark write failed")
                idx += 1
        return time.perf_counter() - t0


def _bench_cache_load_hot(exprs: list[str], rounds: int) -> float:
    with _perf_cache_context():
        keys = [f"perf-load-{i}" for i in range(max(1, len(exprs)))]
        for i, expr in enumerate(exprs):
            if not core.cache_save(keys[i], _cache_payload(expr, i)):
                raise RuntimeError("cache_load benchmark setup write failed")
        _clear_caches()
        t0 = time.perf_counter()
        for _ in range(rounds):
            for key in keys:
                obj = core.cache_load(key)
                if not isinstance(obj, dict):
                    raise RuntimeError("cache_load benchmark read failed")
        return time.perf_counter() - t0


def _bench_queue_schema_hot(rounds: int) -> float:
    with tempfile.TemporaryDirectory(prefix="nautical-perf-queue-") as td:
        with sqlite3.connect(str(Path(td) / "queue.db")) as conn:
            queue_store.init_queue_db(conn)
            t0 = time.perf_counter()
            for _ in range(rounds):
                queue_store.init_queue_db(conn)
            return time.perf_counter() - t0


def _measure(name: str, fn, repeats: int) -> dict:
    samples = []
    # Warmup once for interpreter/cache stabilization.
    _ = fn()
    for _ in range(max(1, repeats)):
        samples.append(float(fn()))
    samples = sorted(samples)
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": statistics.median(samples),
        "max_s": samples[-1],
    }


def _strict_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(text)
    except Exception as exc:
        raise RuntimeError(f"hook stdout is not valid JSON: {exc}") from exc
    if text[end:].strip() or not isinstance(obj, dict):
        raise RuntimeError("hook stdout must contain exactly one JSON object")
    return obj


def _run_hook_timed(
    hook_path: Path,
    *,
    input_text: str,
    env: dict[str, str],
    expected_task: dict | None,
) -> float:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(hook_path)],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=15.0,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{hook_path.name} failed with exit {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    if expected_task is None:
        if (proc.stdout or "").strip():
            raise RuntimeError(f"{hook_path.name} wrote unexpected stdout")
    else:
        actual = _strict_json_object(proc.stdout or "")
        if actual != expected_task:
            raise RuntimeError(f"{hook_path.name} changed the plain passthrough task")
    return elapsed


def _init_empty_queue_db(taskdata: Path) -> None:
    state_dir = taskdata / ".nautical-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(state_dir / ".nautical_queue.db")) as conn:
        conn.execute(
            """
            CREATE TABLE queue_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spawn_intent_id TEXT,
                payload TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                state TEXT NOT NULL DEFAULT 'queued',
                claim_token TEXT,
                claimed_at REAL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.commit()


def _measure_hook_fast_path(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    max_ratio: float,
) -> dict:
    fast_env = dict(base_env)
    fast_env.pop("NAUTICAL_BENCH_FORCE_FULL", None)
    full_env = dict(base_env)
    full_env["NAUTICAL_BENCH_FORCE_FULL"] = "1"

    _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
    _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)

    fast_samples: list[float] = []
    full_samples: list[float] = []
    for index in range(max(1, int(repeats))):
        if index % 2 == 0:
            fast_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
            )
            full_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)
            )
        else:
            full_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)
            )
            fast_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
            )

    fast_samples.sort()
    full_samples.sort()
    fast_median = float(statistics.median(fast_samples))
    full_median = float(statistics.median(full_samples))
    ratio = fast_median / full_median if full_median > 0.0 else 1.0
    return {
        "name": name,
        "samples_s": fast_samples,
        "min_s": fast_samples[0],
        "median_s": fast_median,
        "max_s": fast_samples[-1],
        "full_samples_s": full_samples,
        "full_median_s": full_median,
        "fast_to_full_ratio": ratio,
        "max_ratio": float(max_ratio),
        "budget_s": 0.0,
        "pass": ratio <= float(max_ratio),
    }


def _measure_managed_hook_latency(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    baseline_median_s: float,
    max_ratio: float,
) -> dict:
    env = dict(base_env)
    env.pop("NAUTICAL_CORE_PATH", None)
    env.pop("NAUTICAL_TRUST_CORE_PATH", None)
    env.pop("NAUTICAL_BENCH_FORCE_FULL", None)

    _run_hook_timed(hook_path, input_text=input_text, env=env, expected_task=expected_task)
    samples = sorted(
        _run_hook_timed(hook_path, input_text=input_text, env=env, expected_task=expected_task)
        for _ in range(max(1, int(repeats)))
    )
    median_s = float(statistics.median(samples))
    ratio = median_s / baseline_median_s if baseline_median_s > 0.0 else 1.0
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": median_s,
        "max_s": samples[-1],
        "baseline_median_s": float(baseline_median_s),
        "managed_to_source_ratio": ratio,
        "max_ratio": float(max_ratio),
        "budget_s": 0.0,
        "pass": ratio <= float(max_ratio),
    }


def _bench_hook_fast_paths(cfg: dict) -> dict[str, dict]:
    hook_cfg = cfg.get("hook_fast_path")
    if not isinstance(hook_cfg, dict) or not hook_cfg.get("enabled", True):
        return {}
    repeats = max(1, int(hook_cfg.get("repeats", 7)))
    max_ratios = hook_cfg.get("max_ratio") if isinstance(hook_cfg.get("max_ratio"), dict) else {}
    managed_max_ratio = float(hook_cfg.get("managed_layout_max_ratio", 1.5))

    plain = {
        "uuid": "11111111-1111-1111-1111-111111111111",
        "description": "plain hook latency",
        "status": "pending",
        "entry": "20260101T000000Z",
        "modified": "20260101T000000Z",
    }
    modified = dict(plain, modified="20260101T000001Z")

    with tempfile.TemporaryDirectory(prefix="nautical-hook-perf-") as td:
        temp_root = Path(td)
        config_path = temp_root / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\npanel_mode = "minimal"\n', encoding="utf-8")
        base_env = os.environ.copy()
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TZ": "UTC",
            }
        )
        for key in ("NAUTICAL_DIAG", "NAUTICAL_DIAG_LOG", "NAUTICAL_PROFILE"):
            base_env.pop(key, None)

        cases = []
        add_data = temp_root / "add-data"
        add_data.mkdir()
        cases.append(
            (
                "hook_plain_add",
                ROOT / "on-add-nautical.py",
                json.dumps(plain, ensure_ascii=False),
                plain,
                add_data,
            )
        )
        modify_data = temp_root / "modify-data"
        modify_data.mkdir()
        cases.append(
            (
                "hook_plain_modify",
                ROOT / "on-modify-nautical.py",
                json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False),
                modified,
                modify_data,
            )
        )
        nautical_old = dict(
            plain,
            cp="P1D",
            chain="on",
            chainID="abcd1234",
            link=3,
            due="20270101T090000Z",
        )
        nautical_modified = dict(nautical_old, description="ordinary Nautical edit", modified="20260101T000001Z")
        cases.append(
            (
                "hook_nautical_ordinary_modify",
                ROOT / "on-modify-nautical.py",
                json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(nautical_modified, ensure_ascii=False),
                nautical_modified,
                modify_data,
            )
        )
        exit_data = temp_root / "exit-data"
        exit_data.mkdir()
        _init_empty_queue_db(exit_data)
        cases.append(("hook_empty_exit", ROOT / "on-exit-nautical.py", "", None, exit_data))

        results = {}
        for name, hook_path, input_text, expected_task, taskdata in cases:
            env = dict(base_env)
            env["TASKDATA"] = str(taskdata)
            ratio_budget = float(max_ratios.get(name, 0.8))
            results[name] = _measure_hook_fast_path(
                name,
                hook_path,
                input_text=input_text,
                expected_task=expected_task,
                base_env=env,
                repeats=repeats,
                max_ratio=ratio_budget,
            )

        managed_data = temp_root / "managed-data"
        install_runtime.install_release(
            source=ROOT,
            taskdata=managed_data,
            release_id="perf-managed",
            smoke=False,
        )
        _init_empty_queue_db(managed_data)
        managed_env = dict(base_env)
        managed_env["TASKDATA"] = str(managed_data)
        for name, source_hook, input_text, expected_task, _taskdata in cases:
            managed_name = f"managed_{name}"
            results[managed_name] = _measure_managed_hook_latency(
                managed_name,
                managed_data / "hooks" / source_hook.name,
                input_text=input_text,
                expected_task=expected_task,
                base_env=managed_env,
                repeats=repeats,
                baseline_median_s=float(results[name]["median_s"]),
                max_ratio=managed_max_ratio,
            )
        return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-file", default=str(HERE / "perf_budget.json"))
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--enforce", action="store_true", help="fail non-zero if any budget is exceeded")
    args = ap.parse_args()

    cfg = _load_budget_config(Path(args.budget_file))
    workload = cfg["workload"]
    budgets = cfg["budgets_seconds"]
    exprs = [str(x) for x in workload.get("expressions", []) if str(x).strip()]
    if not exprs:
        raise RuntimeError("No expressions defined in workload.expressions")

    repeats = int(workload.get("repeats", 5))
    parse_rounds = int(workload.get("parse_validate_rounds", 220))
    describe_rounds = int(workload.get("describe_expr_rounds", 220))
    next_after_rounds = int(workload.get("next_after_rounds", 220))
    hints_rounds = int(workload.get("build_hints_rounds", 180))
    cache_key_rounds = int(workload.get("cache_key_rounds", 2500))
    cache_save_rounds = int(workload.get("cache_save_rounds", 120))
    cache_load_rounds = int(workload.get("cache_load_rounds", 300))
    queue_schema_hot_rounds = int(workload.get("queue_schema_hot_rounds", 1000))

    checks = [
        ("parse_validate", lambda: _bench_parse_validate(exprs, parse_rounds), repeats),
        ("describe_expr", lambda: _bench_describe_expr(exprs, describe_rounds), repeats),
        ("next_after", lambda: _bench_next_after(exprs, next_after_rounds), repeats),
        ("build_hints", lambda: _bench_build_hints(exprs, hints_rounds), repeats),
        ("cache_key_hot", lambda: _bench_cache_key_hot(exprs, cache_key_rounds), repeats),
        ("cache_save", lambda: _bench_cache_save(exprs, cache_save_rounds), repeats),
        ("cache_load_hot", lambda: _bench_cache_load_hot(exprs, cache_load_rounds), repeats),
        ("queue_schema_hot", lambda: _bench_queue_schema_hot(queue_schema_hot_rounds), repeats),
    ]

    seasonal = cfg.get("seasonal_workload")
    if isinstance(seasonal, dict):
        seasonal_exprs = [
            str(value)
            for value in seasonal.get("expressions", [])
            if str(value).strip()
        ]
        if seasonal_exprs:
            seasonal_repeats = max(1, int(seasonal.get("repeats", 3)))
            seasonal_parse_rounds = max(1, int(seasonal.get("parse_validate_rounds", 100)))
            seasonal_next_rounds = max(1, int(seasonal.get("next_after_rounds", 100)))
            seasonal_hint_rounds = max(1, int(seasonal.get("build_hints_rounds", 1)))
            checks.extend(
                [
                    (
                        "seasonal_parse_validate",
                        lambda: _bench_parse_validate(
                            seasonal_exprs,
                            seasonal_parse_rounds,
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_next_after",
                        lambda: _bench_next_after(
                            seasonal_exprs,
                            seasonal_next_rounds,
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_build_hints",
                        lambda: _bench_build_hints(
                            seasonal_exprs,
                            seasonal_hint_rounds,
                        ),
                        seasonal_repeats,
                    ),
                ]
            )

    results = {}
    failures = []
    for name, fn, check_repeats in checks:
        r = _measure(name, fn, check_repeats)
        budget = float(budgets.get(name, 0.0))
        r["budget_s"] = budget
        r["pass"] = (budget <= 0.0) or (r["median_s"] <= budget)
        results[name] = r
        if args.enforce and not r["pass"]:
            failures.append(name)

    hook_results = _bench_hook_fast_paths(cfg)
    for name, result in hook_results.items():
        results[name] = result
        if args.enforce and not result["pass"]:
            failures.append(name)

    summary = {
        "budget_file": str(Path(args.budget_file).resolve()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "results": results,
        "enforced": bool(args.enforce),
        "ok": len(failures) == 0,
        "failed_checks": failures,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), indent=2))
    else:
        print("Nautical Performance Budget")
        print(f"Budget file: {summary['budget_file']}")
        for name, r in results.items():
            status = "OK" if r["pass"] else "FAIL"
            if "fast_to_full_ratio" in r:
                print(
                    f"- {name}: fast={r['median_s']:.4f}s full={r['full_median_s']:.4f}s "
                    f"ratio={r['fast_to_full_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            elif "managed_to_source_ratio" in r:
                print(
                    f"- {name}: managed={r['median_s']:.4f}s source={r['baseline_median_s']:.4f}s "
                    f"ratio={r['managed_to_source_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            else:
                print(
                    f"- {name}: median={r['median_s']:.4f}s "
                    f"(min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) "
                    f"budget={r['budget_s']:.4f}s => {status}"
                )
        if args.enforce:
            print("Enforced:", "PASS" if summary["ok"] else f"FAIL ({', '.join(failures)})")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
