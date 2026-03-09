#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic performance budget checks for core anchor paths.

Usage:
  python3 tools/nautical_perf_budget.py
  python3 tools/nautical_perf_budget.py --enforce
  python3 tools/nautical_perf_budget.py --json --enforce
  python3 tools/nautical_perf_budget.py --budget-file tools/perf_budget.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

core = importlib.import_module("nautical_core")


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

    checks = [
        ("parse_validate", lambda: _bench_parse_validate(exprs, parse_rounds)),
        ("describe_expr", lambda: _bench_describe_expr(exprs, describe_rounds)),
        ("next_after", lambda: _bench_next_after(exprs, next_after_rounds)),
        ("build_hints", lambda: _bench_build_hints(exprs, hints_rounds)),
        ("cache_key_hot", lambda: _bench_cache_key_hot(exprs, cache_key_rounds)),
    ]

    results = {}
    failures = []
    for name, fn in checks:
        r = _measure(name, fn, repeats)
        budget = float(budgets.get(name, 0.0))
        r["budget_s"] = budget
        r["pass"] = (budget <= 0.0) or (r["median_s"] <= budget)
        results[name] = r
        if args.enforce and not r["pass"]:
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
        for name in ("parse_validate", "describe_expr", "next_after", "build_hints", "cache_key_hot"):
            r = results[name]
            status = "OK" if r["pass"] else "FAIL"
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
