#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two nautical perf budget reports and detect meaningful regressions.

Usage:
  python3 dev_tools/nautical_perf_compare.py --base base.json --head head.json
  python3 dev_tools/nautical_perf_compare.py --base base.json --head head.json --enforce
  python3 dev_tools/nautical_perf_compare.py --base base.json --head head.json --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def _load(path: str) -> dict:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read '{p}': {e}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid report format in '{p}': expected JSON object")
    return data


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _pct(delta: float, base: float) -> float:
    if base <= 0.0:
        return 0.0
    return (delta / base) * 100.0


def _trend_class(delta: float, pct: float, abs_floor: float, pct_floor: float) -> str:
    if delta <= 0.0:
        return "improved_or_equal"
    if delta < abs_floor:
        return "noise"
    if pct < pct_floor * 100.0:
        return "noise"
    return "regression"


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0.0 else None


def _mapping_list(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _metric_value(result: dict, metric: str) -> float | None:
    """Extract one invocation-level metric without treating absent data as zero.

    Queue samples repeat the same counters per timing sample, so their median is
    representative. Reconcile scale samples contain independent candidate
    shapes, so their exported rows and command counts are summed.
    """
    if metric == "wall_time":
        return _number(result.get("median_s"))

    reports = _mapping_list(result.get("reconcile_reports"))
    if reports:
        keys = {
            "calls": "task_command_calls",
            "rows": "export_rows",
            "transactions": "transaction_count",
        }
        if metric in keys:
            key = keys[metric]
            values = [_number(item.get(key)) for item in reports]
            present = [item for item in values if item is not None]
            if present:
                return sum(present)

    samples = _mapping_list(result.get("task_call_stats"))
    if samples:
        keys = {
            "calls": "run_task_calls",
            "rows": "task_read_rows",
        }
        key = keys.get(metric)
        if key is not None:
            values = [_number(item.get(key)) for item in samples]
            present = [item for item in values if item is not None]
            if present:
                return float(statistics.median(present))

    outbox_samples = _mapping_list(result.get("outbox_stats"))
    if metric == "transactions" and outbox_samples:
        values = [_number(item.get("outbox_transactions")) for item in outbox_samples]
        present = [item for item in values if item is not None]
        if present:
            return float(statistics.median(present))

    if metric in {"cpu_time", "peak_memory"}:
        key = "cpu_median_s" if metric == "cpu_time" else "peak_memory_median_bytes"
        value = _number(result.get(key))
        if value is not None:
            return value

    timing = _mapping_list(result.get("timing_breakdown"))
    timing_keys = {
        "taskwarrior_time": "taskwarrior_seconds",
        "startup_time": "startup_seconds",
        "drain_time": "drain_seconds",
        "presentation_time": "presentation_seconds",
    }
    if metric in timing_keys and timing:
        values = [_number(item.get(timing_keys[metric])) for item in timing]
        present = [item for item in values if item is not None]
        if present:
            return float(statistics.median(present))

    direct_keys = {
        "calls": ("task_command_calls", "run_task_calls"),
        "rows": ("export_rows", "task_read_rows"),
        "transactions": ("transaction_count", "outbox_transactions"),
    }
    for key in direct_keys.get(metric, ()):
        value = _number(result.get(key))
        if value is not None:
            return value
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline perf report JSON path")
    ap.add_argument("--head", required=True, help="current perf report JSON path")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown table")
    ap.add_argument("--enforce", action="store_true", help="exit non-zero on regressions")
    ap.add_argument("--abs-floor-s", type=float, default=0.003, help="absolute regression floor in seconds")
    ap.add_argument(
        "--count-abs-floor",
        type=float,
        default=0.0,
        help="absolute regression floor for call, row, and transaction counts",
    )
    ap.add_argument("--pct-floor", type=float, default=0.15, help="relative regression floor ratio (0.15 = 15%%)")
    args = ap.parse_args()

    base = _load(args.base)
    head = _load(args.head)
    bres = base.get("results") if isinstance(base.get("results"), dict) else {}
    hres = head.get("results") if isinstance(head.get("results"), dict) else {}
    names = sorted(set(bres.keys()) | set(hres.keys()))

    rows = []
    metric_rows = []
    regressions = []
    metric_regressions = []
    for name in names:
        b = bres.get(name) if isinstance(bres.get(name), dict) else {}
        h = hres.get(name) if isinstance(hres.get(name), dict) else {}
        bmed = _as_float(b.get("median_s"), 0.0)
        hmed = _as_float(h.get("median_s"), 0.0)
        budget = _as_float(h.get("budget_s") if "budget_s" in h else b.get("budget_s"), 0.0)
        delta = hmed - bmed
        pct = _pct(delta, bmed)
        trend = _trend_class(delta, pct, args.abs_floor_s, args.pct_floor)
        row = {
            "check": name,
            "base_median_s": bmed,
            "head_median_s": hmed,
            "delta_s": delta,
            "delta_pct": pct,
            "budget_s": budget,
            "trend": trend,
        }
        rows.append(row)
        if trend == "regression":
            regressions.append(name)

        for metric in (
            "calls", "rows", "transactions", "cpu_time", "peak_memory",
            "taskwarrior_time", "startup_time", "drain_time", "presentation_time",
        ):
            base_value = _metric_value(b, metric)
            head_value = _metric_value(h, metric)
            if base_value is None or head_value is None:
                metric_rows.append(
                    {
                        "check": name,
                        "metric": metric,
                        "base": base_value,
                        "head": head_value,
                        "trend": "unavailable",
                    }
                )
                continue
            metric_delta = head_value - base_value
            metric_pct = _pct(metric_delta, base_value)
            if base_value == 0.0 and head_value > 0.0:
                metric_trend = "regression"
            else:
                metric_trend = _trend_class(
                    metric_delta,
                    metric_pct,
                    max(0.0, float(args.count_abs_floor)),
                    args.pct_floor,
                )
            metric_rows.append(
                {
                    "check": name,
                    "metric": metric,
                    "base": base_value,
                    "head": head_value,
                    "delta": metric_delta,
                    "delta_pct": metric_pct,
                    "trend": metric_trend,
                }
            )
            if metric_trend == "regression":
                metric_regressions.append(f"{name}:{metric}")

    summary = {
        "base_report": str(Path(args.base).resolve()),
        "head_report": str(Path(args.head).resolve()),
        "abs_floor_s": float(args.abs_floor_s),
        "pct_floor": float(args.pct_floor),
        "rows": rows,
        "metric_rows": metric_rows,
        "regressions": regressions + metric_regressions,
        "wall_regressions": regressions,
        "metric_regressions": metric_regressions,
        "ok": not regressions and not metric_regressions,
        "enforced": bool(args.enforce),
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), indent=2))
    else:
        print("Nautical Perf Trend Compare")
        print("")
        print("| Check | Base median (s) | Head median (s) | Delta (s) | Delta (%) | Budget (s) | Trend |")
        print("|---|---:|---:|---:|---:|---:|---|")
        for r in rows:
            print(
                f"| {r['check']} | {r['base_median_s']:.6f} | {r['head_median_s']:.6f} | "
                f"{r['delta_s']:+.6f} | {r['delta_pct']:+.2f}% | {r['budget_s']:.6f} | {r['trend']} |"
            )
        print("")
        print("| Check | Metric | Base | Head | Delta | Delta (%) | Trend |")
        print("|---|---|---:|---:|---:|---:|---|")
        for r in metric_rows:
            if r["trend"] == "unavailable":
                print(f"| {r['check']} | {r['metric']} | unavailable | unavailable |  |  | unavailable |")
                continue
            print(
                f"| {r['check']} | {r['metric']} | {r['base']:.0f} | {r['head']:.0f} | "
                f"{r['delta']:+.0f} | {r['delta_pct']:+.2f}% | {r['trend']} |"
            )
        print("")
        if summary["regressions"]:
            print(f"Regressions: {', '.join(summary['regressions'])}")
        else:
            print("Regressions: none")

    if args.enforce and summary["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
