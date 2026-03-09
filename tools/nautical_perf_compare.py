#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two nautical perf budget reports and detect meaningful regressions.

Usage:
  python3 tools/nautical_perf_compare.py --base base.json --head head.json
  python3 tools/nautical_perf_compare.py --base base.json --head head.json --enforce
  python3 tools/nautical_perf_compare.py --base base.json --head head.json --json
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline perf report JSON path")
    ap.add_argument("--head", required=True, help="current perf report JSON path")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of markdown table")
    ap.add_argument("--enforce", action="store_true", help="exit non-zero on regressions")
    ap.add_argument("--abs-floor-s", type=float, default=0.003, help="absolute regression floor in seconds")
    ap.add_argument("--pct-floor", type=float, default=0.15, help="relative regression floor ratio (0.15 = 15%)")
    args = ap.parse_args()

    base = _load(args.base)
    head = _load(args.head)
    bres = base.get("results") if isinstance(base.get("results"), dict) else {}
    hres = head.get("results") if isinstance(head.get("results"), dict) else {}
    names = sorted(set(bres.keys()) | set(hres.keys()))

    rows = []
    regressions = []
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

    summary = {
        "base_report": str(Path(args.base).resolve()),
        "head_report": str(Path(args.head).resolve()),
        "abs_floor_s": float(args.abs_floor_s),
        "pct_floor": float(args.pct_floor),
        "rows": rows,
        "regressions": regressions,
        "ok": len(regressions) == 0,
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
        if regressions:
            print(f"Regressions: {', '.join(regressions)}")
        else:
            print("Regressions: none")

    if args.enforce and regressions:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
