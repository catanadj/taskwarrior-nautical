#!/usr/bin/env python3
"""Run the repository's disposable recurrence stress profile.

The CI workflow intentionally calls this small orchestrator instead of
duplicating fixture setup.  The mixed recurrence loop owns the Taskwarrior
environment and health assertions; this wrapper adds a stable report contract
for CI summaries and artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOOP = ROOT / "dev_tools" / "nautical_mixed_recurrence_loop.py"
PROFILE_BUDGETS = {
    "ci": (8, 300.0),
    "nightly": (24, 300.0),
    "stress": (64, 900.0),
}


def _run_stage(profile: str) -> dict:
    cycles, timeout = PROFILE_BUDGETS[profile]
    command = [sys.executable, str(LOOP), "--cycles", str(cycles), "--json", "--enforce"]
    started = time.perf_counter()
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {
            "stage": "mixed_recurrence",
            "status": "failed",
            "duration_s": round(time.perf_counter() - started, 3),
            "error": f"stage timed out after {timeout:g} seconds",
        }

    raw = (proc.stdout or "").strip()
    try:
        report = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        return {
            "stage": "mixed_recurrence",
            "status": "failed",
            "duration_s": round(time.perf_counter() - started, 3),
            "error": f"stage returned invalid JSON: {exc}",
            "stderr": (proc.stderr or "").strip(),
        }

    ok = proc.returncode == 0 and bool(report.get("ok"))
    return {
        "stage": "mixed_recurrence",
        "status": "passed" if ok else "failed",
        "duration_s": round(time.perf_counter() - started, 3),
        "cycles": report.get("cycles_completed", 0),
        "violations": report.get("violations") or [],
        "error": "" if ok else ((proc.stderr or "").strip() or "mixed recurrence stage failed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Nautical CI stress campaign")
    parser.add_argument("--profile", choices=tuple(PROFILE_BUDGETS), default="ci")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--enforce", action="store_true", help="exit non-zero when a stage fails")
    args = parser.parse_args()

    started = time.perf_counter()
    stage = _run_stage(args.profile)
    failed = [stage["stage"]] if stage["status"] != "passed" else []
    payload = {
        "ok": not failed,
        "profile": args.profile,
        "duration_s": round(time.perf_counter() - started, 3),
        "task_available": shutil.which("task") is not None,
        "failed_stages": failed,
        "stages": [stage],
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"ok={payload['ok']} profile={args.profile} duration_s={payload['duration_s']}")
        for item in payload["stages"]:
            print(f"- {item['stage']}: {item['status']} ({item['duration_s']:.3f}s)")
    return 1 if args.enforce and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
