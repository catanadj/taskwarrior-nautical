#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial stress campaign for Taskwarrior-Nautical.

This runner groups the ugliest scenarios into one repeatable local command:

- hook protocol abuse without requiring Taskwarrior
- deterministic perf budget checks
- reliability smoke / chain churn / queue recovery
- ramp and rate-ramp load stages with threshold enforcement

Usage:
  python3 tools/nautical_stress_campaign.py
  python3 tools/nautical_stress_campaign.py --profile stress --json
  python3 tools/nautical_stress_campaign.py --skip-task-stages
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ON_ADD = ROOT / "on-add-nautical.py"
ON_MODIFY = ROOT / "on-modify-nautical.py"
ON_EXIT = ROOT / "on-exit-nautical.py"
TASK_BIN = shutil.which("task")
RNG = random.Random(1337)


PROFILES = {
    "ci": {
        "hook_fuzz_cases": 4,
        "hook_timeout_s": 5.0,
        "reliability_cmd": [
            sys.executable,
            "dev_tools/nautical_reliability_smoke.py",
            "--load",
            "2",
            "--chain-until-failure",
            "--chain-max",
            "4",
            "--chain-settle",
            "0.02",
        ],
        "load_ramp_cmd": [
            sys.executable,
            "dev_tools/load_test_nautical.py",
            "--tasks",
            "20",
            "--concurrency",
            "1",
            "--done-rate",
            "0.5",
            "--anchor-rate",
            "0.3",
            "--cp-rate",
            "0.3",
            "--limit-p95",
            "1.0",
            "--limit-fail-rate",
            "0.05",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
        "rate_ramp_cmd": [
            sys.executable,
            "dev_tools/load_test_nautical.py",
            "--rate-ramp",
            "--rate-secs",
            "2",
            "--rate-start",
            "5",
            "--rate-step",
            "5",
            "--rate-max",
            "5",
            "--limit-p95",
            "1.0",
            "--limit-fail-rate",
            "0.05",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
    },
    "smoke": {
        "hook_fuzz_cases": 8,
        "hook_timeout_s": 5.0,
        "reliability_cmd": [
            sys.executable,
            "dev_tools/nautical_reliability_smoke.py",
            "--load",
            "5",
            "--chain-until-failure",
            "--chain-max",
            "10",
            "--chain-settle",
            "0.03",
        ],
        "load_ramp_cmd": [
            sys.executable,
            "dev_tools/load_test_nautical.py",
            "--ramp",
            "--ramp-start",
            "50",
            "--ramp-step",
            "50",
            "--ramp-max",
            "50",
            "--concurrency",
            "2",
            "--done-rate",
            "0.7",
            "--anchor-rate",
            "0.35",
            "--cp-rate",
            "0.35",
            "--limit-p95",
            "1.0",
            "--limit-fail-rate",
            "0.05",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
        "rate_ramp_cmd": [
            sys.executable,
            "dev_tools/load_test_nautical.py",
            "--rate-ramp",
            "--rate-secs",
            "4",
            "--rate-start",
            "5",
            "--rate-step",
            "5",
            "--rate-max",
            "10",
            "--limit-p95",
            "1.0",
            "--limit-fail-rate",
            "0.05",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
    },
    "stress": {
        "hook_fuzz_cases": 24,
        "hook_timeout_s": 5.0,
        "reliability_cmd": [
            sys.executable,
            "dev_tools/nautical_reliability_smoke.py",
            "--load",
            "40",
            "--chain-until-failure",
            "--chain-max",
            "80",
            "--chain-settle",
            "0.05",
        ],
        "load_ramp_cmd": [
            sys.executable,
            "dev_tools/load_test_nautical.py",
            "--ramp",
            "--ramp-start",
            "100",
            "--ramp-step",
            "100",
            "--ramp-max",
            "500",
            "--concurrency",
            "2",
            "--done-rate",
            "0.7",
            "--anchor-rate",
            "0.35",
            "--cp-rate",
            "0.35",
            "--limit-p95",
            "1.0",
            "--limit-fail-rate",
            "0.05",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
        "rate_ramp_cmd": [
            sys.executable,
            "dev_tools/load_test_nautical.py",
            "--rate-ramp",
            "--rate-secs",
            "10",
            "--rate-start",
            "5",
            "--rate-step",
            "5",
            "--rate-max",
            "25",
            "--limit-p95",
            "1.0",
            "--limit-fail-rate",
            "0.05",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
    },
    "extreme": {
        "hook_fuzz_cases": 60,
        "hook_timeout_s": 6.0,
        "reliability_cmd": [
            sys.executable,
            "tools/nautical_reliability_smoke.py",
            "--load",
            "120",
            "--durable",
            "--find-limit",
            "--min",
            "60",
            "--max",
            "900",
            "--step",
            "2",
            "--workers",
            "2",
            "--chain-until-failure",
            "--chain-max",
            "300",
            "--chain-settle",
            "0.05",
        ],
        "load_ramp_cmd": [
            sys.executable,
            "tools/load_test_nautical.py",
            "--ramp",
            "--ramp-start",
            "200",
            "--ramp-step",
            "200",
            "--ramp-max",
            "1800",
            "--concurrency",
            "6",
            "--done-rate",
            "0.8",
            "--anchor-rate",
            "0.4",
            "--cp-rate",
            "0.4",
            "--limit-p95",
            "1.25",
            "--limit-fail-rate",
            "0.03",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
        "rate_ramp_cmd": [
            sys.executable,
            "tools/load_test_nautical.py",
            "--rate-ramp",
            "--rate-secs",
            "15",
            "--rate-start",
            "10",
            "--rate-step",
            "10",
            "--rate-max",
            "70",
            "--limit-p95",
            "1.25",
            "--limit-fail-rate",
            "0.03",
            "--limit-queue-bytes",
            "1",
            "--json",
            "--enforce",
        ],
    },
}


def _strict_json_object(stdout_text: str) -> tuple[bool, str]:
    s = (stdout_text or "").strip()
    if not s:
        return False, "stdout empty"
    dec = json.JSONDecoder()
    try:
        obj, idx = dec.raw_decode(s)
    except Exception as exc:
        return False, f"stdout not strict JSON object: {exc}"
    if idx != len(s):
        return False, "stdout had trailing non-JSON content"
    if not isinstance(obj, dict):
        return False, f"stdout JSON was {type(obj).__name__}, expected object"
    return True, ""


def _stdout_contract_ok(stdout_text: str, *, allow_empty: bool) -> tuple[bool, str]:
    s = (stdout_text or "").strip()
    if not s:
        return (True, "") if allow_empty else (False, "stdout empty")
    return _strict_json_object(s)


def _tail(text: str, max_lines: int = 12) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _run(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 30.0,
    cwd: Path = ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        input=input_text,
        env=env,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _base_env(taskdata: Path) -> dict:
    env = os.environ.copy()
    env["TASKDATA"] = str(taskdata)
    env["NAUTICAL_DIAG"] = "1"
    env["NAUTICAL_CONFIG"] = str(taskdata / "missing.toml")
    env["NAUTICAL_CORE_PATH"] = str(ROOT)
    env["PYTHONPATH"] = str(ROOT / "tools") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TZ", "UTC")
    return env


def _make_task(uuid_suffix: int, description: str) -> dict:
    return {
        "uuid": f"00000000-0000-0000-0000-{uuid_suffix:012d}",
        "status": "pending",
        "entry": "20260101T000000Z",
        "description": description,
    }


def _hook_case_result(name: str, ok: bool, detail: str = "", extra: dict | None = None) -> dict:
    payload = {"name": name, "ok": bool(ok)}
    if detail:
        payload["detail"] = detail
    if extra:
        payload.update(extra)
    return payload


def _run_hook_protocol_stage(profile: str) -> dict:
    cfg = PROFILES[profile]
    timeout = float(cfg["hook_timeout_s"])
    cases = []
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="nautical-hook-abuse-") as td:
        td_path = Path(td)
        env = _base_env(td_path)

        add_task = _make_task(101, "Cafe ăîșț protocol ✅")
        p_add = _run([sys.executable, str(ON_ADD)], env=env, input_text=json.dumps(add_task, ensure_ascii=False), timeout=timeout)
        ok_add, msg_add = _strict_json_object(p_add.stdout or "")
        cases.append(
            _hook_case_result(
                "on_add_unicode_diag_passthrough",
                p_add.returncode == 0 and ok_add and "ăîșț protocol ✅" in (p_add.stdout or "") and "\\u" not in (p_add.stdout or ""),
                msg_add or f"rc={p_add.returncode}",
            )
        )

        oversized_desc = "X" * (10 * 1024 * 1024 + 128)
        big_task = _make_task(102, oversized_desc)
        p_big = _run([sys.executable, str(ON_ADD)], env=env, input_text=json.dumps(big_task), timeout=timeout)
        cases.append(
            _hook_case_result(
                "on_add_oversized_input_rejected",
                p_big.returncode != 0 and not (p_big.stdout or "").strip(),
                f"rc={p_big.returncode}",
            )
        )

        p_bad_add = _run(
            [sys.executable, str(ON_ADD)],
            env=env,
            input_text='{"uuid":"00000000-0000-0000-0000-000000000103","status":"pending"',
            timeout=timeout,
        )
        ok_bad_add, msg_bad_add = _stdout_contract_ok(p_bad_add.stdout or "", allow_empty=True)
        cases.append(
            _hook_case_result(
                "on_add_truncated_json_no_stdout_corruption",
                p_bad_add.returncode != 0 and ok_bad_add,
                msg_bad_add or f"rc={p_bad_add.returncode}",
            )
        )

        mod_task = _make_task(201, "Cafe ăîșț modify ✅")
        p_mod = _run(
            [sys.executable, str(ON_MODIFY)],
            env=env,
            input_text=json.dumps(mod_task, ensure_ascii=False),
            timeout=timeout,
        )
        ok_mod, msg_mod = _strict_json_object(p_mod.stdout or "")
        cases.append(
            _hook_case_result(
                "on_modify_unicode_diag_passthrough",
                p_mod.returncode == 0 and ok_mod and "ăîșț modify ✅" in (p_mod.stdout or "") and "\\u" not in (p_mod.stdout or ""),
                msg_mod or f"rc={p_mod.returncode}",
            )
        )

        base_old = json.dumps(_make_task(202, "old state"), ensure_ascii=False)
        base_new = json.dumps(_make_task(203, "new state"), ensure_ascii=False)
        malformed_inputs = [
            base_old + " trailing-garbage",
            base_old + "\n" + base_new[:-5],
            '{"uuid":"00000000-0000-0000-0000-000000000204"}{"uuid"',
            "[1,2,3]",
        ]
        for idx, raw in enumerate(malformed_inputs, start=1):
            proc = _run([sys.executable, str(ON_MODIFY)], env=env, input_text=raw, timeout=timeout)
            ok_stdout, msg_stdout = _stdout_contract_ok(proc.stdout or "", allow_empty=True)
            cases.append(
                _hook_case_result(
                    f"on_modify_malformed_case_{idx}",
                    ok_stdout,
                    msg_stdout or f"rc={proc.returncode}",
                )
            )

        for idx in range(int(cfg["hook_fuzz_cases"])):
            task_a = _make_task(300 + idx * 2, f"fuzz-old-{idx}")
            task_b = _make_task(301 + idx * 2, f"fuzz-new-{idx}")
            valid_a = json.dumps(task_a, ensure_ascii=False)
            valid_b = json.dumps(task_b, ensure_ascii=False)
            raw = RNG.choice(
                [
                    valid_a + RNG.choice(["", "\n", "  "]) + valid_b,
                    valid_a + RNG.choice(["\n", " ", ""]) + valid_b[:-RNG.randint(1, min(12, len(valid_b) - 1))],
                    valid_a[:RNG.randint(1, max(2, len(valid_a) - 1))],
                    RNG.choice(["{", "[", "", "null", "garbage"]) + valid_a,
                    valid_a + "\n" + "{" + valid_b,
                ]
            )
            proc = _run([sys.executable, str(ON_MODIFY)], env=env, input_text=raw, timeout=timeout)
            ok_stdout, msg_stdout = _stdout_contract_ok(proc.stdout or "", allow_empty=True)
            cases.append(
                _hook_case_result(
                    f"on_modify_fuzz_{idx + 1:03d}",
                    ok_stdout,
                    msg_stdout or f"rc={proc.returncode}",
                )
            )

        exit_env = env.copy()
        state_dir = td_path / ".nautical-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        corrupt_db = state_dir / ".nautical_queue.db"
        corrupt_db.write_text("not-a-sqlite-db", encoding="utf-8")
        p_exit = _run([sys.executable, str(ON_EXIT)], env=exit_env, input_text="", timeout=timeout)
        quarantine_candidates = list(state_dir.glob(".nautical_queue.db.corrupt.*"))
        ok_exit_stdout, msg_exit_stdout = _stdout_contract_ok(p_exit.stdout or "", allow_empty=True)
        cases.append(
            _hook_case_result(
                "on_exit_corrupt_queue_db_quarantined",
                p_exit.returncode == 0 and ok_exit_stdout and bool(quarantine_candidates),
                msg_exit_stdout or f"rc={p_exit.returncode}",
                extra={"quarantine_files": len(quarantine_candidates)},
            )
        )

        p_exit_empty = _run([sys.executable, str(ON_EXIT)], env=env, input_text="", timeout=timeout)
        cases.append(
            _hook_case_result(
                "on_exit_empty_stdout_contract",
                p_exit_empty.returncode == 0 and not (p_exit_empty.stdout or ""),
                f"rc={p_exit_empty.returncode}",
            )
        )

    failed = [case for case in cases if not case["ok"]]
    return {
        "stage": "hook_protocol_abuse",
        "status": "passed" if not failed else "failed",
        "profile": profile,
        "duration_s": round(time.perf_counter() - started, 3),
        "cases_total": len(cases),
        "cases_failed": len(failed),
        "cases": cases,
    }


def _run_command_stage(name: str, cmd: list[str], *, parse_json_stdout: bool, timeout: float = 3600.0) -> dict:
    started = time.perf_counter()
    try:
        proc = _run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "stage": name,
            "status": "failed",
            "duration_s": round(time.perf_counter() - started, 3),
            "cmd": cmd,
            "error": f"timeout after {timeout}s",
            "stdout_tail": _tail(exc.stdout or ""),
            "stderr_tail": _tail(exc.stderr or ""),
        }
    payload = None
    parse_error = ""
    if parse_json_stdout:
        try:
            payload = json.loads(proc.stdout or "{}")
        except Exception as exc:
            parse_error = str(exc)
    status = "passed" if proc.returncode == 0 and (payload is not None or not parse_json_stdout) else "failed"
    result = {
        "stage": name,
        "status": status,
        "duration_s": round(time.perf_counter() - started, 3),
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": _tail(proc.stdout or ""),
        "stderr_tail": _tail(proc.stderr or ""),
    }
    if payload is not None:
        result["payload"] = payload
    if parse_error:
        result["parse_error"] = parse_error
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES.keys()), default="stress")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--skip-perf", action="store_true", help="skip perf budget stage")
    ap.add_argument("--skip-task-stages", action="store_true", help="skip Taskwarrior-dependent stages")
    ap.add_argument("--keep-going", action="store_true", help="continue after the first failed stage")
    args = ap.parse_args()

    cfg = PROFILES[args.profile]
    results = []
    started = time.perf_counter()

    def _emit(line: str) -> None:
        if not args.json:
            print(line)

    def _record(stage: dict) -> bool:
        results.append(stage)
        status = stage.get("status", "failed")
        if status == "passed":
            _emit(f"[pass] {stage['stage']} ({stage.get('duration_s', 0):.2f}s)")
            return True
        if status == "skipped":
            _emit(f"[skip] {stage['stage']}: {stage.get('reason', 'skipped')}")
            return True
        _emit(f"[fail] {stage['stage']} ({stage.get('duration_s', 0):.2f}s)")
        detail = stage.get("error") or stage.get("parse_error") or stage.get("stdout_tail") or stage.get("stderr_tail") or ""
        if detail:
            _emit(detail)
        return bool(args.keep_going)

    if not _record(_run_hook_protocol_stage(args.profile)):
        return _finish(args.json, args.profile, results, started)

    if not args.skip_perf:
        perf_stage = _run_command_stage(
            "perf_budget",
            [sys.executable, "dev_tools/nautical_perf_budget.py", "--json", "--enforce"],
            parse_json_stdout=True,
            timeout=600.0,
        )
        if not _record(perf_stage):
            return _finish(args.json, args.profile, results, started)

    if args.skip_task_stages:
        _record({"stage": "task_dependent_stages", "status": "skipped", "reason": "disabled by --skip-task-stages"})
    elif TASK_BIN is None:
        _record({"stage": "task_dependent_stages", "status": "skipped", "reason": "task not found in PATH"})
    else:
        reliability = _run_command_stage("reliability_smoke", cfg["reliability_cmd"], parse_json_stdout=False, timeout=3600.0)
        if not _record(reliability):
            return _finish(args.json, args.profile, results, started)

        load_ramp = _run_command_stage("load_ramp", cfg["load_ramp_cmd"], parse_json_stdout=True, timeout=3600.0)
        if not _record(load_ramp):
            return _finish(args.json, args.profile, results, started)

        rate_ramp = _run_command_stage("load_rate_ramp", cfg["rate_ramp_cmd"], parse_json_stdout=True, timeout=3600.0)
        if not _record(rate_ramp):
            return _finish(args.json, args.profile, results, started)

    return _finish(args.json, args.profile, results, started)


def _finish(as_json: bool, profile: str, stages: list[dict], started: float) -> int:
    failed = [stage for stage in stages if stage.get("status") == "failed"]
    summary = {
        "profile": profile,
        "ok": not failed,
        "failed_stages": [stage.get("stage", "") for stage in failed],
        "duration_s": round(time.perf_counter() - started, 3),
        "task_available": bool(TASK_BIN),
        "stages": stages,
    }
    if as_json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), indent=2))
    else:
        print("")
        print("=" * 60)
        print("STRESS CAMPAIGN SUMMARY")
        print("=" * 60)
        print(f"profile={profile}")
        print(f"task_available={bool(TASK_BIN)}")
        print(f"duration_s={summary['duration_s']:.3f}")
        print(f"failed_stages={summary['failed_stages']}")
        print(f"ok={summary['ok']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
