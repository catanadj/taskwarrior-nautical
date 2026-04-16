#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay recorded hook payloads against Nautical hooks.

The harness replays JSON or raw stdin payloads against the repo-local hook
scripts and checks the stdout/stderr contract. It is intentionally small so
real failing payloads can be dropped into the corpus and rerun later.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CORPUS = HERE / "nautical_hook_replay_corpus.jsonl"
HOOK_FILES = {
    "on-add": ROOT / "on-add-nautical.py",
    "on-modify": ROOT / "on-modify-nautical.py",
    "on-exit": ROOT / "on-exit-nautical.py",
}


@dataclass
class ReplayCase:
    name: str
    hook: str
    input_text: str
    expect_rc: int = 0
    expect_stdout: str = "json_object"
    expect_stderr_contains: tuple[str, ...] = ()


def _load_corpus(path: Path) -> list[ReplayCase]:
    cases: list[ReplayCase] = []
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                obj = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{lineno}: corpus row must be a JSON object")
            name = str(obj.get("name") or "").strip()
            hook = str(obj.get("hook") or "").strip()
            if not name:
                raise ValueError(f"{path}:{lineno}: missing case name")
            if hook not in HOOK_FILES:
                raise ValueError(f"{path}:{lineno}: unsupported hook {hook!r}")
            inp = obj.get("input", "")
            if isinstance(inp, str):
                input_text = inp
            else:
                input_text = json.dumps(inp, ensure_ascii=False, separators=(",", ":"))
            expect = obj.get("expect") if isinstance(obj.get("expect"), dict) else obj
            expect_rc = int(expect.get("expect_rc", obj.get("expect_rc", 0)))
            expect_stdout = str(expect.get("stdout", obj.get("stdout", "json_object"))).strip() or "json_object"
            stderr_contains = obj.get("stderr_contains", expect.get("stderr_contains", []))
            if isinstance(stderr_contains, str):
                stderr_contains = [stderr_contains]
            if not isinstance(stderr_contains, list):
                raise ValueError(f"{path}:{lineno}: stderr_contains must be a list or string")
            cases.append(
                ReplayCase(
                    name=name,
                    hook=hook,
                    input_text=input_text,
                    expect_rc=expect_rc,
                    expect_stdout=expect_stdout,
                    expect_stderr_contains=tuple(str(x) for x in stderr_contains if str(x).strip()),
                )
            )
    return cases


def _strict_json_object(stdout_text: str) -> tuple[bool, str]:
    s = (stdout_text or "").strip()
    if not s:
        return False, "stdout was empty"
    dec = json.JSONDecoder()
    try:
        obj, idx = dec.raw_decode(s)
    except Exception as exc:
        return False, f"invalid JSON object: {exc}"
    if s[idx:].strip():
        return False, "stdout contained trailing non-JSON content"
    if not isinstance(obj, dict):
        return False, f"stdout JSON must be object, got {type(obj).__name__}"
    return True, ""


def _prepare_env(root: Path, taskdata: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["NAUTICAL_CORE_PATH"] = str(root)
    env["NAUTICAL_TRUST_CORE_PATH"] = "1"
    env["NAUTICAL_CONFIG"] = str(root / "config-nautical.toml")
    env["NAUTICAL_TRUST_CONFIG_PATH"] = "1"
    env["TASKDATA"] = str(taskdata)
    env["TZ"] = "UTC"
    env.pop("NAUTICAL_DIAG", None)
    env.pop("NAUTICAL_DIAG_LOG", None)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _run_hook(hook: Path, raw_input: str, env: dict[str, str], timeout_s: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(hook)],
        input=raw_input,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_s,
    )


def _run_case(case: ReplayCase, root: Path, timeout_s: float) -> dict[str, Any]:
    hook = HOOK_FILES[case.hook]
    with tempfile.TemporaryDirectory(prefix="nautical-hook-replay-") as td:
        env = _prepare_env(root, Path(td))
        p = _run_hook(hook, case.input_text, env, timeout_s=timeout_s)
        ok = p.returncode == case.expect_rc
        message = ""
        if not ok:
            message = f"expected rc={case.expect_rc}, got {p.returncode}"
        elif case.expect_stdout == "json_object":
            ok, message = _strict_json_object(p.stdout or "")
        elif case.expect_stdout == "empty":
            s = (p.stdout or "").strip()
            ok = not bool(s)
            message = "stdout must be empty" if not ok else ""
        elif case.expect_stdout == "any":
            ok = True
            message = ""
        else:
            ok = False
            message = f"unknown stdout expectation: {case.expect_stdout!r}"
        if ok and case.expect_stderr_contains:
            stderr = p.stderr or ""
            missing = [frag for frag in case.expect_stderr_contains if frag not in stderr]
            if missing:
                ok = False
                message = f"stderr missing expected fragments: {missing!r}"
        if ok and p.returncode == case.expect_rc and case.expect_rc == 0 and case.expect_stdout != "any":
            stderr = (p.stderr or "").strip()
            if stderr:
                ok = False
                message = f"unexpected stderr: {stderr}"
        return {
            "name": case.name,
            "hook": case.hook,
            "ok": bool(ok),
            "rc": p.returncode,
            "expect_rc": case.expect_rc,
            "stdout": p.stdout or "",
            "stderr": p.stderr or "",
            "message": message,
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay Nautical hook payload corpus")
    ap.add_argument("--root", default=str(ROOT), help="repo root containing hooks and nautical_core/")
    ap.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="JSONL corpus of replay cases")
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--timeout", type=float, default=8.0, help="per-case timeout in seconds")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--verbose", action="store_true", help="print per-case results")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    corpus_path = Path(args.corpus).expanduser().resolve()
    cases = _load_corpus(corpus_path)
    if args.only:
        filt = args.only.lower()
        cases = [c for c in cases if filt in c.name.lower() or filt in c.hook.lower()]

    results = [_run_case(case, root, timeout_s=args.timeout) for case in cases]
    ok = all(bool(r.get("ok")) for r in results)
    payload = {
        "status": "ok" if ok else "fail",
        "root": str(root),
        "corpus": str(corpus_path),
        "total": len(results),
        "passed": sum(1 for r in results if r.get("ok")),
        "failed": sum(1 for r in results if not r.get("ok")),
        "results": results,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"status={payload['status']} total={payload['total']} passed={payload['passed']} failed={payload['failed']}")
        for r in results:
            mark = "✓" if r.get("ok") else "✗"
            ident = f"{r.get('hook')}:{r.get('name')}"
            msg = r.get("message") or ""
            if args.verbose:
                print(f"{mark} {ident} rc={r.get('rc')} {msg}".rstrip())
            elif not r.get("ok"):
                print(f"{mark} {ident}: {msg}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
