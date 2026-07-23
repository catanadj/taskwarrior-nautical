#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment sanity check for Nautical runtime files and hook I/O contracts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


REQUIRED_RUNTIME_FILES = (
    "on-add-nautical.py",
    "on-modify-nautical.py",
    "on-exit-nautical.py",
    "nautical_core/install_runtime.py",
    "nautical_core/hooks/__init__.py",
    "nautical_core/hooks/add_impl.py",
    "nautical_core/hooks/exit_impl.py",
    "nautical_core/hooks/modify_impl.py",
    "nautical_core/native_until.py",
    "nautical_core/modify_expiration.py",
    "nautical_core/tools/nautical_install.py",
)


def _strict_json_object(stdout_text: str) -> tuple[bool, str]:
    s = (stdout_text or "").strip()
    if not s:
        return False, "stdout was empty"
    dec = json.JSONDecoder()
    try:
        obj, idx = dec.raw_decode(s)
    except Exception as e:
        return False, f"invalid JSON object: {e}"
    if s[idx:].strip():
        return False, "stdout contained non-JSON trailing content"
    if not isinstance(obj, dict):
        return False, f"stdout JSON must be object, got {type(obj).__name__}"
    return True, ""


def _run_hook(path: Path, raw_input: str, env: dict[str, str], timeout_s: float = 8.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(path)],
        input=raw_input,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_s,
    )


def _check_required_files(root: Path, require_exec: bool) -> list[dict]:
    out: list[dict] = []
    core_pkg = root / "nautical_core" / "__init__.py"
    out.append(
        {
            "kind": "file",
            "path": "nautical_core/__init__.py",
            "ok": bool(core_pkg.exists() and core_pkg.is_file()),
            "message": "ok" if core_pkg.exists() and core_pkg.is_file() else "missing",
        }
    )
    for rel in REQUIRED_RUNTIME_FILES:
        p = root / rel
        ok = p.exists() and p.is_file()
        msg = "ok"
        if not ok:
            msg = "missing"
        elif require_exec and rel.startswith("on-") and not os.access(str(p), os.X_OK):
            ok = False
            msg = "not executable"
        out.append({"kind": "file", "path": rel, "ok": bool(ok), "message": msg})
    return out


def _check_package_layout(root: Path, env: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    pkg_init = root / "nautical_core" / "__init__.py"
    if not (pkg_init.exists() and pkg_init.is_file()):
        return [{"kind": "layout", "name": "package_core", "ok": False, "message": "nautical_core/__init__.py missing"}]

    hook_names = ("on-add-nautical.py", "on-modify-nautical.py", "on-exit-nautical.py")
    for hook_name in hook_names:
        hook_path = root / hook_name
        if not (hook_path.exists() and hook_path.is_file()):
            out.append({"kind": "layout", "name": hook_name, "ok": False, "message": "hook missing"})
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"_nautical_layout_check_{hook_name.replace('-', '_')}", hook_path)
            if spec is None or spec.loader is None:
                raise RuntimeError("spec_from_file_location failed")
            old_env = os.environ.copy()
            try:
                os.environ.clear()
                os.environ.update(env)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
            finally:
                os.environ.clear()
                os.environ.update(old_env)
            resolved = getattr(mod, "_core_target_from_base")(root)
            ok = resolved == pkg_init
            msg = "ok" if ok else f"resolved to {resolved}"
            out.append({"kind": "layout", "name": hook_name, "ok": bool(ok), "message": msg})
        except Exception as exc:
            out.append({"kind": "layout", "name": hook_name, "ok": False, "message": str(exc)})
    return out


def _check_hook_contracts(root: Path, taskdata: Path) -> list[dict]:
    env = os.environ.copy()
    env["NAUTICAL_CORE_PATH"] = str(root)
    env["NAUTICAL_TRUST_CORE_PATH"] = "1"
    env["TASKDATA"] = str(taskdata)
    env["TZ"] = "UTC"
    env.pop("NAUTICAL_DIAG", None)
    env.pop("NAUTICAL_DIAG_LOG", None)

    hook_add = root / "on-add-nautical.py"
    hook_modify = root / "on-modify-nautical.py"
    hook_exit = root / "on-exit-nautical.py"

    base_task = {
        "uuid": "11111111-1111-1111-1111-111111111111",
        "description": "sanity",
        "status": "pending",
        "entry": "20260101T000000Z",
        "modified": "20260101T000000Z",
    }
    mod_task_old = dict(base_task)
    mod_task_new = dict(base_task)
    mod_task_new["modified"] = "20260101T000001Z"

    checks: list[dict] = []

    p_add = _run_hook(hook_add, json.dumps(base_task), env)
    ok_add = p_add.returncode == 0
    msg_add = ""
    if ok_add:
        ok_add, msg_add = _strict_json_object(p_add.stdout or "")
    else:
        msg_add = (p_add.stderr or "").strip() or f"exit={p_add.returncode}"
    checks.append({"kind": "hook", "name": "on-add", "ok": bool(ok_add), "message": msg_add})

    raw_mod = json.dumps(mod_task_old) + "\n" + json.dumps(mod_task_new)
    p_mod = _run_hook(hook_modify, raw_mod, env)
    ok_mod = p_mod.returncode == 0
    msg_mod = ""
    if ok_mod:
        ok_mod, msg_mod = _strict_json_object(p_mod.stdout or "")
    else:
        msg_mod = (p_mod.stderr or "").strip() or f"exit={p_mod.returncode}"
    checks.append({"kind": "hook", "name": "on-modify", "ok": bool(ok_mod), "message": msg_mod})

    p_exit = _run_hook(hook_exit, "", env)
    ok_exit = p_exit.returncode == 0 and not (p_exit.stdout or "").strip()
    msg_exit = ""
    if not ok_exit:
        if p_exit.returncode != 0:
            msg_exit = (p_exit.stderr or "").strip() or f"exit={p_exit.returncode}"
        else:
            msg_exit = "stdout must be empty"
    checks.append({"kind": "hook", "name": "on-exit", "ok": bool(ok_exit), "message": msg_exit})

    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="Nautical deployment sanity check")
    ap.add_argument("--root", default=str(ROOT), help="project root containing core + hook files")
    ap.add_argument("--taskdata", default="", help="taskdata directory for hook sanity run (defaults to temp dir)")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    ap.add_argument("--no-require-exec", action="store_true", help="do not fail when hook files are not executable")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    require_exec = not bool(args.no_require_exec)

    td_ctx = tempfile.TemporaryDirectory(prefix="nautical-deploy-sanity-")
    try:
        taskdata = Path(args.taskdata).expanduser().resolve() if args.taskdata else Path(td_ctx.name)
        taskdata.mkdir(parents=True, exist_ok=True)

        results = []
        results.extend(_check_required_files(root, require_exec=require_exec))
        layout_env = os.environ.copy()
        layout_env["NAUTICAL_CORE_PATH"] = str(root)
        layout_env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        layout_env["TASKDATA"] = str(taskdata)
        layout_env["TZ"] = "UTC"
        layout_env.pop("NAUTICAL_DIAG", None)
        layout_env.pop("NAUTICAL_DIAG_LOG", None)
        results.extend(_check_package_layout(root, layout_env))
        results.extend(_check_hook_contracts(root, taskdata))
        ok = all(bool(r.get("ok")) for r in results)

        payload = {
            "status": "ok" if ok else "fail",
            "root": str(root),
            "taskdata": str(taskdata),
            "require_exec": require_exec,
            "results": results,
        }

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            print(f"status={payload['status']} root={payload['root']}")
            for r in results:
                ident = r.get("path") or r.get("name") or "unknown"
                print(f"- {r.get('kind')} {ident}: {'ok' if r.get('ok') else 'fail'} {r.get('message') or ''}".rstrip())
        return 0 if ok else 2
    finally:
        td_ctx.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
