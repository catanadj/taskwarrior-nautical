#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment sanity check for Nautical runtime files and hook I/O contracts."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


REQUIRED_RUNTIME_FILES = (
    "on-add.nautical",
    "on-modify.nautical",
    "on-exit.nautical",
    "nautical_core/install_runtime.py",
    "nautical_core/runtime_manifest.py",
    "nautical_core/task_command.py",
    "nautical_core/hooks/__init__.py",
    "nautical_core/hooks/add_impl.py",
    "nautical_core/hooks/exit_impl.py",
    "nautical_core/hooks/modify_impl.py",
    "nautical_core/native_until.py",
    "nautical_core/modify_expiration.py",
    "nautical_core/tools/nautical_install.py",
)


def _load_runtime_manifest(root: Path):
    """Load the manifest from the candidate tree, not from the host checkout."""
    path = root / "nautical_core" / "runtime_manifest.py"
    if not path.is_file():
        return None, f"manifest missing: {path}"
    try:
        loader = importlib.machinery.SourceFileLoader("_nautical_runtime_manifest_check", str(path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        if spec is None or spec.loader is None:
            raise RuntimeError("spec_from_file_location failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


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
    manifest, manifest_error = _load_runtime_manifest(root)
    required_files = list(REQUIRED_RUNTIME_FILES)
    if manifest is not None:
        runtime_files = getattr(manifest, "HOOK_RUNTIME_FILES", {})
        if isinstance(runtime_files, dict):
            for files in runtime_files.values():
                if isinstance(files, (tuple, list)):
                    required_files.extend(
                        str(Path("nautical_core") / str(path))
                        for path in files
                    )
    else:
        out.append({
            "kind": "manifest",
            "path": "nautical_core/runtime_manifest.py",
            "ok": False,
            "message": manifest_error,
        })
    for rel in dict.fromkeys(required_files):
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


def _check_lazy_lifecycle_modules(root: Path, env: dict[str, str]) -> list[dict]:
    """Import every lifecycle module declared by the candidate's own manifest."""
    manifest, manifest_error = _load_runtime_manifest(root)
    if manifest is None:
        return [{"kind": "lazy-modules", "name": "manifest", "ok": False, "message": manifest_error}]
    lazy_modules = getattr(manifest, "HOOK_LAZY_MODULES", {})
    runtime_files = getattr(manifest, "HOOK_RUNTIME_FILES", {})
    if not isinstance(lazy_modules, dict) or not isinstance(runtime_files, dict):
        return [{"kind": "lazy-modules", "name": "manifest", "ok": False, "message": "invalid module manifest"}]

    smoke_script = r'''
import importlib, importlib.util, sys
from pathlib import Path

root = Path(sys.argv[1])
impl_path = Path(sys.argv[2])
event = sys.argv[3]
names = sys.argv[4:]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location(f"_nautical_lazy_{event}", impl_path)
if spec is None or spec.loader is None:
    raise RuntimeError("implementation spec could not be created")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
load_core = getattr(module, "_load_core", None)
if not callable(load_core):
    raise RuntimeError("implementation does not expose _load_core")
load_core()
for name in names:
    if name in {"calendar_feedback", "modify_completion_flow"}:
        importlib.import_module(f"nautical_core.{name}")
    else:
        module._module(name)
'''
    results: list[dict] = []
    for event, names in lazy_modules.items():
        files = runtime_files.get(event)
        impl_rel = files[0] if isinstance(files, (tuple, list)) and files else ""
        if not impl_rel:
            results.append({"kind": "lazy-modules", "name": event, "ok": False, "message": "implementation missing from manifest"})
            continue
        smoke_env = dict(env)
        smoke_env["NAUTICAL_CORE_PATH"] = str(root)
        smoke_env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    smoke_script,
                    str(root),
                    str(root / "nautical_core" / impl_rel),
                    event,
                    *[str(name) for name in names],
                ],
                cwd=str(root),
                env=smoke_env,
                text=True,
                capture_output=True,
                timeout=20.0,
            )
            ok = proc.returncode == 0
            message = "ok" if ok else (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip()
        except Exception as exc:
            ok = False
            message = f"{type(exc).__name__}: {exc}"
        results.append({"kind": "lazy-modules", "name": event, "ok": bool(ok), "message": message})
    return results


def _check_package_layout(root: Path, env: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    pkg_init = root / "nautical_core" / "__init__.py"
    if not (pkg_init.exists() and pkg_init.is_file()):
        return [{"kind": "layout", "name": "package_core", "ok": False, "message": "nautical_core/__init__.py missing"}]

    hook_names = ("on-add.nautical", "on-modify.nautical", "on-exit.nautical")
    for hook_name in hook_names:
        hook_path = root / hook_name
        if not (hook_path.exists() and hook_path.is_file()):
            out.append({"kind": "layout", "name": hook_name, "ok": False, "message": "hook missing"})
            continue
        try:
            loader = importlib.machinery.SourceFileLoader(
                f"_nautical_layout_check_{hook_name.replace('-', '_')}",
                str(hook_path),
            )
            spec = importlib.util.spec_from_loader(loader.name, loader)
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


def _check_performance_workflow(root: Path) -> list[dict]:
    """Ensure the performance job provisions the tools its benchmark invokes."""
    workflow = root / ".github" / "workflows" / "perf-budget.yml"
    if not workflow.is_file():
        return [{"kind": "workflow", "name": "perf-budget", "ok": False, "message": "workflow missing"}]
    text = workflow.read_text(encoding="utf-8")
    requirements_ok = "python3 -m pip install -r requirements.txt" in text
    task_ok = "sudo apt-get install -y taskwarrior" in text
    checks = [
        ("requirements", requirements_ok, "requirements.txt installation"),
        ("taskwarrior", task_ok, "Taskwarrior installation"),
    ]
    return [
        {
            "kind": "workflow",
            "name": f"perf-budget:{name}",
            "ok": bool(ok),
            "message": "ok" if ok else f"missing {description}",
        }
        for name, ok, description in checks
    ]


def _check_workflow_script_references(root: Path) -> list[dict]:
    """Catch CI workflows that invoke a dev tool missing from the checkout."""
    workflows_dir = root / ".github" / "workflows"
    if not workflows_dir.is_dir():
        return [{"kind": "workflow", "name": "script-references", "ok": False, "message": "workflows directory missing"}]
    results: list[dict] = []
    pattern = re.compile(r"python3\s+(dev_tools/[A-Za-z0-9_.-]+\.py)")
    for workflow in sorted((*workflows_dir.glob("*.yml"), *workflows_dir.glob("*.yaml"))):
        text = workflow.read_text(encoding="utf-8")
        for rel in sorted(set(pattern.findall(text))):
            exists = (root / rel).is_file()
            results.append(
                {
                    "kind": "workflow",
                    "name": f"{workflow.name}:{rel}",
                    "ok": bool(exists),
                    "message": "ok" if exists else "referenced script missing",
                }
            )
    return results or [{"kind": "workflow", "name": "script-references", "ok": True, "message": "none"}]
def _check_hook_contracts(root: Path, taskdata: Path) -> list[dict]:
    env = os.environ.copy()
    env["NAUTICAL_CORE_PATH"] = str(root)
    env["NAUTICAL_TRUST_CORE_PATH"] = "1"
    env["TASKDATA"] = str(taskdata)
    env["TZ"] = "UTC"
    env.pop("NAUTICAL_DIAG", None)
    env.pop("NAUTICAL_DIAG_LOG", None)

    hook_add = root / "on-add.nautical"
    hook_modify = root / "on-modify.nautical"
    hook_exit = root / "on-exit.nautical"

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
        results.extend(_check_lazy_lifecycle_modules(root, layout_env))
        results.extend(_check_performance_workflow(root))
        results.extend(_check_workflow_script_references(root))
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
