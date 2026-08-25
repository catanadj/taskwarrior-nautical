#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployment sanity check for Nautical runtime files and hook I/O contracts."""

from __future__ import annotations

import argparse
import ast
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
    "bootstrap.sh",
    "on-add.nautical",
    "on-modify.nautical",
    "on-exit.nautical",
    "nautical_core/install_runtime.py",
    "nautical_core/runtime_manifest.py",
    "nautical_core/runtime_command.py",
    "nautical_core/task_command.py",
    "nautical_core/taskwarrior_client.py",
    "nautical_core/task_read_repository.py",
    "nautical_core/taskwarrior_uow.py",
    "nautical_core/hooks/__init__.py",
    "nautical_core/hooks/add_impl.py",
    "nautical_core/hooks/exit_impl.py",
    "nautical_core/hooks/modify_impl.py",
    "nautical_core/native_until.py",
    "nautical_core/astronomy_validation.py",
    "nautical_core/modify_expiration.py",
    "nautical_core/modify_spawn.py",
    "nautical_core/modify_analytics.py",
    "nautical_core/lifecycle_read_service.py",
    "nautical_core/lifecycle_application.py",
    "nautical_core/query_models.py",
    "nautical_core/query_service.py",
    "nautical_core/tools/nautical_query.py",
    "nautical_core/tools/nautical_install.py",
    "nautical_core/tools/nautical_install_verify.py",
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
        spec.loader.exec_module(module)
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
        operator_files = getattr(manifest, "OPERATOR_RUNTIME_FILES", ())
        if isinstance(operator_files, (tuple, list)):
            required_files.extend(str(path) for path in operator_files)
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


def _module_spec_names(path: Path) -> set[str]:
    """Read hook module names without importing the full implementation."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if not any(isinstance(target, ast.Name) and target.id == "_MODULE_SPECS" for target in targets):
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            return set()
        return {
            key.value
            for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def _check_manifest_alignment(root: Path) -> list[dict]:
    """Ensure staged manifests cover every lazy module used by each hook."""
    manifest, manifest_error = _load_runtime_manifest(root)
    if manifest is None:
        return [{"kind": "manifest", "name": "alignment", "ok": False, "message": manifest_error}]
    lazy_modules = getattr(manifest, "HOOK_LAZY_MODULES", {})
    implementations = {
        "on-add": root / "nautical_core" / "hooks" / "add_impl.py",
        "on-modify": root / "nautical_core" / "hooks" / "modify_impl.py",
        "on-exit": root / "nautical_core" / "hooks" / "exit_impl.py",
    }
    results: list[dict] = []
    for event, implementation in implementations.items():
        declared = {str(name) for name in lazy_modules.get(event, ())}
        if not implementation.is_file():
            results.append({"kind": "manifest", "name": f"alignment:{event}", "ok": False, "message": "implementation missing"})
            continue
        try:
            used = _module_spec_names(implementation)
        except Exception as exc:
            results.append({"kind": "manifest", "name": f"alignment:{event}", "ok": False, "message": f"{type(exc).__name__}: {exc}"})
            continue
        missing = sorted(used - declared)
        results.append({
            "kind": "manifest",
            "name": f"alignment:{event}",
            "ok": not missing,
            "message": "ok" if not missing else f"missing lazy modules: {', '.join(missing)}",
        })
    return results


def _check_removed_ownership(root: Path) -> list[dict]:
    """Reject removed exit modules and reconcile compatibility symbols."""
    manifest, manifest_error = _load_runtime_manifest(root)
    if manifest is None:
        return [{"kind": "ownership", "name": "removed-paths", "ok": False, "message": manifest_error}]
    forbidden_files = tuple(str(path) for path in getattr(manifest, "REMOVED_RUNTIME_FILES", ()))
    forbidden_symbols = frozenset(str(name) for name in getattr(manifest, "REMOVED_RECONCILE_SYMBOLS", ()))
    results: list[dict] = []
    for relative in forbidden_files:
        exists = (root / relative).exists()
        results.append({
            "kind": "ownership",
            "name": f"removed:{relative}",
            "ok": not exists,
            "message": "absent" if not exists else "removed runtime path reintroduced",
        })

    reconcile_path = root / "nautical_core" / "tools" / "nautical_reconcile.py"
    if forbidden_symbols and reconcile_path.is_file():
        try:
            tree = ast.parse(reconcile_path.read_text(encoding="utf-8"), filename=str(reconcile_path))
            used = {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id in forbidden_symbols
            }
            used.update(
                node.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and node.attr in forbidden_symbols
            )
            results.append({
                "kind": "ownership",
                "name": "reconcile-legacy-symbols",
                "ok": not used,
                "message": "absent" if not used else f"removed reconcile symbols reintroduced: {', '.join(sorted(used))}",
            })
        except Exception as exc:
            results.append({
                "kind": "ownership",
                "name": "reconcile-legacy-symbols",
                "ok": False,
                "message": f"{type(exc).__name__}: {exc}",
            })

    # Keep operator fronts as composition/presentation layers.  Importing a
    # hook implementation here would recreate a second lifecycle owner and
    # make staged deployments depend on private hook internals.
    forbidden_imports = frozenset(
        str(name) for name in getattr(manifest, "OPERATOR_FORBIDDEN_HOOK_IMPORTS", ())
    )
    for relative in (
        "nautical_core/tools/nautical_reconcile.py",
        "nautical_core/tools/nautical_doctor.py",
    ):
        path = root / relative
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module)
            violations = sorted(
                name for name in imports
                if name in forbidden_imports or any(name.startswith(f"{prefix}.") for prefix in forbidden_imports)
            )
            results.append({
                "kind": "ownership",
                "name": f"operator-hook-imports:{relative}",
                "ok": not violations,
                "message": "absent" if not violations else f"operator imports hook implementation: {', '.join(violations)}",
            })
        except Exception as exc:
            results.append({
                "kind": "ownership",
                "name": f"operator-hook-imports:{relative}",
                "ok": False,
                "message": f"{type(exc).__name__}: {exc}",
            })

    pure_modules = tuple(getattr(manifest, "PURE_INTEGRITY_MODULES", ()))
    forbidden_pure_tokens = ("hooks", "hook_runtime", "taskwarrior", "sqlite", "rich", "tools")
    for relative in pure_modules:
        path = root / str(relative)
        if not path.is_file():
            results.append({
                "kind": "ownership",
                "name": f"pure-integrity:{relative}",
                "ok": False,
                "message": "declared pure integrity module is missing",
            })
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            pure_imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    pure_imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    pure_imports.add(node.module)
            violations = sorted(
                name for name in pure_imports
                if any(token in name.casefold() for token in forbidden_pure_tokens)
            )
            results.append({
                "kind": "ownership",
                "name": f"pure-integrity:{relative}",
                "ok": not violations,
                "message": "dependency-free" if not violations else f"forbidden dependency: {', '.join(violations)}",
            })
        except Exception as exc:
            results.append({
                "kind": "ownership",
                "name": f"pure-integrity:{relative}",
                "ok": False,
                "message": f"{type(exc).__name__}: {exc}",
            })
    return results


def _check_domain_model_boundaries(root: Path) -> list[dict]:
    """Keep removed mapping/facade construction paths out of shipped code."""
    forbidden = (
        "LifecyclePlan.from_mappings",
        "ParentGuard.from_mapping",
        "LifecycleIdentity.from_mapping",
        "ChildImportPayload.from_mapping",
        "MetadataRepairPayload.from_mapping",
        "sanitize_task_strings",
    )
    violations: list[str] = []
    package = root / "nautical_core"
    if package.is_dir():
        for path in package.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for token in forbidden:
                if token in text:
                    violations.append(f"{path.relative_to(root)}:{token}")
    # JSON decoding is permitted only at explicit protocol, cache, query, or
    # durable-persistence boundaries.  Domain/service modules must consume
    # TaskObservation rather than re-parsing Taskwarrior JSON.
    allowed_json_modules = {
        "nautical_core/task_codec.py",
        "nautical_core/runtime.py",
        "nautical_core/hooks/add_impl.py",
        "nautical_core/hooks/exit_impl.py",
        "nautical_core/tools/nautical_query.py",
        "nautical_core/tools/nautical_install_verify.py",
        "nautical_core/install_runtime.py",
        "nautical_core/lifecycle_outbox.py",
        "nautical_core/position_selection.py",
    }
    direct_json_violations: list[str] = []
    if package.is_dir():
        for path in package.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            uses_json_loads = any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr == "loads"
                for node in ast.walk(tree)
            )
            relative = str(path.relative_to(root))
            if uses_json_loads and relative not in allowed_json_modules:
                direct_json_violations.append(relative)
    if direct_json_violations:
        violations.extend(f"direct-json:{path}" for path in direct_json_violations)

    typed_domain_modules = {
        "nautical_core/add_anchor_preview.py",
        "nautical_core/lifecycle_planner.py",
        "nautical_core/chain_integrity_lifecycle.py",
        "nautical_core/lifecycle_reconciliation.py",
        "nautical_core/hook_engine.py",
        "nautical_core/modify_validation.py",
        "nautical_core/modify_feedback.py",
        "nautical_core/modify_expiration.py",
        "nautical_core/modify_timeline.py",
        "nautical_core/query_service.py",
        "nautical_core/modify_completion_flow.py",
    }
    typed_violations: list[str] = []
    for relative in sorted(typed_domain_modules):
        path = root / relative
        if not path.is_file():
            typed_violations.append(f"missing:{relative}")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            typed_violations.append(f"invalid:{relative}:{type(exc).__name__}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for argument in [*node.args.args, *node.args.kwonlyargs]:
                if argument.arg not in {"task", "parent", "child", "old", "new"}:
                    continue
                annotation = ast.unparse(argument.annotation) if argument.annotation else ""
                if "dict" in annotation.lower() or "mapping" in annotation.lower():
                    typed_violations.append(f"{relative}:{node.name}:{argument.arg}:{annotation}")
    if typed_violations:
        violations.extend(f"untyped-domain:{item}" for item in typed_violations)

    # Remaining task-shaped mappings are explicit protocol/configuration,
    # mutation-adapter, presentation, or durable-serialization boundaries.
    # New domain modules must use TaskPayload/TaskObservation instead.
    task_mapping_allowlist = {
        "nautical_core/business_calendar_api.py",
        "nautical_core/calendar_feedback.py",
        "nautical_core/chain_generation.py",
        "nautical_core/common.py",
        "nautical_core/description_aliases.py",
        "nautical_core/hook_protocol.py",
        "nautical_core/hook_results.py",
        "nautical_core/hooks/add_impl.py",
        "nautical_core/hooks/modify_impl.py",
        "nautical_core/lifecycle_models.py",
        "nautical_core/modify_analytics.py",
        "nautical_core/modify_chain_summary.py",
        "nautical_core/modify_completion_compute.py",
        "nautical_core/modify_protocol.py",
        "nautical_core/modify_runtime.py",
        "nautical_core/natural_language.py",
        "nautical_core/natural_language_api.py",
        "nautical_core/recurrence_context.py",
        "nautical_core/task_codec.py",
        "nautical_core/tools/nautical_doctor.py",
    }
    for path in package.rglob("*.py") if package.is_dir() else ():
        relative = str(path.relative_to(root))
        if relative in task_mapping_allowlist:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for argument in [*node.args.args, *node.args.kwonlyargs]:
                if argument.arg not in {"task", "parent", "child", "old", "new"}:
                    continue
                annotation = ast.unparse(argument.annotation) if argument.annotation else ""
                if "dict" in annotation.lower() or "mapping" in annotation.lower():
                    violations.append(f"mapping-boundary:{relative}:{node.name}:{argument.arg}")

    return [{
        "kind": "domain-model",
        "name": "removed-construction-paths",
        "ok": not violations,
        "message": "absent" if not violations else "removed domain-model paths found: " + ", ".join(sorted(violations)),
    }]


def _check_scheduler_ownership(root: Path) -> list[dict]:
    """Reject operational calls to scheduler aliases removed from the facade."""
    legacy_names = {
        "atom_matches_on",
        "base_next_after_atom",
        "factor_matches_on",
        "next_after_atom_with_mods",
        "next_after_expr",
        "next_after_factor",
        "next_after_term",
        "roll_apply",
    }
    files = [
        *sorted((root / "nautical_core").glob("*.py")),
        *sorted((root / "nautical_core" / "hooks").glob("*.py")),
        root / "nautical_navigator.py",
    ]
    findings: list[dict] = []
    for path in files:
        if not path.is_file() or path.name in {"scheduler_api.py", "scheduler_atom.py", "scheduler_expr.py"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            findings.append({
                "kind": "scheduler-ownership",
                "name": str(path.relative_to(root)),
                "ok": False,
                "message": f"{type(exc).__name__}: {exc}",
            })
            continue
        violations = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "core"
            and node.func.attr in legacy_names
        ]
        if violations:
            names = ", ".join(sorted({node.func.attr for node in violations}))
            findings.append({
                "kind": "scheduler-ownership",
                "name": str(path.relative_to(root)),
                "ok": False,
                "message": f"direct public scheduler calls remain: {names}",
            })
    if findings:
        return findings
    return [{
        "kind": "scheduler-ownership",
        "name": "operational-call-sites",
        "ok": True,
        "message": "ok; scheduler service/private engine boundaries are used",
    }]


def _check_taskwarrior_process_ownership(root: Path) -> list[dict]:
    """Reject direct subprocess boundaries outside their explicit owners."""
    allowed = {
        Path("nautical_core/taskwarrior_client.py"),
        Path("nautical_core/install_runtime.py"),
    }
    violations: list[str] = []
    for path in sorted((root / "nautical_core").rglob("*.py")):
        relative = path.relative_to(root)
        if relative in allowed:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            violations.append(f"{relative}: parse failed: {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "subprocess"
                and node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}
            ):
                violations.append(f"{relative}:{node.lineno} subprocess.{node.func.attr}")
    return [{
        "kind": "process-ownership",
        "name": "taskwarrior-client",
        "ok": not violations,
        "message": "ok" if not violations else "; ".join(violations),
    }]


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
                spec.loader.exec_module(mod)
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


def _check_query_operator(root: Path, env: dict[str, str]) -> list[dict]:
    """Smoke-test the installed-style query launcher without Taskwarrior data."""
    launcher = root / "nautical"
    if not launcher.is_file():
        return [{"kind": "query", "name": "launcher", "ok": False, "message": "nautical launcher missing"}]
    query_env = dict(env)
    query_env["PYTHONPATH"] = str(root) + os.pathsep + str(query_env.get("PYTHONPATH") or "")
    try:
        proc = subprocess.run(
            [sys.executable, str(launcher), "query", "capabilities"],
            cwd=str(root),
            env=query_env,
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        ok, message = _strict_json_object(proc.stdout)
        if proc.returncode != 0:
            ok = False
            message = (proc.stderr or proc.stdout or f"exit={proc.returncode}").strip()
        if ok:
            try:
                payload = json.loads(proc.stdout)
                ok = payload.get("schema") == "nautical.query.capabilities"
                message = "ok" if ok else "unexpected capability schema"
            except Exception as exc:
                ok = False
                message = f"invalid capability JSON: {exc}"
        return [{"kind": "query", "name": "capabilities", "ok": bool(ok), "message": message}]
    except Exception as exc:
        return [{"kind": "query", "name": "capabilities", "ok": False, "message": f"{type(exc).__name__}: {exc}"}]


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
        results.extend(_check_query_operator(root, layout_env))
        results.extend(_check_lazy_lifecycle_modules(root, layout_env))
        results.extend(_check_manifest_alignment(root))
        results.extend(_check_removed_ownership(root))
        results.extend(_check_domain_model_boundaries(root))
        results.extend(_check_scheduler_ownership(root))
        results.extend(_check_taskwarrior_process_ownership(root))
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
