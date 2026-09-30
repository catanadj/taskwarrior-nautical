"""Performance and deployment-profile golden tests."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile

from dev_tools.golden_tests.support import expect, load_hook_module


ROOT = Path(__file__).resolve().parents[2]


def _load_perf_module():
    path = ROOT / "dev_tools" / "nautical_perf_budget.py"
    spec = importlib.util.spec_from_file_location("_nautical_perf_import_profile_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_perf_cold_import_records_module_profile():
    """Cold-import benchmarks should expose loaded-module counts for profiling."""
    perf = _load_perf_module()
    elapsed = perf._bench_cold_import("core", 1)
    if elapsed < 0.0:
        raise AssertionError("cold import benchmark returned an invalid duration")
    if int(perf.IMPORT_PROFILES.get("core", 0)) <= 0:
        raise AssertionError("cold import module profile was not recorded")


def test_deploy_sanity_script_reports_ok():
    """Deployment sanity script should pass on repo-local hooks/core."""
    process = subprocess.run(
        [sys.executable, str(ROOT / "dev_tools" / "nautical_deploy_sanity.py"), "--json"],
        text=True,
        capture_output=True,
        timeout=12.0,
        check=False,
    )
    if process.returncode != 0:
        raise AssertionError(f"deploy sanity returned {process.returncode}: {process.stderr!r}")
    payload = json.loads(process.stdout or "{}")
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    if payload.get("status") != "ok" or not results or not all(item.get("ok") for item in results):
        raise AssertionError(f"unexpected deploy sanity result: {payload}")


def test_deploy_sanity_enforces_removed_lifecycle_ownership():
    """Deployment checks reject reintroduced exit modules and reconcile seams."""
    module = load_hook_module(str(ROOT / "dev_tools" / "nautical_deploy_sanity.py"), "_nautical_removed_ownership_deploy_test")
    failures = [item for item in module._check_removed_ownership(ROOT) if not item.get("ok")]
    expect(not failures, f"removed lifecycle ownership checks failed: {failures!r}")

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        (staged / "nautical_core" / "tools").mkdir(parents=True)
        shutil.copy2(ROOT / "nautical_core" / "runtime_manifest.py", staged / "nautical_core" / "runtime_manifest.py")
        (staged / "nautical_core" / "exit_models.py").write_text("# stale module\n", encoding="utf-8")
        (staged / "nautical_core" / "tools" / "nautical_reconcile.py").write_text(
            "_validate_hook_protocol = object()\n", encoding="utf-8"
        )
        failures = [item for item in module._check_removed_ownership(staged) if not item.get("ok")]
        expect(len(failures) >= 2, f"reintroduced ownership paths were not rejected: {failures!r}")

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        (staged / "nautical_core" / "tools").mkdir(parents=True)
        shutil.copy2(ROOT / "nautical_core" / "runtime_manifest.py", staged / "nautical_core" / "runtime_manifest.py")
        (staged / "nautical_core" / "tools" / "nautical_reconcile.py").write_text(
            "from nautical_core.hooks import modify_impl\n", encoding="utf-8"
        )
        failures = [item for item in module._check_removed_ownership(staged) if not item.get("ok")]
        expect(
            any(item.get("name") == "operator-hook-imports:nautical_core/tools/nautical_reconcile.py" for item in failures),
            f"operator hook import was not rejected: {failures!r}",
        )

    results = module._check_removed_ownership(ROOT)
    expect(
        any(item.get("name") == "pure-integrity:nautical_core/chain_graph.py" for item in results),
        f"pure integrity import checks were not reported: {results!r}",
    )


def test_deploy_sanity_rejects_missing_lazy_lifecycle_module():
    """Deployment sanity fails when a declared lazy module is absent."""
    path = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "candidate"
        shutil.copytree(ROOT, candidate, ignore=shutil.ignore_patterns(".git", "__pycache__", ".nautical-cache", ".nautical_cache"))
        (candidate / "nautical_core" / "modify_completion_compute.py").unlink()
        process = subprocess.run(
            [sys.executable, str(path), "--root", str(candidate), "--no-require-exec", "--json"],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(process.returncode != 0, "deploy sanity accepted a release missing a lazy module")
        payload = json.loads((process.stdout or "{}").strip() or "{}")
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        expect(
            any(item.get("path") == "nautical_core/modify_completion_compute.py" and not item.get("ok") for item in results if isinstance(item, dict)),
            f"missing lazy module was not reported: {results}",
        )
        expect(
            any(item.get("kind") == "lazy-modules" and item.get("name") == "on-modify" and not item.get("ok") for item in results if isinstance(item, dict)),
            f"modify lazy import smoke did not fail: {results}",
        )


def test_deploy_sanity_rejects_missing_operator_runtime_tool():
    """Deployment sanity covers every command dispatched by nautical."""
    path = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "candidate"
        shutil.copytree(ROOT, candidate, ignore=shutil.ignore_patterns(".git", "__pycache__", ".nautical-cache", ".nautical_cache"))
        (candidate / "nautical_core" / "tools" / "nautical_doctor.py").unlink()
        process = subprocess.run(
            [sys.executable, str(path), "--root", str(candidate), "--no-require-exec", "--json"],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(process.returncode != 0, "deploy sanity accepted a release missing an operator tool")
        payload = json.loads((process.stdout or "{}").strip() or "{}")
        results = payload.get("results") if isinstance(payload.get("results"), list) else []
        expect(
            any(item.get("path") == "nautical_core/tools/nautical_doctor.py" and not item.get("ok") for item in results if isinstance(item, dict)),
            f"missing operator tool was not reported: {results}",
        )


def test_deploy_sanity_rejects_unowned_taskwarrior_subprocess():
    """Deployment checks keep Taskwarrior process ownership in one client."""
    module = load_hook_module(str(ROOT / "dev_tools" / "nautical_deploy_sanity.py"), "_nautical_deploy_process_ownership_test")
    with tempfile.TemporaryDirectory() as td:
        core_dir = Path(td) / "nautical_core"
        core_dir.mkdir()
        (core_dir / "bad_runner.py").write_text("import subprocess\nsubprocess.run(['task', 'export'])\n", encoding="utf-8")
        result = module._check_taskwarrior_process_ownership(Path(td))
        expect(result and not result[0]["ok"], f"unowned subprocess was accepted: {result}")
        expect("bad_runner.py:2" in result[0]["message"], f"violation location was lost: {result}")


def test_ops_templates_present_and_runner_executable():
    """Operations templates should exist and the health-check runner be executable."""
    ops = ROOT / "dev_tools" / "ops"
    files = (
        "README.md",
        "nautical-health-check.crontab",
        "nautical-health-check.service",
        "nautical-health-check.timer",
        "nautical_health_check_cron.sh",
    )
    for name in files:
        expect((ops / name).is_file(), f"missing ops template: {ops / name}")
    runner = ops / "nautical_health_check_cron.sh"
    expect(runner.stat().st_mode & 0o111, f"runner should be executable: {runner}")


TESTS = (
    test_perf_cold_import_records_module_profile,
    test_deploy_sanity_script_reports_ok,
    test_deploy_sanity_enforces_removed_lifecycle_ownership,
    test_deploy_sanity_rejects_missing_lazy_lifecycle_module,
    test_deploy_sanity_rejects_missing_operator_runtime_tool,
    test_deploy_sanity_rejects_unowned_taskwarrior_subprocess,
    test_ops_templates_present_and_runner_executable,
)
