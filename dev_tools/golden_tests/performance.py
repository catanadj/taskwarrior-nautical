"""Performance and deployment-profile golden tests."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


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


TESTS = (test_perf_cold_import_records_module_profile, test_deploy_sanity_script_reports_ok)
