"""Performance and deployment-profile golden tests."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
import tempfile

from dev_tools.golden_tests.support import expect, load_hook_module


ROOT = Path(__file__).resolve().parents[2]


def _load_perf_module():
    path = ROOT / "dev_tools" / "nautical_perf_budget.py"
    spec = importlib.util.spec_from_file_location(
        "_nautical_perf_import_profile_test", path
    )
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
        [
            sys.executable,
            str(ROOT / "dev_tools" / "nautical_deploy_sanity.py"),
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=12.0,
        check=False,
    )
    if process.returncode != 0:
        raise AssertionError(
            f"deploy sanity returned {process.returncode}: {process.stderr!r}"
        )
    payload = json.loads(process.stdout or "{}")
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    if (
        payload.get("status") != "ok"
        or not results
        or not all(item.get("ok") for item in results)
    ):
        raise AssertionError(f"unexpected deploy sanity result: {payload}")


def test_deploy_sanity_enforces_removed_lifecycle_ownership():
    """Deployment checks reject reintroduced exit modules and reconcile seams."""
    module = load_hook_module(
        str(ROOT / "dev_tools" / "nautical_deploy_sanity.py"),
        "_nautical_removed_ownership_deploy_test",
    )
    failures = [
        item for item in module._check_removed_ownership(ROOT) if not item.get("ok")
    ]
    expect(not failures, f"removed lifecycle ownership checks failed: {failures!r}")

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        (staged / "nautical_core" / "tools").mkdir(parents=True)
        shutil.copy2(
            ROOT / "nautical_core" / "runtime_manifest.py",
            staged / "nautical_core" / "runtime_manifest.py",
        )
        (staged / "nautical_core" / "exit_models.py").write_text(
            "# stale module\n", encoding="utf-8"
        )
        (staged / "nautical_core" / "tools" / "nautical_reconcile.py").write_text(
            "_validate_hook_protocol = object()\n", encoding="utf-8"
        )
        failures = [
            item
            for item in module._check_removed_ownership(staged)
            if not item.get("ok")
        ]
        expect(
            len(failures) >= 2,
            f"reintroduced ownership paths were not rejected: {failures!r}",
        )

    with tempfile.TemporaryDirectory() as td:
        staged = Path(td)
        (staged / "nautical_core" / "tools").mkdir(parents=True)
        shutil.copy2(
            ROOT / "nautical_core" / "runtime_manifest.py",
            staged / "nautical_core" / "runtime_manifest.py",
        )
        (staged / "nautical_core" / "tools" / "nautical_reconcile.py").write_text(
            "from nautical_core.hooks import modify_impl\n", encoding="utf-8"
        )
        failures = [
            item
            for item in module._check_removed_ownership(staged)
            if not item.get("ok")
        ]
        expect(
            any(
                item.get("name")
                == "operator-hook-imports:nautical_core/tools/nautical_reconcile.py"
                for item in failures
            ),
            f"operator hook import was not rejected: {failures!r}",
        )

    results = module._check_removed_ownership(ROOT)
    expect(
        any(
            item.get("name") == "pure-integrity:nautical_core/chain_graph.py"
            for item in results
        ),
        f"pure integrity import checks were not reported: {results!r}",
    )


def test_deploy_sanity_rejects_missing_lazy_lifecycle_module():
    """Deployment sanity fails when a declared lazy module is absent."""
    path = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "candidate"
        shutil.copytree(
            ROOT,
            candidate,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".nautical-cache", ".nautical_cache"
            ),
        )
        (candidate / "nautical_core" / "modify_completion_compute.py").unlink()
        process = subprocess.run(
            [
                sys.executable,
                str(path),
                "--root",
                str(candidate),
                "--no-require-exec",
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(
            process.returncode != 0,
            "deploy sanity accepted a release missing a lazy module",
        )
        payload = json.loads((process.stdout or "{}").strip() or "{}")
        results = (
            payload.get("results") if isinstance(payload.get("results"), list) else []
        )
        expect(
            any(
                item.get("path") == "nautical_core/modify_completion_compute.py"
                and not item.get("ok")
                for item in results
                if isinstance(item, dict)
            ),
            f"missing lazy module was not reported: {results}",
        )
        expect(
            any(
                item.get("kind") == "lazy-modules"
                and item.get("name") == "on-modify"
                and not item.get("ok")
                for item in results
                if isinstance(item, dict)
            ),
            f"modify lazy import smoke did not fail: {results}",
        )


def test_deploy_sanity_rejects_missing_operator_runtime_tool():
    """Deployment sanity covers every command dispatched by nautical."""
    path = ROOT / "dev_tools" / "nautical_deploy_sanity.py"
    with tempfile.TemporaryDirectory() as td:
        candidate = Path(td) / "candidate"
        shutil.copytree(
            ROOT,
            candidate,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".nautical-cache", ".nautical_cache"
            ),
        )
        (candidate / "nautical_core" / "tools" / "nautical_doctor.py").unlink()
        process = subprocess.run(
            [
                sys.executable,
                str(path),
                "--root",
                str(candidate),
                "--no-require-exec",
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=20.0,
        )
        expect(
            process.returncode != 0,
            "deploy sanity accepted a release missing an operator tool",
        )
        payload = json.loads((process.stdout or "{}").strip() or "{}")
        results = (
            payload.get("results") if isinstance(payload.get("results"), list) else []
        )
        expect(
            any(
                item.get("path") == "nautical_core/tools/nautical_doctor.py"
                and not item.get("ok")
                for item in results
                if isinstance(item, dict)
            ),
            f"missing operator tool was not reported: {results}",
        )


def test_deploy_sanity_rejects_unowned_taskwarrior_subprocess():
    """Deployment checks keep Taskwarrior process ownership in one client."""
    module = load_hook_module(
        str(ROOT / "dev_tools" / "nautical_deploy_sanity.py"),
        "_nautical_deploy_process_ownership_test",
    )
    with tempfile.TemporaryDirectory() as td:
        core_dir = Path(td) / "nautical_core"
        core_dir.mkdir()
        (core_dir / "bad_runner.py").write_text(
            "import subprocess\nsubprocess.run(['task', 'export'])\n", encoding="utf-8"
        )
        result = module._check_taskwarrior_process_ownership(Path(td))
        expect(
            result and not result[0]["ok"], f"unowned subprocess was accepted: {result}"
        )
        expect(
            "bad_runner.py:2" in result[0]["message"],
            f"violation location was lost: {result}",
        )


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


def test_perf_hint_benchmark_isolates_persistent_cache():
    """Hint timing uses a temporary cache and restores production settings."""
    perf = load_hook_module(
        str(ROOT / "dev_tools" / "nautical_perf_budget.py"),
        "_nautical_perf_cache_isolation_test",
    )
    original_build = perf.core.build_and_cache_hints
    original_override = getattr(perf.core, "ANCHOR_CACHE_DIR_OVERRIDE", "")
    seen = []
    try:

        def fake_build(*_args, **_kwargs):
            seen.append(str(getattr(perf.core, "ANCHOR_CACHE_DIR_OVERRIDE", "")))
            payload = perf.core.cache_load("perf-isolation")
            if payload is None:
                payload = {"dnf": []}
                perf.core.cache_save("perf-isolation", payload)
            return payload

        perf.core.build_and_cache_hints = fake_build
        perf._bench_build_hints(["w:mon"], 1, mode="warm")
        expect(
            seen and "nautical-perf-cache-" in seen[0],
            f"benchmark used a non-isolated cache: {seen!r}",
        )
        expect(
            getattr(perf.core, "ANCHOR_CACHE_DIR_OVERRIDE", "") == original_override,
            "benchmark did not restore cache configuration",
        )
    finally:
        perf.core.build_and_cache_hints = original_build


def test_perf_hook_fast_path_ratio_enforcement():
    """Hook latency checks enforce the normalized fast/full median ratio."""
    perf = load_hook_module(
        str(ROOT / "dev_tools" / "nautical_perf_budget.py"),
        "_nautical_hook_perf_ratio_test",
    )

    def timed_latency(_hook_path, *, input_text, env, expected_task):
        _ = (input_text, expected_task)
        return 0.100 if env.get("NAUTICAL_BENCH_FORCE_FULL") == "1" else 0.050

    perf._run_hook_timed = timed_latency
    passing = perf._measure_hook_fast_path(
        "hook_test",
        Path("unused-hook"),
        input_text="{}",
        expected_task={},
        base_env={},
        repeats=3,
        max_ratio=0.8,
    )
    expect(
        passing.get("pass") is True,
        f"clear fast-path improvement should pass: {passing}",
    )
    expect(
        abs(float(passing.get("fast_to_full_ratio")) - 0.5) < 0.001,
        f"unexpected ratio: {passing}",
    )

    def insufficient_gain(_hook_path, *, input_text, env, expected_task):
        _ = (input_text, expected_task)
        return 0.100 if env.get("NAUTICAL_BENCH_FORCE_FULL") == "1" else 0.090

    perf._run_hook_timed = insufficient_gain
    failing = perf._measure_hook_fast_path(
        "hook_test",
        Path("unused-hook"),
        input_text="{}",
        expected_task={},
        base_env={},
        repeats=3,
        max_ratio=0.8,
    )
    expect(
        failing.get("pass") is False, f"insufficient improvement should fail: {failing}"
    )

    perf._run_hook_timed = lambda *_args, **_kwargs: 0.060
    managed = perf._measure_managed_hook_latency(
        "managed_hook_test",
        Path("unused-hook"),
        input_text="{}",
        expected_task={},
        base_env={"NAUTICAL_CORE_PATH": "/source", "NAUTICAL_TRUST_CORE_PATH": "1"},
        repeats=3,
        baseline_median_s=0.050,
        max_ratio=1.5,
    )
    expect(
        managed.get("pass") is True,
        f"reasonable managed-layout overhead should pass: {managed}",
    )
    expect(
        abs(float(managed.get("managed_to_source_ratio")) - 1.2) < 0.001,
        f"unexpected managed/source ratio: {managed}",
    )


def test_hook_replay_harness_reports_ok():
    """Replay harness passes the seeded hook corpus."""
    path = ROOT / "dev_tools" / "nautical_hook_replay.py"
    corpus = ROOT / "dev_tools" / "nautical_hook_replay_corpus.jsonl"
    process = subprocess.run(
        [sys.executable, str(path), "--json", "--corpus", str(corpus)],
        text=True,
        capture_output=True,
        timeout=12.0,
    )
    expect(
        process.returncode == 0,
        f"replay harness returned {process.returncode}: stderr={process.stderr!r}",
    )
    payload = json.loads((process.stdout or "").strip() or "{}")
    expect(
        payload.get("status") == "ok", f"unexpected replay harness status: {payload}"
    )
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    expect(results, "replay harness should report per-case results")
    expect(
        all(bool(result.get("ok")) for result in results if isinstance(result, dict)),
        f"failing replay result: {results}",
    )


def test_mixed_recurrence_loop_harness_reports_ok():
    """Mixed recurrence runner completes a short deterministic cycle run."""
    path = ROOT / "dev_tools" / "nautical_mixed_recurrence_loop.py"
    process = subprocess.run(
        [sys.executable, str(path), "--cycles", "3", "--json"],
        text=True,
        capture_output=True,
        timeout=30.0,
    )
    expect(
        process.returncode == 0,
        f"mixed recurrence loop returned {process.returncode}: stderr={process.stderr!r}",
    )
    payload = json.loads((process.stdout or "").strip() or "{}")
    expect(payload.get("ok") is True, f"unexpected mixed loop status: {payload}")
    expect(
        int(payload.get("cycles_completed") or 0) >= 1,
        f"expected loop progress: {payload}",
    )
    expect(not payload.get("violations"), f"mixed loop reported violations: {payload}")


def test_soak_runner_reports_ok():
    """A short soak run completes without violations."""
    path = ROOT / "dev_tools" / "nautical_soak_test.py"
    process = subprocess.run(
        [
            sys.executable,
            str(path),
            "--seconds",
            "2",
            "--batch-size",
            "4",
            "--anchor-rate",
            "0.5",
            "--cp-rate",
            "0.5",
            "--done-rate",
            "0.5",
            "--progress-every-seconds",
            "0",
            "--json",
            "--enforce",
        ],
        text=True,
        capture_output=True,
        timeout=240,
    )
    expect(
        process.returncode == 0,
        f"soak runner returned {process.returncode}: stderr={process.stderr!r}",
    )
    payload = json.loads((process.stdout or "").strip() or "{}")
    expect(payload.get("ok") is True, f"unexpected soak status: {payload}")
    expect(not payload.get("violations"), f"soak runner reported violations: {payload}")


TESTS = (
    test_perf_cold_import_records_module_profile,
    test_deploy_sanity_script_reports_ok,
    test_deploy_sanity_enforces_removed_lifecycle_ownership,
    test_deploy_sanity_rejects_missing_lazy_lifecycle_module,
    test_deploy_sanity_rejects_missing_operator_runtime_tool,
    test_deploy_sanity_rejects_unowned_taskwarrior_subprocess,
    test_ops_templates_present_and_runner_executable,
    test_perf_hint_benchmark_isolates_persistent_cache,
    test_perf_hook_fast_path_ratio_enforcement,
    test_hook_replay_harness_reports_ok,
    test_mixed_recurrence_loop_harness_reports_ok,
    test_soak_runner_reports_ok,
)


def test_load_benchmark_installs_complete_hook_runtime():
    """The end-to-end benchmark must install on-exit and the Nautical UDAs."""
    load_test = load_hook_module(
        str(ROOT / "dev_tools" / "load_test_nautical.py"),
        "_nautical_load_test_runtime_test",
    )
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_dir = root / "taskdata"
        hooks_dir = data_dir / "hooks"
        taskrc = root / "taskrc"
        config = root / "config-nautical.toml"
        data_dir.mkdir()
        load_test._install_hooks(hooks_dir)
        load_test._write_taskrc(taskrc, data_dir, hooks_dir)
        load_test._write_nautical_config(config)

        for hook_name in ("on-add", "on-modify", "on-exit"):
            hook = hooks_dir / hook_name
            expect(hook.is_file(), f"load benchmark did not install {hook_name}")
            expect(os.access(hook, os.X_OK), f"load benchmark hook is not executable: {hook_name}")
        taskrc_text = taskrc.read_text(encoding="utf-8")
        expect(f"hooks.location={hooks_dir}" in taskrc_text, f"missing hooks.location: {taskrc_text!r}")
        expect(f"include {Path(ROOT) / 'uda.conf'}" in taskrc_text, f"missing UDA include: {taskrc_text!r}")
        expect("verbose=nothing" not in taskrc_text, "benchmark must preserve task IDs in command output")
        expect('tz = "UTC"' in config.read_text(encoding="utf-8"), "benchmark config should be deterministic")


def test_load_benchmark_queue_and_lineage_verification():
    """Benchmark validation should detect active SQLite work and broken parent-child links."""
    load_test = load_hook_module(
        str(ROOT / "dev_tools" / "load_test_nautical.py"),
        "_nautical_load_test_validation_test",
    )
    with tempfile.TemporaryDirectory() as td:
        data_dir = Path(td)
        state_dir = data_dir / ".nautical-state"
        state_dir.mkdir()
        db_path = state_dir / ".nautical_queue.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE queue_entries (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, state TEXT NOT NULL)"
            )
            conn.execute("INSERT INTO queue_entries(payload, state) VALUES (?, ?)", ('{"child":1}', "queued"))
            conn.execute("INSERT INTO queue_entries(payload, state) VALUES (?, ?)", ('{"child":2}', "done"))
        metrics = load_test._queue_metrics(data_dir)
        expect(metrics == {"items": 1, "bytes": len('{"child":1}')}, f"bad active queue metrics: {metrics!r}")

    parent_uuid = "11111111-0000-0000-0000-000000000001"
    child_uuid = "22222222-0000-0000-0000-000000000002"
    rows = [
        {
            "uuid": parent_uuid,
            "status": "completed",
            "chainID": "chain-a",
            "link": 1,
            "nextLink": "22222222",
        },
        {
            "uuid": child_uuid,
            "status": "pending",
            "chainID": "chain-a",
            "link": 2,
            "prevLink": "11111111",
        },
    ]
    valid = load_test._verify_link_rows(rows, [parent_uuid])
    expect(valid == {"expected": 1, "verified": 1, "failures": []}, f"valid lineage was rejected: {valid!r}")
    rows[1]["prevLink"] = "wrong"
    invalid = load_test._verify_link_rows(rows, [parent_uuid])
    expect(invalid.get("verified") == 0 and invalid.get("failures"), f"broken lineage was accepted: {invalid!r}")

TESTS = TESTS + (
    test_load_benchmark_installs_complete_hook_runtime,
    test_load_benchmark_queue_and_lineage_verification,
)
