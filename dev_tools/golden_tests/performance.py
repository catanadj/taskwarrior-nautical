"""Performance and deployment-profile golden tests."""

from __future__ import annotations

import importlib.util
import json
import os
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
