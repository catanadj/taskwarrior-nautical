"""Performance and deployment-profile golden tests."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
import tempfile

from dev_tools.golden_tests.support import expect, load_hook_module


ROOT = Path(__file__).resolve().parents[2]


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
