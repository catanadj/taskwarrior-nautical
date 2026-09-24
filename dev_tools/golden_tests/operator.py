"""Operator/query and installed-layout golden tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def _run(command, *, environment=None, input_text=None, cwd=None, timeout=30):
    env = os.environ.copy()
    if environment:
        env.update(environment)
    return subprocess.run(
        command,
        input=input_text,
        cwd=cwd,
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=timeout,
    )


def test_query_process_boundary_emits_one_json_document():
    """The managed launcher keeps capability and invalid-request stdout strict."""
    launcher = str(ROOT / "nautical")
    env = {"PYTHONPATH": str(ROOT)}
    capability = _run([sys.executable, launcher, "query", "capabilities"], environment=env)
    if capability.returncode != 0 or len(capability.stdout.splitlines()) != 1 or capability.stderr:
        raise AssertionError(f"capability protocol changed: {capability!r}")
    if json.loads(capability.stdout).get("schema") != "nautical.query.capabilities":
        raise AssertionError("capability schema changed")

    inline = _run(
        [sys.executable, launcher, "query", "occurrences", "--request", "{}"],
        environment=env,
    )
    stdin = _run(
        [sys.executable, launcher, "query", "occurrences", "--request", "-"],
        input_text="{}",
        environment=env,
    )
    if inline.returncode != 2 or stdin.returncode != 2 or inline.stdout != stdin.stdout:
        raise AssertionError("inline and stdin invalid responses differ")
    json.loads(inline.stdout)

    diagnostic = _run(
        [sys.executable, launcher, "query", "occurrences", "--request", "{}"],
        environment={**env, "NAUTICAL_DIAG": "1"},
    )
    if diagnostic.returncode != 2 or not diagnostic.stderr.startswith("[nautical] query:"):
        raise AssertionError("query diagnostics changed protocol")


def test_operator_processes_concurrent_contracts_share_taskdata_safely():
    """Concurrent query/reconcile operators keep isolated JSON contracts."""
    with tempfile.TemporaryDirectory(prefix="nautical-concurrent-operators-") as taskdata:
        env = {"TASKDATA": taskdata, "NAUTICAL_CORE_PATH": str(ROOT), "PYTHONPATH": str(ROOT)}
        launcher = str(ROOT / "nautical")
        commands = (
            ([sys.executable, launcher, "query", "capabilities"], True),
            ([sys.executable, launcher, "reconcile", "--json", "--task-bin", "/missing/task"], True),
            ([sys.executable, launcher, "doctor", "--json"], True),
            ([sys.executable, launcher, "query", "integrity", "--all"], True),
            ([sys.executable, str(ROOT / "on-exit.nautical")], False),
        )
        processes = [
            subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, **env})
            for command, _ in commands
        ]
        results = [process.communicate(timeout=30) for process in processes]
    for (stdout, stderr), (command, json_expected) in zip(results, commands):
        if json_expected:
            json.loads(stdout)
        elif stdout:
            raise AssertionError(f"concurrent empty-exit hook wrote stdout: {stdout!r}")
        if "Traceback" in stderr:
            raise AssertionError(f"concurrent operator leaked traceback: {command}: {stderr!r}")


def test_query_installed_layout_runs_outside_checkout():
    """The managed launcher resolves its own staged core package."""
    with tempfile.TemporaryDirectory(prefix="nautical-query-runtime-") as runtime_dir:
        runtime = Path(runtime_dir)
        shutil.copy2(ROOT / "nautical", runtime / "nautical")
        (runtime / "nautical").chmod(0o755)
        shutil.copytree(ROOT / "nautical_core", runtime / "nautical_core")
        process = _run(
            [sys.executable, str(runtime / "nautical"), "query", "capabilities"],
            cwd="/tmp",
            environment={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
        )
    if process.returncode != 0 or process.stderr:
        raise AssertionError(f"installed-layout query failed: {process.stderr or process.stdout}")
    if json.loads(process.stdout).get("schema") != "nautical.query.capabilities":
        raise AssertionError("installed-layout query schema changed")


def test_navigator_import_and_help_are_noninteractive_without_rich():
    """Cold import and non-TTY help must not require the interactive renderer."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    probe = _run(
        [sys.executable, "-c", "import nautical_navigator,sys; print(any(m == 'rich' or m.startswith('rich.') for m in sys.modules))"],
        cwd="/tmp",
        environment=env,
    )
    if probe.returncode != 0 or probe.stdout.strip() != "False" or probe.stderr:
        raise AssertionError(f"Navigator cold import changed: {probe!r}")
    help_result = _run([sys.executable, str(ROOT / "nautical_navigator.py"), "--help"], cwd="/tmp", environment=env)
    if help_result.returncode != 0 or help_result.stderr:
        raise AssertionError(f"Navigator help changed: {help_result!r}")


TESTS = (
    test_query_process_boundary_emits_one_json_document,
    test_operator_processes_concurrent_contracts_share_taskdata_safely,
    test_query_installed_layout_runs_outside_checkout,
    test_navigator_import_and_help_are_noninteractive_without_rich,
)
