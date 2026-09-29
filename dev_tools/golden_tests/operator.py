"""Operator/query and installed-layout golden tests."""

from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile

from dev_tools.golden_tests.support import expect, load_hook_module

ROOT = Path(__file__).resolve().parents[2]
DEV_TOOLS = ROOT / "dev_tools"
CORE_TOOLS = ROOT / "nautical_core" / "tools"


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


def test_health_check_json_ok_empty_taskdata():
    """health check should report ok for empty taskdata."""
    path = DEV_TOOLS / "nautical_health_check.py"
    with tempfile.TemporaryDirectory() as td:
        process = subprocess.run(
            [sys.executable, str(path), "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(process.returncode == 0, f"health check returned {process.returncode}: {process.stderr!r}")
        payload = json.loads((process.stdout or "").strip() or "{}")
        expect(payload.get("status") == "ok", f"unexpected status: {payload}")


def test_queue_status_and_doctor_report_schema_health():
    """Operator diagnostics should distinguish healthy and incompatible outboxes."""
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    status_path = CORE_TOOLS / "nautical_queue_status.py"
    doctor = load_hook_module(
        str(CORE_TOOLS / "nautical_doctor.py"),
        "_nautical_doctor_queue_schema_test",
    )
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        repository = _LifecycleOutboxRepository(taskdata)
        expect(repository.open().ok, "lifecycle outbox did not initialize")
        db_path = repository.path
        process = subprocess.run(
            [sys.executable, str(status_path), "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(process.returncode == 0, f"healthy schema status failed: {process.stderr!r}")
        payload = json.loads(process.stdout)
        expect(payload.get("schema") == "nautical.lifecycle_outbox_status", f"outbox status schema missing: {payload!r}")
        expect(payload.get("schema_version") == 1, f"queue status schema version missing: {payload!r}")
        schema = (payload.get("outbox") or {}).get("schema") or {}
        expect(schema.get("status") == "ok", f"healthy schema was not reported: {payload!r}")
        expect((payload.get("outbox") or {}).get("integrity") == "ok", f"integrity was not checked: {payload!r}")
        with sqlite3.connect(str(db_path)) as connection:
            connection.execute("PRAGMA user_version = 3")
        process = subprocess.run(
            [sys.executable, str(status_path), "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(process.returncode == 2, f"future schema should be an operator error: {process.stdout!r}")
        payload = json.loads(process.stdout)
        expect(payload.get("status") == "error", f"future schema status was not error: {payload!r}")
        findings = []
        doctor._check_lifecycle_outbox(findings, taskdata, 300.0)
        schema_finding = next(item for item in findings if item.get("id") == "outbox.schema")
        expect(schema_finding.get("severity") == "error", f"Doctor missed future schema: {findings!r}")


def test_queue_status_json_ok_empty_taskdata():
    """Lifecycle outbox status should report ok for empty taskdata."""
    path = DEV_TOOLS / "nautical_queue_status.py"
    with tempfile.TemporaryDirectory() as td:
        process = subprocess.run(
            [sys.executable, str(path), "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(process.returncode == 0, f"queue status returned {process.returncode}: {process.stderr!r}")
        payload = json.loads((process.stdout or "").strip() or "{}")
        expect(payload.get("status") == "ok", f"unexpected queue status: {payload}")
        outbox = payload.get("outbox") or {}
        expect(outbox.get("states") == {}, f"unexpected lifecycle states: {payload}")
        expect((outbox.get("schema") or {}).get("status") == "absent", f"unexpected outbox schema: {payload}")


def test_queue_status_explicit_prune_reports_maintenance_result():
    """Retention cleanup is explicit and returns a structured maintenance result."""
    path = DEV_TOOLS / "nautical_queue_status.py"
    with tempfile.TemporaryDirectory() as td:
        process = subprocess.run(
            [sys.executable, str(path), "--taskdata", td, "--prune-acknowledged", "--json"],
            text=True,
            capture_output=True,
            timeout=8,
        )
        expect(process.returncode == 0, f"explicit queue maintenance failed: {process.stderr!r}")
        payload = json.loads(process.stdout)
        maintenance = payload.get("maintenance") or {}
        expect(maintenance.get("ok") is True, f"maintenance result was not successful: {payload!r}")
        expect(maintenance.get("removed") == 0, f"unexpected maintenance removal: {payload!r}")


def test_doctor_installation_json_and_verifier_contract():
    """Installation checks remain bounded and produce a concise report."""
    path = CORE_TOOLS / "nautical_doctor.py"
    with tempfile.TemporaryDirectory() as td:
        env = os.environ.copy()
        env.update(
            {
                "HOME": td,
                "TASKRC": os.path.join(td, ".taskrc"),
                "NAUTICAL_CONFIG": os.path.join(td, "missing-nautical.toml"),
                "TASKDATA": td,
            }
        )
        process = subprocess.run(
            [sys.executable, str(path), "--taskdata", td, "--json", "--installation-only"],
            text=True,
            capture_output=True,
            env=env,
            timeout=10.0,
        )
        expect(process.returncode in (0, 1, 2), f"doctor returned an invalid status: {process.returncode}: {process.stderr!r}")
        payload = json.loads((process.stdout or "").strip() or "{}")
        expect(payload.get("schema") == "nautical.doctor", f"doctor schema missing: {payload!r}")
        expect(payload.get("schema_version") == 1, f"doctor schema version missing: {payload!r}")
        expect(payload.get("scope") == "installation", f"doctor installation scope missing: {payload!r}")
        expect(payload.get("counts") == {"tasks": 0, "nautical_tasks": 0, "chains": 0}, "installation check audited tasks")
        expect(payload.get("outbox") == {}, "installation check audited the lifecycle outbox")
        from nautical_core.tools.nautical_install_verify import build_report, render

        launcher = Path(td) / "nautical"
        launcher.write_text("#!/bin/sh\n", encoding="utf-8")
        launcher.chmod(0o700)
        verifier_payload = {
            "taskdata": td,
            "operator_findings": [
                {"code": "taskwarrior.version", "domain": "taskwarrior", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "taskdata.access", "domain": "taskdata", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "install.runtime", "domain": "install", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "hook.add", "domain": "hook", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "hook.modify", "domain": "hook", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "hook.exit", "domain": "hook", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "uda.anchor", "domain": "uda", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "config.timezone", "domain": "config", "severity": "info", "actionability": "informational", "message": "ok"},
                {"code": "chains.carry.child_relative_offset", "domain": "chains", "severity": "error", "actionability": "actionable", "message": "historical", "guidance": "ignore history"},
            ],
        }
        report = build_report(verifier_payload, platform="Termux", launcher=launcher)
        expect(report.get("status") == "passed", f"operational findings leaked into installation status: {report!r}")
        expect(not report.get("manual_actions"), f"operational findings leaked into install actions: {report!r}")
        rendered = io.StringIO()
        with contextlib.redirect_stdout(rendered):
            render(report)
        expect("\x1b[" not in rendered.getvalue(), "redirected installation report contains terminal styling")
        canonical_payload = dict(verifier_payload)
        canonical_payload["operator_findings"] = [
            item for item in verifier_payload["operator_findings"] if not str(item.get("code") or "").startswith("uda.")
        ]
        canonical_report = build_report(canonical_payload, platform="Linux", launcher=launcher)
        expect(canonical_report.get("status") == "passed", f"healthy canonical evidence was rejected: {canonical_report!r}")


TESTS = (
    test_health_check_json_ok_empty_taskdata,
    test_queue_status_and_doctor_report_schema_health,
    test_queue_status_json_ok_empty_taskdata,
    test_queue_status_explicit_prune_reports_maintenance_result,
    test_doctor_installation_json_and_verifier_contract,
    test_query_process_boundary_emits_one_json_document,
    test_operator_processes_concurrent_contracts_share_taskdata_safely,
    test_query_installed_layout_runs_outside_checkout,
    test_navigator_import_and_help_are_noninteractive_without_rich,
)
