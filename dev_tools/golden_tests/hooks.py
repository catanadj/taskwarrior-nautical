"""Hook-protocol golden tests extracted from the legacy runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def _run(hook: str, payload: str, *, environment: dict[str, str] | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env.setdefault("TZ", "UTC")
    if environment:
        env.update(environment)
    return subprocess.run(
        [sys.executable, str(ROOT / hook)],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=15,
    )


def _assert_one_json(process: subprocess.CompletedProcess[str]) -> dict[str, object]:
    if process.returncode != 0:
        raise AssertionError(process.stderr)
    if len(process.stdout.splitlines()) != 1:
        raise AssertionError(f"expected one stdout line, got {process.stdout!r}")
    value = json.loads(process.stdout)
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object, got {value!r}")
    return value


def test_hook_stdout_strict_json_with_diag_on_add():
    """on-add must keep stdout JSON-only even when diagnostics are enabled."""
    with tempfile.TemporaryDirectory() as temporary:
        process = _run(
            "on-add.nautical",
            json.dumps(
                {
                    "uuid": "00000000-0000-4000-8000-000000000333",
                    "description": "hook test on-add strict stdout",
                    "status": "pending",
                    "entry": "20250101T000000Z",
                }
            ),
            environment={
                "NAUTICAL_DIAG": "1",
                "NAUTICAL_BENCH_FORCE_FULL": "1",
                "NAUTICAL_CONFIG": str(Path(temporary) / "missing.toml"),
            },
        )
    _assert_one_json(process)
    if not process.stderr.strip():
        raise AssertionError("diagnostics enabled for on-add but stderr was empty")


def test_hook_stdout_strict_json_with_diag_on_modify():
    """on-modify must keep stdout JSON-only even when diagnostics are enabled."""
    process = _run(
        "on-modify.nautical",
        json.dumps({"uuid": "00000000-0000-4000-8000-000000000444", "status": "pending"}),
        environment={
            "NAUTICAL_DIAG": "1",
            "NAUTICAL_BENCH_FORCE_FULL": "1",
            "NAUTICAL_CONFIG": str(ROOT / "missing.toml"),
        },
    )
    _assert_one_json(process)
    if not process.stderr.strip():
        raise AssertionError("diagnostics enabled for on-modify but stderr was empty")


def test_hook_stdout_unicode_unescaped_on_add():
    """on-add passthrough stdout should preserve Unicode (ensure_ascii=False)."""
    process = _run(
        "on-add.nautical",
        json.dumps(
            {
                "uuid": "00000000-0000-4000-8000-000000000445",
                "status": "pending",
                "description": "Cafe ăîșț ✅",
            },
            ensure_ascii=False,
        ),
    )
    output = _assert_one_json(process)
    if output["description"] != "Cafe ăîșț ✅" or "\\u" in process.stdout:
        raise AssertionError(f"stdout should preserve raw Unicode: {process.stdout!r}")


def test_hook_stdout_unicode_unescaped_on_modify():
    """on-modify passthrough stdout should preserve Unicode (ensure_ascii=False)."""
    process = _run(
        "on-modify.nautical",
        json.dumps(
            {
                "uuid": "00000000-0000-4000-8000-000000000446",
                "status": "pending",
                "description": "Cafe ăîșț ✅",
            },
            ensure_ascii=False,
        ),
    )
    output = _assert_one_json(process)
    if output["description"] != "Cafe ăîșț ✅" or "\\u" in process.stdout:
        raise AssertionError(f"stdout should preserve raw Unicode: {process.stdout!r}")


def test_hook_stdout_empty_on_exit():
    """on-exit should not emit stdout (stdout is redirected to /dev/null)."""
    with tempfile.TemporaryDirectory() as temporary:
        process = _run(
            "on-exit.nautical",
            "",
            environment={
                "NAUTICAL_DIAG": "1",
                "TASKDATA": temporary,
                "NAUTICAL_CONFIG": str(Path(temporary) / "nautical.toml"),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
            },
        )
    if process.returncode != 0 or process.stdout:
        raise AssertionError(f"on-exit protocol changed: rc={process.returncode}, stdout={process.stdout!r}")


TESTS = (
    test_hook_stdout_strict_json_with_diag_on_add,
    test_hook_stdout_strict_json_with_diag_on_modify,
    test_hook_stdout_unicode_unescaped_on_add,
    test_hook_stdout_unicode_unescaped_on_modify,
    test_hook_stdout_empty_on_exit,
)
