"""Hook-protocol golden tests extracted from the legacy runner."""

from __future__ import annotations

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
    test_hook_stdout_empty_on_exit,
)
