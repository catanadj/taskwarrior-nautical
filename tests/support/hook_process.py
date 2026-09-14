"""Reusable subprocess and temporary-Taskdata fixture for hook contracts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class HookSubprocessFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._taskdata_context = tempfile.TemporaryDirectory()
        self.taskdata = self._taskdata_context.name

    def tearDown(self) -> None:
        self._taskdata_context.cleanup()

    def run_hook(
        self,
        hook: str,
        payload: str,
        *,
        diagnostics: bool = False,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "TASKDATA": self.taskdata,
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TZ": "UTC",
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        if diagnostics:
            environment["NAUTICAL_DIAG"] = "1"
        else:
            environment.pop("NAUTICAL_DIAG", None)
        return subprocess.run(
            [sys.executable, str(ROOT / hook)],
            input=payload,
            text=True,
            capture_output=True,
            env=environment,
            timeout=15,
        )
