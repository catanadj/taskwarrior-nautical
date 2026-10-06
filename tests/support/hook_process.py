"""Reusable subprocess and temporary-Taskdata fixture for hook contracts."""

from __future__ import annotations

import os
from collections.abc import Sequence
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
        timeout: float = 15.0,
        coverage_file: str | Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return self.run_script(
            ROOT / hook,
            payload,
            diagnostics=diagnostics,
            extra_environment=extra_environment,
            timeout=timeout,
            coverage_file=coverage_file,
        )

    def run_python_command(
        self,
        command: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        input: str | None = None,
        cwd: str | Path | None = None,
        clear_environment: Sequence[str] = (),
        instrument_coverage: bool | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float = 15.0,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if len(command) < 2:
            raise ValueError("Python command must include an executable and script")
        if not capture_output or not text:
            raise ValueError("Python subprocess coverage requires captured text output")
        if instrument_coverage is None:
            if command[1] == "-m":
                instrument_coverage = len(command) > 2 and command[2].startswith(
                    "nautical_core."
                )
            else:
                script_path = Path(command[1]).resolve()
                instrument_coverage = script_path.is_relative_to(ROOT)
        environment = os.environ.copy()
        environment.update(env or {})
        process = self.run_script(
            command[1],
            input,
            cwd=cwd,
            diagnostics=environment.get("NAUTICAL_DIAG") == "1",
            extra_environment=environment,
            script_arguments=tuple(command[2:]),
            timeout=timeout,
            clear_environment=clear_environment,
            instrument_coverage=instrument_coverage,
        )
        if check and process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, command, process.stdout, process.stderr
            )
        return process

    def run_python_code(
        self,
        code: str,
        arguments: Sequence[str] = (),
        *,
        env: dict[str, str] | None = None,
        input: str | None = None,
        cwd: str | Path | None = None,
        clear_environment: Sequence[str] = (),
        instrument_coverage: bool = True,
        timeout: float = 15.0,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        script_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                dir=self.taskdata,
                encoding="utf-8",
                delete=False,
            ) as script_file:
                script_file.write(code)
                script_path = Path(script_file.name)
            return self.run_python_command(
                [sys.executable, str(script_path), *arguments],
                env=env,
                input=input,
                cwd=cwd,
                clear_environment=clear_environment,
                instrument_coverage=instrument_coverage,
                timeout=timeout,
                check=check,
            )
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)

    def run_script(
        self,
        script: str | Path,
        payload: str | None = "",
        *,
        diagnostics: bool = False,
        extra_environment: dict[str, str] | None = None,
        script_arguments: tuple[str, ...] = (),
        cwd: str | Path | None = None,
        clear_environment: Sequence[str] = (),
        instrument_coverage: bool = True,
        timeout: float = 15.0,
        coverage_file: str | Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if coverage_file is None and instrument_coverage:
            shared_coverage_dir = os.environ.get("NAUTICAL_SUBPROCESS_COVERAGE_DIR")
            if shared_coverage_dir:
                coverage_name = Path(os.environ.get("COVERAGE_FILE", ".coverage")).name
                coverage_file = Path(shared_coverage_dir) / coverage_name
        environment = os.environ.copy()
        environment.update(
            {
                "TASKDATA": str(self.taskdata),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TZ": "UTC",
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        for name in clear_environment:
            environment.pop(name, None)
        if diagnostics:
            environment["NAUTICAL_DIAG"] = "1"
        else:
            environment.pop("NAUTICAL_DIAG", None)
        command = [sys.executable]
        if coverage_file is not None:
            coverage_path = Path(coverage_file)
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            command.extend(
                [
                    "-m",
                    "coverage",
                    "run",
                    "--parallel-mode",
                    "--data-file",
                    str(coverage_path),
                ]
            )
        command.extend((str(script), *script_arguments))
        return subprocess.run(
            command,
            input=payload,
            cwd=cwd,
            text=True,
            capture_output=True,
            env=environment,
            timeout=timeout,
        )

    def combine_coverage(self, coverage_file: str | Path) -> None:
        import coverage

        coverage_path = Path(coverage_file)
        coverage.Coverage(data_file=str(coverage_path)).combine(
            data_paths=[str(coverage_path.parent)]
        )
