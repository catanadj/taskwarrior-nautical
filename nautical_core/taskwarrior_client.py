"""Single process boundary for Taskwarrior commands."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import tempfile
import time
from typing import BinaryIO, Callable, Mapping, Protocol, Sequence

from .integration_models import CommandFailureKind, TaskCommand, TaskCommandResult


_BUSY_MARKERS = (
    "database is locked",
    "unable to lock",
    "resource temporarily unavailable",
    "another task is running",
    "lock file",
    "lockfile",
    "locked by",
)
_ABSENT_MARKERS = ("no matches",)
_RETRYABLE_KINDS = frozenset({CommandFailureKind.TIMEOUT, CommandFailureKind.BUSY})


@dataclass(frozen=True, slots=True)
class CommandObservation:
    """Content-free telemetry emitted after each process attempt."""

    purpose: str
    attempt: int
    duration: float
    kind: CommandFailureKind


class CommandObserver(Protocol):
    def observe(self, observation: CommandObservation) -> None: ...


@dataclass(frozen=True, slots=True)
class SilentCommandObserver:
    def observe(self, observation: CommandObservation) -> None:
        del observation


class TaskwarriorClient:
    """Execute bounded Taskwarrior commands under one failure policy."""

    def __init__(
        self,
        command_prefix: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        observer: CommandObserver | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        prefix = tuple(str(part) for part in command_prefix)
        if not prefix or not prefix[0].strip():
            raise ValueError("Taskwarrior command prefix is empty")
        self._command_prefix = prefix
        self._env = dict(os.environ if env is None else env)
        self._observer = observer or SilentCommandObserver()
        self._sleeper = sleeper

    def execute(
        self,
        args: Sequence[str],
        *,
        purpose: str,
        timeout: float,
        input_text: str | None = None,
        attempts: int = 1,
        retry_delay: float = 0.0,
        use_tempfiles: bool = False,
    ) -> TaskCommandResult:
        """Run a command, retrying only explicitly transient failures."""
        command = TaskCommand(
            self._command_prefix + tuple(str(part) for part in args),
            purpose,
            timeout,
            input_text,
        )
        max_attempts = max(1, int(attempts))
        delay = max(0.0, float(retry_delay))
        result: TaskCommandResult | None = None
        for attempt in range(1, max_attempts + 1):
            result = self._execute_attempt(command, attempt=attempt, use_tempfiles=use_tempfiles)
            self._observer.observe(
                CommandObservation(command.purpose, attempt, result.duration, result.kind)
            )
            if result.kind not in _RETRYABLE_KINDS or attempt >= max_attempts:
                return result
            if delay:
                self._sleeper(delay * (2 ** (attempt - 1)))
        assert result is not None
        return result

    def _execute_attempt(
        self,
        command: TaskCommand,
        *,
        attempt: int,
        use_tempfiles: bool,
    ) -> TaskCommandResult:
        started = time.monotonic()
        stdout_file, stderr_file = self._temporary_outputs(use_tempfiles)
        proc: subprocess.Popen[bytes] | None = None
        stdout = ""
        stderr = ""
        returncode = 126
        kind = CommandFailureKind.EXECUTION_FAILURE
        try:
            proc = subprocess.Popen(
                list(command.argv),
                stdin=subprocess.PIPE,
                stdout=stdout_file if stdout_file is not None else subprocess.PIPE,
                stderr=stderr_file if stderr_file is not None else subprocess.PIPE,
                env=self._env,
                close_fds=True,
            )
            try:
                out_bytes, err_bytes = proc.communicate(
                    input=command.input_text.encode("utf-8") if command.input_text is not None else None,
                    timeout=command.timeout,
                )
                stdout, stderr = self._collect_output(stdout_file, stderr_file, out_bytes, err_bytes)
                returncode = int(proc.returncode if proc.returncode is not None else 1)
                kind = self._classify(returncode, stdout, stderr)
            except subprocess.TimeoutExpired as exc:
                self._terminate(proc)
                out_bytes, err_bytes = proc.communicate()
                stdout, stderr = self._collect_output(
                    stdout_file,
                    stderr_file,
                    out_bytes or exc.stdout,
                    err_bytes or exc.stderr,
                )
                returncode = 124
                kind = CommandFailureKind.TIMEOUT
        except FileNotFoundError as exc:
            returncode = 127
            stderr = str(exc)
            kind = CommandFailureKind.MISSING_BINARY
        except OSError as exc:
            stderr = str(exc)
            kind = CommandFailureKind.EXECUTION_FAILURE
        finally:
            if stdout_file is not None:
                stdout_file.close()
            if stderr_file is not None:
                stderr_file.close()
        return TaskCommandResult(
            command,
            returncode,
            stdout,
            stderr,
            kind,
            attempt,
            time.monotonic() - started,
        )

    @staticmethod
    def _temporary_outputs(use_tempfiles: bool) -> tuple[BinaryIO | None, BinaryIO | None]:
        if not use_tempfiles:
            return None, None
        stdout_file: BinaryIO | None = None
        try:
            stdout_file = tempfile.TemporaryFile()
            return stdout_file, tempfile.TemporaryFile()
        except OSError:
            if stdout_file is not None:
                stdout_file.close()
            return None, None

    @staticmethod
    def _collect_output(
        stdout_file: BinaryIO | None,
        stderr_file: BinaryIO | None,
        stdout: bytes | None,
        stderr: bytes | None,
    ) -> tuple[str, str]:
        if stdout_file is not None:
            stdout_file.seek(0)
            stdout = stdout_file.read()
        if stderr_file is not None:
            stderr_file.seek(0)
            stderr = stderr_file.read()
        return (
            (stdout or b"").decode("utf-8", errors="replace"),
            (stderr or b"").decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _classify(returncode: int, stdout: str, stderr: str) -> CommandFailureKind:
        if returncode == 0:
            return CommandFailureKind.SUCCESS
        evidence = f"{stderr}\n{stdout}".lower()
        if any(marker in evidence for marker in _BUSY_MARKERS):
            return CommandFailureKind.BUSY
        if any(marker in evidence for marker in _ABSENT_MARKERS):
            return CommandFailureKind.ABSENT
        return CommandFailureKind.REJECTED

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes]) -> None:
        try:
            proc.terminate()
            proc.wait(timeout=0.2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass


__all__ = (
    "CommandObservation",
    "CommandObserver",
    "SilentCommandObserver",
    "TaskwarriorClient",
)
