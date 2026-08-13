"""Validated invocation context for Nautical's Taskwarrior integration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from enum import Enum
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Mapping, Protocol, TypeAlias
import uuid


class IntegrationContextError(RuntimeError):
    """Raised when a full Nautical invocation cannot be validated safely."""

    def __init__(self, stage: str, detail: str):
        self.stage = str(stage or "context").strip()
        self.detail = str(detail or "unavailable").strip()
        super().__init__(f"{self.stage}: {self.detail}")


class IntegrationAccess(str, Enum):
    READ_ONLY = "read_only"
    MUTATION = "mutation"


@dataclass(frozen=True, slots=True)
class DiagnosticEvent:
    stage: str
    message: str

    def __post_init__(self) -> None:
        if not str(self.stage or "").strip() or not str(self.message or "").strip():
            raise IntegrationContextError("diagnostics", "stage and message are required")


class DiagnosticsSink(Protocol):
    def emit(self, event: DiagnosticEvent) -> None: ...


@dataclass(frozen=True, slots=True)
class SilentDiagnostics:
    def emit(self, event: DiagnosticEvent) -> None:
        del event


@dataclass(frozen=True, slots=True)
class StderrDiagnostics:
    prefix: str = "nautical"

    def emit(self, event: DiagnosticEvent) -> None:
        sys.stderr.write(f"[{self.prefix}] {event.stage}: {event.message}\n")


class Clock(Protocol):
    def now_utc(self) -> datetime: ...


@dataclass(frozen=True, slots=True)
class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


ConfigScalar: TypeAlias = str | int | float | bool | None
FrozenConfigValue: TypeAlias = ConfigScalar | tuple["FrozenConfigValue", ...] | tuple[tuple[str, "FrozenConfigValue"], ...]


def _freeze_config_value(value: object) -> FrozenConfigValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_config_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    raise IntegrationContextError("configuration", f"unsupported value type: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class ValidatedNauticalConfiguration:
    source: str
    fingerprint: str
    scheduler_fingerprint: str
    timezone_name: str
    values: tuple[tuple[str, FrozenConfigValue], ...]

    def __post_init__(self) -> None:
        for field in ("source", "fingerprint", "scheduler_fingerprint", "timezone_name"):
            if not str(getattr(self, field) or "").strip():
                raise IntegrationContextError("configuration", f"{field} is required")
        values = tuple(self.values)
        keys = tuple(key for key, _value in values)
        if len(keys) != len(set(keys)):
            raise IntegrationContextError("configuration", "configuration keys must be unique")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, object],
        *,
        scheduler_fingerprint: str,
        timezone_name: str,
    ) -> "ValidatedNauticalConfiguration":
        raw_values = snapshot.get("values")
        if not isinstance(raw_values, Mapping):
            raise IntegrationContextError("configuration", "validated snapshot has no values")
        return cls(
            source=str(snapshot.get("source") or "").strip(),
            fingerprint=str(snapshot.get("fingerprint") or "").strip(),
            scheduler_fingerprint=str(scheduler_fingerprint or "").strip(),
            timezone_name=str(timezone_name or "").strip(),
            values=tuple(sorted((str(key), _freeze_config_value(value)) for key, value in raw_values.items())),
        )


@dataclass(frozen=True, slots=True)
class IntegrationContext:
    taskdata: Path
    taskdata_source: str
    command_prefix: tuple[str, ...]
    configuration: ValidatedNauticalConfiguration
    local_timezone: tzinfo
    diagnostics: DiagnosticsSink
    clock: Clock
    invocation_id: str
    command_budget: int
    access: IntegrationAccess

    def __post_init__(self) -> None:
        taskdata = Path(self.taskdata)
        if not taskdata.is_absolute():
            raise IntegrationContextError("taskdata", "resolved path must be absolute")
        object.__setattr__(self, "taskdata", taskdata)
        if not str(self.taskdata_source or "").strip():
            raise IntegrationContextError("taskdata", "resolution source is required")
        prefix = tuple(str(item) for item in self.command_prefix)
        if not prefix or not prefix[0].strip():
            raise IntegrationContextError("task_binary", "command prefix is empty")
        object.__setattr__(self, "command_prefix", prefix)
        if not isinstance(self.configuration, ValidatedNauticalConfiguration):
            raise IntegrationContextError("configuration", "validated configuration is required")
        if not isinstance(self.local_timezone, tzinfo):
            raise IntegrationContextError("timezone", "validated timezone is required")
        if not callable(getattr(self.diagnostics, "emit", None)):
            raise IntegrationContextError("diagnostics", "diagnostic sink is invalid")
        if not callable(getattr(self.clock, "now_utc", None)):
            raise IntegrationContextError("clock", "clock is invalid")
        invocation_id = str(self.invocation_id or "").strip()
        if not invocation_id:
            raise IntegrationContextError("invocation", "identity is required")
        object.__setattr__(self, "invocation_id", invocation_id)
        if isinstance(self.command_budget, bool) or not isinstance(self.command_budget, int) or self.command_budget < 1:
            raise IntegrationContextError("command_budget", "must be a positive integer")
        try:
            access = IntegrationAccess(self.access)
        except (TypeError, ValueError) as exc:
            raise IntegrationContextError("access", "invalid integration access") from exc
        object.__setattr__(self, "access", access)

    @property
    def mutation_capable(self) -> bool:
        return self.access is IntegrationAccess.MUTATION


def _validated_task_binary(task_binary: str, env: Mapping[str, str]) -> str:
    requested = str(task_binary or "").strip()
    if not requested:
        raise IntegrationContextError("task_binary", "Taskwarrior executable is empty")
    env_path = str(env.get("PATH") or "")
    resolved = shutil.which(requested, path=env_path)
    if resolved is None and os.path.dirname(requested):
        candidate = Path(requested).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = str(candidate.resolve())
    if resolved is None:
        raise IntegrationContextError("task_binary", f"Taskwarrior executable was not found: {requested}")
    return str(Path(resolved).resolve())


def _validated_taskdata(path_value: str, *, mutation: bool) -> Path:
    try:
        taskdata = Path(path_value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IntegrationContextError("taskdata", f"resolved directory is unavailable: {exc}") from exc
    if not taskdata.is_dir():
        raise IntegrationContextError("taskdata", f"resolved path is not a directory: {taskdata}")
    if not os.access(taskdata, os.R_OK | os.X_OK):
        raise IntegrationContextError("taskdata", f"directory is not readable: {taskdata}")
    if mutation and not os.access(taskdata, os.W_OK):
        raise IntegrationContextError("taskdata", f"directory is not writable: {taskdata}")
    return taskdata


def build_integration_context(
    *,
    core: ModuleType,
    argv: tuple[str, ...] = (),
    env: Mapping[str, str] | None = None,
    tw_dir: str = "~/.task",
    task_binary: str = "task",
    access: IntegrationAccess = IntegrationAccess.READ_ONLY,
    command_budget: int = 64,
    diagnostics: DiagnosticsSink | None = None,
    clock: Clock | None = None,
    invocation_id: str | None = None,
    discover_taskdata: bool = False,
) -> IntegrationContext:
    """Resolve and validate all invocation state exactly once."""
    env_map = dict(os.environ if env is None else env)
    try:
        access = IntegrationAccess(access)
    except (TypeError, ValueError) as exc:
        raise IntegrationContextError("access", "invalid integration access") from exc

    task_path = _validated_task_binary(task_binary, env_map)
    resolved_tw_dir = tw_dir
    discovered_taskdata = False
    explicit_taskdata = bool(str(env_map.get("TASKDATA") or "").strip()) or any(
        str(arg).startswith(("data:", "data=", "data.location:", "data.location="))
        for arg in argv
    )
    if discover_taskdata and not explicit_taskdata:
        from . import task_command

        result = task_command.run_task_command(
            task_path,
            ("rc.hooks=off", "rc.verbose=nothing", "_get", "rc.data.location"),
            env=env_map,
            timeout=10.0,
            retry_locks=True,
        )
        if not result.ok:
            raise IntegrationContextError(
                "taskdata",
                task_command.failure_message(result, "Taskwarrior data location lookup"),
            )
        resolved_tw_dir = str(result.stdout or "").strip()
        if not resolved_tw_dir:
            raise IntegrationContextError("taskdata", "Taskwarrior data location is empty")
        discovered_taskdata = True

    resolver = getattr(core, "resolve_task_data_context", None)
    if not callable(resolver):
        raise IntegrationContextError("taskdata", "core resolver is unavailable")
    try:
        raw_taskdata, use_rc_data_location, source = resolver(
            argv=list(argv),
            env=env_map,
            tw_dir=resolved_tw_dir,
        )
    except Exception as exc:
        raise IntegrationContextError("taskdata", str(exc) or type(exc).__name__) from exc
    taskdata = _validated_taskdata(str(raw_taskdata), mutation=access is IntegrationAccess.MUTATION)

    command_prefix: tuple[str, ...] = (task_path,)
    if bool(use_rc_data_location):
        command_prefix += (f"rc.data.location={taskdata}",)

    reload_config = getattr(core, "reload_taskdata_config", None)
    snapshot_fn = getattr(core, "effective_config_snapshot", None)
    scheduling_error_fn = getattr(core, "scheduling_configuration_error", None)
    if not callable(reload_config) or not callable(snapshot_fn) or not callable(scheduling_error_fn):
        raise IntegrationContextError("configuration", "validated core configuration API is unavailable")
    try:
        reload_result = reload_config(taskdata)
        scheduling_error = str(scheduling_error_fn() or "").strip()
        if scheduling_error:
            raise RuntimeError(scheduling_error)
        snapshot = snapshot_fn()
    except Exception as exc:
        raise IntegrationContextError("configuration", str(exc) or type(exc).__name__) from exc
    if not isinstance(reload_result, Mapping) or not bool(reload_result.get("ok")):
        raise IntegrationContextError("configuration", "validated reload did not succeed")
    if not isinstance(snapshot, Mapping):
        raise IntegrationContextError("configuration", "validated snapshot is unavailable")

    local_timezone = getattr(core, "_LOCAL_TZ", None)
    timezone_name = str(getattr(core, "LOCAL_TZ_NAME", "") or "").strip()
    if not isinstance(local_timezone, tzinfo):
        raise IntegrationContextError("timezone", f"configured timezone is unavailable: {timezone_name or 'unknown'}")
    configuration = ValidatedNauticalConfiguration.from_snapshot(
        snapshot,
        scheduler_fingerprint=str(reload_result.get("scheduler_fingerprint") or ""),
        timezone_name=timezone_name,
    )

    diagnostic_sink: DiagnosticsSink
    if diagnostics is not None:
        diagnostic_sink = diagnostics
    elif env_map.get("NAUTICAL_DIAG") == "1":
        diagnostic_sink = StderrDiagnostics()
    else:
        diagnostic_sink = SilentDiagnostics()
    return IntegrationContext(
        taskdata=taskdata,
        taskdata_source="taskwarrior" if discovered_taskdata else str(source or "resolved"),
        command_prefix=command_prefix,
        configuration=configuration,
        local_timezone=local_timezone,
        diagnostics=diagnostic_sink,
        clock=clock or SystemClock(),
        invocation_id=invocation_id or uuid.uuid4().hex,
        command_budget=command_budget,
        access=access,
    )


def build_operator_context(
    *,
    core: ModuleType,
    task_binary: str,
    taskdata: str | None = None,
    env: Mapping[str, str] | None = None,
    access: IntegrationAccess = IntegrationAccess.READ_ONLY,
    command_budget: int = 256,
) -> IntegrationContext:
    """Build one validated context for a Taskwarrior-facing operator command."""
    argv = (f"data:{taskdata}",) if str(taskdata or "").strip() else ()
    return build_integration_context(
        core=core,
        argv=argv,
        env=env,
        tw_dir=str(taskdata or "~/.task"),
        task_binary=task_binary,
        access=access,
        command_budget=command_budget,
        discover_taskdata=not bool(argv),
    )


__all__ = (
    "Clock",
    "DiagnosticEvent",
    "DiagnosticsSink",
    "IntegrationAccess",
    "IntegrationContext",
    "IntegrationContextError",
    "SilentDiagnostics",
    "StderrDiagnostics",
    "SystemClock",
    "ValidatedNauticalConfiguration",
    "build_integration_context",
    "build_operator_context",
)
