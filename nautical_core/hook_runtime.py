from __future__ import annotations

import importlib
from dataclasses import dataclass, field
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Iterator, Protocol

_DIAG_REDACT_KEYS = frozenset({"description", "annotation", "annotations", "note", "notes"})


class DiagnosticEmitter(Protocol):
    def __call__(self, message: str) -> None: ...


def redact_diagnostic_message(msg: object, *, core: Any = None) -> str:
    """Redact task text from diagnostic messages without escaping Unicode."""
    raw = msg if isinstance(msg, str) else str(msg)
    redactor = getattr(core, "diag_log_redact", None) if core is not None else None
    if callable(redactor):
        try:
            redacted = redactor(raw)
            return redacted if isinstance(redacted, str) else str(redacted)
        except Exception:
            pass
    # Keep this boundary independent from the task/domain codec.  Redact
    # scalar JSON values in-place while preserving the original Unicode text.
    keys = "|".join(re.escape(key) for key in sorted(_DIAG_REDACT_KEYS))
    scalar = r'("(?:\\.|[^"\\])*"|null|true|false|-?\d+(?:\.\d+)?)'
    pattern = re.compile(rf'("(?:{keys})"\s*:\s*){scalar}')
    return pattern.sub(r'\1"[redacted]"', raw)


def emit_diagnostic(msg: object, *, hook_name: str, core: Any = None, taskdata: object = "") -> None:
    """Send one redacted diagnostic to the core sink or opt-in stderr."""
    safe_msg = redact_diagnostic_message(msg, core=core)
    if core is not None:
        event_factory = getattr(core, "DiagnosticEvent", None)
        event = event_factory.from_message(safe_msg, hook=hook_name) if event_factory is not None else safe_msg
        core.diag(event, hook_name, str(taskdata))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            import sys
            sys.stderr.write(f"[nautical] {safe_msg}\n")
        except Exception:
            pass


def emit_diagnostic_block(
    title: str,
    items: Iterable[tuple[object, object]] | None,
    *,
    hook_name: str,
    emit: DiagnosticEmitter,
    enabled: bool,
    columns: int = 3,
) -> None:
    """Format a bounded diagnostic block through the shared emitter."""
    if not enabled:
        return
    try:
        pairs = [f"{key}={value}" for key, value in (items or ())]
        emit(f"{title}:")
        step = max(1, int(columns or 1))
        for index in range(0, len(pairs), step):
            emit("  " + "  ".join(pairs[index:index + step]))
    except Exception:
        pass


class HookProfiler:
    """Small opt-in profiler shared by hooks; output is stderr-only."""

    def __init__(self, level: int = 0, import_ms: float | None = None) -> None:
        self.level = int(level or 0)
        self.enabled = self.level > 0
        self.import_ms = float(import_ms) if import_ms is not None else None
        self._t0 = time.perf_counter()
        self._events: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            self._events.append((name, (time.perf_counter() - started) * 1000.0))

    def add_ms(self, name: str, ms: float) -> None:
        if self.enabled:
            self._events.append((name, float(ms)))

    def emit(self) -> None:
        if not self.enabled:
            return
        lines = [f"[NAUTICAL_PROFILE] total={(time.perf_counter() - self._t0) * 1000.0:.1f}ms"]
        if self.import_ms is not None:
            lines.append(f"  import_core={self.import_ms:.1f}ms")
        lines.extend(f"  {name}={ms:.1f}ms" for name, ms in self._events)
        if self.level >= 2 and self._events:
            lines.append("  -- slowest --")
            lines.extend(f"  {name}={ms:.1f}ms" for name, ms in sorted(self._events, key=lambda item: item[1], reverse=True)[:8])
        sys.stderr.write("\n".join(lines) + "\n")


class HookIntegrationContextError(RuntimeError):
    """Retain the loaded core so hooks can explain context validation failures."""

    def __init__(self, core: Any, cause: Exception):
        self.core = core
        self.cause = cause
        self.stage = str(getattr(cause, "stage", "context") or "context")
        self.detail = str(getattr(cause, "detail", "") or cause or type(cause).__name__)
        super().__init__(f"{self.stage}: {self.detail}")


@dataclass(frozen=True, slots=True)
class HookRuntimeState:
    """Validated core/context state shared by every executable hook."""

    core: Any
    target: Path | None
    context: Any
    access: str

    @property
    def taskdata(self) -> Path:
        return Path(self.context.taskdata)

    @property
    def uses_rc_data_location(self) -> bool:
        return len(tuple(getattr(self.context, "command_prefix", ()))) > 1


@dataclass(slots=True)
class HookModuleAccess:
    namespace: dict[str, Any]
    module_specs: dict[str, tuple[str, str, str, str]]
    errors: dict[str, str] = field(default_factory=dict)

    def load_named_module(self, name: str) -> Any | None:
        cache_attr, failed_attr, _rel_name, import_name = self.module_specs[name]
        module = self.namespace.get(cache_attr)
        if module is not None:
            return module
        if self.namespace.get(failed_attr):
            return None
        try:
            module = importlib.import_module(import_name)
            self.namespace[cache_attr] = module
            return module
        except Exception as exc:
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            self.namespace[failed_attr] = True
            return None

    def require_loaded_module(self, module: Any, rel_name: str, error: str = "") -> Any:
        if module is None:
            detail = f" ({error})" if error else ""
            raise RuntimeError(f"nautical_core/{rel_name} is required{detail}")
        return module

    def module(self, name: str, *, required: bool = True) -> Any:
        module = self.load_named_module(name)
        if not required:
            return module
        rel_name = self.module_specs[name][2]
        return self.require_loaded_module(module, rel_name, self.errors.get(name, ""))


def build_hook_runtime_context(
    *,
    module_access: HookModuleAccess,
    hook_name: str,
    integration_context: Any,
    hook_dir: str,
    profile_level: int = 0,
    import_ms: float | None = None,
    business_calendar: Any = None,
) -> Any:
    # Keep hook implementation imports lightweight.  The UoW and its
    # Taskwarrior dependencies are needed only after a validated invocation
    # context exists.
    from .taskwarrior_uow import build_taskwarrior_uow

    hook_context = module_access.module("hook_context")
    uow = build_taskwarrior_uow(integration_context, env=os.environ)
    return hook_context.build_hook_runtime_context(
        hook_name=hook_name,
        integration=integration_context,
        uow=uow,
        hook_dir=hook_dir,
        profile_level=profile_level,
        import_ms=import_ms,
        business_calendar=business_calendar,
    )


def initialize_integration_context(
    *,
    module_access: HookModuleAccess,
    hook_bootstrap: Any,
    core_base: Path,
    argv: tuple[str, ...],
    tw_dir: str,
    access: str,
) -> HookRuntimeState:
    """Import core and construct the sole validated context for a full hook."""
    core, target, import_error = hook_bootstrap.import_core_package(core_base)
    if core is None:
        target_text = str(target or (core_base / "nautical_core" / "__init__.py"))
        if import_error is not None:
            raise RuntimeError(
                f"Failed to import nautical_core from {target_text}: "
                f"{type(import_error).__name__}: {import_error}"
            ) from import_error
        raise ModuleNotFoundError(
            "nautical_core package not found. Expected nautical_core/__init__.py "
            f"in ~/.task or NAUTICAL_CORE_PATH (resolved base: {core_base})"
        )
    context_module = module_access.module("integration_context")
    try:
        context = context_module.build_integration_context(
            runtime=context_module.IntegrationRuntime.from_compatibility_facade(core),
            argv=argv,
            env=os.environ,
            tw_dir=tw_dir,
            task_binary=os.environ.get("NAUTICAL_BENCH_TASK_BIN", "task"),
            access=context_module.IntegrationAccess(access),
        )
    except Exception as exc:
        raise HookIntegrationContextError(core, exc) from exc
    return HookRuntimeState(core=core, target=target, context=context, access=access)
