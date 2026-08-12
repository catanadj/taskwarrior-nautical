"""Canonical, immutable recurrence schedule compilation.

Compilation normalizes task recurrence fields once and produces a stable
fingerprint for task-scoped evaluation and derived-cache invalidation.  It does
not perform Taskwarrior I/O, provider lookup, or occurrence evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .recurrence_spec import RecurrenceSpec


COMPILER_SCHEMA_VERSION = 1


def _freeze_value(value: Any) -> Any:
    """Make parser output immutable while retaining a JSON-like shape."""
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) for item in value):
            return {key: _thaw_value(item) for key, item in value}
        return [_thaw_value(item) for item in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, ProviderInstruction):
        return value.to_dict()
    if isinstance(value, ProjectionInstruction):
        return value.to_dict()
    if isinstance(value, ScheduleLimits):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ProviderInstruction:
    """Typed source instructions for a compiled schedule."""

    kind: str
    anchor_file: str = ""
    cp: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "anchor_file": self.anchor_file, "cp": self.cp}


@dataclass(frozen=True, slots=True)
class ProjectionInstruction:
    """Typed time projection instructions extracted from parser output."""

    modifiers: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"modifiers": [_thaw_value(item) for item in self.modifiers]}


@dataclass(frozen=True, slots=True)
class ScheduleLimits:
    chain_max: int | None = None
    chain_until: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"chain_max": self.chain_max, "chain_until": self.chain_until}


class CompiledScheduleCache:
    """Bounded task/request-scoped reuse cache for compiled schedules."""

    def __init__(self, *, max_entries: int = 32) -> None:
        if max_entries <= 0:
            raise ValueError("compiled schedule cache capacity must be positive")
        self._max_entries = max_entries
        self._entries: dict[str, CompiledSchedule] = {}

    def get_or_compile(self, spec: RecurrenceSpec) -> "CompiledSchedule":
        if not isinstance(spec, RecurrenceSpec):
            raise TypeError("compiled schedule cache requires a RecurrenceSpec")
        key = _spec_cache_key(spec)
        existing = self._entries.get(key)
        if existing is not None:
            return existing
        compiled = CompiledSchedule.from_spec(spec)
        if len(self._entries) >= self._max_entries:
            self._entries.pop(next(iter(self._entries)))
        self._entries[key] = compiled
        return compiled

    def clear(self) -> None:
        self._entries.clear()


def _spec_cache_key(spec: RecurrenceSpec) -> str:
    context = spec.context
    payload = {
        "anchor": spec.anchor,
        "anchor_file": spec.anchor_file,
        "omit": spec.omit,
        "omit_file": spec.omit_file,
        "cp": spec.cp,
        "anchor_mode": spec.anchor_mode,
        "chain_max": spec.chain_max,
        "chain_until": spec.chain_until,
        "context": {
            "chain_id": context.chain_id,
            "timezone": _stable_context_value(context.timezone),
            "business_calendar": _stable_context_value(context.business_calendar),
            "astronomy_config": _stable_context_value(context.astronomy_config),
            "anchor_file_dir": context.anchor_file_dir,
            "namespace": context.namespace,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compile_normalized_parts(spec: RecurrenceSpec) -> dict[str, Any]:
    """Parse expression fields once and describe the provider-facing inputs."""
    if spec.cp and (spec.anchor or spec.anchor_file):
        raise ValueError("A compiled schedule cannot contain both cp and anchor providers.")
    if spec.chain_max is not None and spec.chain_max <= 0:
        raise ValueError("chainMax must be greater than zero in a compiled schedule.")
    if spec.anchor_mode not in {"skip", "all", "flex"}:
        raise ValueError(
            "anchor_mode must be 'skip', 'all', or 'flex' in a compiled schedule."
        )

    from .anchor_omit import validate_omit_expr_strict
    from .parser_api import (
        parse_anchor_expr_to_dnf_cached,
        resolve_anchor_presets,
        validate_anchor_expr_strict,
    )

    anchor_dnf = validate_anchor_expr_strict(spec.anchor) if spec.anchor else None
    if not spec.cp and not anchor_dnf and not spec.anchor_file:
        raise ValueError("A compiled schedule requires an anchor, anchor_file, or cp provider.")
    omit_dnf = None
    if spec.omit:
        omit_dnf = validate_omit_expr_strict(
            spec.omit,
            validate_anchor_expr_cached=parse_anchor_expr_to_dnf_cached,
            resolve_omit_presets=resolve_anchor_presets,
        )
    time_modifiers: list[Any] = []

    def collect_time(value: Any) -> None:
        if isinstance(value, Mapping):
            mods = value.get("mods")
            if isinstance(mods, Mapping) and mods.get("t") is not None:
                time_modifiers.append(mods.get("t"))
            for item in value.values():
                collect_time(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_time(item)

    collect_time(anchor_dnf)
    cp_tokens = None
    if spec.cp:
        from . import parse_cp_sequence_tokens

        cp_tokens = parse_cp_sequence_tokens(spec.cp)
    provider = ProviderInstruction(
        kind="cp" if spec.cp else ("anchor+file" if spec.anchor and spec.anchor_file else "anchor_file" if spec.anchor_file else "anchor"),
        anchor_file=spec.anchor_file,
        cp=spec.cp,
    )
    projection = ProjectionInstruction(
        modifiers=tuple(_freeze_value(item) for item in time_modifiers)
    )
    return {
        "anchor_dnf": _freeze_value(anchor_dnf),
        "omit_dnf": _freeze_value(omit_dnf),
        "provider": provider,
        "time_projection": projection,
        "limits": ScheduleLimits(chain_max=spec.chain_max, chain_until=spec.chain_until),
        "cp_tokens": _freeze_value(cp_tokens),
        "identity": spec.context.chain_id,
    }


def _stable_context_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _stable_context_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_context_value(item) for item in value]
    for attribute in ("key", "zone"):
        marker = getattr(value, attribute, None)
        if marker:
            return str(marker)
    fingerprint = getattr(value, "fingerprint", None)
    if callable(fingerprint):
        try:
            return str(fingerprint())
        except Exception:
            pass
    return f"{type(value).__module__}.{type(value).__qualname__}"


@dataclass(frozen=True, slots=True)
class CompiledSchedule:
    """Immutable canonical schedule consumed by the future evaluator service."""

    spec: RecurrenceSpec
    canonical: tuple[tuple[str, Any], ...]
    fingerprint: str
    normalized: tuple[tuple[str, Any], ...] = ()
    compiler_schema: int = COMPILER_SCHEMA_VERSION

    @classmethod
    def from_spec(cls, spec: RecurrenceSpec) -> "CompiledSchedule":
        if not isinstance(spec, RecurrenceSpec):
            raise TypeError("schedule compilation requires a RecurrenceSpec")
        if not spec.enabled:
            raise ValueError("cannot compile a task without a recurrence expression")
        context = spec.context
        normalized = _compile_normalized_parts(spec)
        payload = {
            "compiler_schema": COMPILER_SCHEMA_VERSION,
            "anchor": spec.anchor,
            "anchor_file": spec.anchor_file,
            "omit": spec.omit,
            "omit_file": spec.omit_file,
            "cp": spec.cp,
            "anchor_mode": spec.anchor_mode,
            "chain_max": spec.chain_max,
            "chain_until": spec.chain_until,
            "normalized": normalized,
            "context": {
                "chain_id": context.chain_id,
                "timezone": _stable_context_value(context.timezone),
                "business_calendar": _stable_context_value(context.business_calendar),
                "astronomy_config": _stable_context_value(context.astronomy_config),
                "anchor_file_dir": context.anchor_file_dir,
                "namespace": context.namespace,
            },
        }
        encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        canonical = _freeze_value(_jsonable(payload))
        fingerprint = "cs1-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        return cls(
            spec=spec,
            canonical=canonical,
            fingerprint=fingerprint,
            normalized=_freeze_value(_jsonable(normalized)),
        )

    @classmethod
    def from_task(cls, task: Mapping[str, Any], **kwargs: Any) -> "CompiledSchedule":
        return cls.from_spec(RecurrenceSpec.from_task(task, **kwargs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiler_schema": self.compiler_schema,
            "fingerprint": self.fingerprint,
            "schedule": _thaw_value(self.canonical),
        }

    @property
    def normalized_payload(self) -> dict[str, Any]:
        """Return normalized instructions for an evaluator session."""
        return _thaw_value(self.normalized)

    def to_diagnostic_json(self) -> str:
        """Return a stable diagnostic snapshot; it is not a runtime cache format."""
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @property
    def cache_key(self) -> str:
        """Namespaced key for derived state; schema changes naturally miss."""
        return f"compiled-schedule:{self.compiler_schema}:{self.fingerprint}"


__all__ = (
    "COMPILER_SCHEMA_VERSION",
    "CompiledSchedule",
    "ProviderInstruction",
    "ProjectionInstruction",
    "ScheduleLimits",
    "CompiledScheduleCache",
)
