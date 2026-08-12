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
    compiler_schema: int = COMPILER_SCHEMA_VERSION

    @classmethod
    def from_spec(cls, spec: RecurrenceSpec) -> "CompiledSchedule":
        if not isinstance(spec, RecurrenceSpec):
            raise TypeError("schedule compilation requires a RecurrenceSpec")
        if not spec.enabled:
            raise ValueError("cannot compile a task without a recurrence expression")
        context = spec.context
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
            "context": {
                "chain_id": context.chain_id,
                "timezone": _stable_context_value(context.timezone),
                "business_calendar": _stable_context_value(context.business_calendar),
                "astronomy_config": _stable_context_value(context.astronomy_config),
                "anchor_file_dir": context.anchor_file_dir,
                "namespace": context.namespace,
            },
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        canonical = tuple(sorted(payload.items(), key=lambda pair: pair[0]))
        fingerprint = "cs1-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
        return cls(spec=spec, canonical=canonical, fingerprint=fingerprint)

    @classmethod
    def from_task(cls, task: Mapping[str, Any], **kwargs: Any) -> "CompiledSchedule":
        return cls.from_spec(RecurrenceSpec.from_task(task, **kwargs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiler_schema": self.compiler_schema,
            "fingerprint": self.fingerprint,
            "schedule": dict(self.canonical),
        }


__all__ = ("COMPILER_SCHEMA_VERSION", "CompiledSchedule")
