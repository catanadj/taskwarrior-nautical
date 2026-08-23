"""The single Taskwarrior JSON codec for the domain-model migration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence

from .task_models import TaskObservation


TASK_CODEC_VERSION = 1
TASK_OBSERVATION_SCHEMA = "nautical.task.observation"


class TaskCodecError(ValueError):
    """A task row or external task payload cannot be represented safely."""


def _reject_constant(value: str) -> Any:
    raise TaskCodecError(f"non-finite JSON number is not supported: {value}")


def _json_safe(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TaskCodecError(f"non-finite number at {path}")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, path=f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise TaskCodecError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _encode(value: Any) -> str:
    try:
        safe = _json_safe(value)
        return json.dumps(safe, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, TaskCodecError):
            raise
        raise TaskCodecError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class TaskCodec:
    """Decode Taskwarrior rows once and encode each external contract explicitly."""

    version: int = TASK_CODEC_VERSION

    def __post_init__(self) -> None:
        if self.version != TASK_CODEC_VERSION:
            raise TaskCodecError(f"unsupported task codec version: {self.version}")

    def decode_row(
        self,
        row: Mapping[str, Any],
        *,
        source_query: str,
        snapshot_id: str = "",
        mutation_epoch: int = 0,
        command_count: int = 0,
    ) -> TaskObservation:
        if not isinstance(row, Mapping):
            raise TaskCodecError("Taskwarrior row must be a JSON object")
        try:
            return TaskObservation.from_mapping(
                row,
                source_query=source_query,
                snapshot_id=snapshot_id,
                mutation_epoch=mutation_epoch,
                command_count=command_count,
            )
        except (TypeError, ValueError) as exc:
            raise TaskCodecError(f"cannot decode Taskwarrior row: {exc}") from exc

    def decode_export(
        self,
        text: str,
        *,
        source_query: str,
        snapshot_id: str = "",
        mutation_epoch: int = 0,
        command_count: int = 0,
    ) -> tuple[TaskObservation, ...]:
        if not isinstance(text, str) or not text.strip():
            raise TaskCodecError("Taskwarrior export is empty")
        try:
            payload = json.loads(text, parse_constant=_reject_constant)
        except (TypeError, json.JSONDecodeError, TaskCodecError) as exc:
            raise TaskCodecError(f"Taskwarrior export contains malformed JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise TaskCodecError("Taskwarrior export must be a JSON array")
        return tuple(
            self.decode_row(
                row,
                source_query=source_query,
                snapshot_id=snapshot_id,
                mutation_epoch=mutation_epoch,
                command_count=command_count,
            )
            for row in payload
        )

    def encode_task_import(self, observation: TaskObservation) -> str:
        """Encode a lossless plain Taskwarrior import object."""
        if not isinstance(observation, TaskObservation):
            raise TaskCodecError("task import requires a TaskObservation")
        return _encode(observation.to_mapping())

    def encode_hook_stdout(self, task: Mapping[str, Any]) -> str:
        """Encode the strict plain JSON object emitted by a hook."""
        if not isinstance(task, Mapping):
            raise TaskCodecError("hook stdout requires a JSON object")
        return _encode(task)

    def encode_query_json(self, payload: Mapping[str, Any]) -> str:
        """Encode a versioned query response without changing its public shape."""
        if not isinstance(payload, Mapping):
            raise TaskCodecError("query response requires a JSON object")
        return _encode(payload)

    def encode_diagnostic(self, observation: TaskObservation) -> str:
        """Encode inspectable observation evidence in a separate versioned schema."""
        if not isinstance(observation, TaskObservation):
            raise TaskCodecError("diagnostic evidence requires a TaskObservation")
        payload = {
            "schema": TASK_OBSERVATION_SCHEMA,
            "version": self.version,
            "task": observation.to_mapping(),
            "semantic_fingerprint": observation.semantic_fingerprint,
            "provenance": {
                "source_query": observation.provenance.source_query,
                "snapshot_id": observation.provenance.snapshot_id,
                "mutation_epoch": observation.provenance.mutation_epoch,
                "command_count": observation.provenance.command_count,
            },
            "issues": [
                {
                    "field": issue.field,
                    "code": issue.code,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "raw": issue.raw,
                }
                for issue in observation.issues
            ],
        }
        return _encode(payload)


DEFAULT_TASK_CODEC = TaskCodec()


__all__ = (
    "DEFAULT_TASK_CODEC",
    "TASK_CODEC_VERSION",
    "TASK_OBSERVATION_SCHEMA",
    "TaskCodec",
    "TaskCodecError",
)
