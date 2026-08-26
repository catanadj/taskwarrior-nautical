"""The single Taskwarrior JSON codec for the domain-model migration."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import sys
from typing import Any, Mapping, MutableMapping, Sequence

try:
    from .task_models import TaskDraft, TaskObservation
except ImportError:  # standalone hook helper loading
    from task_models import TaskDraft as _StandaloneTaskDraft, TaskObservation as _StandaloneTaskObservation
    TaskDraft = _StandaloneTaskDraft  # type: ignore[misc]
    TaskObservation = _StandaloneTaskObservation  # type: ignore[misc]


TASK_CODEC_VERSION = 1
TASK_OBSERVATION_SCHEMA = "nautical.task.observation"
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


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

    @staticmethod
    def normalize_text(value: Any) -> str:
        """Normalize an optional Taskwarrior text field at the codec boundary."""
        text = str(value or "").strip()
        return "" if text.casefold() == "null" else text

    @staticmethod
    def sanitize_task_mapping(task: MutableMapping[str, Any], max_len: int = 1024) -> None:
        """Sanitize string fields at the Taskwarrior output boundary."""
        if not isinstance(task, dict):
            return
        for key, value in list(task.items()):
            if not isinstance(value, str):
                continue
            cleaned = _CONTROL_CHARS_RE.sub("", value)
            if max_len > 0 and len(cleaned) > max_len:
                if os.environ.get("NAUTICAL_DIAG") == "1":
                    print(
                        f"[nautical] UDA value truncated from {len(cleaned)} to {max_len} chars",
                        file=sys.stderr,
                    )
                cleaned = cleaned[:max_len]
            if cleaned != value and os.environ.get("NAUTICAL_DIAG") == "1":
                print(f"[nautical] UDA field truncated: {key}", file=sys.stderr)
            task[key] = cleaned

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

    def decode_object(
        self,
        text: str,
        *,
        source_query: str,
        snapshot_id: str = "",
        mutation_epoch: int = 0,
        command_count: int = 0,
    ) -> TaskObservation:
        """Decode one Taskwarrior JSON object through the same row contract."""
        if not isinstance(text, str) or not text.strip():
            raise TaskCodecError("Taskwarrior task input is empty")
        try:
            payload = json.loads(text, parse_constant=_reject_constant)
        except (TypeError, json.JSONDecodeError, TaskCodecError) as exc:
            raise TaskCodecError(f"Taskwarrior task input contains malformed JSON: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise TaskCodecError("Taskwarrior task input must be a JSON object")
        return self.decode_row(
            payload,
            source_query=source_query,
            snapshot_id=snapshot_id,
            mutation_epoch=mutation_epoch,
            command_count=command_count,
        )

    def decode_leading_rows(
        self,
        text: str,
        *,
        source_query: str,
        max_objects: int = 2,
    ) -> tuple[tuple[TaskObservation, ...], int, str]:
        """Decode concatenated Taskwarrior objects or one JSON row array."""
        if not isinstance(text, str) or not text.strip():
            return (), 0, "Invalid JSON input"
        decoder = json.JSONDecoder(parse_constant=_reject_constant)
        objects: list[Any] = []
        index = 0
        while index < len(text) and len(objects) < max(1, int(max_objects)):
            while index < len(text) and text[index].isspace():
                index += 1
            if index >= len(text):
                break
            try:
                value, end = decoder.raw_decode(text, index)
            except (TypeError, ValueError, TaskCodecError):
                return (), index, "Invalid JSON input"
            if end <= index:
                return (), index, "Invalid JSON input: parser made no progress"
            objects.append(value)
            index = end
        values = objects[0] if len(objects) == 1 and isinstance(objects[0], list) else objects
        try:
            rows = tuple(
                self.decode_row(value, source_query=source_query)
                for value in values
            )
        except (TypeError, TaskCodecError) as exc:
            return (), index, str(exc)
        return rows, index, ""

    def encode_task_import(self, value: TaskObservation | TaskDraft) -> str:
        """Encode a lossless observation or complete child draft."""
        if isinstance(value, TaskObservation):
            return _encode(value.to_mapping())
        if isinstance(value, TaskDraft):
            return _encode(value.to_mapping())
        raise TaskCodecError("task import requires a TaskObservation or TaskDraft")

    def encode_task_import_mapping(self, value: Mapping[str, Any]) -> str:
        """Encode a validated gateway payload at the Taskwarrior boundary."""
        if not isinstance(value, Mapping):
            raise TaskCodecError("task import payload requires an object")
        return _encode(value)

    def prepare_task_import_mapping(self, value: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one validated child payload for Taskwarrior import."""
        if not isinstance(value, Mapping):
            raise TaskCodecError("task import payload requires an object")
        output: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            if key in ("link", "chainMax"):
                try:
                    item = int(item)
                except (TypeError, ValueError):
                    pass
            output[str(key)] = item

        def compact(value: Any) -> Any:
            if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
                return value.replace("-", "").replace(":", "")
            return value

        for key in ("entry", "modified", "due", "end", "wait", "until", "scheduled"):
            if key in output:
                output[key] = compact(output[key])
        annotations = output.get("annotations")
        if isinstance(annotations, list):
            output["annotations"] = [
                {**annotation, "entry": compact(annotation.get("entry"))}
                if isinstance(annotation, Mapping) and annotation.get("entry")
                else annotation
                for annotation in annotations
            ]
        return output

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
