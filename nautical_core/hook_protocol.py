from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, TypeAlias

from .task_models import TaskPayload

if TYPE_CHECKING:
    from .task_codec import TaskCodec
    from .task_models import TaskObservation

class _ProtocolCodecError(Exception):
    """Fallback error type used while the standalone gate is being loaded."""


TaskCodecError: type[Exception]
DEFAULT_TASK_CODEC: TaskCodec | None


try:
    from .task_codec import DEFAULT_TASK_CODEC as _imported_codec, TaskCodecError as _imported_error
    DEFAULT_TASK_CODEC = _imported_codec
    TaskCodecError = _imported_error
except ImportError:  # standalone hook bootstrap loader
    DEFAULT_TASK_CODEC = None
    TaskCodecError = _ProtocolCodecError


def _codec():
    """Resolve the strict codec only when a probe actually decodes a task."""
    global DEFAULT_TASK_CODEC, TaskCodecError
    if DEFAULT_TASK_CODEC is None:
        try:
            from .task_codec import DEFAULT_TASK_CODEC as _codec_a, TaskCodecError as _error_a
            imported_codec, imported_error = _codec_a, _error_a
        except ImportError:  # dynamically loaded protocol test/hook wrapper
            try:
                from nautical_core.task_codec import DEFAULT_TASK_CODEC as _codec_b, TaskCodecError as _error_b
                imported_codec, imported_error = _codec_b, _error_b
            except Exception:
                from task_codec import DEFAULT_TASK_CODEC as _codec_c, TaskCodecError as _error_c
                imported_codec, imported_error = _codec_c, _error_c

        DEFAULT_TASK_CODEC = imported_codec
        TaskCodecError = imported_error
    return DEFAULT_TASK_CODEC


MAX_JSON_BYTES = 10 * 1024 * 1024

_ADD_NAUTICAL_FIELDS = (
    "anchor",
    "anchor_file",
    "anchor_mode",
    "bc",
    "cp",
    "chainID",
    "chainMax",
    "chainUntil",
    "omit",
    "omit_file",
)
_MODIFY_RECURRENCE_FIELDS = ("anchor", "anchor_file", "bc", "cp", "omit", "omit_file")
_MODIFY_CHAIN_FIELDS = ("chainID", "nextLink", "prevLink", "link")
_MODIFY_SAFE_ORDINARY_FIELDS = frozenset(
    (
        "description",
        "project",
        "priority",
        "tags",
        "annotations",
        "depends",
        "modified",
        "urgency",
        "start",
        "parent",
        "blocks",
        "mask",
        "imask",
        "context",
    )
)
_MISSING = object()

class OnAddInput:
    __slots__ = ("task",)

    def __init__(self, task: TaskPayload) -> None:
        self.task = task


class OnModifyInput:
    __slots__ = ("old", "new")

    def __init__(self, old: TaskPayload, new: TaskPayload) -> None:
        self.old = old
        self.new = new


HookRequest: TypeAlias = OnAddInput | OnModifyInput


class ProtocolFailure:
    __slots__ = ("code", "message", "passthrough")

    def __init__(self, code: str, message: str, passthrough: TaskPayload | None = None) -> None:
        self.code = code
        self.message = message
        self.passthrough = passthrough


class HookProtocolResult:
    __slots__ = (
        "event", "raw_bytes", "raw_text", "old", "new", "observation",
        "old_observation", "new_observation", "is_nautical",
        "error", "error_kind", "request", "failure",
    )

    def __init__(
        self,
        *,
        event: str,
        raw_bytes: bytes,
        raw_text: str,
        old: TaskPayload | None = None,
        new: TaskPayload | None = None,
        observation: TaskObservation | None = None,
        old_observation: TaskObservation | None = None,
        new_observation: TaskObservation | None = None,
        is_nautical: bool = False,
        error: str = "",
        error_kind: str = "",
        request: HookRequest | None = None,
        failure: ProtocolFailure | None = None,
    ) -> None:
        self.event = event
        self.raw_bytes = raw_bytes
        self.raw_text = raw_text
        self.old = old
        self.new = new
        self.observation = observation
        self.old_observation = old_observation
        self.new_observation = new_observation
        self.is_nautical = bool(is_nautical)
        self.error = str(error or "")
        self.error_kind = str(error_kind or "")
        self.request = request
        self.failure = failure or (
            ProtocolFailure(self.error_kind or "invalid_input", self.error, new)
            if self.error else None
        )
        if self.request is None and not self.error:
            if event == "on-add" and isinstance(new, dict):
                self.request = OnAddInput(new)
            elif event == "on-modify" and isinstance(old, dict) and isinstance(new, dict):
                self.request = OnModifyInput(old, new)

    @property
    def valid(self) -> bool:
        return self.failure is None and self.request is not None

    @property
    def task(self) -> TaskPayload | None:
        return self.new


def _field_has_value(task: TaskPayload, field: str) -> bool:
    try:
        value = task.get(field)
    except Exception:
        return False
    if value is None:
        return False
    try:
        return bool(str(value).strip())
    except Exception:
        return False


def task_has_add_nautical_fields(task: dict | None) -> bool:
    if not isinstance(task, dict):
        return False
    return any(_field_has_value(task, field) for field in _ADD_NAUTICAL_FIELDS)


def task_has_modify_nautical_fields(task: dict | None) -> bool:
    if not isinstance(task, dict):
        return False
    fields = _MODIFY_RECURRENCE_FIELDS + _MODIFY_CHAIN_FIELDS
    return any(_field_has_value(task, field) for field in fields)


def _observation_has_fields(observation: TaskObservation, fields: tuple[str, ...]) -> bool:
    """Classify from decoded field states without thawing the row."""
    if observation is None or not callable(getattr(observation, "field", None)):
        return False
    return any(
        observation.field(field).presence.value == "value"
        and bool(str(observation.field(field).raw_value() or "").strip())
        for field in fields
    )


def is_safe_nautical_ordinary_modify(old: dict | None, new: dict | None) -> bool:
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if not (task_has_modify_nautical_fields(old) or task_has_modify_nautical_fields(new)):
        return False
    for field in set(old) | set(new):
        if old.get(field, _MISSING) != new.get(field, _MISSING) and field not in _MODIFY_SAFE_ORDINARY_FIELDS:
            return False
    return True


def _raw_input(raw: bytes | str) -> tuple[bytes, str]:
    if isinstance(raw, bytes):
        return raw, raw.decode("utf-8", errors="replace")
    text = str(raw or "")
    return text.encode("utf-8"), text


def _invalid(
    event: str,
    raw_bytes: bytes,
    raw_text: str,
    error: str,
    *,
    new: dict | None = None,
    error_kind: str = "invalid_input",
) -> HookProtocolResult:
    return HookProtocolResult(
        event=event,
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        new=new,
        error=error,
        error_kind=error_kind,
    )


def probe_on_add(raw: bytes | str, *, max_bytes: int = MAX_JSON_BYTES) -> HookProtocolResult:
    raw_bytes, raw_text = _raw_input(raw)
    if len(raw_bytes) > max(0, int(max_bytes)):
        return _invalid("on-add", raw_bytes, raw_text, f"on-add input exceeds {max_bytes} bytes")
    stripped = raw_text.strip()
    if not stripped:
        return _invalid("on-add", raw_bytes, raw_text, "on-add must receive a single JSON task")
    try:
        observation = _codec().decode_object(
            stripped,
            source_query="hook:on-add",
        )
        task = observation.to_mapping()
    except TaskCodecError:
        return _invalid("on-add", raw_bytes, raw_text, "on-add must receive a single JSON task")
    return HookProtocolResult(
        event="on-add",
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        new=task,
        observation=observation,
        is_nautical=_observation_has_fields(observation, _ADD_NAUTICAL_FIELDS),
    )


def _validate_modify_tasks(
    raw_bytes: bytes,
    raw_text: str,
    old: dict,
    new: dict,
    *,
    old_observation: TaskObservation | None = None,
    new_observation: TaskObservation | None = None,
    is_nautical: bool | None = None,
) -> HookProtocolResult:
    if is_nautical is None:
        is_nautical = task_has_modify_nautical_fields(old) or task_has_modify_nautical_fields(new)
    old_uuid = str(old.get("uuid") or "").strip()
    new_uuid = str(new.get("uuid") or "").strip()
    if not old_uuid or not new_uuid:
        if is_nautical:
            return _invalid(
                "on-modify",
                raw_bytes,
                raw_text,
                "Missing task UUID in on-modify input",
                new=new,
                error_kind="protocol",
            )
    elif old_uuid != new_uuid and is_nautical:
        return _invalid(
            "on-modify",
            raw_bytes,
            raw_text,
            "Old and new task UUIDs differ",
            new=new,
            error_kind="protocol",
        )
    return HookProtocolResult(
        event="on-modify",
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        old=old,
        new=new,
        old_observation=old_observation,
        new_observation=new_observation,
        is_nautical=is_nautical,
    )


def probe_on_modify(raw: bytes | str, *, max_bytes: int = MAX_JSON_BYTES) -> HookProtocolResult:
    raw_bytes, raw_text = _raw_input(raw)
    if len(raw_bytes) > max(0, int(max_bytes)):
        return _invalid("on-modify", raw_bytes, raw_text, f"on-modify input exceeds {max_bytes} bytes")
    if not raw_text.strip():
        return _invalid("on-modify", raw_bytes, raw_text, "on-modify must receive two JSON tasks")

    observations, index, decode_error = _codec().decode_leading_rows(
        raw_text,
        source_query="hook:on-modify",
        max_objects=2,
    )
    if decode_error:
        return _invalid("on-modify", raw_bytes, raw_text, decode_error, error_kind="protocol")
    if raw_text[index:].strip():
        return _invalid(
            "on-modify",
            raw_bytes,
            raw_text,
            "Invalid JSON input: trailing content",
            error_kind="protocol",
        )

    tasks = [observation.to_mapping() for observation in observations]

    if len(tasks) >= 2:
        return _validate_modify_tasks(
            raw_bytes,
            raw_text,
            tasks[0],
            tasks[-1],
            old_observation=observations[0],
            new_observation=observations[-1],
            is_nautical=(
                _observation_has_fields(observations[0], _MODIFY_RECURRENCE_FIELDS + _MODIFY_CHAIN_FIELDS)
                or _observation_has_fields(observations[-1], _MODIFY_RECURRENCE_FIELDS + _MODIFY_CHAIN_FIELDS)
            ),
        )
    if len(tasks) == 1:
        return _validate_modify_tasks(
            raw_bytes,
            raw_text,
            tasks[0],
            tasks[0],
            old_observation=observations[0],
            new_observation=observations[0],
            is_nautical=_observation_has_fields(
                observations[0], _MODIFY_RECURRENCE_FIELDS + _MODIFY_CHAIN_FIELDS,
            ),
        )
    return _invalid("on-modify", raw_bytes, raw_text, "on-modify must receive two JSON tasks")


def read_stdin_bytes(*, max_bytes: int = MAX_JSON_BYTES, stream=None) -> bytes:
    source = stream if stream is not None else sys.stdin
    reader = getattr(source, "buffer", source)
    raw = reader.read(max(0, int(max_bytes)) + 1)
    if isinstance(raw, bytes):
        return raw
    return str(raw or "").encode("utf-8")


def read_on_add(*, max_bytes: int = MAX_JSON_BYTES, stream=None) -> HookProtocolResult:
    return probe_on_add(read_stdin_bytes(max_bytes=max_bytes, stream=stream), max_bytes=max_bytes)


def read_on_modify(*, max_bytes: int = MAX_JSON_BYTES, stream=None) -> HookProtocolResult:
    return probe_on_modify(read_stdin_bytes(max_bytes=max_bytes, stream=stream), max_bytes=max_bytes)


def emit_passthrough_json(task: dict | None, *, stream=None) -> None:
    target = stream if stream is not None else sys.stdout
    target.write(json.dumps(task if isinstance(task, dict) else {}, ensure_ascii=False))
    try:
        target.flush()
    except Exception:
        pass


__all__ = (
    "HookRequest",
    "HookProtocolResult",
    "MAX_JSON_BYTES",
    "OnAddInput",
    "OnModifyInput",
    "ProtocolFailure",
    "TaskPayload",
    "emit_passthrough_json",
    "is_safe_nautical_ordinary_modify",
    "probe_on_add",
    "probe_on_modify",
    "read_on_add",
    "read_on_modify",
    "read_stdin_bytes",
    "task_has_add_nautical_fields",
    "task_has_modify_nautical_fields",
)
