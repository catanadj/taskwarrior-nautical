from __future__ import annotations

import json
import sys


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
    ("description", "project", "priority", "tags", "annotations", "depends", "modified", "urgency")
)
_MISSING = object()


class HookProtocolResult:
    __slots__ = ("event", "raw_bytes", "raw_text", "old", "new", "is_nautical", "error", "error_kind")

    def __init__(
        self,
        *,
        event: str,
        raw_bytes: bytes,
        raw_text: str,
        old: dict | None = None,
        new: dict | None = None,
        is_nautical: bool = False,
        error: str = "",
        error_kind: str = "",
    ) -> None:
        self.event = event
        self.raw_bytes = raw_bytes
        self.raw_text = raw_text
        self.old = old
        self.new = new
        self.is_nautical = bool(is_nautical)
        self.error = str(error or "")
        self.error_kind = str(error_kind or "")

    @property
    def valid(self) -> bool:
        return not self.error and isinstance(self.new, dict)

    @property
    def task(self) -> dict | None:
        return self.new


def _field_has_value(task: dict, field: str) -> bool:
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
        task = json.loads(stripped)
    except Exception:
        return _invalid("on-add", raw_bytes, raw_text, "on-add must receive a single JSON task")
    if not isinstance(task, dict):
        return _invalid("on-add", raw_bytes, raw_text, "on-add must receive a single JSON task")
    return HookProtocolResult(
        event="on-add",
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        new=task,
        is_nautical=task_has_add_nautical_fields(task),
    )


def _decode_leading_json_objects(raw: str, *, max_objects: int = 2) -> tuple[list[object], int, str]:
    decoder = json.JSONDecoder()
    objects: list[object] = []
    index = 0
    length = len(raw)
    while index < length and len(objects) < max_objects:
        while index < length and raw[index].isspace():
            index += 1
        if index >= length:
            break
        try:
            obj, end = decoder.raw_decode(raw, index)
        except Exception:
            return objects, index, "Invalid JSON input"
        if end <= index:
            return objects, index, "Invalid JSON input: parser made no progress"
        objects.append(obj)
        index = end
    return objects, index, ""


def _validate_modify_tasks(
    raw_bytes: bytes,
    raw_text: str,
    old: dict,
    new: dict,
) -> HookProtocolResult:
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
        is_nautical=is_nautical,
    )


def probe_on_modify(raw: bytes | str, *, max_bytes: int = MAX_JSON_BYTES) -> HookProtocolResult:
    raw_bytes, raw_text = _raw_input(raw)
    if len(raw_bytes) > max(0, int(max_bytes)):
        return _invalid("on-modify", raw_bytes, raw_text, f"on-modify input exceeds {max_bytes} bytes")
    if not raw_text.strip():
        return _invalid("on-modify", raw_bytes, raw_text, "on-modify must receive two JSON tasks")

    objects, index, decode_error = _decode_leading_json_objects(raw_text, max_objects=2)
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

    if len(objects) == 1 and isinstance(objects[0], list):
        tasks = [item for item in objects[0] if isinstance(item, dict)]
    else:
        tasks = [item for item in objects if isinstance(item, dict)]

    if len(tasks) >= 2:
        return _validate_modify_tasks(raw_bytes, raw_text, tasks[0], tasks[-1])
    if len(tasks) == 1:
        return _validate_modify_tasks(raw_bytes, raw_text, tasks[0], tasks[0])
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
    "HookProtocolResult",
    "MAX_JSON_BYTES",
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
