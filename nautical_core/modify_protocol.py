"""Pure input protocol helpers for the on-modify hook."""

from __future__ import annotations

import json
from collections.abc import Callable


class ModifyProtocolError(ValueError):
    """Raised when the on-modify payload cannot be decoded safely."""


def task_uuid_or_empty(task: dict) -> str:
    if not isinstance(task, dict):
        return ""
    try:
        return str(task.get("uuid") or "").strip()
    except Exception:
        return ""


def validate_modify_pair(
    old: dict,
    new: dict,
    *,
    has_nautical_fields: Callable[[dict, dict], bool],
) -> tuple[dict, dict]:
    old_uuid = task_uuid_or_empty(old)
    new_uuid = task_uuid_or_empty(new)
    if not old_uuid or not new_uuid:
        raise ModifyProtocolError("Missing task UUID in on-modify input")
    if old_uuid != new_uuid:
        if not has_nautical_fields(old, new):
            return old, new
        raise ModifyProtocolError("Old and new task UUIDs differ")
    return old, new


def validate_single_modify_task(
    task: dict,
    *,
    has_nautical_fields: Callable[[dict, dict], bool],
) -> tuple[dict, dict]:
    if not task_uuid_or_empty(task):
        if not has_nautical_fields(task, task):
            return task, task
        raise ModifyProtocolError("Missing task UUID in on-modify input")
    return task, task


def decode_leading_json_objects(raw: str, max_objects: int = 2) -> tuple[list[object], int]:
    """Decode up to ``max_objects`` JSON values from a Taskwarrior stream."""
    decoder = json.JSONDecoder()
    idx = 0
    objs: list[object] = []
    n = len(raw)
    tries = 0
    loop_guard = 0
    max_loops = 10

    while idx < n and len(objs) < max_objects:
        loop_guard += 1
        if loop_guard > max_loops:
            raise ModifyProtocolError("Invalid JSON input: too many parse attempts")
        while idx < n and raw[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except Exception as exc:
            raise ModifyProtocolError("Invalid JSON input") from exc
        objs.append(obj)
        if end <= idx:
            tries += 1
            if tries >= 2:
                raise ModifyProtocolError("Invalid JSON input: parser made no progress")
            idx += 1
            continue
        idx = end

    return objs, idx
