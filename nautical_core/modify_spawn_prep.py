from __future__ import annotations

import json
import re
import uuid


_TW_JISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_UNREC_ATTR_RE = re.compile(r"Unrecognized attribute '([^']+)'", re.I)


class SpawnIdentityError(ValueError):
    """Raised when a persisted child has no canonical Nautical chain ID."""

    def __init__(self) -> None:
        super().__init__(
            "chainID is required for child spawn; UUID-derived legacy identities are unsupported"
        )


def sanitize_unknown_attrs(stderr: str, payload: dict) -> set[str]:
    """Remove Taskwarrior attributes rejected by the current version."""
    removed: set[str] = set()
    for match in _UNREC_ATTR_RE.finditer(stderr or ""):
        name = match.group(1)
        if name in payload:
            payload.pop(name, None)
            removed.add(name)
    return removed


def normalise_datetime_fields(obj: dict) -> None:
    """Use Taskwarrior's compact UTC format for imported datetime fields."""
    def to_tw_compact_isoz(value: str) -> str:
        if isinstance(value, str) and _TW_JISO.fullmatch(value):
            return value.replace("-", "").replace(":", "")
        return value

    for key in ("entry", "modified", "due", "end", "wait", "until", "scheduled"):
        if key in obj and obj[key]:
            obj[key] = to_tw_compact_isoz(obj[key])
    if isinstance(obj.get("annotations"), list):
        for annotation in obj["annotations"]:
            if isinstance(annotation, dict) and annotation.get("entry"):
                annotation["entry"] = to_tw_compact_isoz(annotation["entry"])


def strip_none_and_cast(obj: dict) -> dict:
    out = {}
    for key, value in obj.items():
        if value is None:
            continue
        if key in ("link", "chainMax"):
            try:
                value = int(value)
            except Exception:
                pass
        out[key] = value
    return out


def categorize_spawn_error(returncode: int, stderr: str) -> tuple[str, bool]:
    """Classify an import failure and whether retrying can help."""
    text = (stderr or "").lower()
    if returncode == 0:
        return "success", False
    if "unrecognized attribute" in text:
        return "attribute", True
    if "json" in text or "parse" in text:
        return "parse", False
    if "invalid" in text or "bad date" in text:
        return "validation", False
    if "error" in text or "failed" in text:
        return "taskwarrior", True
    return "unknown", True


def stable_child_uuid(
    parent_task: dict | None,
    child_task: dict | None,
    *,
    task_uuid_or_empty,
    coerce_int,
    stable_child_uuid_namespace,
) -> str:
    """Return a cross-device-stable UUID for a child slot when possible."""
    if not isinstance(parent_task, dict) or not isinstance(child_task, dict):
        return ""
    parent_uuid = task_uuid_or_empty(parent_task)
    if not parent_uuid:
        return ""
    link_no = coerce_int(child_task.get("link"), None)
    if link_no is None:
        return ""
    parent_chain_id = str(parent_task.get("chainID") or "").strip()
    child_chain_id = str(child_task.get("chainID") or "").strip()
    if not parent_chain_id:
        raise SpawnIdentityError()
    if child_chain_id and parent_chain_id != child_chain_id:
        raise SpawnIdentityError()
    chain_id = parent_chain_id
    kind = "anchor" if (parent_task.get("anchor") or "").strip() else "cp" if (parent_task.get("cp") or "").strip() else ""
    slot_key = json.dumps(
        {
            "chain_id": chain_id.lower(),
            "kind": kind,
            "link": int(link_no),
            "parent_uuid": parent_uuid.lower(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return str(uuid.uuid5(stable_child_uuid_namespace, slot_key))


def child_uuid_for_spawn(
    parent_task: dict | None,
    child_task: dict | None,
    env: dict,
    *,
    stable_child_uuid,
    reserve_child_uuid,
) -> str:
    stable = stable_child_uuid(parent_task, child_task)
    if stable:
        return stable
    return reserve_child_uuid(env)


def prepare_spawn_child_payload(
    child_task: dict,
    parent_task: dict | None,
    env: dict,
    *,
    child_uuid_for_spawn,
    fmt_isoz,
    now_utc,
    strip_none_and_cast,
    normalise_datetime_fields,
) -> tuple[object, str, str]:
    from nautical_core.task_codec import DEFAULT_TASK_CODEC
    from nautical_core.task_models import NauticalTask, TaskDraft

    parent_chain_id = str((parent_task or {}).get("chainID") or "").strip()
    child_chain_id = str(child_task.get("chainID") or "").strip()
    if parent_task is not None and not parent_chain_id:
        raise SpawnIdentityError()
    chain_id = parent_chain_id or child_chain_id
    if not chain_id:
        raise SpawnIdentityError()
    if parent_chain_id and child_chain_id and parent_chain_id != child_chain_id:
        raise SpawnIdentityError()
    child_uuid = child_uuid_for_spawn(parent_task, child_task, env)
    child_obj = dict(child_task)
    child_obj["uuid"] = child_uuid
    child_obj["chainID"] = chain_id
    if "entry" not in child_obj:
        child_obj["entry"] = fmt_isoz(now_utc())
    if "modified" not in child_obj:
        child_obj["modified"] = child_obj["entry"]

    child_short = child_uuid[:8]
    child_obj = strip_none_and_cast(child_obj)
    normalise_datetime_fields(child_obj)
    child_task = NauticalTask.from_observation(
        DEFAULT_TASK_CODEC.decode_row(child_obj, source_query="hook child import")
    )
    return TaskDraft.from_task(child_task), child_uuid, child_short
