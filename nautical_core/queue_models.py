from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class QueueEntryError(ValueError):
    pass


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _clean_required_str(value: Any, message: str) -> str:
    text = _clean_str(value)
    if not text:
        raise QueueEntryError(message)
    return text


def _clean_attempts(value: Any) -> int:
    try:
        attempts = int(value or 0)
    except Exception as exc:
        raise QueueEntryError("invalid attempts") from exc
    if attempts < 0:
        raise QueueEntryError("invalid attempts")
    return attempts


@dataclass(frozen=True)
class SpawnQueueEntry:
    parent_uuid: str
    parent_nextlink: str
    child_short: str
    child: dict[str, Any]
    spawn_intent_id: str
    attempts: int = 0

    @classmethod
    def from_mapping(cls, entry: Mapping[str, Any]) -> "SpawnQueueEntry":
        if not isinstance(entry, Mapping):
            raise QueueEntryError("entry not object")
        spawn_intent_id = _clean_required_str(entry.get("spawn_intent_id"), "missing spawn_intent_id")
        child_obj = entry.get("child")
        if not isinstance(child_obj, Mapping):
            raise QueueEntryError("missing child object")
        child = dict(child_obj)
        child["uuid"] = _clean_required_str(child.get("uuid"), "missing child uuid")
        return cls(
            parent_uuid=_clean_str(entry.get("parent_uuid")),
            parent_nextlink=_clean_str(entry.get("parent_nextlink")),
            child_short=_clean_str(entry.get("child_short")),
            child=child,
            spawn_intent_id=spawn_intent_id,
            attempts=_clean_attempts(entry.get("attempts")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "parent_uuid": self.parent_uuid,
            "parent_nextlink": self.parent_nextlink,
            "child_short": self.child_short,
            "child": dict(self.child),
            "spawn_intent_id": self.spawn_intent_id,
        }
        if self.attempts:
            out["attempts"] = self.attempts
        return out


def normalize_spawn_queue_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    return SpawnQueueEntry.from_mapping(entry).to_dict()

@dataclass(slots=True)
class QueueRowClaimResult:
    rows: list[Any]
    lock_busy: bool = False
    err: str = ""


@dataclass(slots=True)
class QueueEntriesBatch:
    entries: list[dict[str, Any]]

    @property
    def entries_total(self) -> int:
        return len(self.entries)


@dataclass(slots=True)
class QueueWriteResult:
    ok: bool
    count: int
    lock_busy: bool = False
    err: str = ""

