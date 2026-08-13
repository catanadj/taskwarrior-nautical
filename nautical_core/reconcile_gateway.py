"""Taskwarrior mutation boundary used by reconcile and doctor.

The gateway deliberately contains Taskwarrior I/O only.  Recurrence decisions
and child payload construction remain in :mod:`chain_generation`.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any

from . import modify_spawn_prep, task_command


_STABLE_CHILD_UUID_NAMESPACE = uuid.UUID("1f4b2396-df58-5a32-a879-33f0d3fe711f")
_TW_JISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _strip_none_and_cast(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if key in {"link", "chainMax"}:
            try:
                value = int(value)
            except Exception:
                pass
        result[key] = value
    return result


def _normalise_datetime_fields(payload: dict[str, Any]) -> None:
    def compact(value: Any) -> Any:
        if isinstance(value, str) and _TW_JISO.fullmatch(value):
            return value.replace("-", "").replace(":", "")
        return value

    for key in ("entry", "modified", "due", "end", "wait", "until", "scheduled"):
        if key in payload and payload[key]:
            payload[key] = compact(payload[key])
    annotations = payload.get("annotations")
    if isinstance(annotations, list):
        for annotation in annotations:
            if isinstance(annotation, dict) and annotation.get("entry"):
                annotation["entry"] = compact(annotation["entry"])


@dataclass(slots=True)
class TaskwarriorMutationGateway:
    """Typed, bounded Taskwarrior mutation operations for operator tools."""

    core: Any
    task_bin: str = "task"

    def __getattr__(self, name: str) -> Any:
        """Expose read-only configuration helpers during migration."""
        return getattr(self.core, name)

    def safe_parse_datetime(self, value: Any):
        if not (value or ""):
            return None, None
        try:
            parsed = self.core.parse_dt_any(value)
        except Exception:
            return None, f"Unrecognized datetime format '{value}'"
        if parsed is None:
            return None, f"Unrecognized datetime format '{value}'"
        return parsed, None

    def _safe_parse_datetime(self, value: Any):
        """Compatibility spelling for existing reconcile diagnostics."""
        return self.safe_parse_datetime(value)

    def stable_child_uuid(self, parent_task: dict[str, Any] | None, child_task: dict[str, Any] | None) -> str:
        return self._stable_child_uuid(parent_task, child_task)

    def _stable_child_uuid(self, parent_task: dict[str, Any] | None, child_task: dict[str, Any] | None) -> str:
        return modify_spawn_prep.stable_child_uuid(
            parent_task,
            child_task,
            task_uuid_or_empty=lambda task: str(task.get("uuid") or "").strip(),
            coerce_int=self.core.coerce_int,
            stable_child_uuid_namespace=_STABLE_CHILD_UUID_NAMESPACE,
        )

    def _child_uuid_for_spawn(
        self,
        parent_task: dict[str, Any] | None,
        child_task: dict[str, Any] | None,
        env: dict[str, str],
    ) -> str:
        stable = self._stable_child_uuid(parent_task, child_task)
        return stable or str(uuid.uuid4())

    def _prepare_child(self, child: dict[str, Any], parent: dict[str, Any] | None) -> tuple[dict[str, Any], str, str]:
        env = os.environ.copy()
        return modify_spawn_prep.prepare_spawn_child_payload(
            child,
            parent,
            env,
            child_uuid_for_spawn=self._child_uuid_for_spawn,
            fmt_isoz=self.core.fmt_isoz,
            now_utc=self.core.now_utc,
            strip_none_and_cast=_strip_none_and_cast,
            normalise_datetime_fields=_normalise_datetime_fields,
        )

    def _spawn_child(self, child: dict[str, Any], parent: dict[str, Any] | None = None) -> tuple[str, set[str]]:
        payload, child_uuid, child_short = self._prepare_child(child, parent)
        result = task_command.run_task_command(
            self.task_bin,
            ["rc.hooks=off", "import", "-"],
            input_text=json.dumps(payload, ensure_ascii=False) + "\n",
            timeout=10.0,
            purpose="import reconciled child",
        )
        if not result.ok:
            raise RuntimeError(task_command.failure_message(result, "child import"))
        return child_short or child_uuid[:8], set()

    def spawn_child(self, child: dict[str, Any], parent: dict[str, Any] | None = None) -> tuple[str, set[str]]:
        return self._spawn_child(child, parent)


__all__ = ("TaskwarriorMutationGateway",)
