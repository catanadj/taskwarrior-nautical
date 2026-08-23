"""Shared recurrence-chain generation for hooks and operator tools.

This module owns the pure part of creating a successor link: resolving the
next recurrence timestamp and constructing the child task payload.  Taskwarrior
I/O remains outside this boundary so reconcile and hooks can share identical
recurrence decisions without importing one another's implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import uuid
from typing import Any, Mapping, MutableMapping

from .scheduler_service import SchedulerService
from .recurrence_context import RecurrenceContext
from .recurrence_spec import normalize_recurrence_text
from .task_models import TaskDraft, NauticalTask


_STABLE_CHILD_UUID_NAMESPACE = uuid.UUID("1f4b2396-df58-5a32-a879-33f0d3fe711f")


_RESERVED_DROP = frozenset(
    {
        "id",
        "uuid",
        "urgency",
        "status",
        "modified",
        "start",
        "end",
        "mask",
        "imask",
        "parent",
        "recur",
        "rc",
        "nextLink",
    }
)
_RESERVED_OVERRIDE = frozenset({"due", "entry", "status", "chain", "prevLink", "link"})
_UDA_CARRY_SKIP_LOWER = frozenset(
    {
        "id",
        "uuid",
        "urgency",
        "status",
        "modified",
        "start",
        "end",
        "mask",
        "imask",
        "parent",
        "recur",
        "rc",
        "nextlink",
        "prevlink",
        "link",
        "chain",
        "chainmax",
        "chainuntil",
        "chainid",
        "cp",
        "anchor",
        "anchor_mode",
        "bc",
        "due",
        "entry",
        "wait",
        "scheduled",
        "until",
    }
)


ChildMetadata = dict[str, Any]
CpChildDueResult = tuple[datetime | None, ChildMetadata | None]
AnchorChildDueResult = tuple[datetime | None, ChildMetadata | None, Any]


class CarryFieldError(RuntimeError):
    """Raised when a child carry field cannot be reconstructed safely."""

    def __init__(self, field: str, reason: str):
        self.field = str(field or "carry")
        self.reason = str(reason or "unknown carry failure")
        super().__init__(f"{self.field} carry failed: {self.reason}")


class ChainIdentityError(ValueError):
    """Raised when generation is requested without Nautical chain identity."""

    def __init__(self) -> None:
        super().__init__(
            "chainID is required for chain generation; UUID-derived legacy identities are unsupported"
        )


@dataclass(slots=True)
class ChainGenerationService:
    """Context-bound recurrence and child-payload generator.

    ``core`` is the configured Nautical facade.  The service deliberately
    accepts it as a dependency instead of importing hook modules, which keeps
    reconcile and doctor independent from the on-modify implementation.
    """

    core: Any
    recurrence_update_udas: tuple[str, ...] = ()
    debug_wait_sched: bool = False
    wait_sched_debug: MutableMapping[str, dict[str, Any]] | None = None
    _evaluator_cache: dict[tuple[Any, ...], SchedulerService] = field(
        default_factory=dict,
        repr=False,
    )

    @classmethod
    def from_core(
        cls,
        core: Any,
        *,
        recurrence_update_udas: tuple[str, ...] | list[str] = (),
        debug_wait_sched: bool = False,
        wait_sched_debug: MutableMapping[str, dict[str, Any]] | None = None,
    ) -> "ChainGenerationService":
        return cls(
            core=core,
            recurrence_update_udas=tuple(
                str(value) for value in recurrence_update_udas if str(value).strip()
            ),
            debug_wait_sched=bool(debug_wait_sched),
            wait_sched_debug=wait_sched_debug,
        )

    @classmethod
    def from_hook(cls, hook: Any) -> "ChainGenerationService":
        """Adapt a hook object using its configured core only.

        Generation decisions must stay owned by this service.  In particular,
        do not capture private callbacks from ``modify_impl`` here: operator
        tools and reconcile must not route back through the heavy hook module.
        """
        core = getattr(hook, "core", None)
        if core is None:
            raise TypeError("chain generation requires a configured Nautical core")
        service = cls.from_core(
            core,
            recurrence_update_udas=tuple(getattr(hook, "_RECURRENCE_UPDATE_UDAS", ()) or ()),
            debug_wait_sched=bool(getattr(hook, "_DEBUG_WAIT_SCHED", False)),
            wait_sched_debug=getattr(hook, "_LAST_WAIT_SCHED_DEBUG", None),
        )
        return service

    def safe_parse_datetime(self, value: Any) -> tuple[datetime | None, str | None]:
        if not (value or ""):
            return None, None
        try:
            parsed = self.core.parse_dt_any(value)
        except Exception:
            return None, f"Unrecognized datetime format '{value}'"
        if parsed is None:
            return None, f"Unrecognized datetime format '{value}'"
        return parsed, None

    @staticmethod
    def _require_chain_id(task: NauticalTask) -> str:
        if not isinstance(task, NauticalTask):
            raise TypeError("chain generation requires a validated NauticalTask")
        chain_id = task.identity.chain_id.value
        if not chain_id:
            raise ChainIdentityError()
        return chain_id

    def _task_scheduler(self, task: NauticalTask) -> SchedulerService:
        identity = self._require_chain_id(task)
        values = task.observation.to_mapping()
        key = (
            identity,
            str(values.get("modified") or ""),
            str(values.get("anchor") or ""),
            str(values.get("anchor_file") or ""),
            str(values.get("omit") or ""),
            str(values.get("omit_file") or ""),
            str(values.get("anchor_mode") or ""),
            str(values.get("chainUntil") or ""),
            str(values.get("bc") or ""),
        )
        cached = self._evaluator_cache.get(key)
        if cached is not None:
            return cached
        context = RecurrenceContext.from_observation(
            task.observation,
            timezone=getattr(self.core, "_LOCAL_TZ", None),
            business_calendar=self.core.business_calendar_for_task(values),
            astronomy_config=getattr(self.core, "ASTRONOMY_CONFIG", None),
            anchor_file_dir=getattr(self.core, "ANCHOR_FILE_DIR", ""),
        )
        service = SchedulerService.from_observation(task.observation, context=context)
        self._evaluator_cache[key] = service
        return service

    def _local(self, value: datetime) -> datetime:
        return self.core.to_local(value)

    def _anchor_parent_local_times(
        self, parent: Mapping[str, Any]
    ) -> tuple[datetime | None, datetime | None, datetime | None]:
        end_dt, error = self.safe_parse_datetime(parent.get("end"))
        if error:
            raise ValueError(f"end field: {error}")
        if end_dt is None:
            return None, None, None
        due_dt, error = self.safe_parse_datetime(parent.get("due"))
        if error:
            raise ValueError(f"due field: {error}")
        scheduled_dt, error = self.safe_parse_datetime(parent.get("scheduled"))
        if error:
            raise ValueError(f"scheduled field: {error}")
        end_local = self._local(end_dt)
        anchor_dt = due_dt or scheduled_dt
        due_local = self._local(anchor_dt) if anchor_dt else end_local
        return end_local, due_local, due_dt

    def compute_cp_child_due(self, parent: NauticalTask) -> CpChildDueResult:
        self._require_chain_id(parent)
        parent_values = parent.observation.to_mapping()
        parent = parent_values
        duration = str(parent.get("cp") or "").strip()
        if not duration:
            return None, None
        chain_id = str(parent_values.get("chainID") or "").strip()
        tokens = self.core.parse_cp_sequence_tokens(duration)
        if not tokens:
            reason = self.core.cp_sequence_parse_error(duration) or (
                f"invalid duration format '{duration}'"
            )
            raise ValueError(f"cp field: {reason} (expected: 3d, 2w, 1h, etc.)")
        link_no = self.core.coerce_int(parent.get("link"), 1)
        seq_idx = (max(1, link_no) - 1) % len(tokens)
        interval = self.core.cp_sequence_interval_for_token(
            tokens[seq_idx],
            cp=duration,
            link_no=link_no,
            token_index=seq_idx,
            chain_id=chain_id,
        )
        if not interval:
            return None, None
        end_dt, error = self.safe_parse_datetime(parent.get("end"))
        if error:
            raise ValueError(f"end field: {error}")
        if end_dt is None:
            return None, None
        due_dt, error = self.safe_parse_datetime(parent.get("due"))
        if error:
            raise ValueError(f"due field: {error}")
        scheduled_dt, error = self.safe_parse_datetime(parent.get("scheduled"))
        if error:
            raise ValueError(f"scheduled field: {error}")
        target_field = "scheduled" if due_dt is None and scheduled_dt is not None else "due"
        candidate = (end_dt + interval).replace(microsecond=0)
        meta: dict[str, Any]
        if int(interval.total_seconds()) % 86400:
            meta = {"period": duration, "basis": "end+cp (exact)", "target_field": target_field}
        else:
            anchor_dt = due_dt or scheduled_dt
            anchor_local = self._local(anchor_dt or end_dt)
            candidate_local = self._local(candidate).replace(
                hour=anchor_local.hour,
                minute=anchor_local.minute,
                second=0,
                microsecond=0,
            )
            candidate = candidate_local.astimezone(timezone.utc)
            meta = {
                "period": duration,
                "basis": "end+cp (preserve clock)",
                "target_field": target_field,
            }
        if len(tokens) > 1 or tokens[seq_idx].get("kind") == "rand":
            meta.update({"cp_sequence_len": len(tokens), "cp_sequence_step": seq_idx + 1})
        return candidate, meta

    def compute_anchor_child_due(self, parent: NauticalTask) -> AnchorChildDueResult:
        task = parent
        self._require_chain_id(task)
        parent = task.observation.to_mapping()
        expression = normalize_recurrence_text(parent.get("anchor"))
        anchor_file = normalize_recurrence_text(parent.get("anchor_file"))
        if not expression and not anchor_file:
            return None, None, None
        scheduler = self._task_scheduler(task)
        evaluator = scheduler.session.evaluator
        end_local, due_local, due_dt = self._anchor_parent_local_times(parent)
        if end_local is None or due_local is None:
            return None, None, None
        try:
            result = scheduler.select_mode(
                str(parent.get("anchor_mode") or "skip").strip().lower() or "skip",
                due_local=due_local,
                end_local=end_local,
                due_explicit=due_dt is not None,
                fallback_hhmm=(due_local.hour, due_local.minute),
                default_seed_date=due_local.date(),
            )
        except ValueError as exc:
            if "Occurrence omission scan exceeded" in str(exc):
                raise ValueError("No valid anchor occurrences found after applying omit rules.") from exc
            raise
        if result.selected_occurrence is None:
            raise ValueError("Could not compute next anchor occurrence")
        target_field = "scheduled" if due_dt is None and parent.get("scheduled") else "due"
        return (
            result.selected_occurrence.astimezone(timezone.utc),
            result.metadata(target_field=target_field),
            evaluator.anchor_dnf or None,
        )

    def _recurrence_anchor_field(self, task: Mapping[str, Any]) -> str:
        return "due" if task.get("due") else "scheduled" if task.get("scheduled") else "due"

    def _configured_recurrence_uda_fields(self, parent: Mapping[str, Any]) -> tuple[str, ...]:
        keys = {str(key).lower(): key for key in parent if isinstance(key, str) and key}
        out: list[str] = []
        seen: set[str] = set()
        for configured in self.recurrence_update_udas:
            lower = str(configured).strip().lower()
            if not lower or lower in seen or lower in _UDA_CARRY_SKIP_LOWER:
                continue
            seen.add(lower)
            actual = keys.get(lower)
            if actual:
                out.append(actual)
        return tuple(out)

    def _carry_relative_datetime(
        self,
        parent: NauticalTask,
        child: dict[str, Any],
        child_due_utc: datetime,
        field: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None:
        parent_values = parent.observation.to_mapping()
        child.pop(field, None)
        if not parent_values.get(field):
            return
        if not parent_values.get(parent_anchor_field):
            reason = f"parent {parent_anchor_field} is missing"
            self._record_carry_debug(field, {"ok": False, "reason": reason})
            raise CarryFieldError(field, reason)
        try:
            parent_anchor = self.core.parse_dt_any(parent_values.get(parent_anchor_field))
            parent_value = self.core.parse_dt_any(parent_values.get(field))
            if not (parent_anchor and parent_value and isinstance(child_due_utc, datetime)):
                raise ValueError("parent or child recurrence timestamp is not parseable")
            parent_delta = self.core.utc_to_local_naive(parent_value) - self.core.utc_to_local_naive(
                parent_anchor
            )
            child_local = self.core.utc_to_local_naive(child_due_utc) + parent_delta
            child[field] = self.core.fmt_isoz(self.core.local_naive_to_utc(child_local))
            self._record_carry_debug(
                field,
                {
                    "ok": True,
                    "parent_anchor": parent_values.get(parent_anchor_field),
                    "parent_val": parent_values.get(field),
                    "child_anchor": child.get(child_anchor_field),
                    "child_val": child.get(field),
                    "delta": str(parent_delta),
                },
            )
        except Exception as exc:
            self._record_carry_debug(field, {"ok": False, "reason": "conversion-failed"})
            if isinstance(exc, CarryFieldError):
                raise
            raise CarryFieldError(field, str(exc) or "timezone conversion failed") from exc

    def carry_relative_datetime(
        self,
        parent: NauticalTask,
        child: dict[str, Any],
        child_due_utc: datetime,
        field: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None:
        """Public service boundary for carrying a relative task field."""
        self._carry_relative_datetime(
            parent,
            child,
            child_due_utc,
            field,
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_anchor_field,
        )

    def _record_carry_debug(self, field: str, payload: dict[str, Any]) -> None:
        if not self.debug_wait_sched or self.wait_sched_debug is None:
            return
        try:
            self.wait_sched_debug[field] = payload
        except Exception:
            return

    def _carry_native_until(
        self,
        parent: NauticalTask,
        child: dict[str, Any],
        child_due_utc: datetime,
        kind: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None:
        parent_values = parent.observation.to_mapping()
        child.pop("until", None)
        if not parent_values.get("until") or not parent_values.get(parent_anchor_field):
            return
        native_until = self.core._import_sibling("native_until")
        try:
            parent_target = self.core.parse_dt_any(parent_values.get(parent_anchor_field))
            parent_until = self.core.parse_dt_any(parent_values.get("until"))
        except Exception as exc:
            raise native_until.NativeUntilCarryError(
                native_until.CARRY_INVALID,
                "native until carry requires valid recurrence timestamps",
            ) from exc
        if not (parent_target and parent_until and isinstance(child_due_utc, datetime)):
            raise native_until.NativeUntilCarryError(
                native_until.CARRY_INVALID,
                "native until carry requires valid recurrence timestamps",
            )
        child["until"] = self.core.fmt_isoz(
            native_until.carry(
                parent_target,
                parent_until,
                child_due_utc,
                kind,
                utc_to_local_naive=self.core.utc_to_local_naive,
                local_naive_to_utc=self.core.local_naive_to_utc,
            )
        )

    def carry_native_until(
        self,
        parent: NauticalTask,
        child: dict[str, Any],
        child_due_utc: datetime,
        kind: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None:
        """Public service boundary for carrying native expiration."""
        self._carry_native_until(
            parent,
            child,
            child_due_utc,
            kind,
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_anchor_field,
        )

    def build_child_draft(
        self,
        parent: NauticalTask,
        child_due_utc: datetime,
        child_field: str,
        next_link_no: int,
        parent_short: str,
        kind: str,
        cpmax: int,
        until_dt: Any,
    ) -> TaskDraft:
        """Build a complete, validated child intent without exposing mappings."""
        parent_chain = self._require_chain_id(parent)
        parent_task = parent
        parent = parent_task.observation.to_mapping()
        if self.debug_wait_sched and self.wait_sched_debug is not None:
            try:
                self.wait_sched_debug.clear()
            except Exception:
                pass
        child = {key: value for key, value in parent.items() if key not in _RESERVED_DROP}
        for key in _RESERVED_OVERRIDE:
            child.pop(key, None)
        child.update(
            {
                "status": "pending",
                "entry": self.core.fmt_isoz(self.core.now_utc()),
                "chain": "on",
                "prevLink": parent_short,
                "link": next_link_no,
            }
        )
        parent_anchor_field = self._recurrence_anchor_field(parent)
        if child_field == "scheduled":
            child.pop("due", None)
            child["scheduled"] = self.core.fmt_isoz(child_due_utc)
        else:
            child["due"] = self.core.fmt_isoz(child_due_utc)
        if kind in {"anchor", "anchor_file"}:
            anchor = normalize_recurrence_text(parent.get("anchor"))
            anchor_file = normalize_recurrence_text(parent.get("anchor_file"))
            if anchor:
                child["anchor"] = anchor
            else:
                child.pop("anchor", None)
            if anchor_file:
                child["anchor_file"] = anchor_file
            else:
                child.pop("anchor_file", None)
            parent_mode = normalize_recurrence_text(parent.get("anchor_mode")) or "skip"
            child["anchor_mode"] = (
                "all" if str(parent_mode).strip().lower() == "flex" else parent_mode
            )
            child.pop("cp", None)
        else:
            child["cp"] = parent.get("cp")
            child.pop("anchor", None)
            child.pop("anchor_file", None)
            child.pop("anchor_mode", None)
        self._carry_relative_datetime(
            parent_task,
            child,
            child_due_utc,
            "wait",
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_field,
        )
        if child_field != "scheduled":
            self._carry_relative_datetime(
                parent_task,
                child,
                child_due_utc,
                "scheduled",
                parent_anchor_field=parent_anchor_field,
                child_anchor_field=child_field,
            )
        self._carry_native_until(
            parent_task,
            child,
            child_due_utc,
            kind,
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_field,
        )
        for field_name in self._configured_recurrence_uda_fields(parent):
            self._carry_relative_datetime(
                parent_task,
                child,
                child_due_utc,
                field_name,
                parent_anchor_field=parent_anchor_field,
                child_anchor_field=child_field,
            )
        if cpmax:
            child["chainMax"] = int(cpmax)
        if until_dt:
            child["chainUntil"] = self.core.fmt_isoz(until_dt)
        child["chainID"] = parent_chain
        child_uuid = uuid.uuid5(
            _STABLE_CHILD_UUID_NAMESPACE,
            json.dumps(
                {
                    "chain_id": parent_chain.lower(),
                    "kind": "anchor" if kind in {"anchor", "anchor_file"} else "cp",
                    "link": int(next_link_no),
                    "parent_uuid": parent_task.identity.task_uuid.value.lower(),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
        child["uuid"] = str(child_uuid)
        child_observation = self.core._import_sibling("task_codec").DEFAULT_TASK_CODEC.decode_row(
            child,
            source_query="chain generation child draft",
        )
        child_task = NauticalTask.from_observation(child_observation)
        target = child_task.temporal.due if child_field != "scheduled" else child_task.temporal.scheduled
        if target is None:
            raise ValueError("generated child draft has no recurrence target")
        return TaskDraft.from_task(child_task, target_field=child_field)


__all__ = (
    "AnchorChildDueResult",
    "CarryFieldError",
    "ChainIdentityError",
    "ChildMetadata",
    "CpChildDueResult",
    "ChainGenerationService",
)
