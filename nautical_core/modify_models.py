from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, TypeAlias


# Hook implementations are intentionally assembled at runtime, but the
# orchestration boundary still needs to distinguish injected callables from
# task data and result values.  ``Any`` remains the payload type because hook
# modules support Taskwarrior's heterogeneous JSON fields.
ServiceCallback: TypeAlias = Callable[..., Any]


@dataclass(slots=True)
class CompletionChainSnapshot:
    mode: str
    rows: list[dict[str, Any]]
    loaded: bool
    chain_id: str = ""
    error: str = ""

    @property
    def coverage(self) -> str:
        """The strongest lookup coverage represented by this snapshot."""
        return self.mode

    @property
    def is_unavailable(self) -> bool:
        return not self.loaded and bool(self.error)


@dataclass(slots=True)
class CompletionSnapshotResult:
    loaded: bool
    rows: list[dict[str, Any]]
    error: str = ""


@dataclass(slots=True)
class CompletionPreflightContext:
    parent_short: str
    base_no: int
    next_no: int
    kind: str
    chain_id: str
    chain_snapshot: CompletionChainSnapshot


@dataclass(slots=True)
class CompletionComputeResult:
    child_due: Any
    meta: Any
    dnf: Any
    until_dt: Any
    cpmax: int
    cap_no: int | None
    finals: list[tuple[str, Any]]
    until_cap_no: int | None




@dataclass(slots=True)
class CompletionPreflightServices:
    short: ServiceCallback
    completion_link_numbers_or_fail: ServiceCallback
    completion_kind_or_stop: ServiceCallback
    completion_chain_id_or_fail: ServiceCallback
    completion_chain_snapshot: ServiceCallback
    completion_existing_next_or_fail: ServiceCallback


@dataclass(slots=True)
class CompletionComputeServices:
    completion_compute_child_due: ServiceCallback
    completion_until_or_fail: ServiceCallback
    completion_until_guard_or_stop: ServiceCallback
    completion_require_child_due_or_fail: ServiceCallback
    completion_warn_unreasonable_duration: ServiceCallback
    completion_caps: ServiceCallback
    completion_cap_guard_or_stop: ServiceCallback


@dataclass(slots=True)
class CpCompletionFeedbackModel:
    new: dict[str, Any]
    child: dict[str, Any]
    child_due: Any
    child_short: str
    next_no: int
    parent_short: str
    cap_no: int | None
    finals: list[tuple[str, Any]]
    now_utc: Any
    until_dt: Any
    until_cap_no: int | None
    meta: dict[str, Any]
    deferred_spawn: bool
    spawn_intent_id: str | None
    chain_by_short: dict[str, Any] | None
    analytics_advice: str | None
    integrity_warnings: list[str] | None
    base_no: int


@dataclass(slots=True)
class AnchorCompletionFeedbackModel:
    new: dict[str, Any]
    child: dict[str, Any]
    child_due: Any
    child_short: str
    next_no: int
    parent_short: str
    cap_no: int | None
    finals: list[tuple[str, Any]]
    now_utc: Any
    until_dt: Any
    until_cap_no: int | None
    dnf: Any
    meta: dict[str, Any]
    stripped_attrs: list[str]
    deferred_spawn: bool
    spawn_intent_id: str | None
    chain_by_short: dict[str, Any] | None
    analytics_advice: str | None
    integrity_warnings: list[str] | None
    base_no: int




@dataclass(slots=True)
class CompletionSpawnResult:
    child: dict[str, Any]
    child_short: str
    stripped_attrs: Any
    verified: bool
    deferred_spawn: bool
    spawn_intent_id: str | None


@dataclass(slots=True)
class CompletionSpawnServices:
    build_child_from_parent: ServiceCallback
    spawn_child_atomic: ServiceCallback
    panel: ServiceCallback
    print_task: ServiceCallback
    diag: ServiceCallback


@dataclass(slots=True)
class AnchorFeedbackServices:
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: Any
    diag_enabled: bool
    format_root_and_age: ServiceCallback
    append_next_wait_sched_rows: ServiceCallback
    timeline_lines: ServiceCallback
    show_timeline_gaps: bool
    root_uuid_from: ServiceCallback
    short: ServiceCallback
    format_next_anchor_rows: ServiceCallback
    format_line_preview: ServiceCallback
    panel_line: ServiceCallback
    text_line: ServiceCallback
    panel: ServiceCallback
    chain_color_per_chain: bool
    chain_colour_for_task: ServiceCallback
    strip_quotes: ServiceCallback
    human_delta: ServiceCallback


@dataclass(slots=True)
class CpFeedbackServices:
    core: Any
    diag_enabled: bool
    format_root_and_age: ServiceCallback
    append_next_wait_sched_rows: ServiceCallback
    timeline_lines: ServiceCallback
    show_timeline_gaps: bool
    format_next_cp_rows: ServiceCallback
    format_line_preview: ServiceCallback
    panel_line: ServiceCallback
    text_line: ServiceCallback
    panel: ServiceCallback
    chain_color_per_chain: bool
    chain_colour_for_task: ServiceCallback
    human_delta: ServiceCallback
