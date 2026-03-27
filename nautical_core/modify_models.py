from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompletionPreflightContext:
    parent_short: str
    base_no: int
    next_no: int
    kind: str
    chain_id: str


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
    short: Any
    completion_link_numbers_or_fail: Any
    completion_kind_or_stop: Any
    completion_chain_id_or_fail: Any
    completion_existing_next_or_fail: Any


@dataclass(slots=True)
class CompletionComputeServices:
    completion_compute_child_due: Any
    completion_until_or_fail: Any
    completion_until_guard_or_stop: Any
    completion_require_child_due_or_fail: Any
    completion_warn_unreasonable_duration: Any
    completion_caps: Any
    completion_cap_guard_or_stop: Any


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
    build_child_from_parent: Any
    spawn_child_atomic: Any
    panel: Any
    print_task: Any
    diag: Any


@dataclass(slots=True)
class AnchorFeedbackServices:
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: Any
    diag_enabled: bool
    format_root_and_age: Any
    append_next_wait_sched_rows: Any
    timeline_lines: Any
    show_timeline_gaps: bool
    root_uuid_from: Any
    short: Any
    format_next_anchor_rows: Any
    format_line_preview: Any
    panel_line: Any
    text_line: Any
    panel: Any
    chain_color_per_chain: bool
    chain_colour_for_task: Any
    strip_quotes: Any
    human_delta: Any


@dataclass(slots=True)
class CpFeedbackServices:
    core: Any
    diag_enabled: bool
    format_root_and_age: Any
    append_next_wait_sched_rows: Any
    timeline_lines: Any
    show_timeline_gaps: bool
    format_next_cp_rows: Any
    format_line_preview: Any
    panel_line: Any
    text_line: Any
    panel: Any
    chain_color_per_chain: bool
    chain_colour_for_task: Any
    human_delta: Any
