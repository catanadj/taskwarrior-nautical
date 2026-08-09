from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol, TypeAlias


# Hook implementations are intentionally assembled at runtime, but the
# orchestration boundary still needs to distinguish injected callables from
# task data and result values.  ``Any`` remains the payload type because hook
# modules support Taskwarrior's heterogeneous JSON fields.
ServiceCallback: TypeAlias = Callable[..., Any]
TaskRow: TypeAlias = dict[str, Any]
ShortUuidCallback: TypeAlias = Callable[[Any], str]


class PanelCallback(Protocol):
    """Render one structured panel without exposing hook implementation details."""

    def __call__(
        self,
        title: str,
        rows: list[tuple[str, Any]],
        *,
        kind: str = "info",
        task: TaskRow | None = None,
    ) -> Any:
        ...


class PrintTaskCallback(Protocol):
    """Emit the current Taskwarrior task through the hook response boundary."""

    def __call__(self, task: TaskRow) -> Any:
        ...


class DiagnosticCallback(Protocol):
    """Record an opt-in diagnostic without affecting hook stdout."""

    def __call__(self, message: str) -> Any:
        ...


CompletionLinkNumbersCallback: TypeAlias = Callable[
    [TaskRow], tuple[int, int] | None
]
CompletionKindCallback: TypeAlias = Callable[[TaskRow, datetime], str | None]
CompletionChainIdCallback: TypeAlias = Callable[[TaskRow], str | None]
CompletionSnapshotCallback: TypeAlias = Callable[
    [str, int, int], "CompletionChainSnapshot"
]
CompletionExistingNextCallback: TypeAlias = Callable[
    [TaskRow, int, "CompletionChainSnapshot | None"], bool
]
CompletionChildDueCallback: TypeAlias = Callable[
    [TaskRow, str], tuple[Any, Any, Any] | None
]
CompletionUntilCallback: TypeAlias = Callable[
    [TaskRow, datetime], datetime | None | bool
]
CompletionUntilGuardCallback: TypeAlias = Callable[[TaskRow, Any, Any, datetime], bool]
CompletionChildRequiredCallback: TypeAlias = Callable[[TaskRow, Any], bool]
CompletionDurationWarningCallback: TypeAlias = Callable[[TaskRow, Any, Any, datetime], None]
CompletionCapsCallback: TypeAlias = Callable[
    [str, TaskRow, Any, Any], tuple[Any, Any, Any, Any, Any]
]
CompletionCapGuardCallback: TypeAlias = Callable[
    [TaskRow, int, int | None, datetime], bool
]
BuildChildCallback: TypeAlias = Callable[
    [TaskRow, Any, str, int, str, str, int, Any], TaskRow
]
SpawnChildCallback: TypeAlias = Callable[
    [TaskRow, TaskRow], tuple[str, Any, bool, bool, str | None, str | None]
]
BuildAndSpawnCallback: TypeAlias = Callable[..., "CompletionSpawnResult | None"]
SeedLookupCallback: TypeAlias = Callable[[TaskRow, TaskRow], None]
ChainExportCallback: TypeAlias = Callable[[str], list[TaskRow]]
ChainIndexesCallback: TypeAlias = Callable[
    [list[TaskRow]], tuple[dict[int, TaskRow], dict[str, TaskRow]]
]
SetChainCacheCallback: TypeAlias = Callable[[str, list[TaskRow]], None]
MergeChainCallback: TypeAlias = Callable[[list[TaskRow], TaskRow, TaskRow, str], list[TaskRow]]
ChainHealthCallback: TypeAlias = Callable[..., str | None]
ChainIntegrityCallback: TypeAlias = Callable[..., list[str] | None]


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
    short: ShortUuidCallback
    completion_link_numbers_or_fail: CompletionLinkNumbersCallback
    completion_kind_or_stop: CompletionKindCallback
    completion_chain_id_or_fail: CompletionChainIdCallback
    completion_chain_snapshot: CompletionSnapshotCallback
    completion_existing_next_or_fail: CompletionExistingNextCallback


@dataclass(slots=True)
class CompletionComputeServices:
    completion_compute_child_due: CompletionChildDueCallback
    completion_until_or_fail: CompletionUntilCallback
    completion_until_guard_or_stop: CompletionUntilGuardCallback
    completion_require_child_due_or_fail: CompletionChildRequiredCallback
    completion_warn_unreasonable_duration: CompletionDurationWarningCallback
    completion_caps: CompletionCapsCallback
    completion_cap_guard_or_stop: CompletionCapGuardCallback


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
    build_child_from_parent: BuildChildCallback
    spawn_child_atomic: SpawnChildCallback
    panel: PanelCallback
    print_task: PrintTaskCallback
    diag: DiagnosticCallback


@dataclass(slots=True)
class CompletionFinalizeServices:
    build_and_spawn_child: BuildAndSpawnCallback
    seed_runtime_lookup_tasks: SeedLookupCallback
    modify_chain_state: ServiceCallback
    get_chain_export: ChainExportCallback
    build_chain_indexes: ChainIndexesCallback
    set_chain_cache: SetChainCacheCallback
    export_uuid_short_cached: Any
    merge_spawned_child_into_chain: MergeChainCallback
    chain_health_advice: ChainHealthCallback
    chain_integrity_warnings: ChainIntegrityCallback
    render_anchor_completion_feedback: ServiceCallback
    render_cp_completion_feedback: ServiceCallback
    print_task: ServiceCallback
    diag_summary: ServiceCallback
    show_analytics: bool
    check_integrity: bool
    analytics_style: str


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
