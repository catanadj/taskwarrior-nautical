from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol, TypeAlias


# Hook implementations are intentionally assembled at runtime, but the
# orchestration boundary still needs to distinguish injected callables from
# task data and result values.  ``Any`` remains the payload type because hook
# modules support Taskwarrior's heterogeneous JSON fields.
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


class RootAgeFormatter(Protocol):
    def __call__(self, task: TaskRow, now_utc: datetime) -> str:
        ...


class WaitScheduleRowsCallback(Protocol):
    def __call__(
        self,
        rows: list[tuple[str, Any]],
        task: TaskRow,
        anchor_due: Any,
        *,
        anchor_field: str = "due",
    ) -> None:
        ...


class TimelineLinesCallback(Protocol):
    def __call__(
        self,
        kind: str,
        task: TaskRow,
        child_due: Any,
        child_short: str,
        dnf: Any,
        *,
        next_count: int = 3,
        cap_no: int | None = None,
        cur_no: int | None = None,
        show_gaps: bool = True,
        round_anchor_gaps: bool = True,
    ) -> list[str]:
        ...


class FeedbackRowsFormatter(Protocol):
    def __call__(
        self,
        rows: list[tuple[str, Any]],
    ) -> list[tuple[str | None, Any]]:
        ...


class PreviewLineFormatter(Protocol):
    def __call__(
        self,
        link_no: int,
        task: TaskRow,
        child_due: Any,
        child_short: str,
        now_utc: Any,
        *,
        child_field: str = "due",
        cap_no: int | None = None,
        until_dt: Any = None,
        until_no: int | None = None,
        child_until_dt: Any = None,
        kind: str = "cp",
        minimal: bool = False,
    ) -> str:
        ...


class PanelLineCallback(Protocol):
    def __call__(
        self,
        title: str,
        line: str,
        *,
        kind: str = "info",
        border_style: str | None = None,
        title_style: str | None = None,
        markup_body: bool = False,
    ) -> None:
        ...


class TextLineCallback(Protocol):
    def __call__(
        self,
        line: str,
        *,
        kind: str = "info",
        markup_body: bool = False,
    ) -> None:
        ...


class FeedbackPanelCallback(Protocol):
    def __call__(
        self,
        title: str,
        rows: list[tuple[str | None, Any]],
        *,
        kind: str = "info",
        border_style: str | None = None,
        title_style: str | None = None,
        label_style: str | None = None,
    ) -> None:
        ...


class ChainColourCallback(Protocol):
    def __call__(self, task: TaskRow, kind: str) -> str:
        ...


class StripQuotesCallback(Protocol):
    def __call__(self, value: str) -> str:
        ...


class HumanDeltaCallback(Protocol):
    def __call__(self, start: Any, end: Any, prefer_months: bool = True) -> str:
        ...


class ComputeAnchorChildDueCallback(Protocol):
    def __call__(self, task: TaskRow) -> tuple[datetime | None, dict[str, Any] | None, Any]:
        ...


class ComputeCpChildDueCallback(Protocol):
    def __call__(self, task: TaskRow) -> tuple[datetime | None, dict[str, Any] | None]:
        ...


class SafeParseDatetimeCallback(Protocol):
    def __call__(self, value: Any) -> tuple[datetime | None, str | None]:
        ...


class ValidateUntilCallback(Protocol):
    def __call__(self, until_dt: datetime, now_utc: Any) -> tuple[bool, str | None]:
        ...


class ValidateChainDurationCallback(Protocol):
    def __call__(self, child_due: Any, until_dt: Any, now_utc: Any) -> tuple[bool, str | None]:
        ...


class CoerceIntCallback(Protocol):
    def __call__(self, value: Any, default: int = 0) -> int:
        ...


class DatetimeParserCallback(Protocol):
    def __call__(self, value: Any) -> datetime | None:
        ...


class EstimateCpFinalCallback(Protocol):
    def __call__(self, task: TaskRow, child_due: Any) -> Any:
        ...


class EstimateAnchorFinalCallback(Protocol):
    def __call__(self, task: TaskRow, child_due: Any, dnf: Any) -> Any:
        ...


class CapFromUntilCpCallback(Protocol):
    def __call__(self, task: TaskRow, child_due: Any) -> tuple[int | None, Any]:
        ...


class CapFromUntilAnchorCallback(Protocol):
    def __call__(self, task: TaskRow, child_due: Any, dnf: Any) -> tuple[int | None, Any]:
        ...


class EndChainSummaryCallback(Protocol):
    def __call__(
        self,
        current: TaskRow,
        reason: str,
        now_utc: Any,
        *,
        current_task: TaskRow | None = None,
    ) -> None:
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
ModifyChainStateCallback: TypeAlias = Callable[[], Any]
SeedLookupCallback: TypeAlias = Callable[[TaskRow, TaskRow], None]
ChainExportCallback: TypeAlias = Callable[[str], list[TaskRow]]
ChainIndexesCallback: TypeAlias = Callable[
    [list[TaskRow]], tuple[dict[int, TaskRow], dict[str, TaskRow]]
]
SetChainCacheCallback: TypeAlias = Callable[[str, list[TaskRow]], None]
MergeChainCallback: TypeAlias = Callable[[list[TaskRow], TaskRow, TaskRow, str], list[TaskRow]]
DiagnosticSummaryCallback: TypeAlias = Callable[[], Any]


class BuildAndSpawnCallback(Protocol):
    """Build and queue one successor without exposing hook internals."""

    def __call__(
        self,
        new: TaskRow,
        *,
        child_due: Any,
        child_field: str,
        next_no: int,
        parent_short: str,
        kind: str,
        cpmax: int,
        until_dt: Any,
        planned_child: TaskRow | None = None,
    ) -> "CompletionSpawnResult | None":
        ...


class ChainHealthCallback(Protocol):
    """Calculate optional chain health advice for completion feedback."""

    def __call__(
        self,
        chain: list[TaskRow],
        kind: str,
        task: TaskRow,
        tol_secs: int = 3600,
        style: str = "compact",
    ) -> str | None:
        ...


class ChainIntegrityCallback(Protocol):
    """Validate chain metadata while retaining the expected-ID keyword."""

    def __call__(
        self,
        chain: list[TaskRow],
        expected_chain_id: str | None = None,
    ) -> list[str]:
        ...


class AnchorCompletionRenderCallback(Protocol):
    """Render anchor completion feedback with its explicit orchestration payload."""

    def __call__(
        self,
        *,
        new: TaskRow,
        child: TaskRow,
        child_due: Any,
        child_short: str,
        next_no: int,
        parent_short: str,
        cap_no: int | None,
        finals: list[tuple[str, Any]],
        now_utc: Any,
        until_dt: Any,
        until_cap_no: int | None,
        dnf: Any,
        meta: dict[str, Any],
        stripped_attrs: list[str],
        deferred_spawn: bool,
        spawn_intent_id: str | None,
        lifecycle_result: "CompletionLifecycleResult",
        chain_by_short: dict[str, Any] | None,
        analytics_advice: str | None,
        integrity_warnings: list[str] | None,
        base_no: int,
    ) -> None:
        ...


class CpCompletionRenderCallback(Protocol):
    """Render CP completion feedback with its explicit orchestration payload."""

    def __call__(
        self,
        *,
        new: TaskRow,
        child: TaskRow,
        child_due: Any,
        child_short: str,
        next_no: int,
        parent_short: str,
        cap_no: int | None,
        finals: list[tuple[str, Any]],
        now_utc: Any,
        until_dt: Any,
        until_cap_no: int | None,
        meta: dict[str, Any],
        deferred_spawn: bool,
        spawn_intent_id: str | None,
        lifecycle_result: "CompletionLifecycleResult",
        chain_by_short: dict[str, Any] | None,
        analytics_advice: str | None,
        integrity_warnings: list[str] | None,
        base_no: int,
    ) -> None:
        ...


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
    planned_child: dict[str, Any] | None = None
    lifecycle_plan: Any = None




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
    lifecycle_result: "CompletionLifecycleResult"
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
    lifecycle_result: "CompletionLifecycleResult"
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
    outcome_state: str = "applied"
    reason: str = ""

    def __post_init__(self) -> None:
        state = str(self.outcome_state or "").strip().lower()
        if state not in {"applied", "manual_review"}:
            raise ValueError(f"unsupported completion spawn state: {self.outcome_state!r}")
        self.outcome_state = state
        self.reason = str(self.reason or "").strip()


@dataclass(frozen=True, slots=True)
class CompletionLifecycleResult:
    """Operational result returned after completion mutation decisions finish."""

    state: str
    child_short: str = ""
    deferred_spawn: bool = False
    spawn_intent_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        state = str(self.state or "").strip().lower()
        if state not in {"applied", "queued", "terminal", "manual_review", "retryable"}:
            raise ValueError(f"unsupported completion lifecycle state: {self.state!r}")
        if state == "queued" and (not self.deferred_spawn or not str(self.spawn_intent_id or "").strip()):
            raise ValueError("queued completion result requires deferred spawn and an intent ID")
        if state in {"applied", "terminal", "manual_review"} and self.deferred_spawn:
            raise ValueError("applied completion result cannot be deferred")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "child_short", str(self.child_short or "").strip())
        object.__setattr__(self, "reason", str(self.reason or "").strip())


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
    modify_chain_state: ModifyChainStateCallback
    get_chain_export: ChainExportCallback
    build_chain_indexes: ChainIndexesCallback
    set_chain_cache: SetChainCacheCallback
    export_uuid_short_cached: Any
    merge_spawned_child_into_chain: MergeChainCallback
    chain_health_advice: ChainHealthCallback
    chain_integrity_warnings: ChainIntegrityCallback
    render_anchor_completion_feedback: AnchorCompletionRenderCallback
    render_cp_completion_feedback: CpCompletionRenderCallback
    print_task: PrintTaskCallback
    diag_summary: DiagnosticSummaryCallback
    show_analytics: bool
    check_integrity: bool
    analytics_style: str


@dataclass(slots=True)
class AnchorFeedbackServices:
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: Any
    diag_enabled: bool
    format_root_and_age: RootAgeFormatter
    append_next_wait_sched_rows: WaitScheduleRowsCallback
    timeline_lines: TimelineLinesCallback
    show_timeline_gaps: bool
    root_uuid_from: Callable[[TaskRow], str]
    short: ShortUuidCallback
    format_next_anchor_rows: FeedbackRowsFormatter
    format_line_preview: PreviewLineFormatter
    panel_line: PanelLineCallback
    text_line: TextLineCallback
    panel: FeedbackPanelCallback
    chain_color_per_chain: bool
    chain_colour_for_task: ChainColourCallback
    strip_quotes: StripQuotesCallback
    human_delta: HumanDeltaCallback


@dataclass(slots=True)
class CpFeedbackServices:
    core: Any
    diag_enabled: bool
    format_root_and_age: RootAgeFormatter
    append_next_wait_sched_rows: WaitScheduleRowsCallback
    timeline_lines: TimelineLinesCallback
    show_timeline_gaps: bool
    format_next_cp_rows: FeedbackRowsFormatter
    format_line_preview: PreviewLineFormatter
    panel_line: PanelLineCallback
    text_line: TextLineCallback
    panel: FeedbackPanelCallback
    chain_color_per_chain: bool
    chain_colour_for_task: ChainColourCallback
    human_delta: HumanDeltaCallback
