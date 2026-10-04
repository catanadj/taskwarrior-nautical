"""Completion preflight and occurrence-limit effects for on-modify."""

from __future__ import annotations

from datetime import date, datetime, timedelta as Timedelta
from dataclasses import dataclass
from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, overload
from uuid import UUID

from .task_models import NauticalTask, TaskDraft, TaskObservation, TaskPayload
from .task_datetime import datetime_value, parser_for_host
from .timeutil import compare_datetimes
from .lifecycle.read_service import ChainSnapshotRepository
from .lifecycle.models import LifecyclePlan
from .task_read_repository import AuthoritativeTaskSnapshot
from .integration_models import TaskRead
from .modify_models import (
    DiagnosticCallback,
    CapFromUntilAnchorCallback,
    CapFromUntilCpCallback,
    CompletionCapGuardCallback,
    CompletionCapsCallback,
    CompletionChildDueCallback,
    CompletionChildRequiredCallback,
    CompletionComputeServices,
    CompletionComputeResult,
    CompletionFinals,
    AnchorDNF,
    CompletionDurationWarningCallback,
    CompletionChainSnapshot,
    CompletionPreflightServices,
    CompletionPreflightContext,
    CompletionSpawnServices,
    CompletionSpawnResult,
    CoerceIntCallback,
    BuildChildDraftCallback,
    ComputeAnchorChildDueCallback,
    ComputeCpChildDueCallback,
    CompletionLifecycleResult,
    CompletionUntilCallback,
    CompletionUntilGuardCallback,
    DatetimeParserCallback,
    EstimateAnchorFinalCallback,
    EstimateCpFinalCallback,
    EndChainSummaryCallback,
    ExistingNextLookupCallback,
    InvalidRelativeCarryReasonCallback,
    ShortUuidCallback,
    PanelCallback,
    PrintTaskCallback,
    SafeParseDatetimeCallback,
    SpawnChildCallback,
    ValidateChainDurationCallback,
    ValidateUntilCallback,
)
from .modify_ui_effects import UIEffectsPorts
from .scheduler_models import OccurrenceSearchExhausted
from .modify_generation_effects import ChainGenerationServicePort

if TYPE_CHECKING:
    from .modify_generation_effects import (
        _ChainGenerationModule,
        GenerationHost,
        GenerationPorts,
        GenerationStatePort,
    )
    from .modify_schedule_effects import (
        _ModifyAnchorCompletionEffects,
        _ModifyRuntimeModule,
        AnchorCompletionPorts,
        CPCompletionPorts,
        SequenceIntervalForToken,
    )
    from .modify_spawn_effects import (
        _LifecycleApplicationModule,
        _LifecycleOutboxModule,
        _ModifyTaskFields,
        _SpawnModule,
        _SpawnPreparationWithPayload,
        SpawnChildPorts,
    )
    from .modify_command_effects import (
        CommandHost,
        CommandPorts,
        DiagCounter,
        RunTaskRecorder,
    )
    from .integration_context import IntegrationContext
    from .task_datetime import TaskDatetimeParser
    from .task_codec import TaskCodec
    from .modify_validation_effects import DurationPorts as DurationPortsModel, UntilPorts as UntilPortsModel
    from .modify_diagnostics_effects import EndChainSummaryPorts
    from .modify_value_effects import DatetimePorts as DatetimePortsModel
    from .cp_parser import CPSequenceToken
    from .modify_models import AnchorFileProviderFactory


class CompletionPreflightService(Protocol):
    """Validated preflight service used by completion effects."""

    def completion_link_numbers_or_fail(
        self,
        task: TaskPayload,
        *,
        coerce_int: CoerceIntCallback,
        max_link_number: int,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
    ) -> tuple[int, int] | None: ...
    def completion_kind_or_stop(
        self,
        task: TaskPayload,
        now_utc: datetime,
        *,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
        end_chain_summary: EndChainSummaryCallback,
    ) -> str | None: ...
    def completion_chain_id_or_fail(
        self,
        task: TaskPayload,
        *,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
    ) -> str | None: ...
    def completion_existing_next_or_fail(
        self,
        task: TaskPayload,
        next_no: int,
        *,
        existing_next_lookup: ExistingNextLookupCallback,
        short: ShortUuidCallback,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
    ) -> bool: ...
    def completion_preflight_context(
        self,
        task: TaskPayload,
        now_utc: datetime,
        *,
        services: CompletionPreflightServices,
    ) -> CompletionPreflightContext | None: ...


class CompletionComputeService(Protocol):
    """Validated compute service used by completion effects."""

    def completion_compute_child_due(
        self,
        task: TaskPayload,
        kind: str,
        *,
        compute_anchor_child_due: ComputeAnchorChildDueCallback,
        compute_cp_child_due: ComputeCpChildDueCallback,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
        diag: DiagnosticCallback | None = None,
        on_terminal: Callable[[OccurrenceSearchExhausted], bool] | None = None,
    ) -> tuple[datetime | None, dict[str, Any] | None, AnchorDNF | None] | None: ...
    def completion_until_or_fail(
        self,
        task: TaskPayload,
        now_utc: datetime,
        *,
        safe_parse_datetime: SafeParseDatetimeCallback,
        validate_until_not_past: ValidateUntilCallback,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
    ) -> datetime | None | Literal[False]: ...
    def completion_until_guard_or_stop(
        self,
        task: TaskPayload,
        child_due: datetime | None,
        until_dt: datetime | None,
        now_utc: datetime,
        *,
        end_chain_summary: EndChainSummaryCallback,
        print_task: PrintTaskCallback,
    ) -> bool: ...
    def completion_require_child_due_or_fail(
        self,
        task: TaskPayload,
        child_due: datetime | None,
        *,
        panel: PanelCallback,
        print_task: PrintTaskCallback,
    ) -> bool: ...
    def completion_warn_unreasonable_duration(
        self,
        task: TaskPayload,
        child_due: datetime | None,
        until_dt: datetime | None,
        now_utc: datetime,
        *,
        validate_chain_duration_reasonable: ValidateChainDurationCallback,
        panel: PanelCallback,
    ) -> None: ...
    def completion_caps(
        self,
        kind: str,
        task: TaskPayload,
        child_due: datetime | None,
        dnf: AnchorDNF | None,
        *,
        coerce_int: CoerceIntCallback,
        dtparse: DatetimeParserCallback,
        estimate_cp_final_by_max: EstimateCpFinalCallback,
        estimate_anchor_final_by_max: EstimateAnchorFinalCallback,
        cap_from_until_cp: CapFromUntilCpCallback,
        cap_from_until_anchor: CapFromUntilAnchorCallback,
    ) -> tuple[int, datetime | None, int | None, CompletionFinals, int | None]: ...
    def completion_cap_guard_or_stop(
        self,
        task: TaskPayload,
        next_no: int,
        cap_no: int | None,
        now_utc: datetime,
        *,
        end_chain_summary: EndChainSummaryCallback,
        print_task: PrintTaskCallback,
    ) -> bool: ...
    def completion_compute_next_and_limits(
        self,
        task: TaskPayload,
        kind: str,
        next_no: int,
        now_utc: datetime,
        *,
        services: CompletionComputeServices,
    ) -> CompletionComputeResult | CompletionLifecycleResult | None: ...
    def attach_lifecycle_plan(
        self,
        task: TaskPayload,
        computed: CompletionComputeResult,
        next_no: int,
        now_utc: datetime,
        *,
        preflight: CompletionPreflightContext | None,
        generation: ChainGenerationServicePort,
        scheduler_fingerprint: str,
        compare_datetimes: Callable[[datetime, datetime], int],
        invalid_relative_carry_reason: InvalidRelativeCarryReasonCallback,
        end_chain_summary: EndChainSummaryCallback,
        ensure_terminal_chain_off: Callable[[TaskPayload, str | None], bool],
        panel: PanelCallback,
        print_task: PrintTaskCallback,
        diag: DiagnosticCallback,
    ) -> CompletionComputeResult | CompletionLifecycleResult: ...


class CompletionSpawnService(Protocol):
    """Validated child-spawn service used by completion effects."""

    def completion_build_and_spawn_child(
        self,
        task: TaskPayload,
        *,
        child_due: datetime | None,
        child_field: str,
        next_no: int,
        parent_short: str,
        kind: str,
        cpmax: int,
        until_dt: datetime | None,
        lifecycle_plan: LifecyclePlan | None = None,
        services: CompletionSpawnServices,
    ) -> CompletionSpawnResult | None: ...


class _CompletionGenerationEffects(Protocol):
    def chain_generation_service(self, ports: GenerationPorts) -> ChainGenerationServicePort: ...

    def generation_ports_for(self, host: GenerationHost) -> GenerationPorts: ...


class _CompletionSpawnCore(Protocol):
    coerce_int: CoerceIntCallback
    fmt_isoz: Callable[[datetime], str]
    now_utc: Callable[[], datetime]


class _CompletionTaskCodecOwner(Protocol):
    DEFAULT_TASK_CODEC: TaskCodec


class _CompletionSpawnModelsOwner(Protocol):
    CompletionSpawnServices: type[CompletionSpawnServices]


class _CompletionSpawnEffectsOwner(Protocol):
    def spawn_child_ports_for(self, host: CompletionSpawnHost) -> SpawnChildPorts: ...

    def spawn_child_atomic(
        self,
        ports: SpawnChildPorts,
        child_task: TaskDraft | dict[str, Any],
        parent_task_with_nextlink: dict[str, Any],
        *,
        lifecycle_plan: LifecyclePlan | None = None,
    ) -> tuple[str, set[str], bool, bool, str | None, str | None]: ...


class _CompletionSpawnCommandOwner(Protocol):
    def command_ports_for(self, host: CommandHost) -> CommandPorts: ...

    def generate_child_uuid_candidate(
        self, ports: CommandPorts, env: Mapping[str, str]
    ) -> str: ...


class CompletionSpawnHost(Protocol):
    _INTEGRATION_CONTEXT: IntegrationContext | None
    TW_DATA_DIR: str
    _TASK_DATETIME_PARSER: TaskDatetimeParser
    _STABLE_CHILD_UUID_NAMESPACE: UUID
    _run_task_diag_bucket: Callable[[list[str]], str]
    _diag_count: DiagCounter
    _diag_record_run_task: RunTaskRecorder
    _diag: Callable[[str], None]
    _task_cmd_prefix: Callable[[], list[str]]
    _RECURRENCE_UPDATE_UDAS: tuple[str, ...]
    _DEBUG_WAIT_SCHED: bool
    _LAST_WAIT_SCHED_DEBUG: MutableMapping[str, dict[str, Any]] | None

    @property
    def core(self) -> _CompletionSpawnCore: ...

    def _modify_runtime_state(self) -> GenerationStatePort: ...

    @overload
    def _module(self, name: Literal["modify_completion_spawn"]) -> CompletionSpawnService: ...

    @overload
    def _module(self, name: Literal["modify_generation_effects"]) -> _CompletionGenerationEffects: ...

    @overload
    def _module(self, name: Literal["task_codec"]) -> _CompletionTaskCodecOwner: ...

    @overload
    def _module(self, name: Literal["modify_models"]) -> _CompletionSpawnModelsOwner: ...

    @overload
    def _module(self, name: Literal["modify_spawn_effects"]) -> _CompletionSpawnEffectsOwner: ...

    @overload
    def _module(self, name: Literal["modify_ui_effects"]) -> _CompletionUIModule: ...

    @overload
    def _module(self, name: Literal["modify_spawn_prep"]) -> _SpawnPreparationWithPayload: ...

    @overload
    def _module(self, name: Literal["modify_command_effects"]) -> _CompletionSpawnCommandOwner: ...

    @overload
    def _module(self, name: Literal["modify_task_fields"]) -> _ModifyTaskFields: ...

    @overload
    def _module(self, name: Literal["modify_spawn"]) -> _SpawnModule: ...

    @overload
    def _module(self, name: Literal["lifecycle_outbox"]) -> _LifecycleOutboxModule: ...

    @overload
    def _module(self, name: Literal["lifecycle_application"]) -> _LifecycleApplicationModule: ...

    @overload
    def _module(self, name: Literal["chain_generation"]) -> _ChainGenerationModule: ...


class TaskRowDecoder(Protocol):
    def __call__(self, row: Mapping[str, Any], *, source_query: str) -> TaskObservation: ...


class CompletionPreflightRepository(ChainSnapshotRepository, Protocol):
    def exact_child_slot(self, chain_id: str, link: int) -> TaskRead[TaskObservation]: ...


@dataclass(frozen=True, slots=True)
class SnapshotPorts:
    repository: ChainSnapshotRepository
    mode: Callable[[], str]
    snapshot_type: type[CompletionChainSnapshot]


@dataclass(frozen=True, slots=True)
class CompletionPreflightPorts:
    preflight: CompletionPreflightService
    coerce_int: CoerceIntCallback
    max_link_number: int
    short_uuid: Callable[[str | None], str]
    panel: PanelCallback
    print_task: PrintTaskCallback
    end_chain_summary: EndChainSummaryCallback
    existing_next_lookup: ExistingNextLookupCallback


@dataclass(frozen=True, slots=True)
class CompletionFeedbackPorts:
    compute: CompletionComputeService
    panel: PanelCallback
    print_task: PrintTaskCallback
    end_chain_summary: EndChainSummaryCallback


@dataclass(frozen=True, slots=True)
class UntilCompletionPorts:
    compute: CompletionComputeService
    parse_datetime: SafeParseDatetimeCallback
    validate_until_not_past: ValidateUntilCallback
    panel: PanelCallback
    print_task: PrintTaskCallback


@dataclass(frozen=True, slots=True)
class CompletionCapsPorts:
    compute: CompletionComputeService
    coerce_int: CoerceIntCallback
    parse_datetime: DatetimeParserCallback
    estimate_cp: EstimateCpFinalCallback
    estimate_anchor: EstimateAnchorFinalCallback
    cap_cp: CapFromUntilCpCallback
    cap_anchor: CapFromUntilAnchorCallback


@dataclass(frozen=True, slots=True)
class ChildDuePorts:
    compute: CompletionComputeService
    generation: ChainGenerationServicePort
    decode_task: TaskRowDecoder
    task_type: type[NauticalTask]
    exhaustion_message: Callable[[OccurrenceSearchExhausted], str]
    ensure_terminal: Callable[[TaskPayload, str | None], bool]
    end_summary: EndChainSummaryCallback
    now_utc: Callable[[], datetime]
    panel: PanelCallback
    print_task: PrintTaskCallback
    diag: DiagnosticCallback


@dataclass(frozen=True, slots=True)
class DurationWarningPorts:
    compute: CompletionComputeService
    validate_duration: ValidateChainDurationCallback
    panel: PanelCallback


@dataclass(frozen=True, slots=True)
class CompletionLifecyclePlanPorts:
    generation: ChainGenerationServicePort
    scheduler_fingerprint: Callable[[], str]
    compare_datetimes: Callable[[datetime, datetime], int]
    invalid_relative_carry_reason: InvalidRelativeCarryReasonCallback
    end_chain_summary: EndChainSummaryCallback
    ensure_terminal_chain_off: Callable[[TaskPayload, str | None], bool]
    panel: PanelCallback
    print_task: PrintTaskCallback
    diagnostic: DiagnosticCallback


@dataclass(frozen=True, slots=True)
class CompletionComputePorts:
    compute: CompletionComputeService
    services_type: type[CompletionComputeServices]
    compute_child_due: CompletionChildDueCallback
    until_or_fail: CompletionUntilCallback
    until_guard_or_stop: CompletionUntilGuardCallback
    require_child_due_or_fail: CompletionChildRequiredCallback
    warn_unreasonable_duration: CompletionDurationWarningCallback
    caps: CompletionCapsCallback
    cap_guard_or_stop: CompletionCapGuardCallback
    lifecycle_plan: CompletionLifecyclePlanPorts


@dataclass(frozen=True, slots=True)
class CompletionPreflightContextPorts:
    preflight: CompletionPreflightService
    services_type: type[CompletionPreflightServices]
    snapshot_type: type[CompletionChainSnapshot]
    snapshot_mode: Callable[[], str]
    coerce_int: CoerceIntCallback
    max_link_number: int
    short_uuid: ShortUuidCallback
    panel: PanelCallback
    print_task: PrintTaskCallback
    end_chain_summary: EndChainSummaryCallback


@dataclass(frozen=True, slots=True)
class CompletionSpawnPorts:
    spawn: CompletionSpawnService
    services_type: type[CompletionSpawnServices]
    build_child_draft: BuildChildDraftCallback
    spawn_child_atomic: SpawnChildCallback
    panel: PanelCallback
    print_task: PrintTaskCallback
    diagnostic: Callable[[str], None]


class _CompletionPreflightCore(Protocol):
    PANEL_MODE: str
    MAX_LINK_NUMBER: int
    coerce_int: CoerceIntCallback
    short_uuid: ShortUuidCallback


class _CompletionModelsOwner(Protocol):
    CompletionPreflightServices: type[CompletionPreflightServices]
    CompletionChainSnapshot: type[CompletionChainSnapshot]
    CompletionComputeServices: type[CompletionComputeServices]


class _CompletionComputeCore(Protocol):
    coerce_int: CoerceIntCallback
    humanize_delta: Callable[..., str]
    fmt_dt_local: Callable[[datetime], str]
    scheduler_config_fingerprint: Callable[[], str] | None
    parse_cp_sequence_tokens: Callable[[str], list[CPSequenceToken] | None]
    cp_sequence_interval_for_token: SequenceIntervalForToken
    build_local_datetime: Callable[[date, tuple[int, int]], datetime]

    def _import_sibling(self, name: Literal["scheduler_models"]) -> _CompletionSchedulerModelsOwner: ...


class _CompletionComputeRuntimeState(Protocol):
    scheduler_services: dict[Any, Any]
    diag_stats: dict[str, Any]
    workflow_context: Any
    chain_generation_service: ChainGenerationServicePort | None


class _CompletionSchedulerModelsOwner(Protocol):
    def occurrence_exhaustion_message(self, error: OccurrenceSearchExhausted) -> str: ...


class _CompletionTaskModelsOwner(Protocol):
    NauticalTask: type[NauticalTask]


class _CompletionValidationEffectsOwner(Protocol):
    UntilPorts: type[UntilPortsModel]
    DurationPorts: type[DurationPortsModel]

    def until_not_past(
        self, ports: UntilPortsModel, until_dt: datetime | None, now_utc: datetime
    ) -> tuple[bool, str | None]: ...

    def chain_duration_reasonable(
        self,
        ports: DurationPortsModel,
        child_due: datetime | None,
        until_dt: datetime | None,
        now_utc: datetime,
    ) -> tuple[bool, str | None]: ...


class _CompletionScheduleEffectsOwner(Protocol):
    def cp_completion_ports_for(self, host: CompletionComputeHost) -> CPCompletionPorts: ...

    def anchor_completion_ports_for(self, host: CompletionComputeHost) -> AnchorCompletionPorts: ...

    def estimate_cp_final_by_max(
        self, ports: CPCompletionPorts, task: TaskPayload, next_due_utc: datetime | None
    ) -> datetime | None: ...

    def estimate_anchor_final_by_max(
        self,
        ports: AnchorCompletionPorts,
        task: TaskPayload,
        next_due_utc: datetime | None,
        dnf: AnchorDNF | None,
    ) -> datetime | None: ...

    def cap_from_until_cp(
        self, ports: CPCompletionPorts, task: TaskPayload, next_due_utc: datetime | None
    ) -> tuple[int | None, datetime | None]: ...

    def cap_from_until_anchor(
        self,
        ports: AnchorCompletionPorts,
        task: TaskPayload,
        next_due_utc: datetime | None,
        dnf: AnchorDNF | None,
    ) -> tuple[int | None, datetime | None]: ...


class _EnsureTerminalChainOffAdapter(Protocol):
    def __call__(
        self,
        compute_host: "CompletionComputeHost",
        task: TaskPayload,
        event: str | None = None,
    ) -> bool: ...


class _CompletionCompositionAdaptersOwner(Protocol):
    ensure_terminal_chain_off_for: _EnsureTerminalChainOffAdapter


class _CompletionDiagnosticsEffectsOwner(Protocol):
    def end_chain_summary_ports_for(self, host: CompletionComputeHost) -> EndChainSummaryPorts: ...

    def end_chain_summary(
        self,
        ports: EndChainSummaryPorts,
        task: TaskPayload,
        reason: str,
        now_utc: datetime,
        current_task: TaskPayload | None = None,
    ) -> None: ...


class _CompletionValueEffectsOwner(Protocol):
    DatetimePorts: type[DatetimePortsModel]

    def compare_datetimes(
        self, ports: DatetimePortsModel, left: datetime, right: datetime
    ) -> int: ...


class _CompletionChainIntegrityLifecycleOwner(Protocol):
    invalid_relative_carry_reason: InvalidRelativeCarryReasonCallback


class CompletionComputeHost(Protocol):
    core: _CompletionComputeCore
    _TASK_DATETIME_PARSER: TaskDatetimeParser
    _tolocal: Callable[[datetime], datetime]
    _to_local_cached: Callable[[datetime], datetime]
    _anchor_file_fallback_hhmm: Callable[[TaskPayload, datetime], tuple[int, int]]
    _anchor_file_provider_for: AnchorFileProviderFactory
    _MAX_ITERATIONS: int
    _RECURRENCE_UPDATE_UDAS: tuple[str, ...]
    _DEBUG_WAIT_SCHED: bool
    _LAST_WAIT_SCHED_DEBUG: MutableMapping[str, dict[str, Any]] | None
    _workflow_now_utc: Callable[[], datetime]
    _MIN_FUTURE_WARN: int
    timedelta: type[Timedelta]
    _diag: DiagnosticCallback

    def _modify_runtime_state(self) -> _CompletionComputeRuntimeState: ...

    @overload
    def _module(self, name: Literal["modify_completion_compute"]) -> CompletionComputeService: ...

    @overload
    def _module(self, name: Literal["modify_models"]) -> _CompletionModelsOwner: ...

    @overload
    def _module(self, name: Literal["modify_schedule_effects"]) -> _CompletionScheduleEffectsOwner: ...

    @overload
    def _module(self, name: Literal["modify_validation_effects"]) -> _CompletionValidationEffectsOwner: ...

    @overload
    def _module(self, name: Literal["modify_generation_effects"]) -> _CompletionGenerationEffects: ...

    @overload
    def _module(self, name: Literal["modify_composition_adapters"]) -> _CompletionCompositionAdaptersOwner: ...

    @overload
    def _module(self, name: Literal["task_codec"]) -> _CompletionTaskCodecOwner: ...

    @overload
    def _module(self, name: Literal["task_models"]) -> _CompletionTaskModelsOwner: ...

    @overload
    def _module(self, name: Literal["modify_ui_effects"]) -> _CompletionUIModule: ...

    @overload
    def _module(self, name: Literal["modify_diagnostics_effects"]) -> _CompletionDiagnosticsEffectsOwner: ...

    @overload
    def _module(self, name: Literal["modify_value_effects"]) -> _CompletionValueEffectsOwner: ...

    @overload
    def _module(self, name: Literal["chain_integrity_lifecycle"]) -> _CompletionChainIntegrityLifecycleOwner: ...

    @overload
    def _module(self, name: Literal["modify_runtime"]) -> _ModifyRuntimeModule: ...

    @overload
    def _module(self, name: Literal["modify_anchor_effects"]) -> _ModifyAnchorCompletionEffects: ...

    @overload
    def _module(self, name: Literal["chain_generation"]) -> _ChainGenerationModule: ...


class CompletionPreflightHost(Protocol):
    _SHOW_ANALYTICS: bool
    _CHECK_CHAIN_INTEGRITY: bool

    @property
    def core(self) -> _CompletionPreflightCore: ...

    @overload
    def _module(
        self, name: Literal["modify_completion_preflight"]
    ) -> CompletionPreflightService: ...

    @overload
    def _module(self, name: Literal["modify_models"]) -> _CompletionModelsOwner: ...


class _CompletionUIModule(Protocol):
    def ui_ports_for(self, host: Any) -> UIEffectsPorts: ...

    def print_task(self, ports: UIEffectsPorts, task: TaskPayload) -> None: ...

    def panel(
        self,
        ports: UIEffectsPorts,
        title: Any,
        rows: Any,
        kind: str = "info",
        border_style: Any = None,
        title_style: Any = None,
        label_style: Any = None,
    ) -> Any: ...


def _ui_ports_for(host: Any) -> tuple[_CompletionUIModule, UIEffectsPorts]:
    ui = host._module("modify_ui_effects")
    return ui, ui.ui_ports_for(host)


def _print_task_port_for(host: Any) -> PrintTaskCallback:
    ui, ports = _ui_ports_for(host)
    return lambda task: ui.print_task(ports, task)


def _end_summary_port_for(host: Any) -> EndChainSummaryCallback:
    diagnostics = host._module("modify_diagnostics_effects")
    ports = diagnostics.end_chain_summary_ports_for(host)
    return lambda task, reason, now, current_task=None: diagnostics.end_chain_summary(
        ports, task, reason, now, current_task
    )


def _feedback_ports_for(
    host: Any, compute: CompletionComputeService, *, summarize: bool = True
) -> CompletionFeedbackPorts:
    return CompletionFeedbackPorts(
        compute=compute,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        end_chain_summary=_end_summary_port_for(host) if summarize else (lambda *args, **kwargs: None),
    )


def _panel_port_for(host: Any) -> PanelCallback:
    ui, ports = _ui_ports_for(host)
    return lambda title, rows, **kwargs: ui.panel(ports, title, rows, **kwargs)


def link_numbers_or_fail(
    ports: CompletionPreflightPorts, new: TaskPayload
) -> tuple[int, int] | None:
    return ports.preflight.completion_link_numbers_or_fail(
        new,
        coerce_int=ports.coerce_int, max_link_number=ports.max_link_number,
        panel=ports.panel, print_task=ports.print_task,
    )


def kind_or_stop(
    ports: CompletionPreflightPorts, new: TaskPayload, now_utc: datetime
) -> str | None:
    return ports.preflight.completion_kind_or_stop(
        new,
        now_utc,
        panel=ports.panel,
        print_task=ports.print_task,
        end_chain_summary=ports.end_chain_summary,
    )


def chain_id_or_fail(ports: CompletionPreflightPorts, new: TaskPayload) -> str | None:
    return ports.preflight.completion_chain_id_or_fail(
        new,
        panel=ports.panel, print_task=ports.print_task,
    )


def existing_next_or_fail(
    ports: CompletionPreflightPorts,
    new: TaskPayload,
    next_no: int,
    chain_snapshot: CompletionChainSnapshot | None,
) -> bool:
    return ports.preflight.completion_existing_next_or_fail(
        new,
        next_no,
        existing_next_lookup=ports.existing_next_lookup,
        short=ports.short_uuid, panel=ports.panel, print_task=ports.print_task,
    )


def _snapshot_mode(ports: SnapshotPorts) -> str:
    return ports.mode()


def chain_snapshot(
    ports: SnapshotPorts, chain_id: str, base_no: int, next_no: int
) -> CompletionChainSnapshot:
    del base_no, next_no
    from .integration_models import Absent, Found, Unavailable

    snapshot = ports.repository.chain_snapshot(chain_id)
    if isinstance(snapshot, Found):
        if isinstance(snapshot.value, AuthoritativeTaskSnapshot):
            value = snapshot.value.rows
        elif isinstance(snapshot.value, tuple):
            value = snapshot.value
        else:
            return ports.snapshot_type(
                mode=_snapshot_mode(ports), rows=[], loaded=False,
                chain_id=chain_id, error="typed chain snapshot rows are unavailable"
            )
        rows = list(value)
        loaded, error = True, ""
    elif isinstance(snapshot, Absent):
        rows, loaded, error = [], True, ""
    elif isinstance(snapshot, Unavailable):
        rows, loaded = [], False
        error = snapshot.evidence.detail or snapshot.evidence.kind.value
    else:
        rows, loaded, error = [], False, "typed chain read returned an unsupported result"
    return ports.snapshot_type(
        mode=_snapshot_mode(ports), rows=rows, loaded=loaded, chain_id=chain_id, error=error
    )


def completion_preflight_context_ports_for(
    host: CompletionPreflightHost,
) -> CompletionPreflightContextPorts:
    preflight = host._module("modify_completion_preflight")
    models = host._module("modify_models")
    return CompletionPreflightContextPorts(
        preflight=preflight,
        services_type=models.CompletionPreflightServices,
        snapshot_type=models.CompletionChainSnapshot,
        snapshot_mode=lambda: (
            "full" if host._SHOW_ANALYTICS or host._CHECK_CHAIN_INTEGRITY else
            "next" if str(getattr(host.core, "PANEL_MODE", "rich") or "rich").strip().lower()
            in {"line", "minimal", "quiet", "text"} else "recent"
        ),
        coerce_int=host.core.coerce_int,
        max_link_number=host.core.MAX_LINK_NUMBER,
        short_uuid=host.core.short_uuid,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        end_chain_summary=_end_summary_port_for(host),
    )


def preflight_context(
    ports: CompletionPreflightContextPorts,
    new: TaskPayload,
    now_utc: datetime,
    repository: CompletionPreflightRepository,
) -> CompletionPreflightContext | None:
    preflight = ports.preflight
    snapshot_ports = SnapshotPorts(
        repository=repository,
        mode=ports.snapshot_mode,
        snapshot_type=ports.snapshot_type,
    )
    preflight_ports = CompletionPreflightPorts(
        preflight=preflight,
        coerce_int=ports.coerce_int,
        max_link_number=ports.max_link_number,
        short_uuid=ports.short_uuid,
        panel=ports.panel,
        print_task=ports.print_task,
        end_chain_summary=ports.end_chain_summary,
        existing_next_lookup=lambda task, link: repository.exact_child_slot(str(task.get("chainID") or ""), link),
    )
    services = ports.services_type(
        short=ports.short_uuid,
        completion_link_numbers_or_fail=lambda task: link_numbers_or_fail(preflight_ports, task),
        completion_kind_or_stop=lambda task, clock: kind_or_stop(
            preflight_ports, task, clock
        ),
        completion_chain_id_or_fail=lambda task: chain_id_or_fail(preflight_ports, task),
        completion_chain_snapshot=lambda chain_id, base_no, next_no: chain_snapshot(snapshot_ports, chain_id, base_no, next_no),
        completion_existing_next_or_fail=lambda task, next_no, snapshot: existing_next_or_fail(preflight_ports, task, next_no, snapshot),
    )
    return preflight.completion_preflight_context(new, now_utc, services=services)


def compute_child_due(
    ports: ChildDuePorts, new: TaskPayload, kind: str
) -> tuple[datetime | None, dict[str, Any] | None, AnchorDNF | None] | None:
    compute = ports.compute

    def typed_task(task: TaskPayload) -> NauticalTask:
        return ports.task_type.from_observation(
            ports.decode_task(task, source_query="on-modify completion")
        )

    def handle_terminal(exc: OccurrenceSearchExhausted) -> bool:
        message = ports.exhaustion_message(exc)
        if exc.is_date_limit:
            ports.ensure_terminal(new, "complete")
            try:
                ports.end_summary(new, message, ports.now_utc(), current_task=new)
            except Exception as summary_exc:
                ports.diag(f"terminal chain summary failed: {summary_exc}")
                ports.panel("⛔ Nautical chain stopped", [("Reason", message), ("Task", str(new.get("uuid") or "")[:8] or "–")], kind="summary")
            ports.print_task(new)
            return True
        ports.panel("⛔ Chain error", [("Scheduler", message), ("Fix", "Use a less sparse rule or adjust its search limits.")], kind="error")
        ports.print_task(new)
        return True

    return compute.completion_compute_child_due(
        new,
        kind,
        compute_anchor_child_due=lambda task: ports.generation.compute_anchor_child_due(typed_task(task)),
        compute_cp_child_due=lambda task: ports.generation.compute_cp_child_due(typed_task(task)),
        panel=ports.panel, print_task=ports.print_task, diag=ports.diag,
        on_terminal=handle_terminal,
    )


def until_or_fail(
    ports: UntilCompletionPorts, new: TaskPayload, now_utc: datetime
) -> datetime | None | Literal[False]:
    return ports.compute.completion_until_or_fail(
        new, now_utc,
        safe_parse_datetime=ports.parse_datetime,
        validate_until_not_past=ports.validate_until_not_past,
        panel=ports.panel, print_task=ports.print_task,
    )


def until_guard_or_stop(
    ports: CompletionFeedbackPorts,
    new: TaskPayload,
    child_due: datetime | None,
    until_dt: datetime | None,
    now_utc: datetime,
) -> bool:
    return ports.compute.completion_until_guard_or_stop(
        new, child_due, until_dt, now_utc,
        end_chain_summary=ports.end_chain_summary, print_task=ports.print_task,
    )


def require_child_due_or_fail(
    ports: CompletionFeedbackPorts, new: TaskPayload, child_due: datetime | None
) -> bool:
    return ports.compute.completion_require_child_due_or_fail(
        new, child_due, panel=ports.panel, print_task=ports.print_task
    )


def warn_unreasonable_duration(
    ports: DurationWarningPorts,
    new: TaskPayload,
    child_due: datetime | None,
    until_dt: datetime | None,
    now_utc: datetime,
) -> None:
    ports.compute.completion_warn_unreasonable_duration(
        new, child_due, until_dt, now_utc,
        validate_chain_duration_reasonable=ports.validate_duration,
        panel=ports.panel,
    )


def caps(
    ports: CompletionCapsPorts,
    kind: str,
    new: TaskPayload,
    child_due: datetime | None,
    dnf: AnchorDNF | None,
) -> tuple[int, datetime | None, int | None, CompletionFinals, int | None]:
    return ports.compute.completion_caps(
        kind, new, child_due, dnf,
        coerce_int=ports.coerce_int, dtparse=ports.parse_datetime,
        estimate_cp_final_by_max=ports.estimate_cp,
        estimate_anchor_final_by_max=ports.estimate_anchor,
        cap_from_until_cp=ports.cap_cp,
        cap_from_until_anchor=ports.cap_anchor,
    )


def cap_guard_or_stop(ports: CompletionFeedbackPorts, new: TaskPayload, next_no: int, cap_no: int | None, now_utc: datetime) -> bool:
    return ports.compute.completion_cap_guard_or_stop(
        new, next_no, cap_no, now_utc,
        end_chain_summary=ports.end_chain_summary, print_task=ports.print_task
    )


def completion_compute_ports_for(host: CompletionComputeHost) -> CompletionComputePorts:
    compute = host._module("modify_completion_compute")
    models = host._module("modify_models")
    schedule = host._module("modify_schedule_effects")
    validation = host._module("modify_validation_effects")
    generation_module = host._module("modify_generation_effects")
    generation = generation_module.chain_generation_service(
        generation_module.generation_ports_for(host)
    )
    cp_schedule_ports = schedule.cp_completion_ports_for(host)
    anchor_schedule_ports = schedule.anchor_completion_ports_for(host)
    feedback = _feedback_ports_for(host, compute)
    feedback_without_summary = _feedback_ports_for(host, compute, summarize=False)
    ensure_terminal: Callable[[TaskPayload, str | None], bool] = partial(
        host._module("modify_composition_adapters").ensure_terminal_chain_off_for,
        host,
    )
    child_due_ports = ChildDuePorts(
        compute=compute,
        generation=generation,
        decode_task=host._module("task_codec").DEFAULT_TASK_CODEC.decode_row,
        task_type=host._module("task_models").NauticalTask,
        exhaustion_message=host.core._import_sibling("scheduler_models").occurrence_exhaustion_message,
        ensure_terminal=ensure_terminal,
        end_summary=_end_summary_port_for(host),
        now_utc=host._workflow_now_utc,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        diag=host._diag,
    )
    until_ports = UntilCompletionPorts(
        compute=compute,
        parse_datetime=host._TASK_DATETIME_PARSER.parse,
        validate_until_not_past=lambda until_dt, now: validation.until_not_past(
            validation.UntilPorts(
                lambda _now: host.timedelta(minutes=1),
                compare_datetimes,
                host.core.humanize_delta,
            ),
            until_dt,
            now,
        ),
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
    )
    duration_ports = DurationWarningPorts(
        compute=compute,
        validate_duration=lambda child_due, until_dt, now: validation.chain_duration_reasonable(
            validation.DurationPorts(host._MIN_FUTURE_WARN, host.core.fmt_dt_local),
            child_due,
            until_dt,
            now,
        ),
        panel=_panel_port_for(host),
    )
    caps_ports = CompletionCapsPorts(
        compute=compute,
        coerce_int=host.core.coerce_int,
        parse_datetime=lambda value: datetime_value(parser_for_host(host), value),
        estimate_cp=lambda task, due: schedule.estimate_cp_final_by_max(
            cp_schedule_ports, task, due
        ),
        estimate_anchor=lambda task, due, expression: schedule.estimate_anchor_final_by_max(
            anchor_schedule_ports, task, due, expression
        ),
        cap_cp=lambda task, due: schedule.cap_from_until_cp(cp_schedule_ports, task, due),
        cap_anchor=lambda task, due, expression: schedule.cap_from_until_anchor(
            anchor_schedule_ports, task, due, expression
        ),
    )
    fingerprint = host.core.scheduler_config_fingerprint
    ensure_terminal_chain_off: Callable[[TaskPayload, str | None], bool] = partial(
        host._module("modify_composition_adapters").ensure_terminal_chain_off_for,
        host,
    )
    plan_ports = CompletionLifecyclePlanPorts(
        generation=generation,
        scheduler_fingerprint=fingerprint if fingerprint is not None else (lambda: ""),
        compare_datetimes=lambda left, right: host._module(
            "modify_value_effects"
        ).compare_datetimes(
            host._module("modify_value_effects").DatetimePorts(
                compare_datetimes
            ),
            left,
            right,
        ),
        invalid_relative_carry_reason=host._module(
            "chain_integrity_lifecycle"
        ).invalid_relative_carry_reason,
        end_chain_summary=_end_summary_port_for(host),
        ensure_terminal_chain_off=ensure_terminal_chain_off,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        diagnostic=host._diag,
    )
    return CompletionComputePorts(
        compute=compute,
        services_type=models.CompletionComputeServices,
        compute_child_due=lambda value, value_kind: compute_child_due(
            child_due_ports, value, value_kind
        ),
        until_or_fail=lambda value, clock: until_or_fail(until_ports, value, clock),
        until_guard_or_stop=lambda value, due, until, clock: until_guard_or_stop(
            feedback, value, due, until, clock
        ),
        require_child_due_or_fail=lambda value, due: require_child_due_or_fail(
            feedback_without_summary, value, due
        ),
        warn_unreasonable_duration=lambda value, due, until, clock: warn_unreasonable_duration(
            duration_ports, value, due, until, clock
        ),
        caps=lambda value_kind, value, due, dnf: caps(
            caps_ports, value_kind, value, due, dnf
        ),
        cap_guard_or_stop=lambda value, number, cap, clock: cap_guard_or_stop(
            feedback, value, number, cap, clock
        ),
        lifecycle_plan=plan_ports,
    )


def compute_next_and_limits(
    ports: CompletionComputePorts,
    new: TaskPayload,
    kind: str,
    next_no: int,
    now_utc: datetime,
    *,
    preflight: CompletionPreflightContext | None = None,
) -> CompletionComputeResult | CompletionLifecycleResult | None:
    services = ports.services_type(
        completion_compute_child_due=ports.compute_child_due,
        completion_until_or_fail=ports.until_or_fail,
        completion_until_guard_or_stop=ports.until_guard_or_stop,
        completion_require_child_due_or_fail=ports.require_child_due_or_fail,
        completion_warn_unreasonable_duration=ports.warn_unreasonable_duration,
        completion_caps=ports.caps,
        completion_cap_guard_or_stop=ports.cap_guard_or_stop,
    )
    computed = ports.compute.completion_compute_next_and_limits(
        new, kind, next_no, now_utc, services=services
    )
    if computed is None:
        return None
    if isinstance(computed, CompletionLifecycleResult):
        return computed
    if not str(new.get("uuid") or "").strip() or not str(new.get("chainID") or "").strip():
        return computed
    plan = ports.lifecycle_plan
    return ports.compute.attach_lifecycle_plan(
        new,
        computed,
        next_no,
        now_utc,
        preflight=preflight,
        generation=plan.generation,
        scheduler_fingerprint=plan.scheduler_fingerprint(),
        compare_datetimes=plan.compare_datetimes,
        invalid_relative_carry_reason=plan.invalid_relative_carry_reason,
        end_chain_summary=plan.end_chain_summary,
        ensure_terminal_chain_off=plan.ensure_terminal_chain_off,
        panel=plan.panel,
        print_task=plan.print_task,
        diag=plan.diagnostic,
    )


def completion_spawn_ports_for(host: CompletionSpawnHost) -> CompletionSpawnPorts:
    spawn = host._module("modify_completion_spawn")
    generation_module = host._module("modify_generation_effects")
    generation = generation_module.chain_generation_service(generation_module.generation_ports_for(host))
    codec = host._module("task_codec")
    models = host._module("modify_models")

    def build_child_draft(
        task: TaskPayload,
        child_due: datetime,
        child_field: str,
        next_no: int,
        parent_short: str,
        kind: str,
        cpmax: int,
        until_dt: datetime | None,
    ) -> TaskDraft:
        typed_task = NauticalTask.from_observation(
            codec.DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify completion")
        )
        return generation.build_child_draft(
            typed_task,
            child_due,
            child_field,
            next_no,
            parent_short,
            kind,
            cpmax,
            until_dt,
        )

    spawn_effects = host._module("modify_spawn_effects")
    spawn_ports = spawn_effects.spawn_child_ports_for(host)
    def spawn_child_atomic(
        child: TaskDraft | TaskPayload,
        parent: TaskPayload,
        *,
        lifecycle_plan: LifecyclePlan | None = None,
    ) -> tuple[str, list[str], bool, bool, str | None, str | None]:
        child_mapping = child.to_mapping() if isinstance(child, TaskDraft) else dict(child)
        result = spawn_effects.spawn_child_atomic(
            spawn_ports,
            child_mapping,
            dict(parent),
            lifecycle_plan=lifecycle_plan,
        )
        return _normalize_completion_spawn_result(result)

    return CompletionSpawnPorts(
        spawn=spawn,
        services_type=models.CompletionSpawnServices,
        build_child_draft=build_child_draft,
        spawn_child_atomic=spawn_child_atomic,
        panel=_panel_port_for(host),
        print_task=_print_task_port_for(host),
        diagnostic=host._diag,
    )


def _normalize_completion_spawn_result(
    result: tuple[str, set[str], bool, bool, str | None, str | None],
) -> tuple[str, list[str], bool, bool, str | None, str | None]:
    child_short, stripped_attrs, verified, deferred, reason, intent_id = result
    return child_short, sorted(stripped_attrs), verified, deferred, reason, intent_id


def build_and_spawn_child(
    ports: CompletionSpawnPorts,
    new: TaskPayload,
    *,
    child_due: datetime | None,
    child_field: str = "due",
    next_no: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt: datetime | None,
    lifecycle_plan: LifecyclePlan | None = None,
) -> CompletionSpawnResult | None:
    services = ports.services_type(
        build_child_draft=ports.build_child_draft,
        spawn_child_atomic=ports.spawn_child_atomic,
        panel=ports.panel,
        print_task=ports.print_task,
        diag=ports.diagnostic,
    )
    return ports.spawn.completion_build_and_spawn_child(
        new,
        child_due=child_due,
        child_field=child_field,
        next_no=next_no,
        parent_short=parent_short,
        kind=kind,
        cpmax=cpmax,
        until_dt=until_dt,
        lifecycle_plan=lifecycle_plan,
        services=services,
    )


__all__ = (
    "CompletionComputePorts", "CompletionLifecyclePlanPorts",
    "CompletionPreflightContextPorts", "CompletionSpawnPorts",
    "completion_compute_ports_for", "completion_preflight_context_ports_for",
    "completion_spawn_ports_for",
    "link_numbers_or_fail", "kind_or_stop", "chain_id_or_fail", "existing_next_or_fail", "chain_snapshot", "preflight_context",
    "compute_child_due", "until_or_fail", "until_guard_or_stop", "require_child_due_or_fail",
    "warn_unreasonable_duration", "caps", "cap_guard_or_stop",
    "compute_next_and_limits", "build_and_spawn_child",
)
