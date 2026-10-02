"""Composition services for the on-modify hook."""

from __future__ import annotations

import re
from dataclasses import dataclass
from contextlib import nullcontext
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence
from .callback_ports import CallbackPort
from .task_datetime import datetime_value, parser_for_core
from .task_models import NauticalTask, TaskObservation, TaskPayload

if TYPE_CHECKING:
    from .lifecycle.models import LifecyclePlan
    from .modify_models import CompletionLifecycleResult, TaskView
    from .modify_workflow import RecurrenceTransitionDecision
    from .modify_validation_effects import (
        AnchorValidationPorts,
        ChainLimitPorts,
        CPValidationPorts as CPValidationPortsContract,
        NativeUntilPorts,
        NativeUntilSlotPorts,
        OmitValidationPorts,
        SharedValidationPorts as SharedValidationPortsContract,
    )


class ModifyCallback(Protocol):
    """Callable service port used by modify-route composition services."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class _NativeUntilGenerationService(Protocol):
    def carry_native_until(
        self,
        parent: NauticalTask,
        child: TaskPayload,
        child_due_utc: datetime,
        kind: str,
        *,
        parent_anchor_field: str,
        child_anchor_field: str,
    ) -> None: ...


class _ModifyGenerationEffects(Protocol):
    def generation_ports_for(self, host: Any) -> object: ...

    def chain_generation_service(self, ports: object) -> _NativeUntilGenerationService: ...


class _DefaultTaskCodec(Protocol):
    def decode_row(
        self,
        row: Mapping[str, Any],
        *,
        source_query: str,
    ) -> TaskObservation: ...


class _TaskCodecModule(Protocol):
    DEFAULT_TASK_CODEC: _DefaultTaskCodec


class _TaskModelsModule(Protocol):
    NauticalTask: type[NauticalTask]


class _ModifySpawnEffects(Protocol):
    def spawn_intent_ports_for(self, host: Any) -> object: ...

    def enqueue_spawn_intent(
        self,
        ports: object,
        plan: "LifecyclePlan",
    ) -> tuple[bool, str]: ...


class _ModifyPresentationEffects(Protocol):
    def lifecycle_result_port_for(self, host: Any) -> object: ...

    def render_lifecycle_result(
        self,
        ports: object,
        result: "CompletionLifecycleResult",
        task: "TaskView",
    ) -> None: ...


class _ModifyDiagnosticsEffects(Protocol):
    def analytics_ports_for(self, host: Any) -> object: ...

    def chain_health_advice(
        self,
        ports: object,
        chain: Sequence[TaskPayload],
        kind: str,
        task: TaskPayload,
        tol_secs: int,
        style: str,
    ) -> Any: ...

    def chain_integrity_warnings(
        self,
        ports: object,
        chain: Sequence[TaskPayload],
        expected_chain_id: str | None = None,
    ) -> list[str]: ...


class _ModifyOrdinaryEffects(Protocol):
    OrdinaryModifyServices: _ServiceFactory
    RecurrenceActivationError: type[Exception]

    def handle_non_completion_modify(
        self,
        old: TaskPayload,
        new: TaskPayload,
        *,
        services: Any,
        lifecycle: Any,
        transition: Any = None,
    ) -> None: ...


class _ModifyCompositionAdapters(Protocol):
    def handle_non_completion(
        self,
        host: Any,
        old: TaskPayload,
        new: TaskPayload,
        unit_of_work: Any,
        *,
        transition: Any = None,
        runtime: Any = None,
    ) -> None: ...

    def handle_completion(
        self,
        host: Any,
        old: TaskPayload,
        new: TaskPayload,
        unit_of_work: Any,
        *,
        transition: Any = None,
        runtime: Any = None,
    ) -> Any: ...

    def handle_deleted(
        self,
        host: Any,
        old: TaskPayload,
        new: TaskPayload,
        unit_of_work: Any,
        *,
        transition: Any = None,
        terminal_decision: Any = None,
        runtime: Any = None,
    ) -> None: ...

    def render_anchor_completion_feedback_for(self, host: Any, *, request: Any) -> None: ...

    def render_cp_completion_feedback_for(self, host: Any, *, request: Any) -> None: ...


class _TaskHookResponseFactory(Protocol):
    def __call__(
        self,
        task: TaskPayload,
        sanitize: bool = False,
        prof: Any | None = None,
    ) -> Any: ...


class _HookResults(Protocol):
    TaskHookResponse: _TaskHookResponseFactory

    def emit_passthrough_json(self, task: TaskPayload) -> None: ...

    def emit_json_result(self, result: Any, *, core: Any = None) -> None: ...


class _HookContext(Protocol):
    def build_on_modify_request(
        self,
        *,
        runtime: Any,
        old: TaskPayload,
        new: TaskPayload,
        old_observation: TaskObservation | None = None,
        new_observation: TaskObservation | None = None,
    ) -> Any: ...


class _HookEngine(Protocol):
    def handle_on_modify(self, request: Any, *, services: Any) -> Any: ...


class _ModifyDeletionDiagnosticsEffects(_ModifyDiagnosticsEffects, Protocol):
    def end_chain_summary_ports_for(self, host: Any) -> object: ...

    def end_chain_summary(
        self,
        ports: object,
        task: Any,
        reason: str,
        now_utc: datetime,
        *,
        current_task: Any = None,
    ) -> Any: ...


class _ModifyCompletionEffects(Protocol):
    def completion_preflight_context_ports_for(self, host: Any) -> object: ...

    def completion_compute_ports_for(self, host: Any) -> object: ...

    def completion_spawn_ports_for(self, host: Any) -> object: ...

    def preflight_context(
        self,
        ports: object,
        new: TaskPayload,
        now_utc: datetime,
        repository: Any,
    ) -> Any: ...

    def compute_next_and_limits(
        self,
        ports: object,
        new: TaskPayload,
        kind: str,
        next_no: int,
        now_utc: datetime,
        *,
        preflight: Any = None,
    ) -> Any: ...

    def build_and_spawn_child(self, ports: object, new: TaskPayload, **kwargs: Any) -> Any: ...


class _ModifyUIEffects(Protocol):
    def ui_ports_for(self, host: Any) -> object: ...

    def print_task(self, ports: object, task: TaskPayload) -> None: ...

    def panel(self, ports: object, title: Any, rows: Any, **kwargs: Any) -> Any: ...


class _ModifyQueries(Protocol):
    def query_ports_for(self, host: Any) -> object: ...

    def cached_format_root_and_age(
        self,
        ports: object,
        task: TaskPayload,
        now_utc: Any,
    ) -> str: ...


class _ModifyLifecycle(Protocol):
    def task_has_nautical_fields(self, task: TaskPayload | None) -> bool: ...

    def task_has_nautical_recurrence_fields(self, task: TaskPayload | None) -> bool: ...

    def apply_nautical_transition(
        self,
        old: TaskPayload | None,
        new: TaskPayload | None,
        *,
        short_uuid: Callable[[Any], str],
    ) -> "RecurrenceTransitionDecision": ...


class _ServiceFactory(Protocol):
    def __call__(self, **kwargs: Any) -> Any: ...


class _ModifyExpirationEffects(Protocol):
    ExpirationServices: _ServiceFactory
    DeletedModifyServices: _ServiceFactory

    def render_recovery_warning(
        self,
        task: TaskPayload,
        reason: str,
        *,
        services: Any,
    ) -> None: ...

    def handle_expired_deleted_modify(self, task: TaskPayload, *, services: Any) -> bool: ...

    def handle_deleted_modify(
        self,
        old: TaskPayload,
        new: TaskPayload,
        *,
        services: Any,
        transition: Any = None,
        terminal_decision: Any = None,
    ) -> None: ...


class _ChainIntegrityLifecycle(Protocol):
    def deleted_chain_disposition(
        self,
        task: TaskObservation,
        *,
        safe_parse_datetime: Callable[[Any], Any],
    ) -> object: ...

    def is_orphan_expiration_candidate(
        self,
        task: TaskObservation,
        *,
        safe_parse_datetime: Callable[[Any], Any],
    ) -> bool: ...

    def plan_recovery_decision(
        self,
        parent: TaskObservation,
        *,
        existing_children: Sequence[TaskObservation],
        hook: Any,
    ) -> object: ...


class _ModifyTransitionEffects(Protocol):
    CPCarryPorts: _ServiceFactory
    NativePreservePorts: _ServiceFactory
    NativeCarryPorts: _ServiceFactory
    CompletionValidationPorts: _ServiceFactory

    def preserve_cp_relative_offsets_on_due_change(
        self,
        ports: object,
        old: TaskPayload,
        new: TaskPayload,
        cp: str,
        **kwargs: Any,
    ) -> Any: ...

    def reject_native_until_carry(self, ports: object, *args: Any, **kwargs: Any) -> Any: ...

    def preserve_native_until_on_target_change(
        self,
        ports: object,
        old: TaskPayload,
        new: TaskPayload,
        kind: str,
        **kwargs: Any,
    ) -> Any: ...

    def validate_completion_cp_and_anchor(
        self,
        ports: object,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...


class _SeedLookupPortsFactory(Protocol):
    def __call__(self, *, service: Any, decode_row: Any, cache_set: Any) -> object: ...


class _ModifyReadEffects(Protocol):
    SeedLookupPorts: _SeedLookupPortsFactory

    def seed_runtime_lookup_tasks(self, ports: object, *tasks: TaskPayload | None) -> None: ...


class _ModifyTaskFields(Protocol):
    def field_changed(self, old: TaskPayload, new: TaskPayload, key: str) -> bool: ...

    def recurrence_anchor_field(self, payload: TaskPayload) -> str: ...

    def strip_quotes(self, value: str) -> str: ...


class _ModifyValidationEffects(Protocol):
    SharedValidationPorts: type["SharedValidationPortsContract"]
    CPValidationPorts: type["CPValidationPortsContract"]

    def omit_validation_ports_for(self, host: Any) -> "OmitValidationPorts": ...

    def chain_limit_ports_for(self, host: Any) -> "ChainLimitPorts": ...

    def anchor_validation_ports_for(self, host: Any) -> "AnchorValidationPorts": ...

    def native_until_ports_for(self, host: Any) -> "NativeUntilPorts": ...

    def native_until_slot_ports_for(self, host: Any) -> "NativeUntilSlotPorts": ...

    def validate_omit(
        self,
        ports: "OmitValidationPorts",
        anchor_expr: str,
        anchor_file_expr: str,
        omit_expr: str,
        omit_file: str,
    ) -> None: ...

    def validate_chain_limits(self, ports: "ChainLimitPorts", task: TaskPayload) -> None: ...

    def validate_anchor(
        self,
        ports: "AnchorValidationPorts",
        old: TaskPayload,
        new: TaskPayload,
        anchor_expr: str,
    ) -> None: ...

    def validate_shared_anchor(self, ports: "SharedValidationPortsContract", expr: str) -> None: ...

    def validate_cp(
        self,
        ports: "CPValidationPortsContract",
        cp_value: str,
        chain_max_value: object,
        chain_until_value: object,
    ) -> None: ...

    def validate_native_until(self, ports: "NativeUntilPorts", task: TaskPayload) -> None: ...

    def validate_native_until_slots(
        self,
        ports: "NativeUntilSlotPorts",
        task: TaskPayload,
    ) -> None: ...

    def semantic_diff_value(self, old_text: str, new_text: str) -> str: ...


class _HookHost:
    """Attribute view over a dynamically loaded hook module's globals."""

    def __init__(self, values: dict[str, Any], name: str) -> None:
        self._values = values
        self.__name__ = name

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class ModifyHookCapabilities:
    """Modules and facilities resolved once at the hook composition root.

    Effects receive this explicit capability set instead of repeatedly reaching
    through the dynamic host module loader.  The loader remains the boundary
    that constructs the set, preserving import-by-file compatibility.
    """

    modify_ordinary: _ModifyOrdinaryEffects
    modify_composition_adapters: _ModifyCompositionAdapters
    hook_results: _HookResults
    hook_context: _HookContext
    hook_engine: _HookEngine
    modify_lifecycle: _ModifyLifecycle
    modify_transition_effects: _ModifyTransitionEffects
    modify_presentation_effects: _ModifyPresentationEffects
    modify_diagnostics_effects: _ModifyDeletionDiagnosticsEffects
    modify_validation_effects: _ModifyValidationEffects
    modify_ui_effects: _ModifyUIEffects
    modify_task_fields: _ModifyTaskFields
    modify_completion_effects: _ModifyCompletionEffects
    modify_read_effects: _ModifyReadEffects
    modify_queries: _ModifyQueries
    modify_expiration: _ModifyExpirationEffects | None
    modify_generation_effects: _ModifyGenerationEffects
    task_codec: _TaskCodecModule
    task_models: _TaskModelsModule
    chain_integrity_lifecycle: _ChainIntegrityLifecycle
    modify_spawn_effects: _ModifySpawnEffects

    @classmethod
    def from_host(cls, host: Any) -> "ModifyHookCapabilities":
        load = host._module
        return cls(
            modify_ordinary=load("modify_ordinary"),
            modify_composition_adapters=load("modify_composition_adapters"),
            hook_results=load("hook_results"),
            hook_context=load("hook_context"),
            hook_engine=load("hook_engine"),
            modify_lifecycle=load("modify_lifecycle"),
            modify_transition_effects=load("modify_transition_effects"),
            modify_presentation_effects=load("modify_presentation_effects"),
            modify_diagnostics_effects=load("modify_diagnostics_effects"),
            modify_validation_effects=load("modify_validation_effects"),
            modify_ui_effects=load("modify_ui_effects"),
            modify_task_fields=load("modify_task_fields"),
            modify_completion_effects=load("modify_completion_effects"),
            modify_read_effects=load("modify_read_effects"),
            modify_queries=load("modify_queries"),
            modify_expiration=load("modify_expiration", required=False),
            modify_generation_effects=load("modify_generation_effects"),
            task_codec=load("task_codec"),
            task_models=load("task_models"),
            chain_integrity_lifecycle=load("chain_integrity_lifecycle"),
            modify_spawn_effects=load("modify_spawn_effects"),
        )


@dataclass(frozen=True)
class NonCompletionRouteCapabilities:
    """Dependencies used only by the ordinary/recurring-edit route."""

    modify_ordinary: _ModifyOrdinaryEffects
    modify_lifecycle: _ModifyLifecycle
    modify_validation_effects: _ModifyValidationEffects
    modify_ui_effects: _ModifyUIEffects
    modify_task_fields: _ModifyTaskFields


@dataclass(frozen=True)
class CompletionRouteCapabilities:
    """Dependencies used only by the completion route."""

    modify_completion_effects: _ModifyCompletionEffects


@dataclass(frozen=True)
class DeletionRouteCapabilities:
    """Dependencies used only by deletion and expiration routes."""

    modify_diagnostics_effects: _ModifyDeletionDiagnosticsEffects
    modify_ui_effects: _ModifyUIEffects
    modify_expiration: _ModifyExpirationEffects | None
    modify_queries: _ModifyQueries


def _cp_carry_ports(host: Any, capabilities: ModifyHookCapabilities) -> Any:
    transition_effects = capabilities.modify_transition_effects
    return transition_effects.CPCarryPorts(
        carry=host._module("modify_carry").preserve_cp_relative_offsets_on_due_change,
        field_changed=capabilities.modify_task_fields.field_changed,
        parse_datetime=lambda value: datetime_value(host._TASK_DATETIME_PARSER, value),
        utc_to_local_naive=host.core.utc_to_local_naive,
        local_naive_to_utc=host.core.local_naive_to_utc,
        format_datetime=host.core.fmt_isoz,
        carry_error=host._module("chain_generation").CarryFieldError,
        workflow=host._module("modify_carry_workflow"),
    )


def _native_preserve_ports(host: Any, capabilities: ModifyHookCapabilities) -> Any:
    transition_effects = capabilities.modify_transition_effects
    generation = capabilities.modify_generation_effects
    task_fields = capabilities.modify_task_fields
    ui = capabilities.modify_ui_effects
    ui_ports = ui.ui_ports_for(host)
    return transition_effects.NativePreservePorts(
        carry=host._module("modify_carry").preserve_native_until_on_target_change,
        field_changed=task_fields.field_changed,
        anchor_field=task_fields.recurrence_anchor_field,
        parse_datetime=lambda value: datetime_value(host._TASK_DATETIME_PARSER, value),
        native_until=host.core._import_sibling("native_until"),
        generation_service=lambda: generation.chain_generation_service(
            generation.generation_ports_for(host)
        ),
        reject_carry=lambda *args: transition_effects.reject_native_until_carry(
            transition_effects.NativeCarryPorts(
                describe_carry=host.core._import_sibling("add_validation").describe_native_until_carry,
                parse_datetime=lambda value: datetime_value(host._TASK_DATETIME_PARSER, value),
                to_local=host.core.to_local,
                format_local=host.core.fmt_dt_local,
                anchor_field=task_fields.recurrence_anchor_field,
                panel=lambda title, rows, **kwargs: ui.panel(ui_ports, title, rows, **kwargs),
                abort=host.sys.exit,
            ),
            *args,
        ),
        diagnostic=host._diag,
        workflow=host._module("modify_carry_workflow"),
    )


def _completion_validation_ports(host: Any, capabilities: ModifyHookCapabilities) -> Any:
    transition_effects = capabilities.modify_transition_effects
    validation_effects = capabilities.modify_validation_effects
    modify_validation = host._module("modify_validation")
    pipeline = host.core._import_sibling("hook_validation_pipeline")
    add_validation = host.core._import_sibling("add_validation")
    shared_ports = validation_effects.SharedValidationPorts(
        pipeline,
        host.core._parser_api.parse_anchor_expr_to_dnf,
        host._validate_anchor_expr_cached,
        host._validate_omit_expr_cached,
    )
    cp_ports = validation_effects.CPValidationPorts(
        modify_validation.validate_cp_on_modify,
        host.core.parse_cp_sequence,
        host.core.cp_sequence_parse_error,
        add_validation.parse_chain_max,
        lambda value: datetime_value(host._TASK_DATETIME_PARSER, value),
    )

    def apply_transition(old_task: TaskPayload, new_task: TaskPayload) -> None:
        capabilities.modify_lifecycle.apply_nautical_transition(
            old_task,
            new_task,
            short_uuid=host.core.short_uuid,
        )

    return transition_effects.CompletionValidationPorts(
        validate=modify_validation.validate_completion_cp_and_anchor,
        strip_quotes=capabilities.modify_task_fields.strip_quotes,
        reject_conflicting_types=pipeline.reject_recurrence_kind_conflict,
        validate_omit=lambda anchor, anchor_file, omit, omit_file: validation_effects.validate_omit(
            validation_effects.omit_validation_ports_for(host), anchor, anchor_file, omit, omit_file
        ),
        validate_chain_limits=lambda task: validation_effects.validate_chain_limits(
            validation_effects.chain_limit_ports_for(host), task
        ),
        parse_cp_sequence=host.core.parse_cp_sequence,
        cp_sequence_parse_error=host.core.cp_sequence_parse_error,
        field_changed=capabilities.modify_task_fields.field_changed,
        validate_anchor=lambda expr: validation_effects.validate_shared_anchor(shared_ports, expr),
        validate_cp=lambda cp, chain_max, chain_until: validation_effects.validate_cp(
            cp_ports, cp, chain_max, chain_until
        ),
        apply_transition=apply_transition,
        fail=host._fail_and_exit,
        diagnostic=host._diag,
    )


@dataclass(frozen=True)
class ModifyRuntimeServices:
    """Explicit runtime services supplied to route effects at composition time."""

    non_completion: NonCompletionRouteCapabilities
    completion: CompletionRouteCapabilities
    deletion: DeletionRouteCapabilities
    runtime_state: ModifyCallback
    import_module: ModifyCallback
    diag_summary: ModifyCallback
    diagnostic: ModifyCallback
    show_analytics: bool
    check_integrity: bool
    analytics_style: str
    seed_runtime_lookup_tasks: ModifyCallback
    lifecycle_read_service: Callable[[], Any]
    chain_health_advice: ModifyCallback
    chain_integrity_warnings: ModifyCallback
    render_anchor_completion_feedback: ModifyCallback
    render_cp_completion_feedback: ModifyCallback
    render_lifecycle_result: ModifyCallback
    print_task: ModifyCallback
    prepare_recurrence: ModifyCallback
    preserve_cp_relative_offsets: ModifyCallback
    preserve_native_until: ModifyCallback
    validate_native_until: ModifyCallback
    validate_native_until_slots: ModifyCallback
    now_utc: Callable[[], Any]
    compute_next_and_limits: ModifyCallback

    @classmethod
    def from_host(cls, host: Any, capabilities: ModifyHookCapabilities | None = None) -> "ModifyRuntimeServices":
        capabilities = capabilities or capabilities_for(host)
        cp_carry_ports = _cp_carry_ports(host, capabilities)
        native_preserve_ports = _native_preserve_ports(host, capabilities)
        completion_validation_ports = _completion_validation_ports(host, capabilities)
        completion_compute_ports = capabilities.modify_completion_effects.completion_compute_ports_for(host)
        return cls(
            non_completion=NonCompletionRouteCapabilities(
                modify_ordinary=capabilities.modify_ordinary,
                modify_lifecycle=capabilities.modify_lifecycle,
                modify_validation_effects=capabilities.modify_validation_effects,
                modify_ui_effects=capabilities.modify_ui_effects,
                modify_task_fields=capabilities.modify_task_fields,
            ),
            completion=CompletionRouteCapabilities(
                modify_completion_effects=capabilities.modify_completion_effects,
            ),
            deletion=DeletionRouteCapabilities(
                modify_diagnostics_effects=capabilities.modify_diagnostics_effects,
                modify_ui_effects=capabilities.modify_ui_effects,
                modify_expiration=capabilities.modify_expiration,
                modify_queries=capabilities.modify_queries,
            ),
            runtime_state=host._modify_runtime_state,
            import_module=host.importlib.import_module,
            diag_summary=host._diag_summary,
            diagnostic=host._diag,
            show_analytics=host._SHOW_ANALYTICS,
            check_integrity=host._CHECK_CHAIN_INTEGRITY,
            analytics_style=host._ANALYTICS_STYLE,
            seed_runtime_lookup_tasks=lambda *tasks: capabilities.modify_read_effects.seed_runtime_lookup_tasks(
                capabilities.modify_read_effects.SeedLookupPorts(
                    service=lifecycle_read_service_for(host),
                    decode_row=capabilities.task_codec.DEFAULT_TASK_CODEC.decode_row,
                    cache_set=host._query_ctx_set,
                ), *tasks
            ),
            lifecycle_read_service=lambda: lifecycle_read_service_for(host),
            chain_health_advice=lambda *args, **kwargs: capabilities.modify_diagnostics_effects.chain_health_advice(
                capabilities.modify_diagnostics_effects.analytics_ports_for(host), *args, **kwargs
            ),
            chain_integrity_warnings=lambda *args, **kwargs: capabilities.modify_diagnostics_effects.chain_integrity_warnings(
                capabilities.modify_diagnostics_effects.analytics_ports_for(host), *args, **kwargs
            ),
            render_anchor_completion_feedback=lambda **kwargs: capabilities.modify_composition_adapters.render_anchor_completion_feedback_for(host, **kwargs),
            render_cp_completion_feedback=lambda **kwargs: capabilities.modify_composition_adapters.render_cp_completion_feedback_for(host, **kwargs),
            render_lifecycle_result=lambda result, task: capabilities.modify_presentation_effects.render_lifecycle_result(
                capabilities.modify_presentation_effects.lifecycle_result_port_for(host), result, task
            ),
            print_task=lambda task: capabilities.modify_ui_effects.print_task(
                capabilities.modify_ui_effects.ui_ports_for(host), task
            ),
            prepare_recurrence=lambda old, new, **kwargs: capabilities.modify_transition_effects.validate_completion_cp_and_anchor(
                completion_validation_ports, old, new, **kwargs
            ),
            preserve_cp_relative_offsets=lambda old, new, cp, **kwargs: capabilities.modify_transition_effects.preserve_cp_relative_offsets_on_due_change(
                cp_carry_ports, old, new, cp, **kwargs
            ),
            preserve_native_until=lambda old, new, kind, **kwargs: capabilities.modify_transition_effects.preserve_native_until_on_target_change(
                native_preserve_ports, old, new, kind, **kwargs
            ),
            validate_native_until=lambda task: capabilities.modify_validation_effects.validate_native_until(
                capabilities.modify_validation_effects.native_until_ports_for(host), task
            ),
            validate_native_until_slots=lambda task: capabilities.modify_validation_effects.validate_native_until_slots(
                capabilities.modify_validation_effects.native_until_slot_ports_for(host), task
            ),
            now_utc=host.core.now_utc,
            compute_next_and_limits=lambda *args, **kwargs: capabilities.modify_completion_effects.compute_next_and_limits(
                completion_compute_ports, *args, **kwargs
            ),
        )


def capabilities_for(host: Any) -> ModifyHookCapabilities:
    """Return the composition-root capability set for ``host``."""
    cached = getattr(host, "_MODIFY_CAPABILITIES", None)
    if cached is None:
        cached = ModifyHookCapabilities.from_host(host)
        try:
            setattr(host, "_MODIFY_CAPABILITIES", cached)
        except Exception:
            pass
    # Construct the datetime port once at the composition root.  Effects may
    # consume it through their narrow parser adapter without rediscovering the
    # live hook/core namespace on every field.
    if getattr(host, "_TASK_DATETIME_PARSER", None) is None:
        try:
            setattr(
                host,
                "_TASK_DATETIME_PARSER",
                parser_for_core(host.core, diagnostic=getattr(host, "_diag", None)),
            )
        except Exception:
            pass
    return cached


def lifecycle_read_service_for(host: Any) -> Any:
    """Construct and retain the invocation's lifecycle read service."""
    state = host._modify_runtime_state()
    existing = getattr(state, "lifecycle_read_service", None)
    if existing is not None:
        # Completion setup attaches the authoritative repository immediately
        # before the first lifecycle read. Refresh a service that was eagerly
        # created by the composition root before that attachment.
        repository = getattr(state, "task_repository", None)
        if repository is not None and getattr(existing, "_repository", None) is None:
            existing._repository = repository
        return existing
    module = host._module("lifecycle_read_service")
    read_effects = host._module("modify_read_effects")
    if getattr(state, "chain_cache_store", None) is None:
        state.chain_cache_store = module.ChainCacheStore()
    capabilities = read_effects.LifecycleReadCapabilities(
        coerce_int=host.core.coerce_int,
        parse_extra_tokens=lambda extra: read_effects.parse_extra_tokens(
            read_effects.ExtraTokenPort(host._module("hook_support", required=False).parse_extra_tokens), extra
        ),
        token_matcher=lambda task, token: read_effects._token_match(host.core.coerce_int, task, token),
        read_query_get=host._read_query_get,
        read_query_missing=host._READ_QUERY_MISSING,
        max_chain_walk=host._MAX_CHAIN_WALK,
        diag=host._diag,
        record_stat=host._record_chain_snapshot_stat,
        cache_store=state.chain_cache_store,
        repository=getattr(state, "task_repository", None),
    )
    service = module.LifecycleReadService(
        coerce_int=capabilities.coerce_int,
        parse_extra_tokens=capabilities.parse_extra_tokens,
        token_matcher=capabilities.token_matcher,
        read_query_get=capabilities.read_query_get,
        chain_cache_get=lambda _chain_id: None,
        repository=capabilities.repository,
        max_chain_walk=capabilities.max_chain_walk,
        diag=capabilities.diag,
        record_stat=capabilities.record_stat,
        cache_store=capabilities.cache_store,
        read_query_missing=capabilities.read_query_missing,
    )
    state.lifecycle_read_service = service
    return service


class ModifyCompositionServices:
    """Bind on-modify effects to the hook's validated composition root."""

    def __init__(self, host: Any, result_cls: CallbackPort) -> None:
        self._host = host
        self._result_cls = result_cls
        self._capabilities = capabilities_for(host)
        self._runtime = ModifyRuntimeServices.from_host(host, self._capabilities)

    def result(self, task: Any, *, sanitize: bool) -> Any:
        return self._result_cls(task=task, sanitize=sanitize)

    def has_nautical_fields(self, task: Any) -> bool:
        return self._capabilities.modify_lifecycle.task_has_nautical_fields(task)

    def load_core(self) -> None:
        self._host._load_core()

    def diag(self, message: str) -> None:
        self._host._diag(message)

    def fail_and_exit(self, title: str, message: str) -> None:
        self._host._fail_and_exit(title, message)

    def handle_non_completion(self, old: Any, new: Any, unit_of_work: Any, transition: Any = None) -> Any:
        self._capabilities.modify_composition_adapters.handle_non_completion(
            self._host, old, new, unit_of_work, transition=transition,
            runtime=self._runtime,
        )

    def handle_completion(self, old: Any, new: Any, unit_of_work: Any, transition: Any = None) -> Any:
        return self._capabilities.modify_composition_adapters.handle_completion(
            self._host, old, new, unit_of_work, transition=transition,
            runtime=self._runtime,
        )

    def handle_deleted(
        self, old: Any, new: Any, unit_of_work: Any, transition: Any = None, terminal_decision: Any = None
    ) -> Any:
        return self._capabilities.modify_composition_adapters.handle_deleted(
            self._host,
            old,
            new,
            unit_of_work,
            transition=transition,
            terminal_decision=terminal_decision,
            runtime=self._runtime,
        )


def hook_host(values: dict[str, Any], name: str) -> Any:
    """Return a live attribute view for import-by-file test harnesses."""
    return _HookHost(values, name)


def run_on_modify(host: Any) -> None:
    """Run the validated on-modify composition root for ``host``."""
    host._reset_modify_runtime_state()
    state = host._modify_runtime_state()
    startup_t0 = host._ptime.perf_counter()
    module_t0 = host._ptime.perf_counter()
    capabilities = capabilities_for(host)
    hook_results = capabilities.hook_results
    state.diag_stats["startup_module_ms"] = round(
        (host._ptime.perf_counter() - module_t0) * 1000.0, 3
    )
    read_t0 = host._ptime.perf_counter()
    old, new = host._read_two()
    alias_candidate = re.search(
        r"(?:^|\s)(?:a|af|am|o|of|cm|cu):",
        str(new.get("description") or ""),
    ) is not None
    lifecycle = capabilities.modify_lifecycle
    has_nautical_fields = lifecycle.task_has_nautical_fields(old) or lifecycle.task_has_nautical_fields(new)
    if not has_nautical_fields and not alias_candidate:
        hook_results.emit_passthrough_json(new)
        host._write_bench_stats()
        return
    host._load_core()
    hook_context = capabilities.hook_context
    hook_engine = capabilities.hook_engine
    host._apply_description_uda_aliases(old, new)
    validation = host.core._import_sibling("hook_validation_pipeline")
    # Alias expansion mutates the canonical task mapping. Refresh the typed
    # observation so transition diffs and recurrence feedback include aliases.
    if host._PARSED_NEW_OBSERVATION is not None:
        task_models = host.core._import_sibling("task_models")
        host._PARSED_NEW_OBSERVATION = task_models.TaskObservation.from_mapping(
            new,
            source_query="on-modify alias-normalized task",
        )
    _validated_observation, validation_report = validation.validate_task_mapping(
        new,
        route=validation.WorkflowRoute.RECURRING_EDIT,
        source_query="on-modify validation",
    )
    if validation_report.status is not validation.ValidationStatus.VALID:
        finding = validation_report.findings[0]
        title = "Invalid chainMax" if finding.code == "chain_max_invalid" else "Invalid Nautical task"
        host._fail_and_exit(title, f"{finding.reason} {finding.correction}")
    if host._PARSED_OLD_OBSERVATION is not None and host._PARSED_NEW_OBSERVATION is not None:
        transition_report = validation.validate_task_transition(
            host._PARSED_OLD_OBSERVATION,
            host._PARSED_NEW_OBSERVATION,
            route=validation.WorkflowRoute.RECURRING_EDIT,
            source_query="on-modify transition validation",
        )
        if transition_report.status is not validation.ValidationStatus.VALID:
            finding = transition_report.findings[0]
            title = "Invalid chainMax" if finding.code == "chain_max_invalid" else "Invalid recurrence transition"
            host._fail_and_exit(title, f"{finding.reason} {finding.correction}")
    config_error = str(getattr(host.core, "scheduling_configuration_error", lambda: "")() or "")
    if config_error and has_nautical_fields:
        host._fail_and_exit(
            "Invalid Nautical configuration",
            f"{config_error}. Fix Nautical configuration before modifying a recurring task.",
        )
    state.diag_stats["startup_read_input_ms"] = round(
        (host._ptime.perf_counter() - read_t0) * 1000.0, 3
    )
    try:
        calendar_context = host.core.use_task_business_calendar(new)
    except Exception as exc:
        host._fail_and_exit("Invalid business calendar", str(exc))
        return
    request_t0 = host._ptime.perf_counter()
    capabilities.modify_read_effects.seed_runtime_lookup_tasks(
        capabilities.modify_read_effects.SeedLookupPorts(
            service=lifecycle_read_service_for(host),
            decode_row=capabilities.task_codec.DEFAULT_TASK_CODEC.decode_row,
            cache_set=host._query_ctx_set,
        ), old, new
    )
    runtime = host._build_hook_runtime_context(new)
    host._modify_runtime_state().workflow_context = runtime.workflow
    request = hook_context.build_on_modify_request(
        runtime=runtime,
        old=old,
        new=new,
        old_observation=host._PARSED_OLD_OBSERVATION,
        new_observation=host._PARSED_NEW_OBSERVATION,
    )
    if host._IMPORT_MS is not None:
        state.diag_stats["startup_import_ms"] = round(float(host._IMPORT_MS), 3)
    state.diag_stats["startup_request_ms"] = round(
        (host._ptime.perf_counter() - request_t0) * 1000.0, 3
    )
    state.diag_stats["startup_total_ms"] = round(
        (host._ptime.perf_counter() - startup_t0) * 1000.0, 3
    )
    displacement_context = (
        host.core.capture_business_calendar_displacements()
        if str(new.get("bc") or "").strip()
        else nullcontext()
    )
    try:
        with calendar_context, displacement_context:
            result = hook_engine.handle_on_modify(
                request,
                services=ModifyCompositionServices(
                    host, hook_results.TaskHookResponse
                ),
            )
        if result is not None:
            hook_results.emit_json_result(result, core=host.core)
    finally:
        runtime.close()
        host._write_bench_stats()


__all__ = ("ModifyCompositionServices",)
