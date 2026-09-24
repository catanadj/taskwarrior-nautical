"""Stable public export contract for the Nautical core facade.

The explicit ``normalize_task_business_calendar_in_place`` name is the public
mutator.  Its shorter predecessor remains a facade-only compatibility alias.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterator
from typing import Any, Callable

from .api_bindings import ApiBinding
from .core_context import CoreContext

PUBLIC_EXPORTS = (
    'AnchorMods', 'AnchorAtom', 'AnchorTerm', 'AnchorDNF', 'TaskDict',
    'AnchorValidationResult', 'chain_colour_root', 'HintMetaCfg', 'HintMeta',
    'HintPerYear', 'HintLimits', 'AnchorHintsPayload', 'ParseError',
    'YearTokenFormatError', 'AndTermUnsatisfiable', 'OccurrenceSearchExhausted', 'BusinessCalendarConfigError',
    'ANCHOR_CACHE_DIR_OVERRIDE', 'BUSINESS_CALENDAR_CONFIG', 'ASTRONOMY_CONFIG',
    'effective_config_snapshot', 'effective_config_fingerprint', 'reload_taskdata_config',
    'scheduling_configuration_error', 'validate_scheduling_configuration',
    'scheduler_config_fingerprint', 'configuration_drift', 'DEFAULT_BUSINESS_CALENDAR',
    'ENABLE_ANCHOR_CACHE', 'LOCAL_TZ_NAME', 'SEASON_HEMISPHERE', 'SEASON_MODE', 'OMIT_FILE_DIR',
    'MAX_LINK_NUMBER', 'PANEL_MODE', 'LIVE_PANEL_DURATION_MS', 'LIVE_PANEL_FOOTER',
    'EXIT_PROGRESS', 'DEFAULT_DUE_HOUR', '_LOCAL_TZ', '_build_anchor_atom_dnf',
    '_cache_atomic_replace', '_cache_lock', '_cache_path', '_clear_all_caches',
    '_doms_allowed_by_year', '_doms_for_monthly_token', '_doms_for_weekly_spec',
    '_emit_cache_metrics', '_fatal_bad_colon_in_year_tail',
    '_first_hit_after_probe_in_month', '_interval_allowed_for_atom', '_month_has_hit',
    '_normalize_anchor_expr_input',
    '_normalize_spec_for_acf_cached', '_parse_anchor_atom_at', '_parse_atom_head',
    '_parse_atom_mods', '_quarter_month_selector_mode', '_quick_weekly_and_check',
    '_quick_yearly_and_check', '_raise_on_bad_colon_year_tokens',
    '_rand_bucket_signature', '_rewrite_quarter_spec_mode', '_rewrite_quarters_in_context',
    '_term_has_any_match_within', '_term_quarter_rewrite_mode',
    '_validate_yearly_spec_token', '_validate_yearly_token_format',
    '_warn_once_per_day', '_warn_once_per_day_any', '_warn_rate_limited_any',
    '_weekly_spec_to_wset', '_weeks_between', '_y_ranges_from_spec',
    'anchor_preset_display', 'build_acf',
    'build_and_cache_hints', 'build_local_datetime', 'business_calendar_fingerprint',
    'business_calendar_displacement_for_date', 'business_calendar_for_task',
    'business_calendar_definitions', 'cache_key_for_task', 'cache_load', 'cache_save',
    'cache_gc', 'capture_business_calendar_displacements', 'coerce_int',
    'describe_anchor_dnf', 'describe_anchor_expr', 'DiagnosticEvent', 'RecurrenceModeResult', 'diag',
    'fcntl', 'fmt_dt_local', 'fmt_isoz',
    'configured_business_calendars', 'get_configured_business_calendar',
    'lint_anchor_expr', 'normalize_task_business_calendar_in_place', 'now_utc',
    'omit_preset_display', 'panel_line', 'parse_anchor_expr_to_dnf',
    'parse_anchor_expr_to_dnf_cached', 'parse_cp_duration', 'parse_cp_sequence',
    'parse_cp_sequence_tokens', 'cp_sequence_parse_error',
    'cp_sequence_interval_for_link', 'cp_sequence_interval_for_token', 'parse_dt_any',
    'pick_hhmm_from_dnf_for_date', 'render_panel',
    'resolve_anchor_presets', 'resolve_business_calendar_config', 'resolve_omit_presets',
    'resolve_task_data_context', 'safe_lock',
    'short_uuid', 'strip_rich_markup', 'tempfile',
    'term_width_stderr', 'to_local', 'utc_to_local_naive', 'local_naive_to_utc',
    'use_business_calendar', 'use_task_business_calendar', 'validate_anchor_expr_strict',
)

__all__ = ('PUBLIC_EXPORTS',)


# WP5 owner registry: explicit canonical owners for the first migrated group.
# Names not yet assigned to a focused owner intentionally retain the root
# facade as their provisional owner until their group is migrated.
PUBLIC_OWNER_MODULES = {name: "nautical_core" for name in PUBLIC_EXPORTS}
PUBLIC_OWNER_MODULES.update({
    "AnchorMods": "nautical_core.parsing.parser_models",
    "AnchorAtom": "nautical_core.parsing.parser_models",
    "AnchorTerm": "nautical_core.parsing.parser_models",
    "AnchorDNF": "nautical_core.parsing.parser_models",
    "AnchorValidationResult": "nautical_core.parsing.parser_models",
    "ParseError": "nautical_core.parsing.parser_models",
    "YearTokenFormatError": "nautical_core.parsing.parser_models",
    "AndTermUnsatisfiable": "nautical_core.parsing.parser_models",
    "OccurrenceSearchExhausted": "nautical_core.scheduler_models",
    "_parse_atom_head": "nautical_core.parsing.parser_support_api",
    "_parse_atom_mods": "nautical_core.parsing.parser_support_api",
    "_parse_anchor_atom_at": "nautical_core.parser_api",
    "_normalize_anchor_expr_input": "nautical_core.parser_api",
    "_fatal_bad_colon_in_year_tail": "nautical_core.parsing.parser_support_api",
    "_raise_on_bad_colon_year_tokens": "nautical_core.parsing.parser_support_api",
    "_validate_yearly_spec_token": "nautical_core.parser_api",
    "_validate_yearly_token_format": "nautical_core.parser_api",
    "build_acf": "nautical_core.parser_api",
    "describe_anchor_dnf": "nautical_core.parser_api",
    "describe_anchor_expr": "nautical_core.parser_api",
    "lint_anchor_expr": "nautical_core.linting_api",
    "parse_anchor_expr_to_dnf": "nautical_core.parser_api",
    "parse_anchor_expr_to_dnf_cached": "nautical_core.parser_api",
    "parse_cp_duration": "nautical_core.parser_api",
    "parse_cp_sequence": "nautical_core.parser_api",
    "parse_cp_sequence_tokens": "nautical_core.parser_api",
    "cp_sequence_parse_error": "nautical_core.parser_api",
    "cp_sequence_interval_for_link": "nautical_core.parser_api",
    "cp_sequence_interval_for_token": "nautical_core.parser_api",
    "resolve_anchor_presets": "nautical_core.parser_api",
    "resolve_omit_presets": "nautical_core.parser_api",
    "validate_anchor_expr_strict": "nautical_core.parser_api",
    "BusinessCalendarConfigError": "nautical_core.business_calendar_config",
    "effective_config_snapshot": "nautical_core.core_config",
    "effective_config_fingerprint": "nautical_core.core_config",
    "reload_taskdata_config": "nautical_core.core_config",
    "scheduling_configuration_error": "nautical_core.core_config",
    "validate_scheduling_configuration": "nautical_core.core_config",
    "scheduler_config_fingerprint": "nautical_core.core_config",
    "configuration_drift": "nautical_core.core_config",
    "ANCHOR_CACHE_DIR_OVERRIDE": "nautical_core.core_config",
    "BUSINESS_CALENDAR_CONFIG": "nautical_core.core_config",
    "ASTRONOMY_CONFIG": "nautical_core.core_config",
    "ENABLE_ANCHOR_CACHE": "nautical_core.core_config",
    "LOCAL_TZ_NAME": "nautical_core.core_config",
    "SEASON_HEMISPHERE": "nautical_core.core_config",
    "SEASON_MODE": "nautical_core.core_config",
    "OMIT_FILE_DIR": "nautical_core.core_config",
    "MAX_LINK_NUMBER": "nautical_core.core_config",
    "PANEL_MODE": "nautical_core.core_config",
    "LIVE_PANEL_DURATION_MS": "nautical_core.core_config",
    "LIVE_PANEL_FOOTER": "nautical_core.core_config",
    "EXIT_PROGRESS": "nautical_core.core_config",
    "DEFAULT_DUE_HOUR": "nautical_core.core_config",
    "DEFAULT_BUSINESS_CALENDAR": "nautical_core.business_calendar",
    "business_calendar_displacement_for_date": "nautical_core.business_calendar",
    "capture_business_calendar_displacements": "nautical_core.business_calendar",
    "business_calendar_definitions": "nautical_core.business_calendar_api",
    "resolve_business_calendar_config": "nautical_core.business_calendar_api",
    "configured_business_calendars": "nautical_core.business_calendar_api",
    "get_configured_business_calendar": "nautical_core.business_calendar_api",
    "business_calendar_for_task": "nautical_core.business_calendar_api",
    "normalize_task_business_calendar_in_place": "nautical_core.business_calendar_api",
    "business_calendar_fingerprint": "nautical_core.business_calendar_api",
    "use_business_calendar": "nautical_core.business_calendar_api",
    "use_task_business_calendar": "nautical_core.business_calendar_api",
    "cache_key_for_task": "nautical_core.cache_api",
    "cache_load": "nautical_core.cache_api",
    "cache_save": "nautical_core.cache_api",
    "cache_gc": "nautical_core.cache_api",
    "_cache_atomic_replace": "nautical_core.cache_api",
    "_cache_lock": "nautical_core.cache_api",
    "_cache_path": "nautical_core.cache_api",
    "_clear_all_caches": "nautical_core.cache_api",
    "_emit_cache_metrics": "nautical_core.cache_api",
    "_normalize_spec_for_acf_cached": "nautical_core.cache_api",
    "parse_anchor_expr_to_dnf_cached": "nautical_core.cache_api",
    "build_and_cache_hints": "nautical_core.hint_builder_api",
    "TaskDict": "nautical_core.task_models",
    "DiagnosticEvent": "nautical_core.diagnostic_models",
    "RecurrenceModeResult": "nautical_core.recurrence_evaluator",
    "build_local_datetime": "nautical_core.time_api",
    "now_utc": "nautical_core.time_api",
    "to_local": "nautical_core.time_api",
    "utc_to_local_naive": "nautical_core.time_api",
    "local_naive_to_utc": "nautical_core.time_api",
    "fmt_dt_local": "nautical_core.time_api",
    "fmt_isoz": "nautical_core.time_api",
    "parse_dt_any": "nautical_core.time_api",
    "coerce_int": "nautical_core.common",
    "short_uuid": "nautical_core.common",
    "pick_hhmm_from_dnf_for_date": "nautical_core.schedule_utils",
    "chain_colour_root": "nautical_core.panel_colours",
    "strip_rich_markup": "nautical_core.ui",
    "term_width_stderr": "nautical_core.ui",
    "panel_line": "nautical_core.ui",
    "render_panel": "nautical_core.ui",
    "diag": "nautical_core.runtime",
    "_warn_once_per_day": "nautical_core.diagnostic_warnings",
    "_warn_once_per_day_any": "nautical_core.diagnostic_warnings",
    "_warn_rate_limited_any": "nautical_core.diagnostic_warnings",
    "_LOCAL_TZ": "nautical_core.core_config",
    "_build_anchor_atom_dnf": "nautical_core.parser_api",
    "_doms_allowed_by_year": "nautical_core.parser_api",
    "_doms_for_monthly_token": "nautical_core.parser_api",
    "_doms_for_weekly_spec": "nautical_core.parser_api",
    "_first_hit_after_probe_in_month": "nautical_core.parser_api",
    "_interval_allowed_for_atom": "nautical_core.scheduler_api",
    "_month_has_hit": "nautical_core.parser_api",
    "_quarter_month_selector_mode": "nautical_core.parser_api",
    "_quick_weekly_and_check": "nautical_core.parser_api",
    "_quick_yearly_and_check": "nautical_core.parser_api",
    "_rand_bucket_signature": "nautical_core.parser_api",
    "_rewrite_quarter_spec_mode": "nautical_core.parser_api",
    "_rewrite_quarters_in_context": "nautical_core.parser_api",
    "_term_has_any_match_within": "nautical_core.parser_api",
    "_term_quarter_rewrite_mode": "nautical_core.parser_api",
    "_weekly_spec_to_wset": "nautical_core.parser_api",
    "_weeks_between": "nautical_core.scheduler_api",
    "_y_ranges_from_spec": "nautical_core.parser_api",
    "anchor_preset_display": "nautical_core.parser_api",
    "omit_preset_display": "nautical_core.parser_api",
    "safe_lock": "nautical_core.cache_api",
    "resolve_task_data_context": "nautical_core.runtime",
    "HintMetaCfg": "nautical_core.hint_models",
    "HintMeta": "nautical_core.hint_models",
    "HintPerYear": "nautical_core.hint_models",
    "HintLimits": "nautical_core.hint_models",
    "AnchorHintsPayload": "nautical_core.hint_models",
    "fcntl": "fcntl",
    "tempfile": "tempfile",
})

# Every exported name is deliberately classified before any compatibility
# removal is considered.  Private helpers remain test seams; hook/bootstrap
# entry points are installed-runtime contracts; the remainder are supported
# public API until a documented deprecation changes that status.
PUBLIC_EXPORT_CATEGORIES = {
    name: (
        "test_seam" if name.startswith("_") or name in {"fcntl", "tempfile"}
        else "installed_runtime" if name in {"diag", "resolve_task_data_context"}
        else "supported_public_api"
    )
    for name in PUBLIC_EXPORTS
}
PUBLIC_EXPORT_CATEGORIES["normalize_task_business_calendar"] = "legacy_compatibility_alias"


PUBLIC_MODEL_NAMES = (
    "AnchorMods", "AnchorAtom", "AnchorTerm", "AnchorDNF", "AnchorValidationResult",
    "ParseError", "YearTokenFormatError", "AndTermUnsatisfiable", "OccurrenceSearchExhausted",
)
_PUBLIC_CALL_PARAMETERS = {
    "parse_anchor_expr_to_dnf": ("s",),
    "parse_anchor_expr_to_dnf_cached": ("s",),
    "validate_anchor_expr_strict": ("expr",),
    "parse_cp_duration": ("dur",),
    "parse_cp_sequence": ("cp",),
    "cp_sequence_interval_for_link": ("cp", "link_no", "chain_id"),
    "build_local_datetime": ("d", "hhmm"),
    "to_local": ("dt_utc",),
    "utc_to_local_naive": ("dt_utc",),
    "local_naive_to_utc": ("dt_local_naive",),
    "parse_dt_any": ("s",),
}
LEGACY_COMPATIBILITY_ALIASES = {
    "normalize_task_business_calendar": (
        "business_calendar_api",
        "normalize_task_business_calendar_in_place",
    ),
}


def ensure_public_models(
    namespace: dict[str, Any],
    package_name: str,
) -> None:
    """Resolve lazily exported public model types into the facade namespace."""
    if "ParseError" in namespace:
        return
    parser_models = importlib.import_module(f"{package_name}.parsing.parser_models")
    scheduler_models = importlib.import_module(f"{package_name}.scheduler_models")
    for name in PUBLIC_MODEL_NAMES[:-1]:
        namespace[name] = getattr(parser_models, name)
    namespace["OccurrenceSearchExhausted"] = scheduler_models.OccurrenceSearchExhausted


class _LazySibling:
    """Resolve a focused sibling only when one of its APIs is used."""

    __slots__ = ("_name", "_module", "_import_sibling")

    def __init__(self, module_name: str, import_sibling: Callable[[str], Any]):
        self._name = module_name
        self._module = None
        self._import_sibling = import_sibling

    def _resolve(self) -> Any:
        if self._module is None:
            self._module = self._import_sibling(self._name)
        return self._module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._resolve(), name, value)


class _LazyPublicExports:
    """Tuple-like export view that preserves deferred wildcard resolution."""

    __slots__ = ("_source",)

    def __init__(self, source: _LazySibling):
        self._source = source

    def _values(self) -> tuple[str, ...]:
        return self._source.PUBLIC_EXPORTS

    def __iter__(self) -> Iterator[str]:
        return iter(self._values())

    def __len__(self) -> int:
        return len(self._values())

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        return self._values()[index]


class _LazyApiBundle:
    """Bind one typed API owner lazily while retaining legacy facade names."""

    __slots__ = (
        "_module_name", "_aliases", "_bindings",
        "_context", "_import_sibling", "_prepare", "_wrappers",
    )

    def __init__(
        self,
        module_name: str,
        aliases: tuple[str | tuple[str, str], ...],
        *,
        core: Any,
        namespace: dict[str, Any],
        import_sibling: Callable[[str], Any],
        prepare: Callable[[], None],
    ):
        self._module_name = module_name
        self._aliases = aliases
        self._bindings: ApiBinding | None = None
        self._import_sibling = import_sibling
        self._prepare = prepare
        self._wrappers: dict[str, Callable[..., Any]] = {}
        self._context = CoreContext(
            namespace,
            import_sibling,
            getattr(core, "__file__", None),
        )

    def _resolve(self) -> ApiBinding:
        if self._bindings is None:
            self._prepare()
            module = self._import_sibling(self._module_name)
            self._bindings = module.for_core(context=self._context)
            for spec in self._aliases:
                alias, source = spec if isinstance(spec, tuple) else (spec, spec)
                wrapper = self._wrappers.get(alias)
                target = getattr(self._bindings, source)
                if wrapper is not None and callable(target):
                    _copy_callable_contract(wrapper, target)
        return self._bindings

    def alias(self, name: str, source_name: str | None = None) -> Callable[..., Any]:
        source_name = source_name or name

        def call(*args: Any, **kwargs: Any) -> Any:
            return getattr(self._resolve(), source_name)(*args, **kwargs)

        call.__name__ = name
        call.__qualname__ = name
        parameter_names = _PUBLIC_CALL_PARAMETERS.get(name)
        if parameter_names is not None:
            setattr(call, "__signature__", inspect.Signature([
                inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for parameter in parameter_names
            ]))
        if source_name.endswith("_cached"):
            for attribute in ("cache_clear", "cache_info", "cache_parameters"):
                def forward_cache_attribute(
                    *args: Any,
                    _attribute: str = attribute,
                    **kwargs: Any,
                ) -> Any:
                    target = getattr(self._resolve(), source_name)
                    return getattr(target, _attribute)(*args, **kwargs)

                setattr(call, attribute, forward_cache_attribute)
        self._wrappers[name] = call
        return call

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resolve(), name)


def _bind_lazy_api_aliases(bundle: _LazyApiBundle, namespace: dict[str, Any]) -> None:
    """Install callable compatibility names while preserving owner aliases."""
    for spec in bundle._aliases:
        alias_name, source_name = spec if isinstance(spec, tuple) else (spec, spec)
        namespace[alias_name] = bundle.alias(alias_name, source_name)


def _copy_callable_contract(wrapper: Callable[..., Any], target: Callable[..., Any]) -> None:
    """Preserve public callable introspection and cache-control attributes."""
    wrapper.__doc__ = getattr(target, "__doc__", None)
    wrapper.__annotations__ = getattr(target, "__annotations__", {})
    wrapper.__module__ = getattr(target, "__module__", wrapper.__module__)
    setattr(wrapper, "__wrapped__", target)
    try:
        setattr(wrapper, "__signature__", inspect.signature(target))
    except (TypeError, ValueError):
        pass
    for name in ("cache_clear", "cache_info", "cache_parameters"):
        value = getattr(target, name, None)
        if callable(value):
            setattr(wrapper, name, value)
