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
