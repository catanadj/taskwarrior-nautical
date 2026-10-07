"""Explicit per-instance runtime context for isolated core services."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypedDict, cast


class ParserDependencies(TypedDict, total=False):
    """Named parser bindings copied from the dynamic core namespace."""

    ANCHOR_PRESETS: dict[str, str]
    AndTermUnsatisfiable: type[Exception]
    MAX_ANCHOR_DNF_TERMS: int
    OMIT_PRESETS: dict[str, str]
    ParseError: type[Exception]
    YearTokenFormatError: type[Exception]
    _QUARTERS: Any
    _WEEKDAYS: Any
    _active_mod_keys: Callable[..., Any]
    _build_acf_impl: Callable[..., Any]
    _clone_dnf: Callable[..., Any]
    _day_offset_re: Any
    _hhmm_re: Any
    _import_sibling: Callable[[str], Any]
    _is_atom_like: Callable[..., Any]
    _month_from_alias: Callable[..., Any]
    _natural_language: Any
    _next_prev_wd_re: Any
    _normalize_spec_for_acf: Callable[..., Any]
    _parse_anchor_expr_to_dnf_cached_impl: Callable[..., Any]
    _parse_group_with_inline_mods: Callable[..., Any]
    _parse_y_token: Callable[..., Any]
    _parser_atoms: Any
    _parser_dnf: Any
    _parser_frontend: Any
    _position_selection: Any
    _rewrite_weekly_multi_time_atoms: Callable[..., Any]
    _rewrite_year_month_aliases_in_context: Callable[..., Any]
    _satisfiability: Any
    _season_support: Any
    _split_csv_lower: Callable[..., Any]
    _split_csv_tokens: Callable[..., Any]
    _strict_validation: Any
    _ttl_lru_cache: Callable[..., Any]
    _unwrap_quotes: Callable[..., Any]
    _validate_monthly_spec: Callable[..., Any]
    _validate_weekly_spec: Callable[..., Any]
    _yearfmt: Callable[..., Any]
    _yearly_validation: Any
    atom_matches_on: Callable[..., Any]
    expand_yearly_cached: Callable[..., Any]
    re: Any


def parser_dependencies(values: Mapping[str, Any]) -> ParserDependencies:
    """Retain immutable per-binding values while declaring known parser keys."""
    return cast(ParserDependencies, MappingProxyType(dict(values)))


class CacheDependencies(TypedDict, total=False):
    """Named cache bindings copied from the dynamic core namespace."""

    ANCHOR_CACHE_DIR_OVERRIDE: str
    ANCHOR_CACHE_TTL: int
    ANCHOR_YEAR_FMT: str
    ENABLE_ANCHOR_CACHE: bool
    LOCAL_TZ_NAME: str
    NAUTICAL_RELEASE_ID: str
    SEASON_HEMISPHERE: str
    WRAND_SALT: str
    _CACHE_LOAD_MEM: Any
    _CACHE_LOAD_MEM_MAX: int
    _CACHE_LOAD_MEM_TTL: float
    _CACHE_LOCK_JITTER: float
    _CACHE_LOCK_RETRIES: int
    _CACHE_LOCK_SLEEP_BASE: float
    _CACHE_LOCK_STALE_AFTER: float
    _cache_atomic_replace: Callable[..., Any]
    _cache_lock: Callable[..., Any]
    _cache_payload_shape_ok: Callable[..., Any]
    _cache_semantic_fingerprint: Callable[..., Any]
    _clone_cache_payload: Callable[..., Any]
    _import_sibling: Callable[[str], Any]
    _is_dnf_like: Callable[..., Any]
    _nautical_cache_dir: Callable[..., Any]
    _normalize_dnf_cached: Callable[..., Any]
    _ttl_lru_cache: Callable[..., Any]
    _validated_user_dir: Any
    _yearfmt: Callable[..., Any]
    base64: Any
    build_acf: Callable[..., Any]
    business_calendar_fingerprint: Callable[..., Any]
    diag: Callable[[str], None]
    effective_config_fingerprint: Callable[..., Any]
    fcntl: Any
    json: Any
    os: Any
    random: Any
    scheduler_config_fingerprint: Callable[..., Any]
    time: Any
    zlib: Any
    __file__: str


def cache_dependencies(values: Mapping[str, Any]) -> CacheDependencies:
    """Retain immutable per-binding values while declaring known cache keys."""
    return cast(CacheDependencies, MappingProxyType(dict(values)))


@dataclass(slots=True)
class CacheState:
    """Mutable per-binding cache entries kept separate from configuration."""

    memory: OrderedDict[str, tuple[tuple[int, int, int, int], dict[str, Any], float]]
    max_entries: int
    ttl: float


@dataclass(frozen=True, slots=True)
class CoreContext:
    """Own one core namespace and its sibling-module loader.

    The context is intentionally small: it provides an explicit hand-off point
    for parser, scheduler, and cache services without changing legacy callers.
    """

    namespace: dict[str, Any]
    import_sibling: Callable[[str], Any]
    source_file: str | None = None

    @classmethod
    def from_core(cls, module: Any, *, namespace: Mapping[str, Any] | None = None) -> "CoreContext":
        values = namespace if namespace is not None else vars(module)
        loader = values.get("_import_sibling", getattr(module, "_import_sibling", None))
        if not callable(loader):
            raise TypeError("core context requires a callable sibling-module loader")
        return cls(dict(values), loader, getattr(module, "__file__", None))

    def get(self, name: str, default: Any = None) -> Any:
        return self.namespace.get(name, default)

    def require(self, name: str) -> Any:
        try:
            return self.namespace[name]
        except KeyError as exc:
            raise RuntimeError(f"core context is missing required binding: {name}") from exc

    def __getattr__(self, name: str) -> Any:
        """Expose legacy ``module._name`` reads through the owned namespace."""
        try:
            return self.namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
