"""Public cache entry points bound to one deps facade instance."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
import random
import tempfile
import time
from typing import Any, Callable
from .api_bindings import ApiBinding, core_namespace
import zlib

from .core_context import CacheDependencies, CacheState, CoreContext

fcntl: Any
try:
    import fcntl
except Exception:
    fcntl = None


def for_core(module: Any = None, *, namespace: dict[str, Any] | None = None, context: CoreContext | None = None) -> ApiBinding:
    """Create cache APIs without sharing cache state across deps loaders."""
    deps = CacheDependencies.from_mapping(
        context.namespace if context is not None
        else core_namespace(module, namespace, context, "cache_api")
    )
    import_sibling: Callable[[str], Any] | None
    if context is not None:
        import_sibling = context.import_sibling
    else:
        import_sibling = deps.get("_import_sibling")
        if not callable(import_sibling):
            import_sibling = getattr(module, "_import_sibling", None)
        if not callable(import_sibling):
            raise TypeError("cache_api.for_core requires a callable sibling-module loader")
    cache_dir_state: list[str | None] = [None]
    cache_state = CacheState(
        memory=deps["_CACHE_LOAD_MEM"],
        max_entries=int(deps["_CACHE_LOAD_MEM_MAX"]),
        ttl=float(deps["_CACHE_LOAD_MEM_TTL"]),
    )
    cache_support = import_sibling("cache_support")
    cache_locking = import_sibling("cache_locking")
    cache_payload = import_sibling("cache_payload")

    def is_atom_like(atom) -> bool:
        return cache_payload.is_factor_like(atom)

    def is_dnf_like(dnf) -> bool:
        return cache_payload.is_dnf_like(dnf, is_atom_like=is_atom_like)

    clone_mod_value = cache_payload.clone_mod_value
    clone_mods = cache_payload.clone_mods
    clone_atom = cache_payload.clone_atom
    clone_dnf = cache_payload.clone_dnf
    clone_cache_payload = cache_payload.clone_cache_payload
    normalize_dnf_cached = cache_payload.normalize_dnf_cached

    def cache_payload_shape_ok(obj: dict) -> bool:
        return cache_payload.cache_payload_shape_ok(
            obj,
            is_dnf_like=deps.get("_is_dnf_like", is_dnf_like),
        )

    def cache_atomic_replace(src: str, dst: str) -> None:
        cache_payload.cache_atomic_replace(src, dst, os_mod=deps["os"])

    def safe_lock_sleep_once(sleep_base: float, jitter: float) -> None:
        cache_locking.safe_lock_sleep_once(
            sleep_base,
            jitter,
            time_mod=deps.get("time", time),
            random_mod=deps.get("random", random),
        )

    def safe_lock_ensure_parent(path_str: str, mkdir: bool) -> None:
        cache_locking.safe_lock_ensure_parent(path_str, mkdir, os_mod=deps["os"])

    def safe_lock_age(path_str: str) -> float | None:
        return cache_locking.safe_lock_age(
            path_str,
            time_mod=deps.get("time", time),
            os_mod=deps["os"],
        )

    def safe_lock_stale_pid(path_str: str, stale_after: float | None) -> bool:
        return cache_locking.safe_lock_stale_pid(
            path_str,
            stale_after,
            time_mod=deps.get("time", time),
            os_mod=deps["os"],
        )

    @contextmanager
    def safe_lock_fcntl_context(
        path_str: str,
        *,
        tries: int,
        sleep_base: float,
        jitter: float,
        mode: int,
        mkdir: bool,
    ):
        with cache_locking.safe_lock_fcntl_context(
            path_str,
            tries=tries,
            sleep_base=sleep_base,
            jitter=jitter,
            mode=mode,
            mkdir=mkdir,
            safe_lock_ensure_parent=safe_lock_ensure_parent,
            safe_lock_sleep_once=safe_lock_sleep_once,
            fcntl_mod=deps.get("fcntl", fcntl),
            os_mod=deps["os"],
        ) as acquired:
            yield acquired

    @contextmanager
    def safe_lock_excl_context(
        path_str: str,
        *,
        tries: int,
        sleep_base: float,
        jitter: float,
        mode: int,
        mkdir: bool,
        stale_after: float | None,
    ):
        with cache_locking.safe_lock_excl_context(
            path_str,
            tries=tries,
            sleep_base=sleep_base,
            jitter=jitter,
            mode=mode,
            mkdir=mkdir,
            stale_after=stale_after,
            safe_lock_ensure_parent=safe_lock_ensure_parent,
            safe_lock_stale_pid=safe_lock_stale_pid,
            safe_lock_age=safe_lock_age,
            safe_lock_sleep_once=safe_lock_sleep_once,
            os_mod=deps["os"],
            time_mod=deps.get("time", time),
        ) as acquired:
            yield acquired

    @contextmanager
    def safe_lock(
        path: str,
        *,
        retries: int = 6,
        sleep_base: float = 0.05,
        jitter: float = 0.0,
        mode: int = 0o600,
        mkdir: bool = True,
        stale_after: float | None = 60.0,
    ):
        with cache_locking.safe_lock(
            path,
            retries=retries,
            sleep_base=sleep_base,
            jitter=jitter,
            mode=mode,
            mkdir=mkdir,
            stale_after=stale_after,
            fcntl_mod=deps.get("fcntl", fcntl),
            os_mod=deps["os"],
            time_mod=deps.get("time", time),
            random_mod=deps.get("random", random),
        ) as acquired:
            yield acquired

    @contextmanager
    def cache_lock(key: str):
        with cache_locking.cache_lock(
            key,
            cache_lock_path=cache_lock_path,
            safe_lock=safe_lock,
            cache_lock_retries=deps["_CACHE_LOCK_RETRIES"],
            cache_lock_sleep_base=deps["_CACHE_LOCK_SLEEP_BASE"],
            cache_lock_jitter=deps["_CACHE_LOCK_JITTER"],
            cache_lock_stale_after=deps["_CACHE_LOCK_STALE_AFTER"],
        ) as acquired:
            yield acquired

    def cache_dir() -> str:
        current = cache_dir_state[0]
        chosen = cache_locking.cache_dir(
            current,
            anchor_cache_dir_override=deps["ANCHOR_CACHE_DIR_OVERRIDE"],
            nautical_cache_dir_path=deps["_nautical_cache_dir"](),
            validated_user_dir=deps["_validated_user_dir"],
            select_cache_dir=cache_support.select_cache_dir,
        )
        cache_dir_state[0] = chosen
        return chosen

    def _source_signature(path: Any) -> str:
        try:
            stat = os.stat(path)
            return f"{getattr(stat, 'st_mtime_ns', 0)}:{stat.st_size}"
        except Exception:
            return "unknown"

    semantic_source_files = (
        "__init__.py",
        "cache_api.py",
        "cache_payload.py",
        "parser_api.py",
        "parsing/parser_atoms.py",
        "parsing/parser_dnf.py",
        "parsing/parser_models.py",
        "parsing/parser_support_api.py",
        "strict_validation.py",
        "scheduler_api.py",
        "scheduler_atom.py",
        "scheduler_expr.py",
        "scheduler_models.py",
        "anchor_inclusion.py",
        "natural_language.py",
        "precompute.py",
        "hint_builder_api.py",
        "recurrence_evaluator.py",
        "time_slots.py",
    )
    semantic_fingerprint_state: list[str | None] = [None]

    def cache_semantic_fingerprint() -> str:
        """Return one process-stable fingerprint for hint semantics."""
        cached = semantic_fingerprint_state[0]
        if cached is not None:
            return cached
        source_file = context.source_file if context is not None else getattr(module, "__file__", "")
        if not isinstance(source_file, str):
            source_file = ""
        package_dir = os.path.dirname(os.path.abspath(source_file))
        release_hint = str(deps.get("NAUTICAL_RELEASE_ID") or os.environ.get("NAUTICAL_RELEASE_ID") or "")
        source_parts = [
            f"{name}:{_source_signature(os.path.join(package_dir, name))}"
            for name in semantic_source_files
        ]
        payload = "|".join(("nautical-hints|semantic-v1", f"release:{release_hint}", *source_parts))
        semantic_fingerprint_state[0] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return str(semantic_fingerprint_state[0])

    def cache_key(
        acf: str,
        anchor_mode: str,
        *,
        business_calendar_fingerprint: str = "",
    ) -> str:
        config_fingerprint = deps["scheduler_config_fingerprint"]()
        semantic_fingerprint = deps.get("_cache_semantic_fingerprint", cache_semantic_fingerprint)()
        profile_fingerprint = (
            f"{business_calendar_fingerprint}|season:{deps['SEASON_HEMISPHERE']}"
            f"|config:{config_fingerprint}|semantic:{semantic_fingerprint}"
        )
        return cache_support.cache_key(
            acf,
            anchor_mode,
            business_calendar_fingerprint=profile_fingerprint,
            anchor_year_fmt=deps["ANCHOR_YEAR_FMT"],
            wrand_salt=deps["WRAND_SALT"],
            local_tz_name=deps["LOCAL_TZ_NAME"],
        )

    def cache_path(key: str) -> str:
        return cache_support.cache_path(cache_dir(), key)

    def cache_lock_path(key: str) -> str:
        return cache_support.cache_lock_path(cache_dir(), key)

    def quarantine_cache(key: str, path: str) -> bool:
        """Move a broken cache entry aside so future reads become clean misses."""
        try:
            with deps.get("_cache_lock", cache_lock)(key) as locked:
                if not locked or not deps["os"].path.exists(path):
                    return False
                target = f"{path}.bad.{deps['os'].getpid()}.{deps.get('time', time).time_ns()}"
                deps["os"].replace(path, target)
                cache_state.memory.pop(key, None)
                return True
        except Exception:
            return False

    def cache_load_impl(key: str) -> dict | None:
        return cache_payload.cache_load(
            key,
            enable_anchor_cache=deps["ENABLE_ANCHOR_CACHE"],
            cache_path=cache_path,
            anchor_cache_ttl=deps["ANCHOR_CACHE_TTL"],
            time_mod=deps.get("time", time),
            cache_load_mem=cache_state.memory,
            cache_load_mem_ttl=cache_state.ttl,
            clone_cache_payload=deps.get("_clone_cache_payload", clone_cache_payload),
            normalize_dnf_cached=deps.get("_normalize_dnf_cached", normalize_dnf_cached),
            cache_payload_shape_ok=deps.get("_cache_payload_shape_ok", cache_payload_shape_ok),
            cache_load_mem_max=cache_state.max_entries,
            diag=deps["diag"],
            quarantine_cache=quarantine_cache,
            os_mod=deps["os"],
            json_mod=deps.get("json", json),
            zlib_mod=deps.get("zlib", zlib),
            base64_mod=deps.get("base64", base64),
        )

    def cache_save_impl(key: str, obj: dict) -> bool:
        return cache_payload.cache_save(
            key,
            obj,
            enable_anchor_cache=deps["ENABLE_ANCHOR_CACHE"],
            json_mod=deps.get("json", json),
            zlib_mod=deps.get("zlib", zlib),
            base64_mod=deps.get("base64", base64),
            cache_path=cache_path,
            cache_dir=cache_dir,
            cache_lock=cache_lock,
            diag=deps["diag"],
            os_mod=deps["os"],
            tempfile_mod=tempfile,
            cache_atomic_replace=deps.get("_cache_atomic_replace", cache_atomic_replace),
            cache_load_mem=cache_state.memory,
        )

    def cache_gc_impl(
        *,
        max_entries: int = 512,
        stale_tmp_age: float = 86400.0,
        stale_lock_age: float = 86400.0,
    ) -> dict:
        """Prune expired and orphaned anchor cache files outside hook hot paths."""
        return cache_payload.cache_gc(
            cache_dir(),
            ttl=deps["ANCHOR_CACHE_TTL"],
            max_entries=max_entries,
            stale_tmp_age=stale_tmp_age,
            stale_lock_age=stale_lock_age,
            cache_lock=cache_lock,
            stale_lock_check=lambda path, age: safe_lock_stale_pid(path, age)
            and (safe_lock_age(path) or 0.0) >= float(age),
            time_mod=deps.get("time", time),
            os_mod=deps["os"],
        )

    ttl_lru_cache = deps["_ttl_lru_cache"]

    @ttl_lru_cache(maxsize=1024)
    def cache_key_for_task_cached(
        anchor_expr: str,
        anchor_mode: str,
        fmt: str,
        business_calendar_fingerprint: str = "",
        config_fingerprint: str = "",
        semantic_fingerprint: str = "",
    ) -> str:
        _ = config_fingerprint
        _ = semantic_fingerprint
        return cache_payload.cache_key_for_task_cached(
            anchor_expr,
            anchor_mode,
            fmt,
            business_calendar_fingerprint,
            build_acf=deps["build_acf"],
            cache_key=cache_key,
        )

    def cache_key_for_task_impl(
        anchor_expr: str,
        anchor_mode: str,
        calendar_fingerprint: str | None = None,
    ) -> str:
        if calendar_fingerprint is None:
            calendar_fingerprint = deps["business_calendar_fingerprint"]()
        semantic_fingerprint = deps.get("_cache_semantic_fingerprint", cache_semantic_fingerprint)()
        return cache_key_for_task_cached(
            anchor_expr or "",
            anchor_mode or "",
            deps["_yearfmt"](),
            calendar_fingerprint,
            deps["effective_config_fingerprint"](),
            semantic_fingerprint,
        )

    def dnf_cache_fingerprint() -> str:
        """Identify parser, cache schema, and installed release inputs."""
        parser_parts = []
        for module_name in (
            "parsing.parser_atoms",
            "parsing.parser_dnf",
            "parsing.parser_frontend",
            "parser_api",
            "parsing.parser_support_api",
            "parsing.parser_models",
            "strict_validation",
        ):
            try:
                sibling = import_sibling(module_name)
                parser_parts.append(f"{module_name}:{_source_signature(getattr(sibling, '__file__', ''))}")
            except Exception:
                parser_parts.append(f"{module_name}:unavailable")
        release = _source_signature(context.source_file if context is not None else getattr(module, "__file__", ""))
        schema = getattr(cache_payload, "CACHE_SCHEMA_VERSION", "unknown")
        return f"parser={'|'.join(parser_parts)}|schema:{schema}|release:{release}"

    def dnf_cache_key(expr: str) -> str:
        payload = "|".join(
            (
                "nautical-dnf",
                str(expr or ""),
                dnf_cache_fingerprint(),
                str(deps["effective_config_fingerprint"]()),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _dnf_cache_enabled() -> bool:
        raw = str(deps["os"].environ.get("NAUTICAL_DNF_DISK_CACHE") or "1").strip().lower()
        return bool(deps.get("ENABLE_ANCHOR_CACHE", True)) and raw in {"1", "true", "yes", "on"}

    def dnf_cache_load(expr: str):
        if not _dnf_cache_enabled():
            return None
        key = dnf_cache_key(expr)
        payload = cache_load_impl(key)
        if not isinstance(payload, dict) or payload.get("kind") != "anchor-dnf":
            if payload is not None:
                quarantine_cache(key, cache_path(key))
            return None
        dnf = payload.get("dnf")
        if not is_dnf_like(dnf):
            quarantine_cache(key, cache_path(key))
            return None
        return normalize_dnf_cached(dnf)

    def dnf_cache_save(expr: str, dnf: Any) -> bool:
        if not _dnf_cache_enabled() or not is_dnf_like(dnf):
            return False
        return cache_save_impl(
            dnf_cache_key(expr),
            {"kind": "anchor-dnf", "dnf": clone_dnf(dnf)},
        )

    # Bind the complete lock port once; compatibility names below continue to
    # expose the same per-deps callables without rebuilding dependencies.
    bound_locking = cache_locking.bind_locking(
        cache_lock_path=cache_lock_path,
        retries=deps["_CACHE_LOCK_RETRIES"],
        sleep_base=deps["_CACHE_LOCK_SLEEP_BASE"],
        jitter=deps["_CACHE_LOCK_JITTER"],
        stale_after=deps["_CACHE_LOCK_STALE_AFTER"],
        fcntl_mod=deps.get("fcntl", fcntl),
        os_mod=deps["os"],
        time_mod=deps.get("time", time),
        random_mod=deps.get("random", random),
    )
    safe_lock = bound_locking.safe_lock
    cache_lock = bound_locking.cache_lock

    return ApiBinding.from_kwargs(
        _safe_lock_sleep_once=safe_lock_sleep_once,
        _safe_lock_ensure_parent=safe_lock_ensure_parent,
        _safe_lock_age=safe_lock_age,
        _safe_lock_stale_pid=safe_lock_stale_pid,
        _safe_lock_fcntl_context=safe_lock_fcntl_context,
        _safe_lock_excl_context=safe_lock_excl_context,
        safe_lock=safe_lock,
        _cache_lock=cache_lock,
        _is_atom_like=is_atom_like,
        _is_dnf_like=is_dnf_like,
        _clone_mod_value=clone_mod_value,
        _clone_mods=clone_mods,
        _clone_atom=clone_atom,
        _clone_dnf=clone_dnf,
        _clone_cache_payload=clone_cache_payload,
        _normalize_dnf_cached=normalize_dnf_cached,
        _cache_payload_shape_ok=cache_payload_shape_ok,
        _cache_atomic_replace=cache_atomic_replace,
        _cache_dir=cache_dir,
        _cache_key=cache_key,
        _cache_path=cache_path,
        _cache_lock_path=cache_lock_path,
        _quarantine_cache=quarantine_cache,
        _cache_load_impl=cache_load_impl,
        _cache_save_impl=cache_save_impl,
        _cache_gc_impl=cache_gc_impl,
        _cache_key_for_task_cached=cache_key_for_task_cached,
        _cache_key_for_task_impl=cache_key_for_task_impl,
        _cache_semantic_fingerprint=cache_semantic_fingerprint,
        cache_load=cache_load_impl,
        cache_save=cache_save_impl,
        cache_gc=cache_gc_impl,
        cache_key_for_task=cache_key_for_task_impl,
        _dnf_cache_fingerprint=dnf_cache_fingerprint,
        _dnf_cache_key=dnf_cache_key,
        _dnf_cache_load=dnf_cache_load,
        _dnf_cache_save=dnf_cache_save,
    )


__all__ = ("for_core",)
