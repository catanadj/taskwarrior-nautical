"""Public cache entry points bound to one core facade instance."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    """Create cache APIs without sharing cache state across core loaders."""
    core = namespace if namespace is not None else vars(module)
    cache_dir_state: list[str | None] = [None]
    import_sibling = core.get("_import_sibling", module._import_sibling)
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
            is_dnf_like=core.get("_is_dnf_like", is_dnf_like),
        )

    def cache_atomic_replace(src: str, dst: str) -> None:
        cache_payload.cache_atomic_replace(src, dst, os_mod=core["os"])

    def safe_lock_sleep_once(sleep_base: float, jitter: float) -> None:
        cache_locking.safe_lock_sleep_once(
            sleep_base,
            jitter,
            time_mod=core["time"],
            random_mod=core["random"],
        )

    def safe_lock_ensure_parent(path_str: str, mkdir: bool) -> None:
        cache_locking.safe_lock_ensure_parent(path_str, mkdir, os_mod=core["os"])

    def safe_lock_age(path_str: str) -> float | None:
        return cache_locking.safe_lock_age(
            path_str,
            time_mod=core["time"],
            os_mod=core["os"],
        )

    def safe_lock_stale_pid(path_str: str, stale_after: float | None) -> bool:
        return cache_locking.safe_lock_stale_pid(
            path_str,
            stale_after,
            time_mod=core["time"],
            os_mod=core["os"],
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
            fcntl_mod=core["fcntl"],
            os_mod=core["os"],
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
            os_mod=core["os"],
            time_mod=core["time"],
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
            fcntl_mod=core["fcntl"],
            os_mod=core["os"],
            time_mod=core["time"],
            random_mod=core["random"],
        ) as acquired:
            yield acquired

    @contextmanager
    def cache_lock(key: str):
        with cache_locking.cache_lock(
            key,
            cache_lock_path=cache_lock_path,
            safe_lock=safe_lock,
            cache_lock_retries=core["_CACHE_LOCK_RETRIES"],
            cache_lock_sleep_base=core["_CACHE_LOCK_SLEEP_BASE"],
            cache_lock_jitter=core["_CACHE_LOCK_JITTER"],
            cache_lock_stale_after=core["_CACHE_LOCK_STALE_AFTER"],
        ) as acquired:
            yield acquired

    def cache_dir() -> str:
        current = core.get("_CACHE_DIR", cache_dir_state[0])
        chosen = cache_locking.cache_dir(
            current,
            anchor_cache_dir_override=core["ANCHOR_CACHE_DIR_OVERRIDE"],
            nautical_cache_dir_path=core["_nautical_cache_dir"](),
            validated_user_dir=core["_validated_user_dir"],
            select_cache_dir=cache_support.select_cache_dir,
        )
        cache_dir_state[0] = chosen
        core["_CACHE_DIR"] = chosen
        return chosen

    def cache_key(
        acf: str,
        anchor_mode: str,
        *,
        business_calendar_fingerprint: str = "",
    ) -> str:
        config_fingerprint = core["scheduler_config_fingerprint"]()
        profile_fingerprint = (
            f"{business_calendar_fingerprint}|season:{core['SEASON_HEMISPHERE']}"
            f"|config:{config_fingerprint}|parser:2|cache:2"
        )
        return cache_support.cache_key(
            acf,
            anchor_mode,
            business_calendar_fingerprint=profile_fingerprint,
            anchor_year_fmt=core["ANCHOR_YEAR_FMT"],
            wrand_salt=core["WRAND_SALT"],
            local_tz_name=core["LOCAL_TZ_NAME"],
            holiday_region=core["HOLIDAY_REGION"],
        )

    def cache_path(key: str) -> str:
        return cache_support.cache_path(cache_dir(), key)

    def cache_lock_path(key: str) -> str:
        return cache_support.cache_lock_path(cache_dir(), key)

    def quarantine_cache(key: str, path: str) -> bool:
        """Move a broken cache entry aside so future reads become clean misses."""
        try:
            with core.get("_cache_lock", cache_lock)(key) as locked:
                if not locked or not core["os"].path.exists(path):
                    return False
                target = f"{path}.bad.{core['os'].getpid()}.{core['time'].time_ns()}"
                core["os"].replace(path, target)
                core["_CACHE_LOAD_MEM"].pop(key, None)
                return True
        except Exception:
            return False

    def cache_load_impl(key: str) -> dict | None:
        return core["_cache_payload"].cache_load(
            key,
            enable_anchor_cache=core["ENABLE_ANCHOR_CACHE"],
            cache_path=cache_path,
            anchor_cache_ttl=core["ANCHOR_CACHE_TTL"],
            time_mod=core["time"],
            cache_load_mem=core["_CACHE_LOAD_MEM"],
            cache_load_mem_ttl=core["_CACHE_LOAD_MEM_TTL"],
            clone_cache_payload=core.get("_clone_cache_payload", clone_cache_payload),
            normalize_dnf_cached=core.get("_normalize_dnf_cached", normalize_dnf_cached),
            cache_payload_shape_ok=core.get("_cache_payload_shape_ok", cache_payload_shape_ok),
            cache_load_mem_max=core["_CACHE_LOAD_MEM_MAX"],
            diag=core["diag"],
            quarantine_cache=quarantine_cache,
            os_mod=core["os"],
            json_mod=core["json"],
            zlib_mod=core["zlib"],
            base64_mod=core["base64"],
        )

    def cache_save_impl(key: str, obj: dict) -> bool:
        return core["_cache_payload"].cache_save(
            key,
            obj,
            enable_anchor_cache=core["ENABLE_ANCHOR_CACHE"],
            json_mod=core["json"],
            zlib_mod=core["zlib"],
            base64_mod=core["base64"],
            cache_path=cache_path,
            cache_dir=cache_dir,
            cache_lock=core.get("_cache_lock", cache_lock),
            diag=core["diag"],
            os_mod=core["os"],
            tempfile_mod=core["tempfile"],
            cache_atomic_replace=core.get("_cache_atomic_replace", cache_atomic_replace),
            cache_load_mem=core["_CACHE_LOAD_MEM"],
        )

    def cache_gc_impl(
        *,
        max_entries: int = 512,
        stale_tmp_age: float = 86400.0,
        stale_lock_age: float = 86400.0,
    ) -> dict:
        """Prune expired and orphaned anchor cache files outside hook hot paths."""
        return core["_cache_payload"].cache_gc(
            cache_dir(),
            ttl=core["ANCHOR_CACHE_TTL"],
            max_entries=max_entries,
            stale_tmp_age=stale_tmp_age,
            stale_lock_age=stale_lock_age,
            cache_lock=core.get("_cache_lock", cache_lock),
            stale_lock_check=lambda path, age: safe_lock_stale_pid(path, age)
            and (safe_lock_age(path) or 0.0) >= float(age),
            time_mod=core["time"],
            os_mod=core["os"],
        )

    ttl_lru_cache = core["_ttl_lru_cache"]

    @ttl_lru_cache(maxsize=1024)
    def cache_key_for_task_cached(
        anchor_expr: str,
        anchor_mode: str,
        fmt: str,
        business_calendar_fingerprint: str = "",
        config_fingerprint: str = "",
    ) -> str:
        _ = config_fingerprint
        return core["_cache_payload"].cache_key_for_task_cached(
            anchor_expr,
            anchor_mode,
            fmt,
            business_calendar_fingerprint,
            build_acf=core["build_acf"],
            cache_key=cache_key,
        )

    def cache_key_for_task_impl(
        anchor_expr: str,
        anchor_mode: str,
        calendar_fingerprint: str | None = None,
    ) -> str:
        if calendar_fingerprint is None:
            calendar_fingerprint = core["business_calendar_fingerprint"]()
        return cache_key_for_task_cached(
            anchor_expr or "",
            anchor_mode or "",
            core["_yearfmt"](),
            calendar_fingerprint,
            core["effective_config_fingerprint"](),
        )

    def _source_signature(path: Any) -> str:
        try:
            stat = os.stat(path)
            return f"{getattr(stat, 'st_mtime_ns', 0)}:{stat.st_size}"
        except Exception:
            return "unknown"

    def dnf_cache_fingerprint() -> str:
        """Identify parser, cache schema, and installed release inputs."""
        parser_parts = []
        for module_name in ("parser_dnf", "parser_api", "parser_support_api", "parser_models", "strict_validation"):
            try:
                sibling = import_sibling(module_name)
                parser_parts.append(f"{module_name}:{_source_signature(getattr(sibling, '__file__', ''))}")
            except Exception:
                parser_parts.append(f"{module_name}:unavailable")
        release = _source_signature(getattr(module, "__file__", ""))
        schema = getattr(cache_payload, "CACHE_SCHEMA_VERSION", "unknown")
        return f"parser={'|'.join(parser_parts)}|schema:{schema}|release:{release}"

    def dnf_cache_key(expr: str) -> str:
        payload = "|".join(
            (
                "nautical-dnf",
                str(expr or ""),
                dnf_cache_fingerprint(),
                str(core["effective_config_fingerprint"]()),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _dnf_cache_enabled() -> bool:
        raw = str(core["os"].environ.get("NAUTICAL_DNF_DISK_CACHE") or "1").strip().lower()
        return bool(core.get("ENABLE_ANCHOR_CACHE", True)) and raw in {"1", "true", "yes", "on"}

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

    return SimpleNamespace(
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
