"""Core-bound Anchor Canonical Form (ACF) operations."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)
    acf = core["_acf_support"]
    ttl_lru_cache = core["_ttl_lru_cache"]

    def atom_sort_key(value: dict) -> tuple:
        return acf.atom_sort_key(value, json_mod=core["json"])

    def acf_unpack(packed: str) -> dict:
        return acf.acf_unpack(
            packed,
            base64_mod=core["base64"],
            zlib_mod=core["zlib"],
            json_mod=core["json"],
        )

    @ttl_lru_cache(maxsize=512)
    def year_pair_cached(a: int, b: int, fmt: str) -> tuple[int, int]:
        return (b, a) if fmt == "MD" else (a, b)

    def year_pair(a: int, b: int) -> tuple[int, int]:
        return year_pair_cached(a, b, core["_yearfmt"]())

    def normalize_spec_for_acf_uncached(typ: str, spec: str):
        return acf.normalize_spec_for_acf_uncached(
            typ,
            spec,
            expand_weekly_aliases=core["_expand_weekly_aliases"],
            split_csv_tokens=core["_split_csv_tokens"],
            normalize_weekday=core["_normalize_weekday"],
            expand_monthly_aliases=core["_expand_monthly_aliases"],
            re_mod=core["re"],
            year_pair=year_pair,
        )

    @ttl_lru_cache(maxsize=512)
    def normalize_spec_for_acf_cached(typ: str, spec: str, fmt: str):
        typ = (typ or "").strip().lower()
        if typ not in ("w", "m", "y"):
            return None
        spec = (spec or "").strip().lower()[:256]
        fmt = "DM" if (fmt or "").upper() == "DM" else "MD"
        return normalize_spec_for_acf_uncached(typ, spec)

    def normalize_spec_for_acf(typ: str, spec: str):
        return acf.normalize_spec_for_acf(
            typ,
            spec,
            normalize_spec_for_acf_cached=lambda t, s: normalize_spec_for_acf_cached(
                t, s, core["_yearfmt"]()
            ),
            clone_mod_value=core["_clone_mod_value"],
        )

    def mods_to_acf(mods: dict) -> dict:
        return acf.mods_to_acf(mods, hhmm_re=core["_hhmm_re"])

    def acf_mods_to_string(value: dict) -> str:
        return acf.acf_mods_to_string(value, wd_abbr=core["_WD_ABBR"])

    def acf_spec_to_string(typ: str, spec) -> str:
        return acf.acf_spec_to_string(
            typ,
            spec,
            tok=core["_tok"],
            tok_range=core["_tok_range"],
        )

    def build_acf_impl(expr: str) -> str:
        return acf.build_acf(
            expr,
            parse_anchor_expr_to_dnf_cached=core["parse_anchor_expr_to_dnf_cached"],
            coerce_int=core["coerce_int"],
            normalize_spec_for_acf=core.get(
                "_normalize_spec_for_acf", normalize_spec_for_acf
            ),
            mods_to_acf=mods_to_acf,
            atom_sort_key=atom_sort_key,
            json_mod=core["json"],
            zlib_mod=core["zlib"],
            base64_mod=core["base64"],
            hashlib_mod=core["hashlib"],
            acf_checksum_len=core["ACF_CHECKSUM_LEN"],
        )

    def is_valid_acf(value: str) -> bool:
        return acf.is_valid_acf(
            value,
            hashlib_mod=core["hashlib"],
            acf_checksum_len=core["ACF_CHECKSUM_LEN"],
            acf_unpack=acf_unpack,
        )

    def acf_to_original_format(value: str) -> str:
        return acf.acf_to_original_format(
            value,
            is_valid_acf=core.get("is_valid_acf", is_valid_acf),
            acf_unpack=acf_unpack,
            acf_spec_to_string=acf_spec_to_string,
            acf_mods_to_string=acf_mods_to_string,
            format_selection_positions=core["_position_selection"].format_positions,
        )

    return SimpleNamespace(
        _atom_sort_key=atom_sort_key,
        _acf_unpack=acf_unpack,
        _year_pair_cached=year_pair_cached,
        _year_pair=year_pair,
        _normalize_spec_for_acf_uncached=normalize_spec_for_acf_uncached,
        _normalize_spec_for_acf_cached=normalize_spec_for_acf_cached,
        _normalize_spec_for_acf=normalize_spec_for_acf,
        _mods_to_acf=mods_to_acf,
        _acf_mods_to_string=acf_mods_to_string,
        _acf_spec_to_string=acf_spec_to_string,
        _build_acf_impl=build_acf_impl,
        is_valid_acf=is_valid_acf,
        acf_to_original_format=acf_to_original_format,
    )


__all__ = ("for_core",)
