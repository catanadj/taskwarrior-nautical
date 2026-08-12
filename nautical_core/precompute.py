from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable


def build_and_cache_hints(
    anchor_expr: str,
    *,
    anchor_mode: str,
    default_due_dt,
    cache_key_for_task,
    cache_load,
    validate_anchor_expr_strict,
    describe_anchor_expr_from_dnf,
    cache_save,
    anchor_year_fmt: str,
    wrand_salt: str,
    local_tz_name: str,
    holiday_region: str,
    business_calendar_fingerprint: str = "",
    include_per_year: bool = True,
    hint_builder: Any | None = None,
    hint_builder_factory: Callable[[], Any] | None = None,
):
    def _canonical(value):
        if isinstance(value, dict):
            return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (list, tuple)):
            return [_canonical(item) for item in value]
        return value

    cache_mode = "annual" if include_per_year else "next-only"
    request_signature = (
        f"{anchor_mode}|hints:{cache_mode}|schema:2|"
        f"start:{default_due_dt.isoformat() if hasattr(default_due_dt, 'isoformat') else default_due_dt}"
    )
    key = cache_key_for_task(
        anchor_expr,
        request_signature,
        business_calendar_fingerprint,
    )
    cached = cache_load(key)
    if cached:
        # A semantic key prevents normal upgrades from reaching this branch,
        # but validate the stored DNF as a second line of defense.  This keeps
        # manually restored or legacy entries from bypassing current parser
        # and satisfiability checks.
        try:
            current_dnf = validate_anchor_expr_strict(anchor_expr)
            if _canonical(cached.get("dnf")) == _canonical(current_dnf):
                return cached
        except Exception:
            pass

    dnf = validate_anchor_expr_strict(anchor_expr)
    natural = describe_anchor_expr_from_dnf(dnf, default_due_dt=default_due_dt)
    if hint_builder is None and hint_builder_factory is not None:
        hint_builder = hint_builder_factory()
    if hint_builder is not None:
        hints = hint_builder.build(
            start_dt=default_due_dt,
            k_next=24,
            sample_days_for_year=366,
            now_local=datetime.now,
            include_per_year=include_per_year,
        )
    else:
        raise TypeError("Hint generation requires the typed HintBuilder service.")

    payload = {
        "meta": {
            "created": int(time.time()),
            "cfg": {
                "fmt": anchor_year_fmt,
                "salt": wrand_salt,
                "tz": local_tz_name,
                "hol": holiday_region,
                "bc": business_calendar_fingerprint,
            },
        },
        "dnf": dnf,
        "natural": natural,
        **hints,
    }
    cache_save(key, payload)
    return payload
