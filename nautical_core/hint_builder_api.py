"""Core-bound hint-builder orchestration."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)

    def _precompute_hints(
        dnf,
        start_dt=None,
        anchor_mode="ALL",
        rand_seed=None,
        k_next=24,
        sample_days_for_year=366,
        business_calendar=None,
        include_per_year: bool = True,
        scheduler_service=None,
    ) -> dict:
        _ = anchor_mode
        business_calendar = core["_business_calendar"].effective_business_calendar(
            business_calendar
        )
        return core["_precompute"].precompute_hints(
            dnf,
            start_dt=start_dt,
            rand_seed=rand_seed,
            k_next=k_next,
            sample_days_for_year=sample_days_for_year,
            now_local=datetime.now,
            next_after_expr=core["_with_business_calendar"](
                core["next_after_expr"], business_calendar
            ),
            next_for_or=core["_with_business_calendar"](
                core["_next_for_or"], business_calendar
            ),
            include_per_year=include_per_year,
            scheduler_service=scheduler_service,
        )

    def build_and_cache_hints(
        anchor_expr: str,
        anchor_mode: str = "ALL",
        default_due_dt=None,
        business_calendar=None,
        include_per_year: bool = True,
    ):
        business_calendar = core["_business_calendar"].effective_business_calendar(
            business_calendar
        )
        calendar_fingerprint = core["business_calendar_fingerprint"](business_calendar)

        def scheduler_service_factory(anchor: str):
            scheduler_service = core["_import_sibling"]("scheduler_service")
            context_type = core["_import_sibling"]("recurrence_context").RecurrenceContext
            context = context_type(
                chain_id="preview",
                timezone=core.get("_LOCAL_TZ"),
                business_calendar=business_calendar,
                astronomy_config=core.get("ASTRONOMY_CONFIG"),
                anchor_file_dir=core.get("ANCHOR_FILE_DIR", ""),
            )
            return scheduler_service.SchedulerService.from_task(
                {"chainID": "preview", "anchor": anchor},
                context=context,
            )

        return core["_precompute"].build_and_cache_hints(
            anchor_expr,
            anchor_mode=anchor_mode,
            default_due_dt=default_due_dt,
            cache_key_for_task=core["cache_key_for_task"],
            cache_load=core["cache_load"],
            validate_anchor_expr_strict=core["validate_anchor_expr_strict"],
            describe_anchor_expr_from_dnf=core["_describe_anchor_expr_from_dnf"],
            precompute_hints=lambda dnf, **kwargs: _precompute_hints(
                dnf,
                business_calendar=business_calendar,
                **kwargs,
            ),
            cache_save=core["cache_save"],
            anchor_year_fmt=core["ANCHOR_YEAR_FMT"],
            wrand_salt=core["WRAND_SALT"],
            local_tz_name=core["LOCAL_TZ_NAME"],
            holiday_region=core["HOLIDAY_REGION"],
            business_calendar_fingerprint=calendar_fingerprint,
            include_per_year=include_per_year,
            scheduler_service_factory=scheduler_service_factory,
        )

    return SimpleNamespace(build_and_cache_hints=build_and_cache_hints)


__all__ = ("for_core",)
