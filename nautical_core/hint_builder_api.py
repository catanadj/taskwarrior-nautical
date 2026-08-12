"""Core-bound hint-builder orchestration."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)

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

        def hint_builder_factory():
            service = scheduler_service_factory(anchor_expr)
            return core["_import_sibling"]("hint_builder").HintBuilder(service)

        return core["_precompute"].build_and_cache_hints(
            anchor_expr,
            anchor_mode=anchor_mode,
            default_due_dt=default_due_dt,
            cache_key_for_task=core["cache_key_for_task"],
            cache_load=core["cache_load"],
            validate_anchor_expr_strict=core["validate_anchor_expr_strict"],
            describe_anchor_expr_from_dnf=core["_describe_anchor_expr_from_dnf"],
            cache_save=core["cache_save"],
            anchor_year_fmt=core["ANCHOR_YEAR_FMT"],
            wrand_salt=core["WRAND_SALT"],
            local_tz_name=core["LOCAL_TZ_NAME"],
            holiday_region=core["HOLIDAY_REGION"],
            business_calendar_fingerprint=calendar_fingerprint,
            include_per_year=include_per_year,
            hint_builder_factory=hint_builder_factory,
        )

    return SimpleNamespace(build_and_cache_hints=build_and_cache_hints)


__all__ = ("for_core",)
