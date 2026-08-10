"""Core-bound business-calendar configuration and task selection API."""

from __future__ import annotations

from functools import lru_cache
from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)

    def business_calendar_definitions():
        return core["_business_calendar_config"].parse_business_calendar_definitions(
            core["BUSINESS_CALENDAR_CONFIG"]
        )

    def validate_business_calendar_omit_expr(expr: str):
        anchor_omit = core["_import_sibling"]("anchor_omit")
        return anchor_omit.validate_omit_expr_strict(
            expr,
            validate_anchor_expr_cached=core["validate_anchor_expr_strict"],
            resolve_omit_presets=core["resolve_omit_presets"],
        )

    def business_calendar_expression_matches_date(dnf, value, name: str) -> bool:
        seed_base = f"business-calendar:{name}"
        return any(
            all(
                core["atom_matches_on"](
                    atom,
                    value,
                    value,
                    seed_base=seed_base,
                    business_calendar=core["_business_calendar"].DEFAULT_BUSINESS_CALENDAR,
                )
                for atom in term
            )
            for term in dnf
        )

    def resolve_business_calendar_config(
        raw_config,
        *,
        anchor_file_dir: str | None = None,
        omit_file_dir: str | None = None,
    ):
        anchor_files = core["_import_sibling"]("anchor_files")
        omit_files = core["_import_sibling"]("omit_files")
        return core["_business_calendar_config"].resolve_business_calendars(
            raw_config,
            anchor_file_dir=(
                core["ANCHOR_FILE_DIR"] if anchor_file_dir is None else anchor_file_dir
            ),
            omit_file_dir=(
                core["OMIT_FILE_DIR"] if omit_file_dir is None else omit_file_dir
            ),
            validate_anchor_expr=core["validate_anchor_expr_strict"],
            validate_omit_expr=validate_business_calendar_omit_expr,
            expression_matches_date=business_calendar_expression_matches_date,
            validate_anchor_file_expr=anchor_files.validate_business_calendar_anchor_file,
            validate_omit_file_expr=omit_files.validate_business_calendar_omit_file,
            unmatched_anchor_file_patterns=anchor_files.unmatched_anchor_file_patterns,
            unmatched_omit_file_patterns=omit_files.unmatched_omit_file_patterns,
            load_anchor_file_dates=anchor_files.load_anchor_file_dates,
            load_omit_file_dates=omit_files.load_omit_file_dates,
        )

    @lru_cache(maxsize=1)
    def configured_business_calendars():
        return resolve_business_calendar_config(core["BUSINESS_CALENDAR_CONFIG"])

    def get_configured_business_calendar(name: str):
        normalized = str(name or "").strip().lower()
        # Resolve through the facade so tests and integrations can replace the
        # registry without reaching into this adapter's closure.
        calendars = core["configured_business_calendars"]()
        try:
            return calendars[normalized]
        except KeyError:
            available = ", ".join(sorted(calendars)) or "none"
            raise core["_business_calendar_config"].BusinessCalendarConfigError(
                f"Unknown business calendar {name!r}; configured calendars: {available}."
            ) from None

    def business_calendar_for_task(task: dict | None):
        raw_name = str((task or {}).get("bc") or "").strip()
        if not raw_name:
            return core["_business_calendar"].DEFAULT_BUSINESS_CALENDAR
        return get_configured_business_calendar(core["_unwrap_quotes"](raw_name))

    def normalize_task_business_calendar(task: dict):
        business_calendar = business_calendar_for_task(task)
        if str(task.get("bc") or "").strip():
            task["bc"] = business_calendar.name
        return business_calendar

    def business_calendar_fingerprint(business_calendar=None) -> str:
        business_calendar = core["_business_calendar"].effective_business_calendar(
            business_calendar
        )
        return str(
            getattr(business_calendar, "fingerprint", "")
            or f"{business_calendar.name}-v1"
        )

    def use_business_calendar(business_calendar):
        return core["_business_calendar"].use_business_calendar(business_calendar)

    def use_task_business_calendar(task: dict):
        return use_business_calendar(normalize_task_business_calendar(task))

    return SimpleNamespace(
        business_calendar_definitions=business_calendar_definitions,
        _validate_business_calendar_omit_expr=validate_business_calendar_omit_expr,
        _business_calendar_expression_matches_date=business_calendar_expression_matches_date,
        resolve_business_calendar_config=resolve_business_calendar_config,
        configured_business_calendars=configured_business_calendars,
        get_configured_business_calendar=get_configured_business_calendar,
        business_calendar_for_task=business_calendar_for_task,
        normalize_task_business_calendar=normalize_task_business_calendar,
        business_calendar_fingerprint=business_calendar_fingerprint,
        use_business_calendar=use_business_calendar,
        use_task_business_calendar=use_task_business_calendar,
    )


__all__ = ("for_core",)
