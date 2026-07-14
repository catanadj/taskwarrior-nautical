from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .business_calendar import ConfiguredBusinessCalendar


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_FIELDS = frozenset({"anchor", "anchor_file", "omit", "omit_file"})
_BUSINESS_ROLLS = frozenset({"pbd", "nbd", "nw"})


class BusinessCalendarConfigError(ValueError):
    pass


@dataclass(frozen=True)
class BusinessCalendarDefinition:
    name: str
    anchor: tuple[str, ...]
    anchor_file: tuple[str, ...]
    omit: tuple[str, ...]
    omit_file: tuple[str, ...]


def _string_values(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise BusinessCalendarConfigError(f"{label} must be a string or array of strings.")

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise BusinessCalendarConfigError(f"{label} must contain only strings.")
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return tuple(out)


def parse_business_calendar_definitions(
    raw_config: Any,
) -> Mapping[str, BusinessCalendarDefinition]:
    if raw_config is None:
        return MappingProxyType({})
    if not isinstance(raw_config, dict):
        raise BusinessCalendarConfigError("business_calendar must be a TOML table.")

    out: dict[str, BusinessCalendarDefinition] = {}
    for raw_name, raw_definition in raw_config.items():
        name = str(raw_name or "").strip().lower()
        if not _NAME_RE.fullmatch(name):
            raise BusinessCalendarConfigError(
                f"Invalid business_calendar name {raw_name!r}; use letters, digits, '_' or '-'."
            )
        if name in out:
            raise BusinessCalendarConfigError(
                f"Duplicate business_calendar name after normalization: {name!r}."
            )
        if not isinstance(raw_definition, dict):
            raise BusinessCalendarConfigError(f"business_calendar.{name} must be a TOML table.")

        normalized: dict[str, Any] = {}
        for raw_key, value in raw_definition.items():
            key = str(raw_key or "").strip().lower()
            if key not in _FIELDS:
                allowed = ", ".join(sorted(_FIELDS))
                raise BusinessCalendarConfigError(
                    f"Unknown business_calendar.{name} field {raw_key!r}; expected one of: {allowed}."
                )
            if key in normalized:
                raise BusinessCalendarConfigError(
                    f"Duplicate business_calendar.{name}.{key} field after normalization."
                )
            normalized[key] = value

        definition = BusinessCalendarDefinition(
            name=name,
            anchor=_string_values(
                normalized.get("anchor"),
                label=f"business_calendar.{name}.anchor",
            ),
            anchor_file=_string_values(
                normalized.get("anchor_file"),
                label=f"business_calendar.{name}.anchor_file",
            ),
            omit=_string_values(
                normalized.get("omit"),
                label=f"business_calendar.{name}.omit",
            ),
            omit_file=_string_values(
                normalized.get("omit_file"),
                label=f"business_calendar.{name}.omit_file",
            ),
        )
        if not definition.anchor and not definition.anchor_file:
            raise BusinessCalendarConfigError(
                f"business_calendar.{name} must define anchor or anchor_file."
            )
        out[name] = definition

    return MappingProxyType(out)


def validate_calendar_rule_modifiers(mods: Mapping[str, Any], *, label: str) -> None:
    if mods.get("t"):
        raise BusinessCalendarConfigError(f"{label} does not support time modifiers (@t).")
    if (
        mods.get("bd")
        or mods.get("wd") is True
        or int(mods.get("business_day_offset", 0) or 0)
        or mods.get("roll") in _BUSINESS_ROLLS
    ):
        raise BusinessCalendarConfigError(
            f"{label} cannot use business-day modifiers while defining a business calendar."
        )


def _validate_rule_dnf(dnf: Any, *, label: str) -> None:
    for term in dnf or ():
        for atom in term or ():
            if atom.get("kind") == "select":
                raise BusinessCalendarConfigError(
                    f"{label} does not support positional selection modifiers."
                )
            if int(atom.get("ival", atom.get("intv", 1)) or 1) != 1:
                raise BusinessCalendarConfigError(
                    f"{label} does not support interval recurrences (/N)."
                )
            typ = str(atom.get("typ") or atom.get("type") or "").strip().lower()
            spec = str(atom.get("spec") or atom.get("value") or "").strip().lower()
            if "rand" in spec:
                raise BusinessCalendarConfigError(f"{label} does not support random selectors.")
            if typ == "m" and any(
                token.strip() == "lbd" or re.fullmatch(r"-?\d+bd", token.strip())
                for token in spec.split(",")
            ):
                raise BusinessCalendarConfigError(
                    f"{label} cannot use business-day ordinals while defining a business calendar."
                )
            validate_calendar_rule_modifiers(atom.get("mods") or {}, label=label)


def _validated_rules(
    expressions: tuple[str, ...],
    *,
    field_label: str,
    validate_expr: Callable[[str], Any],
) -> tuple[Any, ...]:
    out: list[Any] = []
    for expression in expressions:
        try:
            dnf = validate_expr(expression)
            _validate_rule_dnf(dnf, label=field_label)
        except BusinessCalendarConfigError:
            raise
        except Exception as exc:
            raise BusinessCalendarConfigError(f"Invalid {field_label}: {exc}") from exc
        out.append(dnf)
    return tuple(out)


def _loaded_file_dates(
    expressions: tuple[str, ...],
    *,
    field_label: str,
    directory: str,
    validate_file_expr: Callable[[str], None],
    unmatched_patterns: Callable[[str, str], tuple[str, ...]],
    load_dates: Callable[[str, str], frozenset[date]],
) -> frozenset[date]:
    out: set[date] = set()
    for expression in expressions:
        try:
            validate_file_expr(expression)
            unmatched = unmatched_patterns(expression, directory)
            if unmatched:
                joined = ", ".join(repr(pattern) for pattern in unmatched)
                raise BusinessCalendarConfigError(
                    f"{field_label} pattern(s) matched no files: {joined}."
                )
            out.update(load_dates(expression, directory))
        except BusinessCalendarConfigError as exc:
            if field_label in str(exc):
                raise
            raise BusinessCalendarConfigError(f"Invalid {field_label}: {exc}") from exc
        except Exception as exc:
            raise BusinessCalendarConfigError(f"Invalid {field_label}: {exc}") from exc
    return frozenset(out)


def resolve_business_calendars(
    raw_config: Any,
    *,
    anchor_file_dir: str,
    omit_file_dir: str,
    validate_anchor_expr: Callable[[str], Any],
    validate_omit_expr: Callable[[str], Any],
    expression_matches_date: Callable[[Any, date, str], bool],
    validate_anchor_file_expr: Callable[[str], None],
    validate_omit_file_expr: Callable[[str], None],
    unmatched_anchor_file_patterns: Callable[[str, str], tuple[str, ...]],
    unmatched_omit_file_patterns: Callable[[str, str], tuple[str, ...]],
    load_anchor_file_dates: Callable[[str, str], frozenset[date]],
    load_omit_file_dates: Callable[[str, str], frozenset[date]],
) -> Mapping[str, ConfiguredBusinessCalendar]:
    definitions = parse_business_calendar_definitions(raw_config)
    calendars: dict[str, ConfiguredBusinessCalendar] = {}
    for name, definition in definitions.items():
        anchor_label = f"business_calendar.{name}.anchor"
        omit_label = f"business_calendar.{name}.omit"
        anchor_rules = _validated_rules(
            definition.anchor,
            field_label=anchor_label,
            validate_expr=validate_anchor_expr,
        )
        omit_rules = _validated_rules(
            definition.omit,
            field_label=omit_label,
            validate_expr=validate_omit_expr,
        )
        anchor_dates = _loaded_file_dates(
            definition.anchor_file,
            field_label=f"business_calendar.{name}.anchor_file",
            directory=anchor_file_dir,
            validate_file_expr=validate_anchor_file_expr,
            unmatched_patterns=unmatched_anchor_file_patterns,
            load_dates=load_anchor_file_dates,
        )
        omit_dates = _loaded_file_dates(
            definition.omit_file,
            field_label=f"business_calendar.{name}.omit_file",
            directory=omit_file_dir,
            validate_file_expr=validate_omit_file_expr,
            unmatched_patterns=unmatched_omit_file_patterns,
            load_dates=load_omit_file_dates,
        )

        def anchor_matches(value: date, rules=anchor_rules, calendar_name=name) -> bool:
            return any(expression_matches_date(rule, value, calendar_name) for rule in rules)

        def omit_matches(value: date, rules=omit_rules, calendar_name=name) -> bool:
            return any(expression_matches_date(rule, value, calendar_name) for rule in rules)

        fingerprint_payload = {
            "name": name,
            "anchor": anchor_rules,
            "anchor_file": sorted(item.isoformat() for item in anchor_dates),
            "omit": omit_rules,
            "omit_file": sorted(item.isoformat() for item in omit_dates),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        calendars[name] = ConfiguredBusinessCalendar(
            name=name,
            fingerprint=fingerprint,
            anchor_dates=anchor_dates,
            omit_dates=omit_dates,
            _anchor_matches=anchor_matches,
            _omit_matches=omit_matches,
        )
    return MappingProxyType(calendars)
