from __future__ import annotations

import os
import re
from bisect import bisect_right
from datetime import date, datetime, timedelta
from datetime import timezone
from typing import Callable

from .business_calendar import (
    DEFAULT_BUSINESS_CALENDAR,
    BusinessCalendar,
    effective_business_calendar,
)
from .business_calendar_config import validate_calendar_rule_modifiers
from . import file_resource_limits as resource_limits
from .file_backed_dates import load_file_date_data
from .file_source_expr import (
    FileSourceResolution,
    ResolvedFileSource,
    parse_file_source_expression,
    resolve_file_source_expression,
    resolve_file_sources,
)
from .schedule_utils import apply_day_offset, roll_apply
from .recurrence_context import RecurrenceContext
from .occurrence_provider import (
    Occurrence,
    OccurrenceBatch,
    ProviderCapabilities,
    ProviderContract,
    _cursor_before,
)
from .timeutil import compare_datetimes
from .time_windows import parse_clock_value, parse_random_time_window_spec, parse_time_schedule_spec, parse_time_window_spec


class AnchorFileOccurrenceExhausted(LookupError):
    """Raised when a cursor has no further anchor-file occurrence."""

    def __init__(self, anchor_file: str, skipped: int) -> None:
        self.anchor_file = str(anchor_file)
        self.skipped = int(skipped)
        super().__init__(
            f"anchor_file '{self.anchor_file}' exhausted after skipping "
            f"{self.skipped} omitted occurrences."
        )


_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_NEXT_PREV_WD_RE = re.compile(r"^(next|prev)-(mon|tue|wed|thu|fri|sat|sun)$")
_DAY_OFFSET_RE = re.compile(r"^([+-]\d+)d$")
_HHMM_RE = re.compile(r"^(\d{2}):(\d{2})$")
_HOUR_PAD_RE = re.compile(r"^(\d):(\d{2})(?::\d{2})?$")


def _default_mods() -> dict:
    return {
        "t": None,
        "roll": None,
        "wd": None,
        "bd": False,
        "day_offset": 0,
        "business_day_offset": 0,
    }


def validate_anchor_file_name(value: str | None) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if name in {".", ".."} or "/" in name or "\\" in name or os.path.isabs(name):
        raise ValueError("anchor_file must be a file name, not a path.")
    return name


def _parse_hhmm(text: str) -> tuple[int, int] | None:
    raw = str(text or "").strip()
    match = _HHMM_RE.match(raw)
    if not match:
        pad = _HOUR_PAD_RE.match(raw)
        if pad:
            raise ValueError(f"Time '{raw}' needs a leading zero. Use '0{pad.group(1)}:{pad.group(2)}'.")
        return parse_clock_value(raw)
    hh = int(match.group(1))
    mm = int(match.group(2))
    if hh > 23 or mm > 59:
        return None
    return (hh, mm)


def parse_anchor_file_spec(value: str | None) -> tuple[str, dict]:
    raw = str(value or "").strip()
    if not raw:
        return "", _default_mods()
    name, mods_str = (raw.split("@", 1) + [""])[:2]
    file_name = validate_anchor_file_name(name.strip())
    mods = _default_mods()
    if not mods_str:
        return file_name, mods

    for raw_tok in mods_str.split("@"):
        tok = raw_tok.strip().lower()
        if not tok:
            continue
        if tok.startswith("t="):
            if mods["t"] is not None:
                raise ValueError("Duplicate '@t=' modifier. Use a single '@t=HH:MM,HH:MM,...' list.")
            values = [part.strip() for part in tok.split("=", 1)[1].split(",") if part.strip()]
            random_window = parse_random_time_window_spec(tok.split("=", 1)[1].strip())
            if random_window is not None:
                mods["t"] = []
                mods["time_random"] = random_window.canonical
                continue
            try:
                window = parse_time_window_spec(tok.split("=", 1)[1].strip())
            except ValueError as exc:
                raise ValueError(f"anchor_file @t: {exc}") from None
            if window is not None:
                mods["t"] = list(window.slots)
                mods["time_window"] = window.canonical
                if window.crosses_midnight:
                    mods["time_window_offsets"] = list(window.slots_with_offsets)
                continue
            try:
                schedule = parse_time_schedule_spec(tok.split("=", 1)[1].strip())
            except ValueError as exc:
                raise ValueError(f"anchor_file @t: {exc}") from None
            if schedule is not None:
                mods["t"] = list(schedule.slots)
                mods["time_schedule"] = schedule.canonical
                continue
            times: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for item in values:
                hhmm = _parse_hhmm(item)
                if not hhmm:
                    raise ValueError(f"Invalid time in anchor_file @t=HH[:MM][,HH[:MM]...]: '{item}'")
                if hhmm not in seen:
                    times.append(hhmm)
                    seen.add(hhmm)
            if not times:
                raise ValueError(f"Invalid time in anchor_file @t=HH[:MM][,HH[:MM]...]: '{tok}'")
            mods["t"] = times[0] if len(times) == 1 else times
            continue
        if tok in ("nw", "pbd", "nbd"):
            mods["roll"] = tok
            continue
        if tok == "bd":
            mods["bd"] = True
            continue
        match = _NEXT_PREV_WD_RE.match(tok)
        if match:
            mods["roll"] = f"{match.group(1)}-wd"
            mods["wd"] = _WEEKDAYS[match.group(2)]
            continue
        match = _DAY_OFFSET_RE.match(tok)
        if match:
            mods["day_offset"] += int(match.group(1))
            continue
        match = re.fullmatch(r"([+-]\d+)bd", tok)
        if match:
            mods["business_day_offset"] += int(match.group(1))
            continue
        raise ValueError(f"Unknown anchor_file modifier '@{tok}'")
    return file_name, mods


def resolve_anchor_file_path(name: str | None, anchor_file_dir: str | None) -> str:
    file_name, _mods = parse_anchor_file_spec(name)
    if not file_name:
        return ""
    resolution = resolve_file_source_expression(file_name, anchor_file_dir, label="anchor_file")
    if len(resolution.sources) != 1:
        raise ValueError("resolve_anchor_file_path requires exactly one matching anchor_file.")
    return resolution.sources[0].path


def _resolved_anchor_sources(name: str | None, anchor_file_dir: str | None) -> FileSourceResolution:
    parsed = parse_file_source_expression(name, label="anchor_file")
    for source in parsed:
        _parse_source_mod_layers(source.pattern, source.modifier_layers)
    return resolve_file_sources(parsed, anchor_file_dir, label="anchor_file")


def unmatched_anchor_file_patterns(name: str | None, anchor_file_dir: str | None) -> tuple[str, ...]:
    return _resolved_anchor_sources(name, anchor_file_dir).unmatched_patterns


def validate_business_calendar_anchor_file(value: str) -> None:
    parsed = parse_file_source_expression(value, label="anchor_file")
    for source in parsed:
        layers, _source_time = _parse_source_mod_layers(
            source.pattern,
            source.modifier_layers,
        )
        for mods in layers:
            validate_calendar_rule_modifiers(mods, label="business calendar anchor_file")


def _parse_source_mod_layers(
    display_name: str,
    modifier_layers: tuple[str, ...],
) -> tuple[list[dict], object | None]:
    layers: list[dict] = []
    source_time: object | None = None
    for modifier_text in modifier_layers:
        _file_name, mods = parse_anchor_file_spec(f"source{modifier_text}")
        tval = mods.get("t")
        if mods.get("time_random"):
            if source_time is not None:
                raise ValueError(
                    f"anchor_file '{display_name}' has more than one @t modifier across its expression groups."
                )
            source_time = {"t": [], "time_random": str(mods["time_random"])}
        elif tval is not None:
            if source_time is not None:
                raise ValueError(
                    f"anchor_file '{display_name}' has more than one @t modifier across its expression groups."
                )
            window_spec = mods.get("time_window")
            window = parse_time_window_spec(str(window_spec)) if window_spec else None
            source_time = list(window.slots_with_offsets) if window is not None and window.crosses_midnight else tval
        layers.append(mods)
    if not layers:
        layers.append(_default_mods())
    return layers, source_time


def _load_anchor_source_data(
    source: ResolvedFileSource,
    business_calendar: BusinessCalendar,
) -> tuple[frozenset[date], dict[date, str], object | None]:
    dates, descriptions = load_file_date_data(
        source.path,
        label=f"anchor_file '{source.display_name}'",
    )
    layers, source_time = _parse_source_mod_layers(source.display_name, source.modifier_layers)
    for mods in layers:
        dates, descriptions = _apply_anchor_file_mods(
            dates,
            descriptions,
            mods,
            business_calendar=business_calendar,
        )
    return dates, descriptions, source_time


def _load_anchor_file_data(
    name: str | None,
    anchor_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> tuple[frozenset[date], dict[date, str]]:
    business_calendar = effective_business_calendar(business_calendar)
    resolution = _resolved_anchor_sources(name, anchor_file_dir)
    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for source in resolution.sources:
        dates, descriptions, _source_time = _load_anchor_source_data(source, business_calendar)
        out_dates.update(dates)
        if len(out_dates) > resource_limits.MAX_RESOLVED_DATES:
            raise ValueError(
                f"anchor_file resolves to more than {resource_limits.MAX_RESOLVED_DATES} unique dates."
            )
        for item_date, text in descriptions.items():
            if text:
                out_descriptions.setdefault(item_date, text)
    return frozenset(out_dates), out_descriptions


def _apply_anchor_file_mods(
    dates: frozenset[date],
    descriptions: dict[date, str],
    mods: dict,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> tuple[frozenset[date], dict[date, str]]:
    if not dates:
        return frozenset(), {}
    if not any(
        (
            mods.get("bd"),
            mods.get("roll"),
            int(mods.get("day_offset", 0) or 0),
            int(mods.get("business_day_offset", 0) or 0),
        )
    ):
        return dates, dict(descriptions)
    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for item_date in sorted(dates):
        transformed = _transform_anchor_file_date(
            item_date,
            mods,
            business_calendar=business_calendar,
        )
        if transformed is None:
            continue
        out_dates.add(transformed)
        text = str(descriptions.get(item_date) or "").strip()
        if text:
            out_descriptions.setdefault(transformed, text)
    return frozenset(out_dates), out_descriptions


def _transform_anchor_file_date(
    d: date,
    mods: dict,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> date | None:
    rolled = roll_apply(
        d,
        mods,
        parse_error_cls=ValueError,
        business_calendar=business_calendar,
    )
    if mods.get("bd") and not business_calendar.is_business_day(rolled):
        return None
    return apply_day_offset(rolled, mods, business_calendar=business_calendar)


def _norm_t_list(tval) -> list[tuple[int, int]]:
    if not tval:
        return []
    if isinstance(tval, tuple):
        return [tval]
    if isinstance(tval, list):
        return [item for item in tval if isinstance(item, tuple)]
    return []


def load_anchor_file_dates(
    name: str | None,
    anchor_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> frozenset[date]:
    dates, _descriptions = _load_anchor_file_data(
        name,
        anchor_file_dir,
        business_calendar=business_calendar,
    )
    return dates


def load_anchor_file_descriptions(
    name: str | None,
    anchor_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> dict[date, str]:
    _dates, descriptions = _load_anchor_file_data(
        name,
        anchor_file_dir,
        business_calendar=business_calendar,
    )
    return descriptions


def anchor_file_description_for_date(
    name: str | None,
    anchor_file_dir: str | None,
    target: date,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> str | None:
    text = str(
        load_anchor_file_descriptions(
            name,
            anchor_file_dir,
            business_calendar=business_calendar,
        ).get(target)
        or ""
    ).strip()
    return text or None


def _load_anchor_file_occurrence_records(
    name: str | None,
    anchor_file_dir: str | None,
    fallback_hhmm: tuple[int, int],
    *,
    business_calendar: BusinessCalendar | None = None,
    context: RecurrenceContext | None = None,
) -> list[tuple[date, tuple[int, int], str]]:
    business_calendar = effective_business_calendar(business_calendar)
    resolution = _resolved_anchor_sources(name, anchor_file_dir)
    out: list[tuple[date, tuple[int, int], str]] = []
    seen: dict[tuple[date, tuple[int, int]], int] = {}
    for source in resolution.sources:
        dates, _descriptions, source_time = _load_anchor_source_data(source, business_calendar)
        if isinstance(source_time, dict) and source_time.get("time_random"):
            if context is None:
                raise ValueError("anchor_file random time windows require recurrence context with chain ID.")
            times = None
        elif isinstance(source_time, (list, tuple)) and source_time and all(
            isinstance(item, tuple) and len(item) == 3 for item in source_time
        ):
            times = [item for item in source_time if isinstance(item, tuple) and len(item) == 3]
        else:
            times = _norm_t_list(source_time) or [fallback_hhmm]
        for item_date in sorted(dates):
            if isinstance(source_time, dict) and source_time.get("time_random"):
                from .time_slots import resolve_time_slots_with_offsets
                times = resolve_time_slots_with_offsets(source_time, item_date, context=context)
            for slot in times:
                if isinstance(slot, tuple) and len(slot) == 3:
                    day_offset, hour, minute = slot
                    occurrence = (item_date + timedelta(days=int(day_offset)), (int(hour), int(minute)))
                else:
                    occurrence = (item_date, slot)
                description = str(_descriptions.get(item_date) or "").strip()
                existing_index = seen.get(occurrence)
                if existing_index is not None:
                    if not out[existing_index][2] and description:
                        out[existing_index] = (occurrence[0], occurrence[1], description)
                    continue
                seen[occurrence] = len(out)
                if len(seen) > resource_limits.MAX_RESOLVED_DATES:
                    raise ValueError(
                        f"anchor_file resolves to more than "
                        f"{resource_limits.MAX_RESOLVED_DATES} occurrences."
                    )
                out.append((occurrence[0], occurrence[1], description))
    out.sort()
    return out


def load_anchor_file_occurrence_specs(
    name: str | None,
    anchor_file_dir: str | None,
    fallback_hhmm: tuple[int, int],
    *,
    business_calendar: BusinessCalendar | None = None,
    context: RecurrenceContext | None = None,
    _records_sink: list[tuple[date, tuple[int, int], str]] | None = None,
) -> list[tuple[date, tuple[int, int]]]:
    records = _load_anchor_file_occurrence_records(
        name,
        anchor_file_dir,
        fallback_hhmm,
        business_calendar=business_calendar,
        context=context,
    )
    if _records_sink is not None:
        _records_sink.extend(records)
    return [
        (item_date, hhmm)
        for item_date, hhmm, _description in records
    ]


class AnchorFileOccurrenceProvider:
    """Typed adapter over the legacy tuple-based anchor-file expansion API."""

    def __init__(
        self,
        name: str | None,
        anchor_file_dir: str | None,
        fallback_hhmm: tuple[int, int],
        *,
        business_calendar: BusinessCalendar | None = None,
        context: RecurrenceContext | None = None,
    ) -> None:
        self.name = name
        self.anchor_file_dir = anchor_file_dir
        self.fallback_hhmm = fallback_hhmm
        self.business_calendar = business_calendar
        self.context = context
        self._spec_cache: list[tuple[date, tuple[int, int]]] | None = None
        self._record_cache: list[tuple[date, tuple[int, int], str]] | None = None
        self._next_index = 0
        self._last_after: datetime | None = None
        self._last_candidate: datetime | None = None
        self._last_candidate_index: int | None = None
        self._conversion_key: tuple[object, ...] | None = None
        self._candidate_records: list[tuple[datetime, str]] | None = None
        self._candidate_keys: list[datetime] = []

    @property
    def contract(self) -> ProviderContract:
        return ProviderContract(
            source="anchor_file",
            cursor="inclusive",
            finite=True,
            capabilities=ProviderCapabilities(batch_generation=True, cursor_reuse=True),
        )

    def _records(self) -> list[tuple[date, tuple[int, int], str]]:
        if self._record_cache is None:
            if self._spec_cache is not None:
                self._record_cache = [(item_date, hhmm, "") for item_date, hhmm in self._spec_cache]
                return self._record_cache
            records: list[tuple[date, tuple[int, int], str]] = []
            specs = load_anchor_file_occurrence_specs(
                self.name,
                self.anchor_file_dir,
                self.fallback_hhmm,
                business_calendar=self.business_calendar,
                context=self.context,
                _records_sink=records,
            )
            self._record_cache = records or [(item_date, hhmm, "") for item_date, hhmm in specs]
        return self._record_cache

    def _specs(self) -> list[tuple[date, tuple[int, int]]]:
        if self._spec_cache is None:
            self._spec_cache = [(item_date, hhmm) for item_date, hhmm, _description in self._records()]
        return self._spec_cache

    def occurrences(self) -> list[Occurrence]:
        values: list[Occurrence] = []
        for d0, hhmm, description in self._records():
            values.append(
                Occurrence(
                    day=d0,
                    hour=hhmm[0],
                    minute=hhmm[1],
                    source="anchor_file",
                    description=description,
                )
            )
        return values

    def first_after(
        self,
        after_local: datetime,
        *,
        build_local_datetime: Callable[[date, tuple[int, int]], datetime],
        to_local: Callable[[datetime], datetime],
        inclusive: bool = False,
    ) -> Occurrence | None:
        """Find one file occurrence through the provider's cached cursor."""
        return self.next_after(
            after_local,
            build_local_datetime=build_local_datetime,
            to_local=to_local,
            inclusive=inclusive,
        )

    def collect_after(
        self,
        after_local: datetime,
        *,
        limit: int,
        inclusive: bool,
        build_local_datetime: Callable[[date, tuple[int, int]], datetime],
        to_local: Callable[[datetime], datetime],
    ) -> OccurrenceBatch[Occurrence]:
        """Return a bounded cached slice without repeated provider callbacks."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Anchor-file batch limit must be a non-negative integer.")
        if limit == 0:
            return OccurrenceBatch()
        first = self.next_after(
            after_local,
            build_local_datetime=build_local_datetime,
            to_local=to_local,
            inclusive=inclusive,
        )
        if first is None or self._last_candidate_index is None or self._candidate_records is None:
            return OccurrenceBatch()
        start = self._last_candidate_index
        values = [
            Occurrence(
                day=value.date(),
                hour=value.hour,
                minute=value.minute,
                source="anchor_file",
                description=description,
                local_datetime=value,
            )
            for value, description in self._candidate_records[start:start + limit]
        ]
        return OccurrenceBatch(values)

    def next_after(
        self,
        after_local: datetime,
        *,
        build_local_datetime: Callable[[date, tuple[int, int]], datetime],
        to_local: Callable[[datetime], datetime],
        inclusive: bool = False,
    ) -> Occurrence | None:
        # Inclusive lookups use an instant just before the requested value so
        # the shared strict-progress guard remains valid.
        cursor_after = _cursor_before(after_local) if inclusive else after_local
        records = self._records()
        conversion_key = (
            getattr(build_local_datetime, "__self__", None),
            getattr(build_local_datetime, "__func__", build_local_datetime),
            getattr(to_local, "__self__", None),
            getattr(to_local, "__func__", to_local),
        )
        if self._conversion_key != conversion_key:
            self._candidate_records = None
            self._candidate_keys = []
            self._next_index = 0
            self._last_after = None
            self._last_candidate = None
            self._last_candidate_index = None
        if self._candidate_records is None:
            candidates: list[datetime] = []
            descriptions: list[str] = []
            for d0, hhmm, description in records:
                raw_candidate = build_local_datetime(d0, hhmm)
                if not isinstance(raw_candidate, datetime):
                    raise TypeError("Anchor-file provider returned a non-datetime candidate.")
                candidate = to_local(raw_candidate)
                if not isinstance(candidate, datetime):
                    raise TypeError("Anchor-file provider returned a non-datetime local value.")
                candidates.append(candidate)
                descriptions.append(description)
            try:
                aware = [
                    value.tzinfo is not None and value.utcoffset() is not None
                    for value in candidates
                ]
                if any(flag != aware[0] for flag in aware[1:]):
                    raise ValueError("mixed aware and naive datetimes")
                order_keys = [
                    value.astimezone(timezone.utc) if aware and aware[0] else value
                    for value in candidates
                ]
                ordered_pairs = sorted(
                    zip(order_keys, candidates, descriptions),
                    key=lambda item: item[0],
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("Anchor-file provider returned incomparable local datetimes.") from exc
            self._candidate_records = []
            self._candidate_keys = []
            for order_key, candidate, description in ordered_pairs:
                if not self._candidate_records or compare_datetimes(candidate, self._candidate_records[-1][0]) != 0:
                    self._candidate_records.append((candidate, description))
                    self._candidate_keys.append(order_key)
                elif not self._candidate_records[-1][1] and description:
                    self._candidate_records[-1] = (self._candidate_records[-1][0], description)
        candidate_records = self._candidate_records
        if candidate_records is None:
            raise RuntimeError("Anchor-file candidate cache was not initialized.")
        try:
            cursor_aware = cursor_after.tzinfo is not None and cursor_after.utcoffset() is not None
            if self._candidate_keys and cursor_aware != (
                self._candidate_keys[0].tzinfo is not None
                and self._candidate_keys[0].utcoffset() is not None
            ):
                raise ValueError("mixed aware and naive datetimes")
            cursor_key = (
                cursor_after.astimezone(timezone.utc)
                if cursor_aware
                else cursor_after
            )
            selected_index = bisect_right(self._candidate_keys, cursor_key)
        except (TypeError, ValueError) as exc:
            raise ValueError("Anchor-file provider returned an incomparable datetime.") from exc
        if selected_index >= len(candidate_records):
            self._next_index = len(candidate_records)
            self._last_after = after_local
            self._last_candidate = None
            self._last_candidate_index = None
            self._conversion_key = conversion_key
            return None
        value, description = candidate_records[selected_index]
        local = value
        from .occurrence_provider import _require_forward_progress

        _require_forward_progress(cursor_after, local)
        self._next_index = selected_index + 1 if selected_index is not None else len(candidate_records)
        self._last_after = after_local
        self._last_candidate = local
        self._last_candidate_index = selected_index
        self._conversion_key = conversion_key
        return Occurrence(
            day=local.date(),
            hour=local.hour,
            minute=local.minute,
            source="anchor_file",
            description=description,
            local_datetime=local,
        )


def next_anchor_file_occurrence_after(
    name: str | None,
    anchor_file_dir: str | None,
    after_dt_local: datetime,
    fallback_hhmm: tuple[int, int],
    *,
    build_local_datetime: Callable[[date, tuple[int, int]], datetime],
    to_local: Callable[[datetime], datetime],
    business_calendar: BusinessCalendar | None = None,
    context: RecurrenceContext | None = None,
) -> datetime | None:
    provider = AnchorFileOccurrenceProvider(
        name,
        anchor_file_dir,
        fallback_hhmm,
        business_calendar=business_calendar,
        context=context,
    )
    occurrence = provider.next_after(
        after_dt_local,
        build_local_datetime=build_local_datetime,
        to_local=to_local,
    )
    return occurrence.local_datetime if occurrence is not None else None
