"""Immutable recurrence evaluation boundary.

The evaluator owns normalized recurrence identity, parsing, occurrence lookup,
limits, and local-time conversion while callers migrate incrementally.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from .recurrence_context import RecurrenceContext
from .recurrence_spec import RecurrenceSpec
from .occurrence_provider import Occurrence
from .timeutil import build_local_datetime as _build_local_datetime
from .timeutil import local_naive_to_utc as _local_naive_to_utc
from .timeutil import to_local as _to_local
from .timeutil import utc_to_local_naive as _utc_to_local_naive


@dataclass(frozen=True, slots=True)
class RecurrenceLimits:
    """Normalized chain limits owned by an evaluator."""

    chain_max: int | None = None
    chain_until: datetime | None = None

    def allows(self, link_no: int, candidate_utc: datetime) -> bool:
        """Return whether a projected link remains inside chain limits."""
        if isinstance(link_no, bool) or not isinstance(link_no, int) or link_no <= 0:
            raise ValueError("Recurrence link number must be a positive integer.")
        if not isinstance(candidate_utc, datetime):
            raise TypeError("Recurrence limit checks require a datetime candidate.")
        if self.chain_max is not None and link_no > self.chain_max:
            return False
        if self.chain_until is not None:
            from .timeutil import compare_datetimes

            if compare_datetimes(candidate_utc, self.chain_until) > 0:
                return False
        return True


@dataclass(frozen=True, slots=True)
class RecurrenceEvaluator:
    """Context-bound recurrence facade used by future lookup consumers."""

    spec: RecurrenceSpec
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, Any],
        *,
        context: RecurrenceContext | None = None,
        fallback_chain_id: str | None = None,
        timezone: Any | None = None,
        business_calendar: Any | None = None,
        astronomy_config: Mapping[str, Any] | None = None,
        anchor_file_dir: str = "",
        namespace: str = "nautical",
    ) -> "RecurrenceEvaluator":
        """Build an evaluator from a task without performing any I/O."""
        recurrence_context = context or RecurrenceContext.from_task(
            task,
            fallback_chain_id=fallback_chain_id,
            timezone=timezone,
            business_calendar=business_calendar,
            astronomy_config=astronomy_config,
            anchor_file_dir=anchor_file_dir,
            namespace=namespace,
        )
        return cls(RecurrenceSpec.from_task(task, context=recurrence_context))

    @classmethod
    def from_spec(cls, spec: RecurrenceSpec) -> "RecurrenceEvaluator":
        """Build an evaluator around an already-normalized specification."""
        if not isinstance(spec, RecurrenceSpec):
            raise TypeError("Recurrence evaluator requires a RecurrenceSpec.")
        return cls(spec)

    @property
    def context(self) -> RecurrenceContext:
        return self.spec.context

    @property
    def chain_id(self) -> str:
        return self.context.chain_id

    @property
    def seed_base(self) -> str:
        return self.context.seed_base

    @property
    def kind(self) -> str | None:
        return self.spec.kind

    @property
    def enabled(self) -> bool:
        return self.spec.enabled

    @property
    def anchor_mode(self) -> str:
        """Return the validated anchor mode without re-parsing it."""
        from .add_validation import validate_anchor_mode

        mode, reason = validate_anchor_mode(self.spec.anchor_mode)
        if reason:
            raise ValueError(reason)
        return mode

    @property
    def anchor_dnf(self) -> list:
        """Lazily parse the anchor expression through Nautical's cached parser."""
        if not self.spec.anchor:
            return []
        return copy.deepcopy(self._get_cached("anchor_dnf", self._parse_anchor))

    @property
    def omit_dnf(self) -> list:
        """Lazily parse the omit expression with omit-specific validation."""
        if not self.spec.omit:
            return []
        return copy.deepcopy(self._get_cached("omit_dnf", self._parse_omit))

    @property
    def cp_tokens(self) -> list | None:
        """Lazily parse CP into fixed/random tokens without resolving randomness."""
        if not self.spec.cp:
            return None
        return copy.deepcopy(self._get_cached("cp_tokens", self._parse_cp_tokens))

    @property
    def limits(self) -> RecurrenceLimits:
        """Return parsed chain limits, keeping native task ``until`` separate."""
        return self._get_cached("limits", self._parse_limits)

    def _get_cached(self, key: str, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def _parse_anchor(self) -> list:
        from . import parse_anchor_expr_to_dnf_cached

        return parse_anchor_expr_to_dnf_cached(self.spec.anchor)

    def _parse_omit(self) -> list:
        from . import parse_anchor_expr_to_dnf_cached, resolve_omit_presets
        from .anchor_omit import validate_omit_expr_strict

        return validate_omit_expr_strict(
            self.spec.omit,
            validate_anchor_expr_cached=parse_anchor_expr_to_dnf_cached,
            resolve_omit_presets=resolve_omit_presets,
        )

    def _parse_cp_tokens(self) -> list | None:
        from . import parse_cp_sequence_tokens

        return parse_cp_sequence_tokens(self.spec.cp)

    def _parse_limits(self) -> RecurrenceLimits:
        if self.spec.chain_until:
            from . import parse_dt_any

            chain_until = parse_dt_any(self.spec.chain_until)
            if chain_until is None:
                raise ValueError(
                    f"Unrecognized chainUntil datetime format '{self.spec.chain_until}'."
                )
        else:
            chain_until = None
        if self.spec.chain_max is not None and self.spec.chain_max <= 0:
            raise ValueError("chainMax must be greater than zero.")
        return RecurrenceLimits(chain_max=self.spec.chain_max, chain_until=chain_until)

    def cp_interval_for_link(self, link_no: int) -> timedelta | None:
        """Resolve one CP interval using this evaluator's stable chain identity."""
        if self.kind != "cp":
            raise ValueError("CP interval lookup requires a CP recurrence.")
        if isinstance(link_no, bool) or not isinstance(link_no, int) or link_no <= 0:
            raise ValueError("CP interval lookup requires a positive link number.")
        from . import cp_sequence_interval_for_link

        return cp_sequence_interval_for_link(self.spec.cp, link_no, chain_id=self.chain_id)

    def project_cp(self, base_utc: datetime, link_no: int) -> datetime:
        """Project a CP link from an existing UTC due instant."""
        if not isinstance(base_utc, datetime):
            raise TypeError("CP projection requires a datetime base.")
        if base_utc.tzinfo is None or base_utc.utcoffset() is None:
            raise ValueError("CP projection requires a timezone-aware UTC datetime.")
        interval = self.cp_interval_for_link(link_no)
        if interval is None:
            raise ValueError(f"Unable to resolve CP interval for link {link_no}.")
        base_utc = base_utc.astimezone(timezone.utc)
        if interval.total_seconds() % 86400 == 0 and self.timezone is not None:
            local = self.to_local(base_utc)
            shifted_day = local.date() + timedelta(days=int(interval.total_seconds() // 86400))
            return self.build_local_datetime(
                shifted_day,
                (local.hour, local.minute),
            )
        return base_utc + interval

    def limits_allow(self, candidate: datetime, link_no: int) -> bool:
        """Check chain limits for a local or UTC candidate occurrence."""
        if not isinstance(candidate, datetime):
            raise TypeError("Recurrence limit checks require a datetime candidate.")
        candidate_utc = (
            self.local_naive_to_utc(candidate)
            if candidate.tzinfo is None
            else candidate.astimezone(timezone.utc)
        )
        return self.limits.allows(link_no, candidate_utc)

    def next_after(
        self,
        after_local: datetime,
        *,
        next_occurrence_after_local_dt: Callable[..., Any],
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        inclusive: bool = False,
        pick_occurrence_local: Callable[..., Any] | None = None,
        anchor_file_provider: Any | None = None,
        max_file_skips: int = 512,
    ) -> Occurrence | None:
        """Return the next included expression/file occurrence.

        The scheduler callback remains injectable during migration; all source
        merging, omit handling, identity, and timezone conversion live here.
        """
        if self.kind != "anchor":
            raise ValueError("Occurrence lookup requires an anchor recurrence.")
        if not isinstance(after_local, datetime):
            raise TypeError("Occurrence lookup requires a datetime cursor.")
        self._validate_hhmm(fallback_hhmm)
        if isinstance(max_file_skips, bool) or not isinstance(max_file_skips, int) or max_file_skips <= 0:
            raise ValueError("Anchor-file omission scan limit must be a positive integer.")
        from . import anchor_inclusion

        return anchor_inclusion.next_included_occurrence(
            dnf=self.anchor_dnf,
            anchor_file_str=self.spec.anchor_file,
            after_local_dt=after_local,
            inclusive=inclusive,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date or after_local.date(),
            seed_base=self.seed_base,
            omit_dnf=self.omit_dnf,
            core=self._core_module(),
            next_occurrence_after_local_dt=next_occurrence_after_local_dt,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_dir=self.context.anchor_file_dir,
            anchor_file_provider=anchor_file_provider,
            recurrence_context=self.context,
            business_calendar=self.context.business_calendar,
            max_file_skips=max_file_skips,
        )

    def collect_after(
        self,
        after_local: datetime,
        *,
        limit: int,
        next_occurrence_after_local_dt: Callable[..., Any],
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        inclusive: bool = False,
        pick_occurrence_local: Callable[..., Any] | None = None,
        anchor_file_provider: Any | None = None,
        max_iterations: int = 512,
        max_file_skips: int = 512,
    ) -> list[Occurrence]:
        """Collect a bounded, merged stream of included occurrences."""
        self._validate_hhmm(fallback_hhmm)
        from .occurrence_provider import AnchorOccurrenceProvider, collect_after

        provider = AnchorOccurrenceProvider(
            lambda cursor: self.next_after(
                cursor,
                next_occurrence_after_local_dt=next_occurrence_after_local_dt,
                fallback_hhmm=fallback_hhmm,
                default_seed_date=default_seed_date,
                pick_occurrence_local=pick_occurrence_local,
                anchor_file_provider=anchor_file_provider,
                max_file_skips=max_file_skips,
            )
        )
        return collect_after(
            provider,
            after_local,
            limit=limit,
            inclusive=inclusive,
            max_iterations=max_iterations,
            build_local_datetime=self.build_local_datetime,
            to_local=self.to_local,
        )

    @staticmethod
    def _core_module():
        from . import _PKG_PROXY

        return _PKG_PROXY

    @staticmethod
    def _validate_hhmm(value: tuple[int, int]) -> None:
        if (
            not isinstance(value, tuple)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            or not 0 <= value[0] <= 23
            or not 0 <= value[1] <= 59
        ):
            raise ValueError("Fallback occurrence time must be an (hour, minute) tuple.")

    @property
    def timezone(self) -> Any | None:
        return self.context.timezone

    def to_local(self, dt_utc: datetime) -> datetime:
        return _to_local(dt_utc, self.timezone)

    def utc_to_local_naive(self, dt_utc: datetime) -> datetime:
        return _utc_to_local_naive(dt_utc, self.timezone)

    def local_naive_to_utc(self, dt_local_naive: datetime) -> datetime:
        return _local_naive_to_utc(dt_local_naive, self.timezone)

    def build_local_datetime(self, day: date, hhmm: tuple[int, int]) -> datetime:
        return _build_local_datetime(day, hhmm, self.timezone)


__all__ = ("RecurrenceEvaluator", "RecurrenceLimits")
