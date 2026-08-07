"""Immutable recurrence evaluation boundary.

This first slice owns only normalized recurrence identity and local-time
conversion. Occurrence lookup and limit projection are added in later slices.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping

from .recurrence_context import RecurrenceContext
from .recurrence_spec import RecurrenceSpec
from .timeutil import build_local_datetime as _build_local_datetime
from .timeutil import local_naive_to_utc as _local_naive_to_utc
from .timeutil import to_local as _to_local
from .timeutil import utc_to_local_naive as _utc_to_local_naive


@dataclass(frozen=True, slots=True)
class RecurrenceLimits:
    """Normalized chain limits owned by an evaluator."""

    chain_max: int | None = None
    chain_until: datetime | None = None


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
