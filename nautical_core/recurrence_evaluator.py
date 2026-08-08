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
class RecurrenceModeResult:
    """Typed result of selecting the next occurrence for an anchor mode.

    The hook still consumes the historical metadata dictionary for now.  This
    value keeps the selection and missed-occurrence evidence typed while the
    mode engine is migrated into the evaluator.
    """

    selected_occurrence: datetime | None
    mode: str
    basis: str | None
    source: str
    missed_count: int = 0
    missed_preview: tuple[datetime, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or not self.mode:
            raise ValueError("Recurrence mode result requires a mode.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Recurrence mode result requires a source.")
        if isinstance(self.missed_count, bool) or not isinstance(self.missed_count, int):
            raise TypeError("Recurrence missed count must be an integer.")
        if self.missed_count < 0:
            raise ValueError("Recurrence missed count cannot be negative.")
        if not isinstance(self.missed_preview, tuple):
            raise TypeError("Recurrence missed preview must be a tuple.")
        if any(not isinstance(value, datetime) for value in self.missed_preview):
            raise TypeError("Recurrence missed preview values must be datetimes.")

    def metadata(self, *, target_field: str | None = None) -> dict[str, object]:
        """Return the legacy metadata shape used by existing feedback panels."""
        result: dict[str, object] = {
            "mode": self.mode,
            "basis": self.basis,
            "source": self.source,
            "missed_count": self.missed_count,
            "missed_preview": [value.isoformat() for value in self.missed_preview[:5]],
        }
        if target_field:
            result["target_field"] = target_field
        return result

    def get(self, key: str, default: object = None) -> object:
        """Read legacy metadata keys during the incremental migration."""
        return self.metadata().get(key, default)


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
        return self._get_cached("anchor_dnf", self._parse_anchor, clone=True)

    @property
    def omit_dnf(self) -> list:
        """Lazily parse the omit expression with omit-specific validation."""
        if not self.spec.omit and not self.spec.omit_file:
            return []
        return self._get_cached("omit_dnf", self._parse_omit, clone=True)

    @property
    def cp_tokens(self) -> list | None:
        """Lazily parse CP into fixed/random tokens without resolving randomness."""
        if not self.spec.cp:
            return None
        return self._get_cached("cp_tokens", self._parse_cp_tokens, clone=True)

    @property
    def limits(self) -> RecurrenceLimits:
        """Return parsed chain limits, keeping native task ``until`` separate."""
        return self._get_cached("limits", self._parse_limits)

    def _get_cached(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        clone: bool = False,
    ) -> Any:
        if key not in self._cache:
            value = loader()
            self._cache[key] = copy.deepcopy(value) if clone else value
        return self._cache[key]

    def _parse_anchor(self) -> list[Any]:
        from . import parse_anchor_expr_to_dnf_cached

        return parse_anchor_expr_to_dnf_cached(self.spec.anchor)

    def _parse_omit(self) -> list[Any]:
        from . import parse_anchor_expr_to_dnf_cached, resolve_omit_presets
        from .anchor_omit import validate_omit_expr_strict

        omit_dnf = None
        if self.spec.omit:
            omit_dnf = validate_omit_expr_strict(
                self.spec.omit,
                validate_anchor_expr_cached=parse_anchor_expr_to_dnf_cached,
                resolve_omit_presets=resolve_omit_presets,
            )
        omit_dates: frozenset[date] = frozenset()
        omit_descriptions: dict[date, str] = {}
        if self.spec.omit_file:
            core = self._core_module()
            omit_files = core._import_sibling("omit_files")
            omit_dates, omit_descriptions = omit_files.load_omit_file_data(
                self.spec.omit_file,
                getattr(core, "OMIT_FILE_DIR", ""),
                business_calendar=self.context.business_calendar,
            )
        if not omit_dnf and not omit_dates and not omit_descriptions:
            return []
        from .anchor_omit import combine_omit_state

        return combine_omit_state(
            omit_dnf=omit_dnf,
            omit_dates=omit_dates,
            omit_descriptions=omit_descriptions,
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

    def _anchor_file_provider_for(self, fallback_hhmm: tuple[int, int]) -> Any | None:
        """Build one context-bound anchor-file provider per evaluator session."""
        if not self.spec.anchor_file:
            return None
        key = f"anchor_file_provider:{fallback_hhmm[0]:02d}:{fallback_hhmm[1]:02d}"
        return self._get_cached(key, lambda: self._build_anchor_file_provider(fallback_hhmm))

    def _build_anchor_file_provider(self, fallback_hhmm: tuple[int, int]) -> Any:
        from . import anchor_inclusion

        return anchor_inclusion._build_anchor_file_provider(
            self.spec.anchor_file,
            anchor_file_dir=self.context.anchor_file_dir,
            fallback_hhmm=fallback_hhmm,
            seed_base=self.seed_base,
            core=self._core_module(),
            recurrence_context=self.context,
            business_calendar=self.context.business_calendar,
        )

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
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        inclusive: bool = False,
        pick_occurrence_local: Callable[..., Any] | None = None,
        anchor_file_provider: Any | None = None,
        max_file_skips: int = 512,
    ) -> Occurrence | None:
        """Return the next included expression/file occurrence.

        The evaluator-bound scheduler owns date, time, astronomy, and timezone
        resolution for every caller.
        """
        if self.kind != "anchor":
            raise ValueError("Occurrence lookup requires an anchor recurrence.")
        if not isinstance(after_local, datetime):
            raise TypeError("Occurrence lookup requires a datetime cursor.")
        self._validate_hhmm(fallback_hhmm)
        if isinstance(max_file_skips, bool) or not isinstance(max_file_skips, int) or max_file_skips <= 0:
            raise ValueError("Anchor-file omission scan limit must be a positive integer.")
        from . import anchor_inclusion
        next_occurrence_after_local_dt = self._default_next_occurrence_after_local_dt
        anchor_file_provider = anchor_file_provider or self._anchor_file_provider_for(fallback_hhmm)

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

    def next_event_after(
        self,
        after_local: datetime,
        *,
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        inclusive: bool = False,
        pick_occurrence_local: Callable[..., Any] | None = None,
        anchor_file_provider: Any | None = None,
        include_omitted: bool = False,
        max_file_skips: int = 512,
    ) -> Occurrence | None:
        """Return the next occurrence, optionally retaining omitted events.

        The default is equivalent to :meth:`next_after`.  When
        ``include_omitted`` is true, the returned :class:`Occurrence` keeps its
        ``omitted`` marker so callers can inspect the complete event stream.
        """
        if self.kind != "anchor":
            raise ValueError("Occurrence lookup requires an anchor recurrence.")
        if not isinstance(after_local, datetime):
            raise TypeError("Occurrence lookup requires a datetime cursor.")
        self._validate_hhmm(fallback_hhmm)
        if isinstance(max_file_skips, bool) or not isinstance(max_file_skips, int) or max_file_skips <= 0:
            raise ValueError("Anchor-file omission scan limit must be a positive integer.")
        from . import anchor_inclusion
        from .occurrence_provider import _require_forward_progress
        next_occurrence_after_local_dt = self._default_next_occurrence_after_local_dt
        anchor_file_provider = anchor_file_provider or self._anchor_file_provider_for(fallback_hhmm)

        cursor = after_local
        first = inclusive
        for _ in range(max_file_skips):
            event = anchor_inclusion.next_occurrence_event_local(
                dnf=self.anchor_dnf,
                anchor_file_str=self.spec.anchor_file,
                after_local_dt=cursor,
                inclusive=first,
                fallback_hhmm=fallback_hhmm,
                default_seed_date=default_seed_date or after_local.date(),
                seed_base=self.seed_base,
                # The event stream must see omitted anchor dates so it can
                # retain or skip them explicitly; omission is applied by the
                # event merger rather than by the date scheduler.
                omit_dnf=self.omit_dnf,
                scheduler_omit_dnf=None,
                core=self._core_module(),
                next_occurrence_after_local_dt=next_occurrence_after_local_dt,
                pick_occurrence_local=pick_occurrence_local,
                anchor_file_dir=self.context.anchor_file_dir,
                anchor_file_provider=anchor_file_provider,
                recurrence_context=self.context,
                business_calendar=self.context.business_calendar,
            )
            if event is None or event.local_datetime is None:
                return None
            if include_omitted or not event.omitted:
                return event
            _require_forward_progress(cursor, event.local_datetime)
            cursor = event.local_datetime
            first = False
        raise ValueError(
            f"Occurrence omission scan exceeded {max_file_skips} events; "
            "narrow the anchor or omit rule."
        )

    def events_between(
        self,
        start_local: datetime,
        end_local: datetime,
        *,
        limit: int,
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        inclusive: bool = True,
        pick_occurrence_local: Callable[..., Any] | None = None,
        anchor_file_provider: Any | None = None,
        include_omitted: bool = False,
        max_iterations: int = 512,
        max_file_skips: int = 512,
    ) -> list[Occurrence]:
        """Return a bounded event stream in ``[start_local, end_local]``.

        ``limit`` counts included occurrences.  Omitted events are retained in
        the returned list only when ``include_omitted`` is true, so callers can
        inspect why a stream advanced without changing normal limits.
        """
        if not isinstance(start_local, datetime) or not isinstance(end_local, datetime):
            raise TypeError("Occurrence ranges require datetime boundaries.")
        from .timeutil import compare_datetimes

        if compare_datetimes(end_local, start_local) < 0:
            raise ValueError("Occurrence range end must not precede its start.")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("Occurrence range limit must be a non-negative integer.")
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("Occurrence range iteration limit must be a positive integer.")
        if limit == 0:
            return []

        events: list[Occurrence] = []
        cursor = start_local
        first = inclusive
        included_count = 0
        for _ in range(max_iterations):
            event = self.next_event_after(
                cursor,
                fallback_hhmm=fallback_hhmm,
                default_seed_date=default_seed_date,
                inclusive=first,
                pick_occurrence_local=pick_occurrence_local,
                anchor_file_provider=anchor_file_provider,
                include_omitted=include_omitted,
                max_file_skips=max_file_skips,
            )
            if event is None or event.local_datetime is None:
                break
            if compare_datetimes(event.local_datetime, end_local) > 0:
                break
            events.append(event)
            if not event.omitted:
                included_count += 1
            if included_count >= limit:
                break
            cursor = event.local_datetime
            first = False
        else:
            raise ValueError("Occurrence provider exceeded its range iteration limit.")
        return events

    def select_mode(
        self,
        mode: str,
        *,
        due_local: datetime,
        end_local: datetime,
        due_explicit: bool = True,
        fallback_hhmm: tuple[int, int] = (9, 0),
        default_seed_date: date | None = None,
        pick_occurrence_local: Callable[..., Any] | None = None,
        anchor_file_provider: Any | None = None,
        missed_limit: int = 25,
        max_iterations: int = 512,
        max_file_skips: int = 512,
    ) -> RecurrenceModeResult:
        """Select the next occurrence using Nautical's three anchor modes.

        This is intentionally a pure recurrence boundary: callers provide
        local completion/due cursors, while Taskwarrior mutation remains
        outside the evaluator.
        """
        if mode not in {"skip", "all", "flex"}:
            raise ValueError("Recurrence mode must be 'skip', 'all', or 'flex'.")
        if not isinstance(due_local, datetime) or not isinstance(end_local, datetime):
            raise TypeError("Mode selection requires datetime due and end values.")
        if isinstance(missed_limit, bool) or not isinstance(missed_limit, int) or missed_limit <= 0:
            raise ValueError("Missed-occurrence limit must be a positive integer.")
        source = "anchor+anchor_file" if self.spec.anchor and self.spec.anchor_file else (
            "anchor_file" if self.spec.anchor_file else "anchor"
        )

        missed: list[Occurrence] = []
        from .timeutil import compare_datetimes

        if mode in {"all", "flex"} and due_explicit and compare_datetimes(end_local, due_local) > 0:
            missed = [
                event
                for event in self.events_between(
                    due_local,
                    end_local,
                    limit=missed_limit,
                    fallback_hhmm=fallback_hhmm,
                    default_seed_date=default_seed_date,
                    inclusive=False,
                    pick_occurrence_local=pick_occurrence_local,
                    anchor_file_provider=anchor_file_provider,
                    include_omitted=True,
                    max_iterations=max_iterations,
                    max_file_skips=max_file_skips,
                )
                if not event.omitted
            ]

        if mode == "all" and missed:
            selected = missed[0].local_datetime
            return RecurrenceModeResult(
                selected_occurrence=selected,
                mode=mode,
                basis="missed",
                source=source,
                missed_count=len(missed),
                missed_preview=tuple(
                    event.local_datetime for event in missed[:5] if event.local_datetime is not None
                ),
            )

        if mode == "flex":
            cursor = end_local
            basis = "flex"
        elif mode == "skip":
            cursor = max(due_local, end_local, key=lambda value: value.astimezone(timezone.utc) if value.tzinfo else value)
            basis = "after_end"
        else:
            cursor = due_local
            basis = "after_due"
        selected_event = self.next_event_after(
            cursor,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            inclusive=False,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_provider=anchor_file_provider,
            max_file_skips=max_file_skips,
        )
        selected = selected_event.local_datetime if selected_event is not None else None
        return RecurrenceModeResult(
            selected_occurrence=selected,
            mode=mode,
            basis=basis,
            source=source,
            missed_count=len(missed),
            missed_preview=tuple(
                event.local_datetime for event in missed[:5] if event.local_datetime is not None
            ),
        )

    def collect_after(
        self,
        after_local: datetime,
        *,
        limit: int,
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
    def _core_module() -> Any:
        from . import _PKG_PROXY

        return _PKG_PROXY

    def _default_next_occurrence_after_local_dt(
        self,
        dnf,
        after_local_dt: datetime,
        *,
        default_seed_date: date | None,
        seed_base: str,
        omit_dnf=None,
        fallback_hhmm: tuple[int, int] | None = None,
    ):
        """Resolve the shared date/time scheduler for evaluator consumers."""
        scheduler = self._get_cached("scheduler_binding", self._build_scheduler_binding)
        return scheduler(
            dnf,
            after_local_dt,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            fallback_hhmm=fallback_hhmm,
        )

    def _build_scheduler_binding(self) -> Callable[..., Any]:
        """Build the evaluator-bound scheduler once per evaluator session."""
        from .add_anchor_compute import anchor_next_occurrence_after_local_dt
        from .anchor_inclusion import _norm_t_mod
        from .time_slots import resolve_time_slots

        core = self._core_module()

        class SchedulerCoreProxy:
            """Expose core helpers while binding all wall-clock conversion to this evaluator."""

            def __getattr__(self, name: str) -> Any:
                return getattr(core, name)

            def to_local(self, value: Any) -> Any:
                return self_evaluator.to_local(value)

            def build_local_datetime(self, day: Any, hhmm: tuple[int, int]) -> Any:
                return self_evaluator.build_local_datetime(day, hhmm)

            def factor_matches_on(
                self,
                factor: Any,
                day: Any,
                default_seed: Any,
                *,
                seed_base: Any = None,
                business_calendar: Any = None,
            ) -> Any:
                return core.factor_matches_on(
                    factor,
                    day,
                    default_seed,
                    seed_base=seed_base,
                    business_calendar=(
                        business_calendar
                        if business_calendar is not None
                        else self_evaluator.context.business_calendar
                    ),
                )

            def next_after_expr(
                self,
                expression: Any,
                ref_date: Any,
                *,
                default_seed: Any = None,
                seed_base: Any = None,
                date_is_excluded: Any = None,
                business_calendar: Any = None,
            ) -> Any:
                return core.next_after_expr(
                    expression,
                    ref_date,
                    default_seed=default_seed,
                    seed_base=seed_base,
                    date_is_excluded=date_is_excluded,
                    business_calendar=(
                        business_calendar
                        if business_calendar is not None
                        else self_evaluator.context.business_calendar
                    ),
                )

        self_evaluator = self
        scheduler_core = SchedulerCoreProxy()

        def resolve_slots(value: Any, target_date: Any) -> Any:
            """Resolve slots with the evaluator's astronomy and timezone context."""
            config = self.context.astronomy_config
            if config is None:
                config = getattr(core, "ASTRONOMY_CONFIG", {})
            return resolve_time_slots(
                value,
                target_date,
                config=config,
                to_local=self.to_local,
            )

        def scheduler(
            dnf: Any,
            after_local_dt: datetime,
            *,
            default_seed_date: date | None,
            seed_base: str,
            omit_dnf: Any = None,
            fallback_hhmm: tuple[int, int] | None = None,
        ) -> Any:
            return anchor_next_occurrence_after_local_dt(
                dnf,
                after_local_dt,
                fallback_hhmm=fallback_hhmm or (9, 0),
                interval_seed=default_seed_date,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                default_seed_date=default_seed_date,
                core=scheduler_core,
                norm_t_mod=_norm_t_mod,
                resolve_time_slots=resolve_slots,
            )

        return scheduler

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


__all__ = ("RecurrenceEvaluator", "RecurrenceLimits", "RecurrenceModeResult")
