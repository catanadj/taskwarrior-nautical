"""Pure hint aggregation over typed scheduler collections."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from .occurrence_outcomes import OccurrenceCollectionResult
from .scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from .scheduler_models import OccurrenceSearchExhausted, occurrence_exhaustion_message


@dataclass(slots=True)
class HintBuilder:
    """Build serializable hint data from one task-scoped scheduler service."""

    service: Any

    def _collect_dates(
        self,
        start: date,
        end: date,
        *,
        limit: int,
    ) -> OccurrenceCollectionResult:
        if end <= start or limit <= 0:
            timezone = self.service.session.evaluator.context.timezone
            cursor = OccurrenceCursor.strict_after(
                datetime.combine(start, time.max, tzinfo=timezone),
                timezone=timezone,
            )
            return OccurrenceCollectionResult(occurrences=(), cursor=cursor)
        timezone = self.service.session.evaluator.context.timezone
        cursor = OccurrenceCursor.strict_after(
            datetime.combine(start, time.max, tzinfo=timezone),
            timezone=timezone,
        )
        request = OccurrenceRangeRequest(
            cursor,
            end_local=datetime.combine(end - timedelta(days=1), time.max, tzinfo=timezone),
            limit=limit,
            max_iterations=max(limit, 512),
        )
        result = self.service.collect_request(request)
        if not isinstance(result, OccurrenceCollectionResult):
            raise TypeError("Scheduler service returned an invalid hint collection.")
        if result.failure is not None:
            raise RuntimeError(
                f"Hint scheduling {result.status}: {result.failure.reason}"
            )
        return result

    @staticmethod
    def _unique_dates(result: OccurrenceCollectionResult, *, start: date, end: date) -> list[date]:
        seen: set[date] = set()
        dates: list[date] = []
        for occurrence in result:
            local = occurrence.local_datetime
            if local is None:
                continue
            day = local.date()
            if start < day < end and day not in seen:
                seen.add(day)
                dates.append(day)
        return dates

    def build(
        self,
        *,
        start_dt: datetime | date | None,
        k_next: int,
        sample_days_for_year: int,
        now_local,
        include_per_year: bool = True,
    ) -> dict[str, Any]:
        today = now_local().date()
        start = (start_dt.date() if isinstance(start_dt, datetime) else start_dt) or today
        preview_limit = max(1, int(k_next))
        preview_end = start + timedelta(days=365 * 5)
        preview = self._collect_dates(
            start,
            preview_end,
            limit=max(preview_limit * 16, 64),
        )
        next_dates = [
            value.isoformat() + "T00:00"
            for value in self._unique_dates(preview, start=start, end=preview_end)[:preview_limit]
        ]
        terminal: OccurrenceSearchExhausted | None = preview.terminal
        hints: dict[str, Any] = {
            "next_dates": next_dates,
            "limits": {
                "stop": terminal.kind if terminal is not None else "none",
                "max_left": 0,
                "until": "",
                "message": occurrence_exhaustion_message(terminal) if terminal is not None else "",
            },
            "rand_preview": next_dates[:10],
        }
        if not include_per_year:
            return hints

        sample_horizon = max(1, int(sample_days_for_year or 1))
        sample_end = today + timedelta(days=sample_horizon)
        annual = self._collect_dates(today, sample_end, limit=max(sample_horizon * 8, 64))
        annual_dates = self._unique_dates(annual, start=today, end=sample_end)
        if terminal is None:
            terminal = annual.terminal
        hints["per_year"] = {
            "est": len(annual_dates),
            "first": annual_dates[0].isoformat() + "T00:00" if annual_dates else "",
            "last": annual_dates[-1].isoformat() + "T00:00" if annual_dates else "",
        }
        if terminal is not None:
            hints["limits"] = {
                **hints["limits"],
                "stop": terminal.kind,
                "message": occurrence_exhaustion_message(terminal),
            }
        return hints


__all__ = ("HintBuilder",)
