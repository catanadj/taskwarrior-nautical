"""Read-only occurrence query orchestration for local consumers."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from types import ModuleType
from typing import Any, Mapping

from .integration_models import Absent, Found, Unavailable
from .integration_context import IntegrationAccess
from .occurrence_outcomes import OccurrenceCollectionResult
from .recurrence_context import RecurrenceContext
from .recurrence_spec import normalize_recurrence_text
from .scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from .scheduler_service import SchedulerService
from .chain_generation import ChainGenerationService
from .task_read_repository import ACTIVE_TASK_STATUSES, ALL_TASK_STATUSES
from .query_models import (
    OccurrenceQueryRequest,
    OccurrenceQueryResponse,
    OccurrenceRecord,
    QueryFailure,
    TaskIdentity,
    TaskOccurrenceResult,
)
from .parser_models import ParseError


class QueryServiceError(RuntimeError):
    """Raised when a query cannot be safely constructed or executed."""


def _timezone_name(value: tzinfo) -> str:
    return str(getattr(value, "key", "") or value)


def _boundary_local(value: date | datetime, date_only: bool, local_timezone: tzinfo, *, end: bool) -> datetime:
    if date_only:
        if not isinstance(value, date) or isinstance(value, datetime):
            raise QueryServiceError("query date boundary is invalid")
        if end:
            next_day = value + timedelta(days=1)
            return datetime.combine(next_day, time.min, tzinfo=local_timezone) - timedelta(microseconds=1)
        return datetime.combine(value, time.min, tzinfo=local_timezone)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise QueryServiceError("query timestamp boundary must be timezone-aware")
    return value.astimezone(local_timezone)


def _link_value(value: object) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _task_identity(task: Mapping[str, Any]) -> TaskIdentity:
    uuid_value = str(task.get("uuid") or "").strip()
    chain_id = str(task.get("chainID") or "").strip()
    if not uuid_value:
        raise QueryServiceError("Taskwarrior row has no UUID")
    if not chain_id:
        raise QueryServiceError("Nautical task has no chainID; recurrence identity is incomplete")
    anchor = normalize_recurrence_text(task.get("anchor"))
    anchor_file = normalize_recurrence_text(task.get("anchor_file"))
    cp = normalize_recurrence_text(task.get("cp"))
    kind = "cp" if cp else "anchor" if anchor else "anchor_file" if anchor_file else ""
    expression = cp or anchor or anchor_file
    return TaskIdentity(
        uuid=uuid_value,
        chain_id=chain_id,
        link=_link_value(task.get("link")),
        description=str(task.get("description") or ""),
        recurrence_kind=kind,
        expression=expression,
        current_due=str(task.get("due") or ""),
        current_scheduled=str(task.get("scheduled") or ""),
    )


def _has_recurrence_identity(task: Mapping[str, Any]) -> bool:
    return bool(
        str(task.get("chainID") or "").strip()
        and any(
            normalize_recurrence_text(task.get(field))
            for field in ("anchor", "anchor_file", "cp")
        )
    )


def _ordered_rows(rows: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
    def key(row: Mapping[str, Any]) -> tuple[str, int, str]:
        link = _link_value(row.get("link"))
        return (
            str(row.get("chainID") or "").strip().lower(),
            link if link is not None else 2**31 - 1,
            str(row.get("uuid") or "").strip().lower(),
        )

    return tuple(sorted(rows, key=key))


def _failure(code: str, message: str, *, task_uuid: str | None = None, retryable: bool = False, **details: Any) -> QueryFailure:
    return QueryFailure(code=code, message=message, task_uuid=task_uuid, retryable=retryable, details=details)


def _terminal(result: OccurrenceCollectionResult) -> Mapping[str, Any] | None:
    value = result.terminal
    if value is None:
        return None
    return {
        "kind": value.kind,
        "scope": value.scope,
        "reference": value.reference,
        "limit": value.limit,
        "message": str(value),
    }


class OccurrenceQueryService:
    """Resolve bounded occurrence queries without mutation or subprocesses."""

    def __init__(self, unit_of_work: Any, *, core: ModuleType) -> None:
        context = getattr(unit_of_work, "context", None)
        if context is None or getattr(context, "access", None) is not IntegrationAccess.READ_ONLY:
            raise QueryServiceError("occurrence queries require a read-only Taskwarrior unit of work")
        self._uow = unit_of_work
        self._core = core
        self._timezone = getattr(context, "local_timezone", None)
        if not isinstance(self._timezone, tzinfo):
            raise QueryServiceError("validated local timezone is unavailable")

    def _context_for(self, task: Mapping[str, Any]) -> RecurrenceContext:
        chain_id = str(task.get("chainID") or "").strip()
        if not chain_id:
            raise QueryServiceError("Nautical task has no chainID; recurrence identity is incomplete")
        calendar = None
        resolver = getattr(self._core, "business_calendar_for_task", None)
        if callable(resolver):
            calendar = resolver(dict(task))
        return RecurrenceContext(
            chain_id=chain_id,
            timezone=self._timezone,
            business_calendar=calendar,
            astronomy_config=getattr(self._core, "ASTRONOMY_CONFIG", {}),
            anchor_file_dir=str(getattr(self._core, "ANCHOR_FILE_DIR", "") or ""),
        )

    def _rows_for(self, request: OccurrenceQueryRequest) -> tuple[Mapping[str, Any], ...] | QueryFailure:
        repository = self._uow.repository
        selector = request.selector
        if selector.all_tasks:
            read = repository.broad_snapshot(
                identity="query:all-active",
                filters=("chain:on",),
                statuses=ACTIVE_TASK_STATUSES,
                complete_chain_history=False,
            )
            if isinstance(read, Found):
                rows = tuple(row for row in read.value.rows if _has_recurrence_identity(row))
                return _ordered_rows(rows)[: request.max_tasks]
            if isinstance(read, Absent):
                return ()
            return _failure("task_read_unavailable", read.evidence.detail, retryable=read.retryable)
        if selector.chain_id:
            read = repository.chain_snapshot(selector.chain_id, statuses=ALL_TASK_STATUSES, complete_history=True)
            if isinstance(read, Found):
                return _ordered_rows(tuple(read.value))[: request.max_tasks]
            if isinstance(read, Absent):
                return _failure("chain_absent", read.reason)
            return _failure("task_read_unavailable", read.evidence.detail, retryable=read.retryable)
        if len(selector.uuids) > 1:
            read = repository.broad_snapshot(
                identity="query:uuids",
                filters=(),
                statuses=ALL_TASK_STATUSES,
                complete_chain_history=True,
            )
            if not isinstance(read, Found):
                if isinstance(read, Absent):
                    return _failure("task_snapshot_absent", read.reason)
                return _failure("task_read_unavailable", read.evidence.detail, retryable=read.retryable)
            uuid_rows: list[Mapping[str, Any]] = []
            for uuid_value in selector.uuids:
                matches = read.value.uuid_matches(uuid_value)
                if not matches:
                    uuid_rows.append({"uuid": uuid_value, "_query_absent": True})
                elif len(matches) == 1:
                    uuid_rows.append(matches[0])
                else:
                    uuid_rows.append({"uuid": uuid_value, "_query_ambiguous": True})
            return tuple(uuid_rows[: request.max_tasks])
        single_rows: list[Mapping[str, Any]] = []
        for uuid_value in selector.uuids:
            read = repository.by_uuid(uuid_value, statuses=ALL_TASK_STATUSES)
            if isinstance(read, Found):
                single_rows.append(read.value)
                continue
            if isinstance(read, Absent):
                single_rows.append({"uuid": uuid_value, "_query_absent": True})
                continue
            return _failure(
                "task_read_unavailable",
                read.evidence.detail,
                task_uuid=uuid_value,
                retryable=read.retryable,
            )
        return tuple(single_rows[: request.max_tasks])

    def _records(self, items: Any, timezone_name: str) -> tuple[OccurrenceRecord, ...]:
        records: list[OccurrenceRecord] = []
        for item in items:
            if item.local_datetime is None:
                continue
            local = item.local_datetime
            records.append(
                OccurrenceRecord(
                    local=local,
                    utc=local.astimezone(timezone.utc),
                    timezone=timezone_name,
                    source=item.source,
                    description=item.description,
                    omitted=bool(item.omitted),
                )
            )
        return tuple(records)

    def _task_reference_local(self, task: Mapping[str, Any]) -> datetime | None:
        """Return the task's current recurrence reference in query timezone."""
        raw = task.get("due") or task.get("scheduled")
        if not raw:
            return None
        parser = getattr(self._core, "parse_dt_any", None)
        if not callable(parser):
            raise QueryServiceError("Nautical datetime parser is unavailable")
        parsed = parser(raw)
        if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
            raise QueryServiceError("task due/scheduled value is not a valid timezone-aware datetime")
        return parsed.astimezone(self._timezone)

    def _query_cp_task(
        self,
        task: Mapping[str, Any],
        identity: TaskIdentity,
        request: OccurrenceQueryRequest,
    ) -> TaskOccurrenceResult:
        """Project CP slots from the task's current due/end without mutation."""
        reference = self._task_reference_local(task)
        if reference is None:
            raise QueryServiceError("CP task has no due or scheduled reference")
        start = _boundary_local(request.start.value, request.start.date_only, self._timezone, end=False)
        end = (
            _boundary_local(request.end.value, request.end.date_only, self._timezone, end=True)
            if request.end is not None else None
        )
        chain_until = task.get("chainUntil")
        if chain_until:
            parser = getattr(self._core, "parse_dt_any", None)
            if not callable(parser):
                raise QueryServiceError("Nautical datetime parser is unavailable")
            parsed_until = parser(chain_until)
            if not isinstance(parsed_until, datetime) or parsed_until.tzinfo is None or parsed_until.utcoffset() is None:
                raise QueryServiceError("chainUntil is not a valid timezone-aware datetime")
            until_local = parsed_until.astimezone(self._timezone)
            end = until_local if end is None else min(end, until_local)
        if end is not None and end < start:
            return TaskOccurrenceResult(identity, "empty")
        link = _link_value(task.get("link")) or 1
        chain_max = task.get("chainMax")
        max_link = None if chain_max in (None, "") else int(float(str(chain_max)))
        limit = request.count or request.max_occurrences
        current = reference
        records: list[OccurrenceRecord] = []
        generator = ChainGenerationService.from_core(self._core)
        while len(records) < limit:
            within_start = current > start or (request.start_inclusive and current == start)
            within_end = end is None or current <= end
            if within_start and within_end:
                records.append(
                    OccurrenceRecord(
                        local=current,
                        utc=current.astimezone(timezone.utc),
                        timezone=_timezone_name(self._timezone),
                        source="cp",
                    )
                )
            if end is not None and current >= end:
                break
            if max_link is not None and link >= max_link:
                break
            parent = dict(task)
            stamp = self._core.fmt_isoz(current.astimezone(timezone.utc))
            parent["end"] = stamp
            parent["due"] = stamp
            parent["link"] = link
            child_due, _metadata = generator.compute_cp_child_due(parent)
            if child_due is None:
                break
            current = child_due.astimezone(self._timezone)
            link += 1
        return TaskOccurrenceResult(
            identity,
            "found" if records else "empty",
            tuple(records),
        )

    def _query_task(self, task: Mapping[str, Any], request: OccurrenceQueryRequest) -> TaskOccurrenceResult:
        if task.get("_query_absent"):
            return TaskOccurrenceResult(
                task=None,
                status="absent",
                failure=_failure("task_absent", "Taskwarrior returned no task for the requested UUID", task_uuid=str(task.get("uuid") or "")),
            )
        if task.get("_query_ambiguous"):
            uuid_value = str(task.get("uuid") or "")
            return TaskOccurrenceResult(
                task=None,
                status="invalid",
                failure=_failure(
                    "ambiguous_uuid",
                    "UUID selector matched more than one task",
                    task_uuid=uuid_value,
                ),
            )
        try:
            identity = _task_identity(task)
            if identity.recurrence_kind == "cp":
                return self._query_cp_task(task, identity, request)
            context = self._context_for(task)
            scheduler = SchedulerService.from_task(task, context=context)
            identity = replace(identity, schedule_fingerprint=scheduler.fingerprint)
            start = _boundary_local(request.start.value, request.start.date_only, self._timezone, end=False)
            task_reference = self._task_reference_local(task)
            if task_reference is not None and task_reference > start:
                start = task_reference
                start_inclusive = True
            else:
                start_inclusive = request.start_inclusive
            end = (
                _boundary_local(request.end.value, request.end.date_only, self._timezone, end=True)
                if request.end is not None else None
            )
            if end is not None and end < start:
                return TaskOccurrenceResult(task=identity, status="empty")
            range_request = OccurrenceRangeRequest(
                cursor=OccurrenceCursor(start, inclusive=start_inclusive, timezone=self._timezone),
                end_local=end,
                limit=request.count or request.max_occurrences,
                omission_policy=request.omission_policy,
                max_iterations=request.max_iterations,
                max_file_skips=request.max_file_skips,
            )
            collected = scheduler.collect_request(range_request)
            failure = None
            if collected.failure is not None:
                failure = _failure(
                    "scheduler_unavailable" if collected.status == "unavailable" else "scheduler_invalid",
                    collected.failure.reason,
                    task_uuid=identity.uuid,
                    retryable=collected.status == "unavailable",
                    error_type=collected.failure.error_type,
                )
            return TaskOccurrenceResult(
                task=identity,
                status=collected.status,
                occurrences=self._records(collected, _timezone_name(self._timezone)),
                omitted_occurrences=self._records(collected.omitted_occurrences, _timezone_name(self._timezone)),
                failure=failure,
                terminal=_terminal(collected),
            )
        except (ParseError, QueryServiceError, LookupError, OSError, TypeError, ValueError) as exc:
            uuid_value = str(task.get("uuid") or "") or None
            return TaskOccurrenceResult(
                task=None,
                status="invalid" if isinstance(exc, (ParseError, TypeError, ValueError, QueryServiceError)) else "unavailable",
                failure=_failure("task_invalid", str(exc), task_uuid=uuid_value),
            )

    def query(self, request: OccurrenceQueryRequest) -> OccurrenceQueryResponse:
        if not isinstance(request, OccurrenceQueryRequest):
            raise QueryServiceError("occurrence query requires a validated request")
        rows = self._rows_for(request)
        if isinstance(rows, QueryFailure):
            return OccurrenceQueryResponse(
                request=request,
                timezone=_timezone_name(self._timezone),
                status="unavailable",
                configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
                failure=rows,
            )
        raw_results = tuple(self._query_task(row, request) for row in rows)
        results_list: list[TaskOccurrenceResult] = []
        total = 0
        for result in raw_results:
            available = max(0, request.max_total_occurrences - total)
            if len(result.occurrences) > available:
                results_list.append(
                    replace(
                        result,
                        status="exhausted",
                        occurrences=result.occurrences[:available],
                        failure=_failure(
                            "total_occurrence_limit",
                            "query total occurrence limit was reached",
                            task_uuid=result.task.uuid if result.task is not None else None,
                            limit=request.max_total_occurrences,
                        ),
                        terminal={
                            "kind": "total_query_limit",
                            "limit": request.max_total_occurrences,
                        },
                    )
                )
                total = request.max_total_occurrences
                continue
            results_list.append(result)
            total += len(result.occurrences)
        results = tuple(results_list)
        statuses = {result.status for result in results}
        if "found" in statuses:
            status = "found"
        elif "unavailable" in statuses:
            status = "unavailable"
        elif "invalid" in statuses:
            status = "invalid"
        elif "exhausted" in statuses:
            status = "exhausted"
        elif "absent" in statuses:
            status = "absent"
        else:
            status = "empty"
        return OccurrenceQueryResponse(
            request=request,
            timezone=_timezone_name(self._timezone),
            results=results,
            status=status,
            configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
        )

    def _reference_utc(self, task: Mapping[str, Any]) -> datetime:
        parser = getattr(self._core, "parse_dt_any", None)
        if not callable(parser):
            raise QueryServiceError("Nautical datetime parser is unavailable")
        for field in (("end",) if normalize_recurrence_text(task.get("cp")) else ()) + ("due", "scheduled"):
            value = task.get(field)
            if not value:
                continue
            parsed = parser(value)
            if isinstance(parsed, datetime) and parsed.tzinfo is not None and parsed.utcoffset() is not None:
                return parsed.astimezone(timezone.utc)
            raise QueryServiceError(f"task {field} is not a valid timezone-aware datetime")
        raise QueryServiceError("task has no due, scheduled, or completion reference")

    def _query_next_task(self, task: Mapping[str, Any]) -> TaskOccurrenceResult:
        if task.get("_query_absent"):
            return TaskOccurrenceResult(None, "absent", failure=_failure("task_absent", "Taskwarrior returned no task", task_uuid=str(task.get("uuid") or "")))
        if task.get("_query_ambiguous"):
            return TaskOccurrenceResult(None, "invalid", failure=_failure("ambiguous_uuid", "UUID selector matched more than one task", task_uuid=str(task.get("uuid") or "")))
        identity: TaskIdentity | None = None
        try:
            identity = _task_identity(task)
            context = self._context_for(task)
            reference_utc = self._reference_utc(task)
            link = _link_value(task.get("link")) or 1
            chain_metadata: dict[str, Any] = {
                "chainID": identity.chain_id,
                "link": link,
                "prevLink": str(task.get("prevLink") or "") or None,
                "nextLink": str(task.get("nextLink") or "") or None,
                "status": str(task.get("status") or "") or None,
                "chainMax": task.get("chainMax") or None,
                "chainUntil": task.get("chainUntil") or None,
            }
            recurrence_kind = identity.recurrence_kind
            reference_field = "end" if recurrence_kind == "cp" and task.get("end") else (
                "due" if task.get("due") else "scheduled"
            )
            lifecycle_metadata: dict[str, Any] = {
                "projected": True,
                "basis": "completion-end" if recurrence_kind == "cp" and task.get("end") else "task-reference",
                "reference_field": reference_field,
                "reference_utc": reference_utc.isoformat().replace("+00:00", "Z"),
                "target_field": "scheduled" if not task.get("due") and task.get("scheduled") else "due",
                "child_created": False,
            }
            chain_max = task.get("chainMax")
            if chain_max not in (None, ""):
                try:
                    if link >= int(chain_max):
                        return TaskOccurrenceResult(identity, "empty", chain=chain_metadata, lifecycle=lifecycle_metadata)
                except (TypeError, ValueError) as exc:
                    raise QueryServiceError("chainMax is not an integer") from exc

            def bounded(candidate: datetime) -> bool:
                chain_until = task.get("chainUntil")
                if not chain_until:
                    return True
                parser = getattr(self._core, "parse_dt_any", None)
                if not callable(parser):
                    raise QueryServiceError("Nautical datetime parser is unavailable")
                limit = parser(chain_until)
                if not isinstance(limit, datetime) or limit.tzinfo is None or limit.utcoffset() is None:
                    raise QueryServiceError("chainUntil is not a valid timezone-aware datetime")
                return candidate.astimezone(timezone.utc) <= limit.astimezone(timezone.utc)

            if normalize_recurrence_text(task.get("cp")):
                parent = dict(task)
                if not parent.get("end"):
                    formatter = getattr(self._core, "fmt_isoz", None)
                    if not callable(formatter):
                        raise QueryServiceError("Nautical datetime formatter is unavailable")
                    parent["end"] = formatter(reference_utc)
                child_due, _metadata = ChainGenerationService.from_core(self._core).compute_cp_child_due(parent)
                if child_due is None:
                    return TaskOccurrenceResult(identity, "empty", chain=chain_metadata, lifecycle=lifecycle_metadata)
                if not bounded(child_due):
                    return TaskOccurrenceResult(identity, "empty", chain=chain_metadata, lifecycle=lifecycle_metadata)
                lifecycle_metadata["basis_detail"] = str((_metadata or {}).get("basis") or "end+cp")
                lifecycle_metadata["target_field"] = str((_metadata or {}).get("target_field") or lifecycle_metadata["target_field"])
                local = child_due.astimezone(self._timezone)
                record = OccurrenceRecord(
                    local=local,
                    utc=child_due.astimezone(timezone.utc),
                    timezone=_timezone_name(self._timezone),
                    source="cp",
                )
                return TaskOccurrenceResult(identity, "found", (record,), chain=chain_metadata, lifecycle=lifecycle_metadata)
            scheduler = SchedulerService.from_task(task, context=context)
            lifecycle_metadata["basis_detail"] = "calendar-schedule"
            identity = replace(identity, schedule_fingerprint=scheduler.fingerprint)
            result = scheduler.collect_request(
                OccurrenceRangeRequest(
                    cursor=OccurrenceCursor(
                        reference_utc.astimezone(self._timezone),
                        inclusive=False,
                        timezone=self._timezone,
                    ),
                    limit=1,
                    omission_policy="exclude",
                )
            )
            if result.failure is not None:
                return TaskOccurrenceResult(
                    identity,
                    result.status,
                    failure=_failure("scheduler_unavailable", result.failure.reason, task_uuid=identity.uuid, retryable=result.status == "unavailable"),
                    chain=chain_metadata,
                    lifecycle=lifecycle_metadata,
                )
            records = self._records(result, _timezone_name(self._timezone))
            if records and not bounded(records[0].utc):
                return TaskOccurrenceResult(identity, "empty", chain=chain_metadata, lifecycle=lifecycle_metadata)
            return TaskOccurrenceResult(identity, result.status, records, terminal=_terminal(result), chain=chain_metadata, lifecycle=lifecycle_metadata)
        except (ParseError, QueryServiceError, LookupError, OSError, TypeError, ValueError) as exc:
            return TaskOccurrenceResult(
                identity,
                "invalid",
                failure=_failure("next_projection_invalid", str(exc), task_uuid=identity.uuid if identity else str(task.get("uuid") or "")),
                chain=({"chainID": identity.chain_id} if identity is not None else {}),
                lifecycle={"projected": True, "child_created": False},
            )

    def query_next(self, request: OccurrenceQueryRequest) -> OccurrenceQueryResponse:
        if not isinstance(request, OccurrenceQueryRequest) or request.operation != "next":
            raise QueryServiceError("next query requires a request with operation 'next'")
        rows = self._rows_for(request)
        if isinstance(rows, QueryFailure):
            return OccurrenceQueryResponse(
                request=request,
                timezone=_timezone_name(self._timezone),
                status="unavailable",
                configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
                failure=rows,
                schema="nautical.query.next",
            )
        results = tuple(self._query_next_task(row) for row in rows)
        statuses = {item.status for item in results}
        status = "found" if "found" in statuses else "unavailable" if "unavailable" in statuses else "invalid" if "invalid" in statuses else "absent" if "absent" in statuses else "empty"
        return OccurrenceQueryResponse(
            request=request,
            timezone=_timezone_name(self._timezone),
            results=results,
            status=status,
            configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
            schema="nautical.query.next",
        )


__all__ = ("OccurrenceQueryService", "QueryServiceError")
