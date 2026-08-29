"""Read-only occurrence query orchestration for local consumers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone, tzinfo
import hashlib
import json
from types import ModuleType
from typing import Any, Literal, Mapping, TypeAlias, cast

from .integration_models import Absent, Found, Unavailable
from .integration_context import IntegrationAccess
from .occurrence_outcomes import OccurrenceCollectionResult
from .recurrence_context import RecurrenceContext
from .task_codec import TaskCodec
from .scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from .scheduler_service import SchedulerService
from .chain_generation import ChainGenerationService
from .task_read_repository import ACTIVE_TASK_STATUSES, ALL_TASK_STATUSES
from .task_models import FieldPresence, NauticalTask, TaskObservation
from .task_codec import DEFAULT_TASK_CODEC
from .query_models import (
    HARD_MAX_FILE_SKIPS,
    HARD_MAX_ITERATIONS,
    HARD_MAX_OCCURRENCES,
    HARD_MAX_TASKS,
    OccurrenceQueryRequest,
    OccurrenceQueryResponse,
    OccurrenceRecord,
    QueryFailure,
    TaskIdentity,
    TaskOccurrenceResult,
)
from .operator_models import OperatorCursor, OperatorContractError
from .parser_models import ParseError
from .hook_validation_pipeline import ValidationStatus, validate_task_mapping
from .hook_workflow_models import WorkflowRoute


class QueryServiceError(RuntimeError):
    """Raised when a query cannot be safely constructed or executed."""


@dataclass(frozen=True, slots=True)
class _AbsentTask:
    uuid: str


@dataclass(frozen=True, slots=True)
class _AmbiguousTask:
    uuid: str


TaskRow: TypeAlias = TaskObservation | _AbsentTask | _AmbiguousTask


def _task_value(task: TaskRow, name: str) -> object:
    """Read a task field from the authoritative observation boundary."""
    if not isinstance(task, TaskObservation):
        return None
    state = task.field(name)
    if state.presence is FieldPresence.ABSENT:
        return None
    return getattr(state.value, "value", state.value)


def _task_raw_value(task: TaskObservation, name: str) -> object:
    state = task.field(name)
    return state.raw_value() if state.presence is FieldPresence.VALUE else None


def _task_with_overrides(task: TaskObservation, **overrides: Any) -> NauticalTask:
    """Build one validated domain task for a projected scheduler step."""
    values = task.to_mapping()
    values.update(overrides)
    observation = DEFAULT_TASK_CODEC.decode_row(values, source_query="query projected recurrence")
    return NauticalTask.from_observation(observation)


def _decode_repository_row(value: Any, *, source_query: str) -> TaskObservation:
    """Normalize a repository boundary row before it reaches query logic.

    Production repositories already return observations.  The explicit mapping
    branch keeps lightweight repository adapters honest at this single boundary
    without allowing raw rows into scheduling or query consumers.
    """
    if isinstance(value, TaskObservation):
        return value
    if isinstance(value, Mapping):
        return DEFAULT_TASK_CODEC.decode_row(value, source_query=source_query)
    raise TypeError("TaskReadRepository returned a non-task row")


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


def _task_identity(task: TaskObservation) -> TaskIdentity:
    uuid_value = str(_task_value(task, "uuid") or "").strip()
    chain_id = str(_task_value(task, "chainID") or "").strip()
    if not uuid_value:
        raise QueryServiceError("Taskwarrior row has no UUID")
    if not chain_id:
        raise QueryServiceError("Nautical task has no chainID; recurrence identity is incomplete")
    anchor = TaskCodec.normalize_text(_task_value(task, "anchor"))
    anchor_file = TaskCodec.normalize_text(_task_value(task, "anchor_file"))
    cp = TaskCodec.normalize_text(_task_value(task, "cp"))
    kind = "cp" if cp else "anchor" if anchor else "anchor_file" if anchor_file else ""
    expression = cp or anchor or anchor_file
    return TaskIdentity(
        uuid=uuid_value,
        chain_id=chain_id,
        link=_link_value(_task_value(task, "link")),
        description=str(_task_value(task, "description") or ""),
        recurrence_kind=kind,
        expression=expression,
        current_due=str(_task_raw_value(task, "due") or ""),
        current_scheduled=str(_task_raw_value(task, "scheduled") or ""),
    )


def _has_recurrence_identity(task: TaskObservation) -> bool:
    return bool(
        str(_task_value(task, "chainID") or "").strip()
        and any(
            TaskCodec.normalize_text(_task_value(task, field))
            for field in ("anchor", "anchor_file", "cp")
        )
    )


def _ordered_rows(rows: tuple[TaskObservation, ...]) -> tuple[TaskObservation, ...]:
    def key(row: TaskObservation) -> tuple[str, int, str]:
        link = _link_value(_task_value(row, "link"))
        return (
            str(_task_value(row, "chainID") or "").strip().lower(),
            link if link is not None else 2**31 - 1,
            str(_task_value(row, "uuid") or "").strip().lower(),
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
        local_timezone = getattr(context, "local_timezone", None)
        if not isinstance(local_timezone, tzinfo):
            raise QueryServiceError("validated local timezone is unavailable")
        self._timezone: tzinfo = local_timezone
        self._scheduler_cache: dict[tuple[str, tuple[tuple[str, str], ...]], SchedulerService] = {}

    def _scheduler_for(self, task: TaskObservation, domain_task: NauticalTask, context: RecurrenceContext) -> SchedulerService:
        """Reuse one scheduler session for identical recurrence inputs in this invocation."""
        key = (
            str(_task_value(task, "chainID") or "").strip().lower(),
            tuple(
                (field, TaskCodec.normalize_text(_task_value(task, field)))
                for field in ("anchor", "anchor_file", "anchor_mode", "omit", "omit_file", "cp")
            ),
        )
        scheduler = self._scheduler_cache.get(key)
        if scheduler is None:
            scheduler = SchedulerService.from_task(domain_task, context=context)
            self._scheduler_cache[key] = scheduler
        return scheduler

    def _context_for(self, task: TaskObservation) -> RecurrenceContext:
        chain_id = str(_task_value(task, "chainID") or "").strip()
        if not chain_id:
            raise QueryServiceError("Nautical task has no chainID; recurrence identity is incomplete")
        calendar = None
        resolver = getattr(self._core, "business_calendar_for_task", None)
        if callable(resolver):
            calendar = resolver(task)
        return RecurrenceContext(
            chain_id=chain_id,
            timezone=self._timezone,
            business_calendar=calendar,
            astronomy_config=getattr(self._core, "ASTRONOMY_CONFIG", {}),
            anchor_file_dir=str(getattr(self._core, "ANCHOR_FILE_DIR", "") or ""),
        )

    def _starts_after_request_end(
        self,
        task: TaskObservation,
        request: OccurrenceQueryRequest,
    ) -> bool:
        if request.end is None:
            return False
        try:
            reference = self._task_reference_local(task)
            end = _boundary_local(request.end.value, request.end.date_only, self._timezone, end=True)
        except (QueryServiceError, OSError, TypeError, ValueError):
            return False
        return reference is not None and reference > end

    def _rows_for(self, request: OccurrenceQueryRequest) -> tuple[TaskRow, ...] | QueryFailure:
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
                if len(read.value.rows) > HARD_MAX_TASKS:
                    return _failure(
                        "task_scope_exhausted",
                        f"whole-system snapshot contains {len(read.value.rows)} tasks; limit is {HARD_MAX_TASKS}; use an explicit chain or UUID scope",
                        limit=HARD_MAX_TASKS,
                        observed=len(read.value.rows),
                    )
                selected: list[TaskObservation] = []
                for raw_row in read.value.rows:
                    row = _decode_repository_row(raw_row, source_query="query all-active")
                    if _has_recurrence_identity(row) and not self._starts_after_request_end(row, request):
                        selected.append(row)
                rows = tuple(selected)
                return _ordered_rows(rows)
            if isinstance(read, Absent):
                return ()
            return _failure("task_read_unavailable", read.evidence.detail, retryable=read.retryable)
        if selector.chain_id:
            read = repository.chain_snapshot(selector.chain_id, statuses=ALL_TASK_STATUSES, complete_history=True)
            if isinstance(read, Found):
                return _ordered_rows(
                    tuple(_decode_repository_row(row, source_query="query chain") for row in read.value)
                )
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
            uuid_rows: list[TaskRow] = []
            for uuid_value in selector.uuids:
                matches = read.value.uuid_matches(uuid_value)
                if not matches:
                    uuid_rows.append(_AbsentTask(uuid_value))
                elif len(matches) == 1:
                    uuid_rows.append(_decode_repository_row(matches[0], source_query="query UUID snapshot"))
                else:
                    uuid_rows.append(_AmbiguousTask(uuid_value))
            return tuple(uuid_rows)
        single_rows: list[TaskRow] = []
        for uuid_value in selector.uuids:
            read = repository.by_uuid(uuid_value, statuses=ALL_TASK_STATUSES)
            if isinstance(read, Found):
                single_rows.append(_decode_repository_row(read.value, source_query="query UUID"))
                continue
            if isinstance(read, Absent):
                single_rows.append(_AbsentTask(uuid_value))
                continue
            return _failure(
                "task_read_unavailable",
                read.evidence.detail,
                task_uuid=uuid_value,
                retryable=read.retryable,
            )
        return tuple(single_rows)

    def _page_rows(
        self,
        rows: tuple[TaskRow, ...],
        request: OccurrenceQueryRequest,
    ) -> tuple[tuple[TaskRow, ...], OperatorCursor | None, bool]:
        """Slice a deterministic task snapshot and bind continuation to its content."""
        if not request.selector.all_tasks:
            if request.cursor is not None:
                raise QueryServiceError("query cursors are supported only for --all task queries")
            return rows, None, True
        snapshot_id = self._snapshot_id(rows)
        configuration = str(getattr(self._uow.context.configuration, "fingerprint", ""))
        epoch = str(getattr(self._uow, "mutation_epoch", 0))
        if request.cursor is not None:
            try:
                request.cursor.assert_compatible(snapshot_id, configuration, epoch)
            except OperatorContractError as exc:
                raise QueryServiceError(str(exc)) from exc
            page_size = request.cursor.page_size
            position = request.cursor.position
        else:
            page_size = request.max_tasks
            position = 0
        page = rows[position : position + page_size]
        complete = position + len(page) >= len(rows)
        next_cursor = None if complete else OperatorCursor(
            snapshot_id,
            configuration,
            epoch,
            position=position + len(page),
            page_size=page_size,
        )
        return page, next_cursor, complete

    @staticmethod
    def _snapshot_id(rows: tuple[TaskRow, ...]) -> str:
        evidence = [
            row.to_mapping() if isinstance(row, TaskObservation) else {"uuid": row.uuid}
            for row in rows
        ]
        return "query-snapshot-" + hashlib.sha256(
            json.dumps(evidence, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:32]

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

    def _task_reference_local(self, task: TaskObservation) -> datetime | None:
        """Return the task's current recurrence reference in query timezone."""
        raw = _task_value(task, "due") or _task_value(task, "scheduled")
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
        task: TaskObservation,
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
        chain_until = _task_value(task, "chainUntil")
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
        link = _link_value(_task_value(task, "link")) or 1
        chain_max = _task_value(task, "chainMax")
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
            stamp = self._core.fmt_isoz(current.astimezone(timezone.utc))
            parent = _task_with_overrides(task, end=stamp, due=stamp, link=link)
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

    def _query_task(self, task: TaskRow, request: OccurrenceQueryRequest) -> TaskOccurrenceResult:
        if isinstance(task, _AbsentTask):
            return TaskOccurrenceResult(
                task=None,
                status="absent",
                failure=_failure("task_absent", "Taskwarrior returned no task for the requested UUID", task_uuid=task.uuid),
            )
        if isinstance(task, _AmbiguousTask):
            uuid_value = task.uuid
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
            _validated, validation_report = validate_task_mapping(
                task.to_mapping(),
                route=WorkflowRoute.RECURRING_EDIT,
                source_query="query validation",
            )
            if validation_report.status is not ValidationStatus.VALID:
                finding = validation_report.findings[0]
                return TaskOccurrenceResult(
                    task=None,
                    status="invalid" if validation_report.status.value == "invalid" else "unavailable",
                    failure=_failure(
                        finding.code,
                        f"{finding.reason} {finding.correction}".strip(),
                        task_uuid=str(_task_value(task, "uuid") or "") or None,
                        retryable=validation_report.status is ValidationStatus.UNAVAILABLE,
                    ),
                )
            identity = _task_identity(task)
            if identity.recurrence_kind == "cp":
                return self._query_cp_task(task, identity, request)
            domain_task = NauticalTask.from_observation(task)
            context = self._context_for(task)
            scheduler = self._scheduler_for(task, domain_task, context)
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
            error_uuid: str | None = str(_task_value(task, "uuid") or "") or None
            return TaskOccurrenceResult(
                task=None,
                status="invalid" if isinstance(exc, (ParseError, TypeError, ValueError, QueryServiceError)) else "unavailable",
                failure=_failure("task_invalid", str(exc), task_uuid=error_uuid),
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
        try:
            page_rows, next_cursor, complete = self._page_rows(rows, request)
        except QueryServiceError as exc:
            return OccurrenceQueryResponse(
                request=request,
                timezone=_timezone_name(self._timezone),
                status="unavailable",
                configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
                failure=_failure("cursor_unavailable", str(exc), retryable=False),
                coverage={"kind": "unavailable", "reason": str(exc)},
            )
        raw_results = tuple(self._query_task(row, request) for row in page_rows)
        if request.selector.all_tasks:
            raw_results = tuple(
                result
                for result in raw_results
                if not (
                    result.status == "empty"
                    and not result.occurrences
                    and not result.omitted_occurrences
                    and result.failure is None
                )
            )
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
            status=cast(
                Literal["found", "empty", "exhausted", "absent", "unavailable", "invalid"],
                status,
            ),
            configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
            cursor=next_cursor,
            complete=complete,
            coverage={
                "kind": "complete" if complete else "bounded",
                "source": "taskwarrior.authoritative_export",
                "observed": tuple(
                    str(_task_value(row, "uuid") or "")
                    for row in page_rows
                    if isinstance(row, TaskObservation) or isinstance(row, (_AbsentTask, _AmbiguousTask))
                ),
                "omitted_count": max(0, len(rows) - len(page_rows)),
                "snapshot_id": self._snapshot_id(rows),
                "mutation_epoch": str(getattr(self._uow, "mutation_epoch", 0)),
            },
        )

    def _reference_utc(self, task: TaskObservation) -> datetime:
        parser = getattr(self._core, "parse_dt_any", None)
        if not callable(parser):
            raise QueryServiceError("Nautical datetime parser is unavailable")
        for field in (("end",) if TaskCodec.normalize_text(_task_value(task, "cp")) else ()) + ("due", "scheduled"):
            value = _task_value(task, field)
            if not value:
                continue
            parsed = parser(value)
            if isinstance(parsed, datetime) and parsed.tzinfo is not None and parsed.utcoffset() is not None:
                return parsed.astimezone(timezone.utc)
            raise QueryServiceError(f"task {field} is not a valid timezone-aware datetime")
        raise QueryServiceError("task has no due, scheduled, or completion reference")

    def _daily_anchor_summary(
        self,
        scheduler: SchedulerService,
        *,
        due_local: datetime,
        evaluated_local: datetime,
        mode: str,
    ) -> tuple[dict[str, Any], list[str]]:
        day_start = datetime.combine(evaluated_local.date(), time.min, tzinfo=self._timezone)
        day_end = day_start + timedelta(days=1) - timedelta(microseconds=1)
        collected = scheduler.collect_request(
            OccurrenceRangeRequest(
                cursor=OccurrenceCursor(day_start, inclusive=True, timezone=self._timezone),
                end_local=day_end,
                limit=HARD_MAX_OCCURRENCES,
                omission_policy="exclude",
                max_iterations=HARD_MAX_ITERATIONS,
                max_file_skips=HARD_MAX_FILE_SKIPS,
            )
        )
        if collected.failure is not None:
            raise QueryServiceError(collected.failure.reason)
        occurrences = [item.local_datetime for item in collected.occurrences if item.local_datetime is not None]
        current_position = next(
            (index for index, occurrence in enumerate(occurrences, 1) if occurrence == due_local),
            None,
        )
        missed = (
            [occurrence for occurrence in occurrences if due_local < occurrence <= evaluated_local]
            if mode in {"skip", "flex"}
            else []
        )
        upcoming = [occurrence for occurrence in occurrences if occurrence > evaluated_local]
        return (
            {
                "date": evaluated_local.date().isoformat(),
                "total": len(occurrences),
                "current_position": current_position,
                "missed": len(missed),
                "upcoming": len(upcoming),
            },
            [occurrence.isoformat() for occurrence in missed],
        )

    def _query_next_task(
        self,
        task: TaskRow,
        request: OccurrenceQueryRequest,
    ) -> TaskOccurrenceResult:
        if isinstance(task, _AbsentTask):
            return TaskOccurrenceResult(None, "absent", failure=_failure("task_absent", "Taskwarrior returned no task", task_uuid=task.uuid))
        if isinstance(task, _AmbiguousTask):
            return TaskOccurrenceResult(None, "invalid", failure=_failure("ambiguous_uuid", "UUID selector matched more than one task", task_uuid=task.uuid))
        identity: TaskIdentity | None = None
        try:
            identity = _task_identity(task)
            domain_task = NauticalTask.from_observation(task)
            context = self._context_for(task)
            reference_utc = self._reference_utc(task)
            evaluated_utc = (
                _boundary_local(
                    request.evaluation_at.value,
                    request.evaluation_at.date_only,
                    self._timezone,
                    end=False,
                ).astimezone(timezone.utc)
                if request.evaluation_at is not None
                else reference_utc
            )
            link = _link_value(_task_value(task, "link")) or 1
            chain_metadata: dict[str, Any] = {
                "chainID": identity.chain_id,
                "link": link,
                "prevLink": str(_task_value(task, "prevLink") or "") or None,
                "nextLink": str(_task_value(task, "nextLink") or "") or None,
                "status": str(_task_value(task, "status") or "") or None,
                "chainMax": _task_value(task, "chainMax") or None,
                "chainUntil": _task_value(task, "chainUntil") or None,
            }
            recurrence_kind = identity.recurrence_kind
            reference_field = "end" if recurrence_kind == "cp" and _task_value(task, "end") else (
                "due" if _task_value(task, "due") else "scheduled"
            )
            lifecycle_metadata: dict[str, Any] = {
                "projected": True,
                "basis": "completion-end" if recurrence_kind == "cp" and _task_value(task, "end") else "task-reference",
                "reference_field": reference_field,
                "reference_utc": reference_utc.isoformat().replace("+00:00", "Z"),
                "target_field": "scheduled" if not _task_value(task, "due") and _task_value(task, "scheduled") else "due",
                "child_created": False,
            }
            if request.evaluation_at is not None:
                lifecycle_metadata["evaluated_at"] = evaluated_utc.astimezone(self._timezone).isoformat()
            chain_max = _task_value(task, "chainMax")
            if chain_max not in (None, ""):
                try:
                    if link >= int(str(chain_max)):
                        return TaskOccurrenceResult(identity, "empty", chain=chain_metadata, lifecycle=lifecycle_metadata)
                except (TypeError, ValueError) as exc:
                    raise QueryServiceError("chainMax is not an integer") from exc

            def bounded(candidate: datetime) -> bool:
                chain_until = _task_value(task, "chainUntil")
                if not chain_until:
                    return True
                parser = getattr(self._core, "parse_dt_any", None)
                if not callable(parser):
                    raise QueryServiceError("Nautical datetime parser is unavailable")
                limit = parser(chain_until)
                if not isinstance(limit, datetime) or limit.tzinfo is None or limit.utcoffset() is None:
                    raise QueryServiceError("chainUntil is not a valid timezone-aware datetime")
                return candidate.astimezone(timezone.utc) <= limit.astimezone(timezone.utc)

            if TaskCodec.normalize_text(_task_value(task, "cp")):
                if request.evaluation_at is not None:
                    formatter = getattr(self._core, "fmt_isoz", None)
                    if not callable(formatter):
                        raise QueryServiceError("Nautical datetime formatter is unavailable")
                    projected_end = formatter(evaluated_utc)
                elif not _task_value(task, "end"):
                    formatter = getattr(self._core, "fmt_isoz", None)
                    if not callable(formatter):
                        raise QueryServiceError("Nautical datetime formatter is unavailable")
                    projected_end = formatter(reference_utc)
                else:
                    projected_end = str(_task_value(task, "end") or "")
                parent = _task_with_overrides(task, end=projected_end)
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
            scheduler = self._scheduler_for(task, domain_task, context)
            lifecycle_metadata["basis_detail"] = "calendar-schedule"
            identity = replace(identity, schedule_fingerprint=scheduler.fingerprint)
            if request.evaluation_at is not None:
                due_local = reference_utc.astimezone(self._timezone)
                evaluated_local = evaluated_utc.astimezone(self._timezone)
                mode = str(_task_value(task, "anchor_mode") or "skip").strip().lower() or "skip"
                selected = scheduler.select_mode(
                    mode,
                    due_local=due_local,
                    end_local=evaluated_local,
                    due_explicit=bool(_task_value(task, "due")),
                    fallback_hhmm=(due_local.hour, due_local.minute),
                    default_seed_date=due_local.date(),
                )
                if selected.selected_occurrence is None:
                    return TaskOccurrenceResult(
                        identity,
                        "empty",
                        chain=chain_metadata,
                        lifecycle=lifecycle_metadata,
                    )
                daily_instances, missed_occurrences = self._daily_anchor_summary(
                    scheduler,
                    due_local=due_local,
                    evaluated_local=evaluated_local,
                    mode=mode,
                )
                candidate = selected.selected_occurrence
                lifecycle_metadata.update(
                    {
                        "anchor_mode": mode,
                        "basis_detail": selected.basis,
                        "daily_instances": daily_instances,
                        "missed_occurrences": missed_occurrences,
                        "next": candidate.isoformat(),
                    }
                )
                if not bounded(candidate):
                    return TaskOccurrenceResult(identity, "empty", chain=chain_metadata, lifecycle=lifecycle_metadata)
                return TaskOccurrenceResult(
                    identity,
                    "found",
                    (
                        OccurrenceRecord(
                            local=candidate,
                            utc=candidate.astimezone(timezone.utc),
                            timezone=_timezone_name(self._timezone),
                            source="anchor",
                        ),
                    ),
                    chain=chain_metadata,
                    lifecycle=lifecycle_metadata,
                )
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
                failure=_failure("next_projection_invalid", str(exc), task_uuid=identity.uuid if identity else str(_task_value(task, "uuid") or "")),
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
        results = tuple(self._query_next_task(row, request) for row in rows)
        statuses = {item.status for item in results}
        status = "found" if "found" in statuses else "unavailable" if "unavailable" in statuses else "invalid" if "invalid" in statuses else "absent" if "absent" in statuses else "empty"
        return OccurrenceQueryResponse(
            request=request,
            timezone=_timezone_name(self._timezone),
            results=results,
            status=cast(
                Literal["found", "empty", "exhausted", "absent", "unavailable", "invalid"],
                status,
            ),
            configuration_fingerprint=str(getattr(self._uow.context.configuration, "fingerprint", "")),
            schema="nautical.query.next",
        )


__all__ = ("OccurrenceQueryService", "QueryServiceError")
