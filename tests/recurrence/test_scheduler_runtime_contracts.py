"""Direct contracts for recurrence sessions, services, and diagnostics."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import nautical_core as core
from nautical_core.evaluation_session import EvaluationSession
from nautical_core.occurrence_outcomes import FoundOccurrence, OccurrenceCollectionResult
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.recurrence_spec import RecurrenceSpec
from nautical_core.scheduler_cursor import OccurrenceCursor, OccurrenceRangeRequest
from nautical_core.scheduler_service import SchedulerService
from nautical_core.scheduler_trace import SchedulerTrace
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.task_models import NauticalTask


def _evaluator_for_task(values: dict[str, str], *, zone=timezone.utc, astronomy_config=None):
    from nautical_core.recurrence_evaluator import RecurrenceEvaluator

    task = dict(values)
    chain_id = task.setdefault("chainID", "runtime-contract")
    task.setdefault("uuid", "00000000-0000-4000-8000-000000000001")
    task.setdefault("status", "pending")
    task.setdefault("link", 1)
    observation = DEFAULT_TASK_CODEC.decode_row(task, source_query="test:scheduler-runtime")
    return RecurrenceEvaluator.from_observation(
        observation,
        context=RecurrenceContext(
            chain_id=chain_id, timezone=zone, astronomy_config=astronomy_config
        ),
    )


class SchedulerRuntimeContractTests(unittest.TestCase):
    def test_last_weekday_selector_advances_through_friday_occurrences(self) -> None:
        dnf = core.validate_anchor_expr_strict("m:last-fri")
        cursor = date(2026, 1, 1)
        occurrences = []
        for _ in range(5):
            occurrence, _metadata = core.next_after_expr(
                dnf,
                cursor,
                default_seed=date(2026, 1, 1),
            )
            occurrences.append(occurrence)
            cursor = occurrence

        self.assertEqual(len({(value.year, value.month) for value in occurrences}), 5)
        self.assertTrue(all(value.weekday() == 4 for value in occurrences))

    def test_recurrence_spec_normalizes_fields_and_enforces_context_identity(self) -> None:
        spec = RecurrenceSpec.from_observation(
            DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "00000000-0000-4000-8000-000000000505",
                    "status": "pending",
                    "chainID": "spec-chain",
                    "link": 1,
                    "anchor": " w:mon ",
                    "anchor_file": " events.csv ",
                    "omit": " y:12-25 ",
                    "cp": "",
                    "anchor_mode": "ALL",
                    "chainMax": "4",
                    "chainUntil": " 20261231T230000Z ",
                },
                source_query="test:recurrence-spec",
            )
        )
        self.assertEqual(spec.context.chain_id, "spec-chain")
        self.assertEqual((spec.anchor, spec.anchor_file), ("w:mon", "events.csv"))
        self.assertEqual((spec.omit, spec.chain_max), ("y:12-25", 4))
        self.assertEqual((spec.anchor_mode, spec.kind, spec.enabled), ("all", "anchor", True))
        self.assertEqual(spec.chain_until, "2026-12-31T23:00:00Z")

        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000501",
                "chainID": "spec-observation-chain",
                "link": 1,
                "status": "pending",
                "anchor": " w:mon ",
                "anchor_mode": "ALL",
                "chainMax": "4",
            },
            source_query="test:recurrence-spec",
        )
        typed_spec = RecurrenceSpec.from_observation(observation)
        self.assertEqual(typed_spec.context.chain_id, "spec-observation-chain")
        self.assertEqual((typed_spec.anchor, typed_spec.chain_max), ("w:mon", 4))

        null_spec = RecurrenceSpec.from_observation(
            DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "00000000-0000-4000-8000-000000000506",
                    "status": "pending",
                    "chainID": "spec-null-chain",
                    "link": 1,
                    "anchor": "w:mon",
                    "anchor_file": "null",
                    "anchor_mode": "skip",
                },
                source_query="test:recurrence-spec",
            )
        )
        self.assertEqual(null_spec.anchor_file, "")
        self.assertEqual(null_spec.kind, "anchor")

        supplied = RecurrenceContext(chain_id="supplied")
        supplied_observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000502",
                "status": "pending",
                "chainID": "supplied",
                "link": 1,
                "anchor": "w:fri",
            },
            source_query="test:recurrence-spec",
        )
        self.assertIs(
            RecurrenceSpec.from_observation(supplied_observation, context=supplied).context,
            supplied,
        )

        conflicting_observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000504",
                "status": "pending",
                "chainID": "task-chain",
                "link": 1,
                "anchor": "w:fri",
            },
            source_query="test:recurrence-spec",
        )
        with self.assertRaisesRegex(ValueError, "Conflicting recurrence identities"):
            RecurrenceSpec.from_observation(conflicting_observation, context=supplied)

    def test_evaluator_loads_omit_file_without_text_rule(self) -> None:
        from types import SimpleNamespace

        from nautical_core import omit_files
        from nautical_core.recurrence_evaluator import RecurrenceEvaluator

        with TemporaryDirectory() as directory:
            Path(directory, "holidays.csv").write_text(
                "date,description\n2025-01-06,Holiday\n", encoding="utf-8"
            )
            runtime = SimpleNamespace(
                parse_dt_any=core.parse_dt_any,
                OMIT_FILE_DIR=directory,
                _import_sibling=lambda _name: omit_files,
            )
            with patch.object(RecurrenceEvaluator, "_core_module", return_value=runtime):
                evaluator = _evaluator_for_task(
                    {
                        "chainID": "omit-file-only",
                        "anchor": "w:mon",
                        "omit_file": "holidays.csv",
                    }
                )
                state = evaluator.omit_dnf

        self.assertEqual(state.dates, frozenset({date(2025, 1, 6)}))
        self.assertEqual(state.descriptions, {date(2025, 1, 6): "Holiday"})

    def test_evaluator_combines_and_caches_omit_expression_and_file(self) -> None:
        from types import SimpleNamespace

        from nautical_core import omit_files
        from nautical_core.recurrence_evaluator import RecurrenceEvaluator

        original_loader = omit_files.load_omit_file_data
        with TemporaryDirectory() as directory:
            Path(directory, "holidays.csv").write_text(
                "date,description\n2025-01-06,Holiday\n", encoding="utf-8"
            )
            runtime = SimpleNamespace(
                parse_dt_any=core.parse_dt_any,
                OMIT_FILE_DIR=directory,
                _import_sibling=lambda _name: omit_files,
            )
            with patch.object(RecurrenceEvaluator, "_core_module", return_value=runtime):
                with patch.object(
                    omit_files, "load_omit_file_data", wraps=original_loader
                ) as loader:
                    evaluator = _evaluator_for_task(
                        {
                            "chainID": "omit-file-combined",
                            "anchor": "w:mon",
                            "omit": "w:sun",
                            "omit_file": "holidays.csv",
                        }
                    )
                    state = evaluator.omit_dnf
                    self.assertIs(evaluator.omit_dnf, state)

        self.assertEqual(loader.call_count, 1)
        self.assertIn(date(2025, 1, 6), state.dates)
        with self.assertRaises((AttributeError, TypeError)):
            state.dnf[0] = ()

    def test_evaluation_session_is_identity_scoped_and_refreshes_changed_schedule(self) -> None:
        first_observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000512",
                "status": "pending",
                "link": 1,
                "chainID": "session-a",
                "anchor": "w:mon",
            },
            source_query="test:evaluation-session",
        )
        session = EvaluationSession.from_observation(
            first_observation,
            context=RecurrenceContext(chain_id="session-a"),
        )
        evaluator = session.evaluator

        self.assertIs(session.evaluator, evaluator)
        provider = session.get_or_create("provider", object)
        self.assertIs(session.get_or_create("provider", object), provider)

        other = RecurrenceSpec.from_observation(
            DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "00000000-0000-4000-8000-000000000513",
                    "status": "pending",
                    "link": 1,
                    "chainID": "session-b",
                    "anchor": "w:mon",
                },
                source_query="test:evaluation-session",
            )
        )
        self.assertFalse(session.matches(other))

        changed = RecurrenceSpec.from_observation(
            DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "00000000-0000-4000-8000-000000000514",
                    "status": "pending",
                    "link": 1,
                    "chainID": "session-a",
                    "anchor": "w:tue",
                },
                source_query="test:evaluation-session",
            )
        )
        self.assertTrue(session.refresh(changed))
        self.assertEqual(session.evaluator.spec.anchor, "w:tue")
        self.assertIsNotNone(session.next_outcome)
        self.assertIsNotNone(session.collect_after_cursor)

    def test_scheduler_service_exposes_typed_lookup_collection_and_preview(self) -> None:
        zone = ZoneInfo("Europe/Sofia")
        context = RecurrenceContext(chain_id="service-chain", timezone=zone)
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000502",
                "chainID": "service-chain",
                "link": 1,
                "status": "pending",
                "anchor": "w:mon@t=09:00",
            },
            source_query="test:scheduler-service",
        )
        service = SchedulerService.from_observation(observation, context=context)
        self.assertTrue(service.fingerprint)

        task = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "00000000-0000-4000-8000-000000000515",
                    "chainID": "service-task-chain",
                    "link": 1,
                    "status": "pending",
                    "anchor": "w:mon@t=09:00",
                },
                source_query="test:scheduler-service-task",
            )
        )
        task_service = SchedulerService.from_task(
            task,
            context=RecurrenceContext(chain_id="service-task-chain", timezone=zone),
        )
        self.assertTrue(task_service.fingerprint)

        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 2, 9, 0, tzinfo=zone), timezone=zone
        )
        self.assertIsInstance(service.next(cursor), FoundOccurrence)
        collected = service.collect(cursor, limit=2)
        self.assertIsInstance(collected, OccurrenceCollectionResult)
        self.assertEqual((len(collected), collected.status), (2, "found"))

        preview = service.preview(cursor.local_datetime, limit=1, timezone=zone)
        self.assertIsInstance(preview, OccurrenceCollectionResult)
        self.assertEqual(len(preview), 1)
        self.assertEqual(preview.to_dict()["status"], "found")

    def test_scheduler_trace_is_bounded_disabled_by_default_and_redacted(self) -> None:
        disabled = SchedulerTrace()
        disabled.record("selected", provider="anchor", term="w:mon")
        self.assertFalse(disabled.events)

        trace = SchedulerTrace(enabled=True, max_events=2)
        trace.record("proposed", provider="anchor_file", term="/private/calendar.csv")
        trace.record("selected", provider="anchor_file", term="calendar.csv")
        trace.record("selected", provider="anchor_file", term="third")
        payload = json.dumps(trace.summary(), ensure_ascii=False)

        self.assertEqual(trace.dropped, 1)
        self.assertNotIn("/private/calendar.csv", payload)
        self.assertIn("<redacted>", payload)

    def test_range_request_enforces_bounds_and_omission_policy(self) -> None:
        from datetime import timedelta

        zone = ZoneInfo("Europe/Sofia")
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 2, 9, 0, tzinfo=zone), timezone=zone
        )
        with self.assertRaises(ValueError):
            OccurrenceRangeRequest(
                cursor, end_local=cursor.local_datetime - timedelta(days=1)
            )
        with self.assertRaisesRegex(ValueError, "omission policy"):
            OccurrenceRangeRequest(cursor, omission_policy="unknown")

        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000519",
                "chainID": "range-chain",
                "link": 1,
                "status": "pending",
                "anchor": "w:mon@t=09:00",
            },
            source_query="test:occurrence-range",
        )
        service = SchedulerService.from_observation(
            observation,
            context=RecurrenceContext(chain_id="range-chain", timezone=zone),
        )
        included = service.collect_request(
            OccurrenceRangeRequest(cursor, omission_policy="include")
        )
        self.assertEqual(included.status, "found")
        self.assertEqual(included.request.omission_policy, "include")

    def test_range_request_preserves_omitted_occurrence_provenance(self) -> None:
        zone = ZoneInfo("Europe/Sofia")
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000520",
                "chainID": "omission-range-chain",
                "link": 1,
                "status": "pending",
                "anchor": "w:mon",
                "omit": "w:mon",
            },
            source_query="test:occurrence-range",
        )
        service = SchedulerService.from_observation(
            observation,
            context=RecurrenceContext(chain_id="omission-range-chain", timezone=zone),
        )
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 2, 23, 59, tzinfo=zone), timezone=zone
        )
        end = datetime(2026, 8, 31, 23, 59, tzinfo=zone)

        excluded = service.collect_request(
            OccurrenceRangeRequest(cursor, end_local=end, limit=4)
        )
        included = service.collect_request(
            OccurrenceRangeRequest(
                cursor, end_local=end, limit=4, omission_policy="include"
            )
        )
        reported = service.collect_request(
            OccurrenceRangeRequest(
                cursor, end_local=end, limit=4, omission_policy="report"
            )
        )

        self.assertFalse(excluded.occurrences)
        self.assertEqual(len(included.occurrences), 4)
        self.assertTrue(all(item.omitted for item in included))
        self.assertFalse(reported.occurrences)
        self.assertEqual(len(reported.omitted_occurrences), 4)

    def test_range_collection_wraps_unavailable_and_invalid_failures(self) -> None:
        from types import SimpleNamespace

        from nautical_core.occurrence_outcomes import InvalidOccurrence, UnavailableOccurrence

        zone = ZoneInfo("UTC")
        context = RecurrenceContext(chain_id="range-failure", timezone=zone)
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 1, tzinfo=zone), timezone=zone
        )
        request = OccurrenceRangeRequest(cursor, limit=1)

        class FailingSession:
            evaluator = SimpleNamespace(context=context, kind="anchor")

            def __init__(self, error: Exception) -> None:
                self.error = error

            def collect_after_cursor(self, *_args, **_kwargs):
                raise self.error

        unavailable = SchedulerService(
            FailingSession(LookupError("astronomy profile unavailable"))
        ).collect_request(request)
        self.assertIsInstance(unavailable.failure, UnavailableOccurrence)
        self.assertEqual(unavailable.status, "unavailable")

        invalid = SchedulerService(
            FailingSession(ValueError("malformed recurrence result"))
        ).collect_request(request)
        self.assertIsInstance(invalid.failure, InvalidOccurrence)
        self.assertEqual(invalid.status, "invalid")

    def test_service_collection_preserves_prefix_and_terminal_evidence(self) -> None:
        from types import SimpleNamespace

        from nautical_core.occurrence_provider import Occurrence, OccurrenceBatch
        from nautical_core.scheduler_models import OccurrenceSearchExhausted

        zone = ZoneInfo("UTC")
        cursor = OccurrenceCursor.strict_after(
            datetime(9998, 12, 30, 9, tzinfo=zone), timezone=zone
        )
        occurrence = Occurrence(
            datetime(9998, 12, 31, tzinfo=zone).date(),
            9,
            0,
            local_datetime=datetime(9998, 12, 31, 9, tzinfo=zone),
        )
        terminal = OccurrenceSearchExhausted(
            "finite test provider",
            reference="9999-12-31",
            limit=2,
            kind=OccurrenceSearchExhausted.DATE_LIMIT,
        )

        class FiniteSession:
            evaluator = SimpleNamespace(context=SimpleNamespace(timezone=zone), kind="finite")

            def collect_after_cursor(self, *_args, **_kwargs):
                return OccurrenceBatch((occurrence,), terminal=terminal)

            def collect_events_after_cursor(self, *_args, **_kwargs):
                return OccurrenceBatch((occurrence,), terminal=terminal)

        service = SchedulerService(FiniteSession())
        result = service.collect(cursor, limit=3)
        self.assertIs(result.terminal, terminal)
        self.assertEqual(result.occurrences, (occurrence,))
        ranged = service.collect_request(OccurrenceRangeRequest(cursor, limit=3))
        self.assertIs(ranged.terminal, terminal)

        class EmptyFiniteSession(FiniteSession):
            def collect_after_cursor(self, *_args, **_kwargs):
                return OccurrenceBatch((), terminal=terminal)

            def collect_events_after_cursor(self, *_args, **_kwargs):
                return OccurrenceBatch((), terminal=terminal)

        empty = SchedulerService(EmptyFiniteSession()).collect(cursor, limit=1)
        self.assertEqual(empty.status, "exhausted")

    def test_evaluator_owns_normalized_recurrence_context_without_io(self) -> None:
        from datetime import date, timezone

        context = RecurrenceContext(
            chain_id="evaluator-chain",
            timezone=ZoneInfo("America/New_York"),
            anchor_file_dir="/tmp/evaluator-anchor-files",
        )
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000521",
                "chainID": "evaluator-chain",
                "link": 1,
                "status": "pending",
                "anchor": "w:mon@t=02:30",
                "anchor_mode": "SKIP",
                "chainMax": "4",
            },
            source_query="test:recurrence-evaluator",
        )
        from nautical_core.recurrence_evaluator import RecurrenceEvaluator

        evaluator = RecurrenceEvaluator.from_observation(observation, context=context)
        self.assertEqual(evaluator.chain_id, "evaluator-chain")
        self.assertEqual(evaluator.seed_base, "evaluator-chain")
        self.assertEqual(evaluator.kind, "anchor")
        self.assertTrue(evaluator.enabled)
        self.assertEqual(evaluator.spec.anchor_mode, "skip")
        self.assertEqual(evaluator.spec.chain_max, 4)

        shifted = evaluator.build_local_datetime(date(2025, 3, 9), (2, 30))
        self.assertEqual((evaluator.to_local(shifted).hour, evaluator.to_local(shifted).minute), (3, 30))
        self.assertEqual(
            evaluator.utc_to_local_naive(shifted), datetime(2025, 3, 9, 3, 30)
        )

        parsed_observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000522",
                "chainID": "parsed-chain",
                "link": 1,
                "status": "pending",
                "anchor": "w:mon@t=02:30",
                "omit": "w:sun",
                "anchor_mode": "ALL",
                "chainMax": "4",
                "chainUntil": "2025-12-31T23:00:00Z",
            },
            source_query="test:recurrence-evaluator",
        )
        parsed = RecurrenceEvaluator.from_observation(
            parsed_observation,
            context=RecurrenceContext(chain_id="parsed-chain", timezone=timezone.utc),
        )
        self.assertEqual(parsed.anchor_mode, "all")
        anchor_dnf = parsed.anchor_dnf
        omit_dnf = parsed.omit_dnf
        self.assertEqual((len(anchor_dnf), len(anchor_dnf[0])), (1, 1))
        self.assertEqual(len(omit_dnf), 1)
        self.assertIs(parsed.anchor_dnf, anchor_dnf)
        self.assertIs(parsed.omit_dnf, omit_dnf)

        with self.assertRaises(TypeError):
            anchor_dnf[0][0]["spec"] = "corrupted"
        with self.assertRaises((AttributeError, TypeError)):
            anchor_dnf.clear()
        self.assertEqual(parsed.limits.chain_until, datetime(2025, 12, 31, 23, tzinfo=timezone.utc))
        self.assertTrue(parsed.limits_allow(datetime(2025, 12, 31, 22), 4))
        self.assertFalse(parsed.limits_allow(datetime(2026, 1, 1), 4))
        self.assertFalse(parsed.limits_allow(datetime(2025, 12, 31, 22), 5))

    def test_evaluator_projects_cp_and_reuses_its_file_provider(self) -> None:
        cp_evaluator = _evaluator_for_task(
            {"chainID": "cp-chain", "cp": "1d,rand(2d..3d)"}
        )
        cp_tokens = cp_evaluator.cp_tokens
        self.assertEqual(len(cp_tokens or ()), 2)
        with self.assertRaises(TypeError):
            cp_tokens[0]["kind"] = "corrupted"
        self.assertEqual(cp_evaluator.cp_interval_for_link(1), timedelta(days=1))
        random_interval = cp_evaluator.cp_interval_for_link(2)
        self.assertGreaterEqual(random_interval, timedelta(days=2))
        self.assertLessEqual(random_interval, timedelta(days=3))
        self.assertEqual(
            cp_evaluator.project_cp(datetime(2025, 1, 1, tzinfo=timezone.utc), 1),
            datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        with self.assertRaisesRegex(ValueError, "positive link"):
            cp_evaluator.cp_interval_for_link(0)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            cp_evaluator.project_cp(datetime(2025, 1, 1), 1)

        file_evaluator = _evaluator_for_task(
            {"chainID": "file-chain", "anchor_file": "events.csv"}
        )
        provider = file_evaluator._anchor_file_provider_for((9, 0))
        self.assertIs(file_evaluator._anchor_file_provider_for((9, 0)), provider)

    def test_evaluator_stream_modes_and_guards_preserve_occurrence_semantics(self) -> None:
        parsed = _evaluator_for_task(
            {
                "chainID": "parsed-stream",
                "anchor": "w:mon@t=02:30",
                "omit": "w:sun",
                "anchor_mode": "ALL",
                "chainMax": "4",
                "chainUntil": "2025-12-31T23:00:00Z",
            }
        )
        stream = parsed.collect_after(datetime(2025, 1, 1, tzinfo=timezone.utc), limit=2)
        self.assertEqual(
            [item.local_datetime for item in stream],
            [
                datetime(2025, 1, 6, 2, 30, tzinfo=timezone.utc),
                datetime(2025, 1, 13, 2, 30, tzinfo=timezone.utc),
            ],
        )

        event_evaluator = _evaluator_for_task(
            {"chainID": "event-chain", "anchor": "w:mon..tue", "omit": "w:mon"}
        )
        start = datetime(2025, 1, 5, tzinfo=timezone.utc)
        omitted_event = event_evaluator.next_event_after(start, include_omitted=True)
        self.assertTrue(omitted_event.omitted)
        self.assertEqual(omitted_event.local_datetime, datetime(2025, 1, 6, 9, tzinfo=timezone.utc))
        included_event = event_evaluator.next_event_after(start)
        self.assertFalse(included_event.omitted)
        self.assertEqual(included_event.local_datetime, datetime(2025, 1, 7, 9, tzinfo=timezone.utc))
        ranged_events = event_evaluator.events_between(
            start,
            datetime(2025, 1, 20, tzinfo=timezone.utc),
            limit=1,
            inclusive=False,
            include_omitted=True,
        )
        self.assertEqual(
            [item.local_datetime for item in ranged_events],
            [datetime(2025, 1, 6, 9, tzinfo=timezone.utc), datetime(2025, 1, 7, 9, tzinfo=timezone.utc)],
        )

        mode_evaluator = _evaluator_for_task({"chainID": "mode-chain", "anchor": "w:mon"})
        mode_common = {
            "due_local": datetime(2025, 1, 1, tzinfo=timezone.utc),
            "end_local": datetime(2025, 1, 3, tzinfo=timezone.utc),
        }
        self.assertEqual(mode_evaluator.select_mode("all", **mode_common).basis, "after_due")
        self.assertEqual(mode_evaluator.select_mode("skip", **mode_common).basis, "after_end")
        flex = mode_evaluator.select_mode("flex", **mode_common)
        self.assertEqual((flex.basis, flex.missed_count), ("flex", 0))
        for result, basis in (
            (mode_evaluator.select_mode("all", **mode_common), "after_due"),
            (mode_evaluator.select_mode("skip", **mode_common), "after_end"),
            (flex, "flex"),
        ):
            self.assertEqual(result.selected_occurrence, datetime(2025, 1, 6, 9, tzinfo=timezone.utc))
            self.assertEqual(result.basis, basis)
        self.assertEqual(mode_evaluator.select_mode("all", **mode_common).source, "anchor")

        with self.assertRaisesRegex(ValueError, "range iteration limit"):
            event_evaluator.events_between(
                start,
                datetime(2025, 1, 10, tzinfo=timezone.utc),
                limit=2,
                max_iterations=1,
            )

        limited = _evaluator_for_task(
            {
                "chainID": "limited-chain",
                "anchor": "w:mon",
                "chainMax": "2",
                "chainUntil": "2025-01-10T00:00:00Z",
            }
        )
        self.assertTrue(limited.limits_allow(datetime(2025, 1, 9), 2))
        self.assertFalse(limited.limits_allow(datetime(2025, 1, 11), 2))
        self.assertFalse(limited.limits_allow(datetime(2025, 1, 9), 3))

        astronomy = _evaluator_for_task({"chainID": "astronomy-chain", "anchor": "moon:full"})
        with self.assertRaisesRegex(ValueError, "Fallback occurrence time"):
            astronomy.collect_after(
                datetime(2025, 1, 1), limit=1, fallback_hhmm=(24, 0)
            )

        import nautical_core.astronomy as astronomy_module
        from unittest.mock import patch

        def resolve_sunrise(_event, day, config=None):
            return datetime(day.year, day.month, day.day, 6, 30, tzinfo=timezone.utc)

        with patch.object(astronomy_module, "resolve_event", resolve_sunrise):
            astronomical_time = _evaluator_for_task(
                {"chainID": "evaluator-astronomy-time", "anchor": "w:mon@t=sunrise"},
                astronomy_config={"default_location": "test"},
            )
            next_event = astronomical_time.next_after(
                datetime(2025, 1, 6, 7, 0, tzinfo=timezone.utc)
            )
        self.assertEqual(
            (next_event.local_datetime.date(), next_event.local_datetime.hour, next_event.local_datetime.minute),
            (datetime(2025, 1, 13).date(), 6, 30),
        )

        with self.assertRaisesRegex(ValueError, "anchor_mode"):
            _evaluator_for_task(
                {"chainID": "invalid-mode", "anchor": "w:mon", "anchor_mode": "bad"}
            )
        from nautical_core.recurrence_evaluator import RecurrenceEvaluator

        missing_chain = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000523",
                "link": 1,
                "status": "pending",
                "anchor": "w:mon",
            },
            source_query="test:recurrence-evaluator",
        )
        with self.assertRaisesRegex(ValueError, "chainID"):
            RecurrenceEvaluator.from_observation(missing_chain)

    def test_recurrence_sessions_are_isolated_from_collection_order(self) -> None:
        import random

        zone = ZoneInfo("Europe/Sofia")
        cases = (
            ("w:mon", "shuffle-weekly"),
            ("m/2:15", "shuffle-monthly"),
            ("y:02-29", "shuffle-leap"),
            ("w:mon | w:fri", "shuffle-or"),
            ("m:rand", "shuffle-random"),
        )
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 1, 1, tzinfo=zone), timezone=zone
        )

        def collect(anchor: str, chain_id: str) -> tuple[datetime | None, ...]:
            observation = DEFAULT_TASK_CODEC.decode_row(
                {
                    "uuid": "00000000-0000-4000-8000-000000000001",
                    "chainID": chain_id,
                    "link": 1,
                    "status": "pending",
                    "anchor": anchor,
                },
                source_query="test:scheduler-session-order",
            )
            service = SchedulerService.from_observation(
                observation,
                context=RecurrenceContext(chain_id=chain_id, timezone=zone),
            )
            return tuple(
                item.local_datetime
                for item in service.collect(cursor, limit=3, max_iterations=2048)
            )

        baseline = {anchor: collect(anchor, chain_id) for anchor, chain_id in cases}
        for seed in (11, 29, 47):
            order = list(cases)
            random.Random(seed).shuffle(order)
            for anchor, chain_id in order:
                with self.subTest(seed=seed, anchor=anchor):
                    self.assertEqual(collect(anchor, chain_id), baseline[anchor])


if __name__ == "__main__":
    unittest.main()
