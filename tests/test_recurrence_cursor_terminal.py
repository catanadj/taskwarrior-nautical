from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch, PropertyMock
from zoneinfo import ZoneInfo

from nautical_core.occurrence_provider import Occurrence, OccurrenceBatch
from nautical_core.recurrence_evaluator import RecurrenceEvaluator
from nautical_core.recurrence_context import RecurrenceContext
from nautical_core.scheduler_cursor import OccurrenceCursor
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.task_codec import DEFAULT_TASK_CODEC


class CursorTerminalEvidenceTests(unittest.TestCase):
    def test_anchor_file_all_mode_orders_repeated_hour_by_instant(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000962",
                "status": "pending",
                "link": 1,
                "chainID": "dst-fold",
                "anchor_file": "fold.csv",
                "anchor_mode": "all",
            },
            source_query="test:dst-fold-mode",
        )
        evaluator = RecurrenceEvaluator.from_observation(
            observation,
            context=RecurrenceContext(chain_id="dst-fold", timezone=zone),
        )
        due = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        end = datetime(2026, 10, 25, 3, 30, tzinfo=zone, fold=1)
        first_fold = datetime(2026, 10, 25, 3, 20, tzinfo=zone, fold=0)
        second_fold = datetime(2026, 10, 25, 3, 20, tzinfo=zone, fold=1)
        after_end = datetime(2026, 10, 25, 3, 45, tzinfo=zone, fold=1)

        class FoldProvider:
            def next_after(self, after_local, *, build_local_datetime, to_local):
                for value in (first_fold, second_fold, after_end):
                    if value.astimezone(timezone.utc) > after_local.astimezone(timezone.utc):
                        return Occurrence(
                            day=value.date(),
                            hour=value.hour,
                            minute=value.minute,
                            source="anchor_file",
                            local_datetime=value,
                        )
                return None

        result = evaluator.select_mode(
            "all",
            due_local=due,
            end_local=end,
            due_explicit=True,
            fallback_hhmm=(3, 15),
            default_seed_date=due.date(),
            anchor_file_provider=FoldProvider(),
        )

        self.assertIs(result.selected_occurrence, second_fold)
        self.assertEqual(result.source, "anchor_file")
        self.assertEqual(result.missed_count, 1)

    def test_cursor_lookup_policy_and_timezone_are_explicit(self) -> None:
        from zoneinfo import ZoneInfo

        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.task_codec import DEFAULT_TASK_CODEC

        zone = ZoneInfo("Europe/Sofia")
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000516",
                "status": "pending",
                "link": 1,
                "chainID": "cursor-chain",
                "anchor": "w:mon@t=09:00",
            },
            source_query="test:occurrence-cursor",
        )
        evaluator = RecurrenceEvaluator.from_observation(
            observation,
            context=RecurrenceContext(chain_id="cursor-chain", timezone=zone),
        )
        instant = datetime(2026, 8, 3, 9, 0, tzinfo=zone)
        strict = OccurrenceCursor.strict_after(instant, timezone=zone)
        inclusive = OccurrenceCursor.inclusive_at(instant, timezone=zone)

        self.assertFalse(strict.inclusive)
        self.assertTrue(inclusive.inclusive)
        self.assertIsNotNone(evaluator.next_after_cursor(strict))
        self.assertEqual(len(evaluator.collect_after_cursor(strict, limit=1)), 1)
        with self.assertRaisesRegex(ValueError, "timezone"):
            evaluator.next_after_cursor(
                OccurrenceCursor.strict_after(instant, timezone=ZoneInfo("UTC"))
            )

    def test_strict_weekday_cursor_returns_first_adjacent_weekday(self) -> None:
        from zoneinfo import ZoneInfo

        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.task_codec import DEFAULT_TASK_CODEC

        zone = ZoneInfo("Europe/Sofia")
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000517",
                "status": "pending",
                "link": 1,
                "chainID": "cursor-weekday",
                "anchor": "w:mon..fri",
            },
            source_query="test:occurrence-cursor",
        )
        evaluator = RecurrenceEvaluator.from_observation(
            observation,
            context=RecurrenceContext(chain_id="cursor-weekday", timezone=zone),
        )

        result = evaluator.next_after_cursor(
            OccurrenceCursor.strict_after(
                datetime(2026, 8, 7, 9, 0, tzinfo=zone), timezone=zone
            )
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.day.isoformat(), "2026-08-10")

    def test_omission_composition_uses_one_canonical_state_shape(self) -> None:
        from nautical_core.anchor_omit import OmitState, combine_omit_state

        state = combine_omit_state(omit_dnf=[[{"w": "mon"}]], omit_dates={"2026-01-02"})
        self.assertIsInstance(state, OmitState)
        self.assertEqual(state.dates, frozenset({"2026-01-02"}))

    def test_collect_after_cursor_preserves_terminal_evidence(self) -> None:
        evaluator = object.__new__(RecurrenceEvaluator)
        terminal = OccurrenceSearchExhausted("date limit", kind=OccurrenceSearchExhausted.DATE_LIMIT)
        batch = OccurrenceBatch([], terminal=terminal)
        cursor = OccurrenceCursor(datetime(2026, 1, 1), inclusive=False, date_limit=date(2026, 1, 31))
        with patch.object(RecurrenceEvaluator, "context", new_callable=PropertyMock, return_value=SimpleNamespace(timezone=None)), patch.object(RecurrenceEvaluator, "collect_after", return_value=batch) as collect:
            result = evaluator.collect_after_cursor(cursor, limit=1)
        self.assertIs(result.terminal, terminal)
        self.assertIs(collect.call_args.args[0], cursor)

    def test_event_range_preserves_valid_prefix_before_terminal_exhaustion(self) -> None:
        from unittest.mock import patch
        from zoneinfo import ZoneInfo

        from nautical_core.occurrence_provider import Occurrence
        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.task_codec import DEFAULT_TASK_CODEC

        zone = ZoneInfo("UTC")
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000524",
                "status": "pending",
                "link": 1,
                "chainID": "terminal-events",
                "anchor": "w:mon",
            },
            source_query="test:occurrence-terminal",
        )
        evaluator = RecurrenceEvaluator.from_observation(
            observation,
            context=RecurrenceContext(chain_id="terminal-events", timezone=zone),
        )
        first_local = datetime(2026, 1, 5, 9, 0, tzinfo=zone)
        terminal = OccurrenceSearchExhausted(
            "terminal event stream", reference=date(9999, 12, 31), limit=1
        )
        calls = 0

        def next_event(_self, _cursor, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return Occurrence(
                    first_local.date(),
                    first_local.hour,
                    first_local.minute,
                    local_datetime=first_local,
                )
            raise terminal

        with patch.object(RecurrenceEvaluator, "next_event_after", next_event):
            events = evaluator.events_between(
                datetime(2026, 1, 1, 9, 0, tzinfo=zone),
                datetime(9999, 12, 31, 9, 0, tzinfo=zone),
                limit=3,
            )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].local_datetime, first_local)
        self.assertIs(events.terminal, terminal)


class TypedOccurrenceOutcomeContractTests(unittest.TestCase):
    def test_evaluator_exposes_found_absent_and_invalid_outcomes(self) -> None:
        from zoneinfo import ZoneInfo

        from nautical_core.occurrence_outcomes import (
            AbsentOccurrence,
            FoundOccurrence,
            InvalidOccurrence,
            outcome_from_occurrence,
        )
        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.task_codec import DEFAULT_TASK_CODEC

        zone = ZoneInfo("Europe/Sofia")
        observation = DEFAULT_TASK_CODEC.decode_row(
            {
                "uuid": "00000000-0000-4000-8000-000000000518",
                "status": "pending",
                "link": 1,
                "chainID": "outcome-chain",
                "anchor": "w:mon@t=09:00",
            },
            source_query="test:occurrence-outcome",
        )
        evaluator = RecurrenceEvaluator.from_observation(
            observation,
            context=RecurrenceContext(chain_id="outcome-chain", timezone=zone),
        )
        cursor = OccurrenceCursor.strict_after(
            datetime(2026, 8, 2, 9, 0, tzinfo=zone), timezone=zone
        )

        self.assertIsInstance(evaluator.next_outcome(cursor), FoundOccurrence)
        self.assertIsInstance(outcome_from_occurrence(None), AbsentOccurrence)
        self.assertIsInstance(
            evaluator.next_outcome(cursor, max_file_skips=0), InvalidOccurrence
        )
        self.assertIsInstance(evaluator.next_outcome("bad-cursor"), InvalidOccurrence)

    def test_exhaustion_outcome_retains_terminal_bounds_and_differs_from_absence(self) -> None:
        from nautical_core.occurrence_outcomes import AbsentOccurrence, ExhaustedOccurrence

        error = OccurrenceSearchExhausted(
            "yearly atom scheduling",
            reference="9999-01-01",
            limit=12,
            kind=OccurrenceSearchExhausted.DATE_LIMIT,
        )
        exhausted = ExhaustedOccurrence(error)

        self.assertEqual(exhausted.terminal_evidence["kind"], "date_limit")
        self.assertEqual(exhausted.to_dict()["limit"], 12)
        self.assertEqual(AbsentOccurrence().status, "absent")
        self.assertNotEqual(exhausted.status, AbsentOccurrence().status)

    def test_mutation_candidate_rejects_absent_outcome(self) -> None:
        from nautical_core.occurrence_outcomes import AbsentOccurrence, mutation_candidate

        with self.assertRaisesRegex(RuntimeError, "requires a found occurrence"):
            mutation_candidate(AbsentOccurrence())

    def test_presentation_summary_is_compact_for_absent_outcome(self) -> None:
        from nautical_core.occurrence_outcomes import AbsentOccurrence, presentation_summary

        self.assertEqual(presentation_summary(AbsentOccurrence("no match")), "absent: no match")


if __name__ == "__main__":
    unittest.main()
