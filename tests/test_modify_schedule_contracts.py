from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, get_type_hints
from unittest.mock import patch
from zoneinfo import ZoneInfo

import nautical_core as core
import nautical_core.modify_completion_compute as modify_completion_compute
import nautical_core.modify_runtime as modify_runtime
import nautical_core.modify_schedule_effects as modify_schedule_effects
import nautical_core.modify_models as modify_models
from nautical_core.modify_models import AnchorFileProviderFactory
import nautical_core.timezone_facade as timezone_facade
from nautical_core.add_anchor_compute import anchor_next_occurrence_after_local_dt
from nautical_core.recurrence_evaluator import RecurrenceEvaluator
from nautical_core.task_datetime import parser_for_core
from nautical_core.timeutil import compare_datetimes


class ModifyScheduleContractTests(unittest.TestCase):
    def test_completion_projection_adapters_use_typed_temporal_inputs(self) -> None:
        from nautical_core.parsing.parser_models import AnchorDNF

        expected = {
            modify_schedule_effects.estimate_cp_final_by_max: {
                "next_due_utc": datetime | None,
            },
            modify_schedule_effects.cap_from_until_cp: {
                "next_due_utc": datetime | None,
            },
            modify_schedule_effects.estimate_anchor_final_by_max: {
                "next_due_utc": datetime | None,
                "dnf": AnchorDNF | None,
            },
            modify_schedule_effects.cap_from_until_anchor: {
                "next_due_utc": datetime | None,
                "dnf": AnchorDNF | None,
            },
        }
        for function, expected_parameters in expected.items():
            annotations = get_type_hints(function)
            with self.subTest(function=function.__name__):
                for parameter, expected_type in expected_parameters.items():
                    self.assertEqual(annotations[parameter], expected_type)

    def _scheduler_ports(
        self, state: modify_runtime.ModifyRuntimeState
    ) -> modify_schedule_effects.SchedulerPorts:
        return modify_schedule_effects.SchedulerPorts(
            service_for_task=lambda task: modify_runtime.scheduler_service_for_task(
                task,
                state=state,
                core=core,
                recurrence_seed_base=modify_schedule_effects.recurrence_seed_base,
            )
        )

    def test_scheduler_ports_use_task_service_factory_contract(self) -> None:
        annotations = get_type_hints(modify_schedule_effects.SchedulerPorts)
        self.assertEqual(
            tuple(annotations),
            ("service_for_task",),
        )
        self.assertIs(
            annotations["service_for_task"],
            modify_schedule_effects.SchedulerServiceForTask,
        )

    def test_schedule_projection_helpers_have_domain_result_types(self) -> None:
        self.assertEqual(
            get_type_hints(modify_schedule_effects.anchor_included_occurrences)[
                "return"
            ],
            list[datetime],
        )
        self.assertEqual(
            get_type_hints(modify_schedule_effects.estimate_cp_final_by_max)["return"],
            datetime | None,
        )
        self.assertEqual(
            get_type_hints(modify_schedule_effects.cap_from_until_cp)["return"],
            tuple[int | None, datetime | None],
        )

    def test_sequence_interval_port_uses_named_protocol(self) -> None:
        self.assertIs(
            get_type_hints(modify_schedule_effects.SequencePorts)["sequence_interval"],
            modify_schedule_effects.SequenceIntervalForToken,
        )

    def test_occurrence_port_uses_named_protocol(self) -> None:
        self.assertIs(
            get_type_hints(modify_schedule_effects.OccurrencePorts)["next_occurrence"],
            modify_schedule_effects.NextOccurrenceAfterLocalDateTime,
        )

    def test_completion_ports_type_shared_runtime_callbacks(self) -> None:
        for ports_type in (
            modify_schedule_effects.CPCompletionPorts,
            modify_schedule_effects.AnchorCompletionPorts,
        ):
            annotations = get_type_hints(ports_type)
            self.assertIs(annotations["parse_datetime"], modify_models.DatetimeParserCallback)
            self.assertIs(annotations["coerce_int"], modify_models.CoerceIntCallback)
            self.assertIs(annotations["diagnostic"], modify_models.DiagnosticCallback)

        self.assertIs(
            get_type_hints(modify_schedule_effects.CPCompletionPorts)["compute"],
            modify_schedule_effects.CPCompletionCompute,
        )
        self.assertIs(
            get_type_hints(modify_schedule_effects.AnchorCompletionPorts)["compute"],
            modify_schedule_effects.AnchorCompletionCompute,
        )

        self.assertEqual(
            get_type_hints(modify_schedule_effects.CPCompletionPorts)[
                "parse_cp_sequence_tokens"
            ],
            Callable[[str], list[dict[str, Any]] | None],
        )

    def test_anchor_completion_ports_type_projection_callbacks(self) -> None:
        annotations = get_type_hints(modify_schedule_effects.AnchorCompletionPorts)
        self.assertIs(annotations["safe_parse_datetime"], modify_models.SafeParseDatetimeCallback)
        self.assertEqual(
            annotations["to_local_cached"],
            Callable[[datetime], datetime],
        )
        self.assertEqual(
            annotations["anchor_file_fallback_hhmm"],
            Callable[[dict[str, Any], datetime], tuple[int, int]],
        )
        self.assertEqual(
            annotations["omit_dnf_from_parent"],
            Callable[[dict[str, Any]], tuple[str, Any]],
        )
        self.assertEqual(
            annotations["compare_datetimes"],
            Callable[[datetime, datetime], int],
        )
        self.assertIs(
            annotations["anchor_file_provider_for"],
            AnchorFileProviderFactory,
        )

    def test_schedule_ports_have_concrete_callback_signatures(self) -> None:
        annotations = get_type_hints(modify_schedule_effects.SchedulePorts)
        self.assertEqual(annotations["to_local"], Callable[[datetime], datetime])
        self.assertEqual(
            annotations["build_local_datetime"],
            Callable[[date, tuple[int, int]], datetime],
        )

    def test_on_modify_reuses_task_scoped_evaluator_and_scheduler_binding(self) -> None:
        task = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "description": "task-scoped evaluator fixture",
            "chainID": "session-chain",
            "status": "pending",
            "link": 1,
            "anchor": "w:mon@t=09:00",
            "anchor_mode": "skip",
            "due": "20250106T090000Z",
            "end": "20250106T100000Z",
        }
        state = modify_runtime.new_runtime_state()
        ports = self._scheduler_ports(state)
        with (
            patch.object(timezone_facade, "_local_timezone", timezone.utc),
            patch.object(
                core,
                "business_calendar_for_task",
                return_value=core.DEFAULT_BUSINESS_CALENDAR,
            ),
        ):
            evaluator_for_task, _service_for_task = modify_schedule_effects.scheduler_callbacks(ports)
            first = evaluator_for_task(task)
            second = evaluator_for_task(dict(task))
            binding_a = first._get_cached("scheduler_binding", first._build_scheduler_binding)
            binding_b = first._get_cached("scheduler_binding", first._build_scheduler_binding)

        self.assertIs(first, second)
        self.assertIs(binding_a, binding_b)

    def test_overnight_window_advances_past_second_dst_fold(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        dnf = core.validate_anchor_expr_strict("w:sat@t=22:20..03:20/6")
        cursor = datetime(2026, 10, 25, 3, 15, tzinfo=zone, fold=1)
        def next_occurrence(
            expression: Any,
            after: datetime,
            *,
            fallback_hhmm: tuple[int, int],
            interval_seed: date | None,
            seed_base: str,
            omit_dnf: Any,
            default_seed_date: date | None,
        ) -> datetime | None:
            return anchor_next_occurrence_after_local_dt(
                expression,
                after,
                fallback_hhmm=fallback_hhmm,
                interval_seed=interval_seed,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                default_seed_date=default_seed_date,
                core=core,
            )

        ports = modify_schedule_effects.OccurrencePorts(next_occurrence)

        with patch.object(timezone_facade, "_local_timezone", zone):
            result = modify_schedule_effects.next_occurrence_after_local_dt(
                ports,
                dnf,
                cursor,
                default_seed_date=date(2026, 10, 24),
                seed_base="dst-overnight-second-fold",
                fallback_hhmm=(22, 20),
            )

        self.assertEqual(result.date(), date(2026, 10, 31))
        self.assertEqual((result.hour, result.minute), (22, 20))

    def test_cp_chain_max_estimate_advances_through_sequence_intervals(self) -> None:
        ports = modify_schedule_effects.CPCompletionPorts(
            compute=modify_completion_compute,
            parse_datetime=core.parse_dt_any,
            coerce_int=core.coerce_int,
            parse_cp_sequence_tokens=core.parse_cp_sequence_tokens,
            sequence=modify_schedule_effects.SequencePorts(
                core.cp_sequence_interval_for_token
            ),
            schedule=modify_schedule_effects.SchedulePorts(
                core.to_local, core.build_local_datetime
            ),
            max_iterations=100,
            diagnostic=lambda _message: None,
        )
        task = {
            "cp": "3d,20d,7d",
            "link": 1,
            "chainMax": 4,
            "chainID": "chainmax-sequence-test",
        }
        next_due = core.build_local_datetime(date(2026, 1, 4), (9, 0)).astimezone(
            timezone.utc
        )

        final_due = modify_schedule_effects.estimate_cp_final_by_max(
            ports, task, next_due
        )

        final_local = core.to_local(final_due)
        self.assertEqual(final_local.date(), date(2026, 1, 31))
        self.assertEqual((final_local.hour, final_local.minute), (9, 0))

    def test_cp_chain_max_forecast_stops_at_iteration_budget(self) -> None:
        diagnostics: list[str] = []
        ports = modify_schedule_effects.CPCompletionPorts(
            compute=modify_completion_compute,
            parse_datetime=core.parse_dt_any,
            coerce_int=core.coerce_int,
            parse_cp_sequence_tokens=core.parse_cp_sequence_tokens,
            sequence=modify_schedule_effects.SequencePorts(
                core.cp_sequence_interval_for_token
            ),
            schedule=modify_schedule_effects.SchedulePorts(
                core.to_local, core.build_local_datetime
            ),
            max_iterations=3,
            diagnostic=diagnostics.append,
        )

        final_due = modify_schedule_effects.estimate_cp_final_by_max(
            ports,
            {"cp": "1d", "link": 1, "chainMax": 5000},
            datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
        )

        self.assertIsNone(final_due)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("final date is unavailable", diagnostics[0])

    def test_anchor_chain_max_forecast_stops_at_iteration_budget(self) -> None:
        diagnostics = []
        calls = []

        def next_daily(_self, _dnf, value, **_kwargs):
            calls.append(value)
            return value + timedelta(days=1)

        ports = replace(
            self._completion_ports(max_iterations=3), diagnostic=diagnostics.append
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000118",
            "description": "chain max bound",
            "status": "completed",
            "anchor": "w:mon",
            "link": 1,
            "chainMax": 5000,
            "chainID": "bound-test",
            "due": "2026-01-01T09:00:00Z",
        }

        with patch.object(
            RecurrenceEvaluator,
            "_default_next_occurrence_after_local_dt",
            next_daily,
        ):
            final_due = modify_schedule_effects.estimate_anchor_final_by_max(
                ports,
                task,
                datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
                None,
            )

        self.assertIsNone(final_due)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(diagnostics), 1)
        self.assertIn("final date is unavailable", diagnostics[0])

    def test_runtime_anchor_provider_cache_keys_effective_fallback(self) -> None:
        with TemporaryDirectory() as directory:
            Path(directory, "calendar.csv").write_text(
                "date\n2026-08-03\n", encoding="utf-8"
            )
            state = modify_runtime.new_runtime_state()
            with patch.object(core, "ANCHOR_FILE_DIR", directory):
                morning = modify_runtime.anchor_file_provider_for(
                    "calendar.csv",
                    fallback_hhmm=(9, 0),
                    seed_base="fallback-provider-test",
                    state=state,
                    core=core,
                )
                same_morning = modify_runtime.anchor_file_provider_for(
                    "calendar.csv",
                    fallback_hhmm=(9, 0),
                    seed_base="fallback-provider-test",
                    state=state,
                    core=core,
                )
                early = modify_runtime.anchor_file_provider_for(
                    "calendar.csv",
                    fallback_hhmm=(6, 0),
                    seed_base="fallback-provider-test",
                    state=state,
                    core=core,
                )

        self.assertIs(morning, same_morning)
        self.assertIsNot(morning, early)
        self.assertEqual(morning.fallback_hhmm, (9, 0))
        self.assertEqual(early.fallback_hhmm, (6, 0))

    def _completion_ports(self, *, max_iterations: int, anchor_file_provider_for=None):
        parser = parser_for_core(core)
        scheduler = self._scheduler_ports(modify_runtime.new_runtime_state())
        return modify_schedule_effects.AnchorCompletionPorts(
            compute=modify_completion_compute,
            parse_datetime=core.parse_dt_any,
            coerce_int=core.coerce_int,
            scheduler=scheduler,
            to_local_cached=core.to_local,
            safe_parse_datetime=parser.parse,
            anchor_file_fallback_hhmm=lambda _task, _next: (9, 0),
            omit_dnf_from_parent=lambda _task: ("", None),
            anchor_file_provider_for=(
                anchor_file_provider_for
                if anchor_file_provider_for is not None
                else lambda *_args, **_kwargs: None
            ),
            compare_datetimes=compare_datetimes,
            max_iterations=max_iterations,
            diagnostic=lambda _message: None,
        )

    def test_included_occurrence_collection_rejects_non_advancing_scheduler(self) -> None:
        ports = modify_schedule_effects.AnchorOccurrencePorts(
            self._scheduler_ports(modify_runtime.new_runtime_state())
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000960",
            "status": "pending",
            "link": 1,
            "anchor": "w:mon",
            "chainID": "provider-guard-test",
        }

        with patch.object(
            RecurrenceEvaluator,
            "_default_next_occurrence_after_local_dt",
            lambda _self, _dnf, value, **_kwargs: value,
        ):
            with self.assertRaisesRegex(ValueError, "non-advancing"):
                modify_schedule_effects.anchor_included_occurrences(
                    ports,
                    task,
                    after_local_dt=datetime(2026, 8, 3, 9, 0),
                    inclusive=False,
                    limit=1,
                    fallback_hhmm=(9, 0),
                    omit_dnf=None,
                    seed_base="provider-guard-test",
                    default_seed_date=date(2026, 8, 3),
                    dnf=[[{"kind": "w", "value": "mon", "mods": {}}]],
                )

    def test_until_projection_fails_closed_at_iteration_limit(self) -> None:
        ports = self._completion_ports(max_iterations=3)
        task = {
            "uuid": "00000000-0000-4000-8000-000000000962",
            "status": "pending",
            "chainID": "projection-limit",
            "anchor": "w:mon",
            "link": 1,
            "due": "20260801T090000Z",
            "chainUntil": "20350801T090000Z",
        }

        with patch.object(
            RecurrenceEvaluator,
            "_default_next_occurrence_after_local_dt",
            lambda _self, _dnf, value, **_kwargs: value + timedelta(days=1),
        ):
            with self.assertRaisesRegex(ValueError, "projection exceeded"):
                modify_schedule_effects.cap_from_until_anchor(
                    ports,
                    task,
                    datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
                    [[{"kind": "w", "value": "mon", "mods": {}}]],
                )

    def test_until_projection_builds_and_reuses_one_anchor_file_provider(self) -> None:
        built = []
        providers = []

        def build_provider(anchor_file, *, fallback_hhmm, seed_base):
            provider = object()
            built.append((anchor_file, fallback_hhmm, seed_base, provider))
            return provider

        def included(_ports, _task, *, after_local_dt, anchor_file_provider=None, **_kwargs):
            providers.append(anchor_file_provider)
            return [after_local_dt + timedelta(days=1)]

        ports = self._completion_ports(
            max_iterations=10,
            anchor_file_provider_for=build_provider,
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000961",
            "status": "pending",
            "chainID": "provider-reuse",
            "link": 1,
            "due": "20260801T090000Z",
            "chainUntil": "20260805T090000Z",
            "anchor_file": "calendar.csv",
        }

        with patch.object(modify_schedule_effects, "anchor_included_occurrences", included):
            final_no, final_dt = modify_schedule_effects.cap_from_until_anchor(
                ports,
                task,
                datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
                None,
            )

        self.assertEqual(final_no, 6)
        self.assertIsNotNone(final_dt)
        self.assertEqual(len(built), 1)
        self.assertTrue(providers)
        self.assertTrue(all(provider is built[0][3] for provider in providers))

    def test_overnight_window_until_includes_the_final_morning_slot(self) -> None:
        due_local = core.build_local_datetime(date(2025, 12, 15), (22, 30))
        until_local = core.build_local_datetime(date(2025, 12, 16), (6, 30))
        task = {
            "uuid": "00000000-0000-4000-8000-000000000963",
            "description": "overnight completion",
            "status": "completed",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "overnight-until-contract",
            "anchor": "w:mon@t=22:30..06:30/7",
            "link": 1,
            "due": core.fmt_isoz(due_local),
            "end": core.fmt_isoz(due_local + timedelta(minutes=10)),
            "chainUntil": core.fmt_isoz(until_local),
        }

        final_link, final_due = modify_schedule_effects.cap_from_until_anchor(
            self._completion_ports(max_iterations=32),
            task,
            due_local.astimezone(timezone.utc),
            core.validate_anchor_expr_strict(task["anchor"]),
        )

        self.assertEqual(final_link, 8)
        self.assertIsNotNone(final_due)
        self.assertEqual(
            core.to_local(final_due),
            until_local,
        )

    def test_fall_back_overnight_until_includes_final_slot(self) -> None:
        from zoneinfo import ZoneInfo

        from nautical_core.recurrence_context import RecurrenceContext
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        from nautical_core.task_models import NauticalTask

        zone = ZoneInfo("America/New_York")
        due_local = datetime(2026, 10, 31, 22, 30, tzinfo=zone)
        until_local = datetime(2026, 11, 1, 2, 30, tzinfo=zone)
        task = {
            "uuid": "00000000-0000-4000-8000-000000000964",
            "description": "DST fallback chain end",
            "status": "completed",
            "anchor": "w:sat@t=22:30..02:30/5",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "dstuntil-contract",
            "link": 1,
            "chainUntil": until_local.isoformat(),
            "due": due_local.isoformat(),
            "end": due_local.isoformat(),
        }
        observation = DEFAULT_TASK_CODEC.decode_row(
            task, source_query="DST chainUntil contract"
        )
        evaluator = RecurrenceEvaluator.from_task(
            NauticalTask.from_observation(observation),
            context=RecurrenceContext.from_observation(observation, timezone=zone),
        )

        def parse_datetime(value):
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

        final_link, final_due = modify_completion_compute.cap_from_until_anchor(
            task,
            due_local.astimezone(timezone.utc),
            core.validate_anchor_expr_strict(task["anchor"]),
            parse_datetime=parse_datetime,
            coerce_int=core.coerce_int,
            recurrence_seed_base=lambda value: str(value["chainID"]),
            to_local_cached=lambda value: value.astimezone(zone),
            safe_parse_datetime=lambda value: (parse_datetime(value), None),
            anchor_file_fallback_hhmm=lambda *_args: (9, 0),
            omit_dnf_from_parent=lambda _task: (None, None),
            recurrence_evaluator_for_task=lambda _task: evaluator,
            anchor_file_provider_for=lambda *_args, **_kwargs: None,
            anchor_included_occurrences=lambda *_args, **_kwargs: (),
            compare_datetimes=compare_datetimes,
            max_iterations=32,
        )

        self.assertEqual(final_link, 6)
        self.assertIsNotNone(final_due)
        self.assertEqual(final_due.astimezone(zone), until_local)


if __name__ == "__main__":
    unittest.main()
