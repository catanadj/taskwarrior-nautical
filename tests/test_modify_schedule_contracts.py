from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import nautical_core as core
from nautical_core import modify_completion_compute, modify_runtime, modify_schedule_effects
from nautical_core.recurrence_evaluator import RecurrenceEvaluator
from nautical_core.task_datetime import parser_for_core
from nautical_core.timeutil import compare_datetimes


class ModifyScheduleContractTests(unittest.TestCase):
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
        scheduler = modify_schedule_effects.SchedulerPorts(
            runtime_module=modify_runtime,
            state=modify_runtime.new_runtime_state(),
            core=core,
        )
        return modify_schedule_effects.AnchorCompletionPorts(
            compute=modify_completion_compute,
            parse_datetime=core.parse_dt_any,
            coerce_int=core.coerce_int,
            scheduler=scheduler,
            to_local_cached=core.to_local,
            safe_parse_datetime=parser.parse,
            anchor_file_fallback_hhmm=lambda _task, _next: (9, 0),
            omit_dnf_from_parent=lambda _task: (None, None),
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
            modify_schedule_effects.SchedulerPorts(
                runtime_module=modify_runtime,
                state=modify_runtime.new_runtime_state(),
                core=core,
            )
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


if __name__ == "__main__":
    unittest.main()
