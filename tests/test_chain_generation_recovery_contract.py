from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, get_type_hints
from unittest.mock import patch

import nautical_core as core
import nautical_core.anchor_inclusion as anchor_inclusion
import nautical_core.timezone_facade as timezone_facade
from nautical_core.chain_generation import ChainGenerationService
from nautical_core.chain_integrity_recovery import IntegrityRecoveryService
from nautical_core.cp_parser import cp_sequence_interval_for_token, parse_cp_sequence_tokens
from nautical_core.integration_models import MutationOperation
from nautical_core.occurrence_provider import Occurrence
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.task_models import NauticalTask
from nautical_core.timeutil import fmt_isoz


UUID = "11111111-1111-4111-8111-111111111111"


class _Core:
    DEFAULT_BUSINESS_CALENDAR = None
    ASTRONOMY_CONFIG = None
    ANCHOR_FILE_DIR = ""

    @staticmethod
    def parse_dt_any(value):
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text)

    @staticmethod
    def to_local(value):
        return value.astimezone(timezone.utc)

    @staticmethod
    def utc_to_local_naive(value):
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @staticmethod
    def local_naive_to_utc(value):
        return value.replace(tzinfo=timezone.utc)

    @staticmethod
    def fmt_isoz(value):
        return fmt_isoz(value)

    @staticmethod
    def now_utc():
        return datetime(2026, 1, 1, 12, tzinfo=timezone.utc)

    @staticmethod
    def coerce_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    parse_cp_sequence_tokens = staticmethod(parse_cp_sequence_tokens)
    cp_sequence_interval_for_token = staticmethod(cp_sequence_interval_for_token)

    @staticmethod
    def cp_sequence_parse_error(_value):
        return "invalid cp sequence"

    @staticmethod
    def _import_sibling(name):
        if name == "task_codec":
            import nautical_core.task_codec as task_codec

            return task_codec
        if name == "native_until":
            import nautical_core.native_until as native_until

            return native_until
        raise AssertionError(name)


def _task(**updates) -> NauticalTask:
    row = {
        "uuid": UUID,
        "status": "completed",
        "chain": "on",
        "chainID": "chain-a",
        "link": 1,
        "cp": "1d,2d",
        "due": "2026-01-02T09:00:00Z",
        "end": "2026-01-02T10:00:00Z",
        "description": "parent",
    }
    row.update(updates)
    return NauticalTask.from_observation(
        DEFAULT_TASK_CODEC.decode_row(row, source_query="chain-contract-test")
    )


def _observation(**updates):
    row = {
        "uuid": UUID,
        "status": "pending",
        "chain": "on",
        "chainID": "chain-a",
        "link": 1,
        "cp": "1d",
        "due": "2026-01-02T09:00:00Z",
        "modified": "2026-01-02T10:00:00Z",
        "until": "2026-01-03T09:00:00Z",
    }
    row.update(updates)
    return DEFAULT_TASK_CODEC.decode_row(row, source_query="recovery-contract-test")


class ChainGenerationContractTests(unittest.TestCase):
    def setUp(self):
        timezone_override = patch.object(
            timezone_facade, "_local_timezone", timezone.utc
        )
        timezone_override.start()
        self.addCleanup(timezone_override.stop)
        self.service = ChainGenerationService.from_core(_Core())

    def test_expiration_recovery_helpers_expose_typed_child_timestamps(self):
        from nautical_core.chain_integrity_lifecycle import (
            _build_expiration_child_with_day_end,
            compute_expiration_child_due,
        )

        compute_hints = get_type_hints(compute_expiration_child_due)
        self.assertEqual(
            compute_hints["return"], tuple[datetime | None, dict[str, Any]]
        )

        fallback_hints = get_type_hints(_build_expiration_child_with_day_end)
        self.assertIs(fallback_hints["child_due"], datetime)
        self.assertEqual(fallback_hints["until_dt"], datetime | None)
        self.assertIs(fallback_hints["hook"], object)

    def test_reconcile_expiration_anchor_advances_from_recurrence_target(self):
        from nautical_core.chain_integrity_lifecycle import compute_expiration_child_due

        parent = {
            "uuid": UUID,
            "status": "deleted",
            "anchor": "w:mon@t=09:00",
            "anchor_mode": "skip",
            "chain": "on",
            "chainID": "chain-a",
            "link": 1,
            "due": "20260706T090000Z",
            "end": "20260715T180000Z",
        }

        child_due, metadata = compute_expiration_child_due(
            parent, generation=self.service
        )

        self.assertEqual(child_due, datetime(2026, 7, 13, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata.get("basis"), "due recurrence target (expired)")

    def test_reconcile_expiration_cp_advances_from_recurrence_target(self):
        from nautical_core.chain_integrity_lifecycle import compute_expiration_child_due

        parent = {
            "uuid": UUID,
            "status": "deleted",
            "cp": "7d",
            "chain": "on",
            "chainID": "chain-a",
            "link": 1,
            "due": "20260720T090000Z",
            "end": "20260726T235900Z",
        }

        child_due, metadata = compute_expiration_child_due(
            parent, generation=self.service
        )

        self.assertEqual(child_due, datetime(2026, 7, 27, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata.get("basis"), "due recurrence target (expired)")

        scheduled_parent = dict(parent)
        scheduled_parent.pop("due")
        scheduled_parent["scheduled"] = "20260720T090000Z"
        scheduled_child_due, scheduled_metadata = compute_expiration_child_due(
            scheduled_parent, generation=self.service
        )

        self.assertEqual(
            scheduled_child_due, datetime(2026, 7, 27, 9, tzinfo=timezone.utc)
        )
        self.assertEqual(scheduled_metadata.get("target_field"), "scheduled")

    def test_native_until_carry_does_not_relabel_parser_defects_as_invalid_input(self):
        class BrokenParser:
            def parse(self, _value):
                raise RuntimeError("datetime parser defect")

        service = ChainGenerationService.from_core(
            _Core(), datetime_parser=BrokenParser()
        )

        with self.assertRaisesRegex(RuntimeError, "datetime parser defect"):
            service.carry_native_until(
                _task(until="2026-01-03T09:00:00Z"),
                {},
                datetime(2026, 1, 3, 9, tzinfo=timezone.utc),
                "cp",
                parent_anchor_field="due",
                child_anchor_field="due",
            )

    def test_cp_generation_uses_link_sequence_and_durable_metadata(self):
        parent = _task(link=1)
        due, metadata = self.service.compute_cp_child_due(parent)
        self.assertEqual(due, datetime(2026, 1, 3, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["cp_sequence_len"], 2)
        self.assertEqual(metadata["cp_sequence_step"], 1)
        due2, metadata2 = self.service.compute_cp_child_due(_task(link=2))
        self.assertEqual(due2, datetime(2026, 1, 4, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata2["cp_sequence_step"], 2)

    def test_cp_generation_selects_middle_interval_and_preserves_wall_clock(self):
        parent = _task(
            cp="3d,20d,7d",
            link=2,
            due="2026-01-01T09:00:00Z",
            end="2026-01-01T10:00:00Z",
        )

        due, metadata = self.service.compute_cp_child_due(parent)

        self.assertEqual(due, datetime(2026, 1, 21, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["cp_sequence_step"], 2)
        self.assertEqual(metadata["cp_sequence_len"], 3)

    def test_cp_generation_uses_chain_scoped_random_interval(self):
        parent = _task(
            cp="rand(3d..7d)",
            chainID="abcd1234",
            link=2,
            due="2026-01-01T09:00:00Z",
            end="2026-01-01T10:00:00Z",
        )

        due, metadata = self.service.compute_cp_child_due(parent)

        self.assertEqual(due, datetime(2026, 1, 6, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["cp_sequence_step"], 1)
        self.assertEqual(metadata["cp_sequence_len"], 1)

    def test_cp_generation_uses_scheduled_when_due_is_missing(self):
        parent = _task(
            cp="P1D",
            due=None,
            scheduled="2025-01-01T09:00:00Z",
            end="2025-01-01T17:00:00Z",
        )

        due, metadata = self.service.compute_cp_child_due(parent)

        self.assertEqual(due, datetime(2025, 1, 2, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["target_field"], "scheduled")

    def test_anchor_generation_uses_scheduled_seed_for_all_mode(self):
        scheduled = core.build_local_datetime(date(2025, 1, 6), (9, 0))
        ended = core.build_local_datetime(date(2025, 1, 8), (10, 0))
        parent = _task(
            anchor="w:mon..sun@t=09:00",
            anchor_mode="all",
            cp=None,
            due=None,
            scheduled=fmt_isoz(scheduled),
            end=fmt_isoz(ended),
        )
        service = ChainGenerationService.from_core(core)

        due, metadata, _dnf = service.compute_anchor_child_due(parent)

        due_local = core.to_local(due)
        self.assertEqual(due_local.date(), date(2025, 1, 7))
        self.assertEqual((due_local.hour, due_local.minute), (9, 0))
        self.assertEqual(metadata["target_field"], "scheduled")

    def test_positional_anchor_generation_advances_to_next_selected_date(self):
        parent = _task(
            anchor="(w:tue | w:thu)@in-month=last",
            anchor_mode="skip",
            cp=None,
            due="2026-07-30T09:00:00Z",
            end="2026-07-30T10:00:00Z",
        )

        due, metadata, dnf = self.service.compute_anchor_child_due(parent)

        self.assertEqual(due, datetime(2026, 8, 27, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["basis"], "after_end")
        self.assertEqual(dnf[0][0]["kind"], "select")

    def test_anchor_post_selection_modifiers_transform_completion_occurrence(self):
        parent = _task(
            anchor="(w:tue | w:thu)@in-month=last@+2d@t=09:00",
            anchor_mode="skip",
            cp=None,
            due="2026-08-01T09:00:00Z",
            end="2026-08-01T10:00:00Z",
        )

        due, metadata, dnf = self.service.compute_anchor_child_due(parent)

        self.assertEqual(due, datetime(2026, 8, 29, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["basis"], "after_end")
        self.assertEqual(dnf[0][0]["mods"]["day_offset"], 2)

    def test_yearly_positional_selection_applies_post_selection_offset(self):
        parent = _task(
            anchor="(w:mon)@in-year=last@+7d@t=09:00",
            anchor_mode="skip",
            cp=None,
            due="2027-01-04T09:00:00Z",
            end="2027-01-04T10:00:00Z",
        )

        due, metadata, dnf = self.service.compute_anchor_child_due(parent)

        self.assertEqual(due, datetime(2028, 1, 3, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["basis"], "after_end")
        self.assertEqual(dnf[0][0]["scope"], "year")

    def test_year_ordinal_anchor_reconcile_calculation(self):
        parent = _task(
            anchor="y:d60@t=09:00",
            anchor_mode="skip",
            cp=None,
            due="2024-02-29T09:00:00Z",
            end="2024-02-29T10:00:00Z",
        )

        test_core = _Core()
        test_core.YearTokenFormatError = core.YearTokenFormatError
        service = ChainGenerationService.from_core(test_core)
        child_due, metadata, _dnf = service.compute_anchor_child_due(parent)

        child_local = test_core.to_local(child_due)
        self.assertEqual(child_local.date(), date(2025, 3, 1))
        self.assertEqual((child_local.hour, child_local.minute), (9, 0))
        self.assertEqual(metadata.get("basis"), "after_end")

    def test_on_modify_compute_anchor_child_due_accepts_scheduled_after_due(self):
        due = core.build_local_datetime(date(2025, 1, 6), (9, 0))
        scheduled = core.build_local_datetime(date(2025, 1, 8), (12, 0))
        ended = core.build_local_datetime(date(2025, 1, 8), (10, 0))
        parent = _task(
            anchor="w:mon..sun@t=09:00",
            anchor_mode="all",
            cp=None,
            due=fmt_isoz(due),
            scheduled=fmt_isoz(scheduled),
            end=fmt_isoz(ended),
        )

        service = ChainGenerationService.from_core(core)
        child_due, metadata, _dnf = service.compute_anchor_child_due(parent)

        self.assertEqual(core.to_local(child_due).date(), date(2025, 1, 7))
        self.assertEqual((core.to_local(child_due).hour, core.to_local(child_due).minute), (9, 0))
        self.assertEqual(metadata["target_field"], "due")

    def test_on_modify_compute_anchor_child_due_skips_omit_date(self):
        due = core.build_local_datetime(date(2025, 1, 6), (9, 0))
        ended = core.build_local_datetime(date(2025, 1, 6), (10, 0))
        parent = _task(
            anchor="w:mon,wed,fri@t=09:00",
            omit="w:wed",
            anchor_mode="skip",
            cp=None,
            due=fmt_isoz(due),
            end=fmt_isoz(ended),
        )

        service = ChainGenerationService.from_core(core)
        child_due, metadata, _dnf = service.compute_anchor_child_due(parent)

        child_local = core.to_local(child_due)
        self.assertEqual(child_local.date(), date(2025, 1, 10))
        self.assertEqual((child_local.hour, child_local.minute), (9, 0))
        self.assertEqual(metadata["target_field"], "due")

    def test_on_modify_compute_anchor_child_due_unsatisfiable_omit_fails(self):
        due = core.build_local_datetime(date(2025, 1, 6), (9, 0))
        ended = core.build_local_datetime(date(2025, 1, 6), (10, 0))
        parent = _task(
            anchor="w:mon",
            omit="w:mon",
            anchor_mode="skip",
            cp=None,
            due=fmt_isoz(due),
            end=fmt_isoz(ended),
        )

        service = ChainGenerationService.from_core(core)

        with self.assertRaisesRegex(
            ValueError,
            "No valid anchor occurrences found after applying omit rules",
        ):
            service.compute_anchor_child_due(parent)

    def test_on_modify_compute_counted_random_advances_within_period(self):
        chain_id = "abcd1234"
        dnf = core.validate_anchor_expr_strict("m:2rand")
        seed = date(2026, 1, 1)
        first, _metadata = core.next_after_expr(
            dnf,
            seed,
            default_seed=seed,
            seed_base=chain_id,
        )
        expected, _metadata = core.next_after_expr(
            dnf,
            first,
            default_seed=seed,
            seed_base=chain_id,
        )
        parent_due = core.build_local_datetime(first, (9, 0))
        parent_end = core.build_local_datetime(first, (10, 0))
        parent = _task(
            anchor="m:2rand",
            anchor_mode="skip",
            cp=None,
            due=fmt_isoz(parent_due),
            end=fmt_isoz(parent_end),
            chainID=chain_id,
        )

        service = ChainGenerationService.from_core(core)
        child_due, metadata, _child_dnf = service.compute_anchor_child_due(parent)

        self.assertEqual(core.to_local(child_due).date(), expected)
        self.assertEqual(expected.month, first.month)
        self.assertEqual(metadata["target_field"], "due")

    def test_anchor_generation_selects_next_local_timed_slot(self):
        due_local = core.build_local_datetime(date(2026, 7, 4), (9, 0))
        end_local = core.build_local_datetime(date(2026, 7, 4), (10, 0))
        parent = _task(
            anchor="w:mon..sun@t=05:00,09:00,14:00,19:00",
            cp=None,
            due=fmt_isoz(due_local),
            end=fmt_isoz(end_local),
        )
        service = ChainGenerationService.from_core(core)

        child_due, _metadata, _dnf = service.compute_anchor_child_due(parent)

        child_local = core.to_local(child_due)
        self.assertEqual(child_local.date(), date(2026, 7, 4))
        self.assertEqual((child_local.hour, child_local.minute), (14, 0))

    def test_anchor_generation_advances_to_next_window_slot_after_completion(self):
        due = datetime(2025, 12, 17, 6, tzinfo=timezone.utc)
        ended = datetime(2025, 12, 17, 6, 30, tzinfo=timezone.utc)
        parent = _task(
            anchor="w:mon..sun@t=06..18/3h",
            cp=None,
            due=fmt_isoz(due),
            end=fmt_isoz(ended),
        )

        child_due, metadata, _dnf = self.service.compute_anchor_child_due(parent)

        child_local = self.service.core.to_local(child_due)
        self.assertEqual(child_local, datetime(2025, 12, 17, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["basis"], "after_end")

    def test_partitioned_anchor_window_uses_remaining_slots_then_rolls_day(self):
        def child_after(due_hour: int, due_minute: int):
            due = datetime(2025, 12, 17, due_hour, due_minute, tzinfo=timezone.utc)
            parent = _task(
                anchor="w:mon..sun@t=04:30..19:30/3",
                cp=None,
                due=fmt_isoz(due),
                end=fmt_isoz(due + timedelta(minutes=10)),
            )
            child_due, _metadata, _dnf = self.service.compute_anchor_child_due(parent)
            return self.service.core.to_local(child_due)

        self.assertEqual(child_after(4, 30), datetime(2025, 12, 17, 12, tzinfo=timezone.utc))
        self.assertEqual(child_after(19, 30), datetime(2025, 12, 18, 4, 30, tzinfo=timezone.utc))

    def test_overnight_window_completion_advances_across_window_boundary(self):
        due = datetime(2025, 12, 15, 22, 30, tzinfo=timezone.utc)
        anchor = "w:mon@t=22:30..06:30/7"

        def child_after(ended: datetime) -> datetime:
            parent = _task(
                anchor=anchor,
                cp=None,
                due=fmt_isoz(due),
                end=fmt_isoz(ended),
            )
            child_due, _metadata, _dnf = self.service.compute_anchor_child_due(parent)
            return self.service.core.to_local(child_due)

        self.assertEqual(
            child_after(due + timedelta(minutes=10)),
            datetime(2025, 12, 15, 23, 50, tzinfo=timezone.utc),
        )
        self.assertEqual(
            child_after(datetime(2025, 12, 16, 6, 40, tzinfo=timezone.utc)),
            datetime(2025, 12, 22, 22, 30, tzinfo=timezone.utc),
        )

    def test_random_window_completion_reuses_the_next_chain_scoped_slot(self):
        from nautical_core.time_slots import resolve_time_slots_with_offsets

        chain_id = "randommodify1"
        target_date = date(2025, 12, 15)
        slots = resolve_time_slots_with_offsets(
            {"time_random": "rand(06:00..18:00/3)", "t": []},
            target_date,
            seed_base=chain_id,
        )

        def utc_slot(slot):
            day_offset, hour, minute = slot
            return datetime.combine(
                target_date + timedelta(days=day_offset),
                datetime.min.time(),
                tzinfo=timezone.utc,
            ).replace(hour=hour, minute=minute)

        first, expected = utc_slot(slots[0]), utc_slot(slots[1])
        parent = _task(
            anchor="w:mon@t=rand(06..18/3)",
            cp=None,
            chainID=chain_id,
            due=fmt_isoz(first),
            end=fmt_isoz(first + timedelta(minutes=10)),
        )

        child_due, _metadata, _dnf = self.service.compute_anchor_child_due(parent)

        self.assertEqual(self.service.core.to_local(child_due), expected)

    def test_anchor_file_projection_reuses_one_provider(self):
        builders = []
        occurrence = Occurrence(
            date(2026, 8, 4),
            9,
            0,
            source="anchor_file",
            local_datetime=core.to_local(
                core.build_local_datetime(date(2026, 8, 4), (9, 0))
            ),
        )

        def build_provider(*_args, **_kwargs):
            provider = SimpleNamespace(
                next_after=lambda after_local, **_kwargs: (
                    occurrence if occurrence.local_datetime > after_local else None
                )
            )
            builders.append(provider)
            return provider

        due = core.build_local_datetime(date(2026, 8, 3), (9, 0))
        parent = _task(
            anchor="w:mon@t=09:00",
            anchor_file="calendar.csv@t=09:00",
            anchor_mode="all",
            cp=None,
            due=fmt_isoz(due),
            end=fmt_isoz(due + timedelta(hours=1)),
        )
        service = ChainGenerationService.from_core(core)

        with patch.object(anchor_inclusion, "_build_anchor_file_provider", build_provider):
            child_due, _metadata, _dnf = service.compute_anchor_child_due(parent)

        self.assertEqual(len(builders), 1)
        self.assertEqual(core.to_local(child_due).hour, 9)

    def test_pure_anchor_file_projection_reuses_one_provider(self):
        builders = []
        occurrence = Occurrence(
            date(2026, 8, 4),
            9,
            0,
            source="anchor_file",
            local_datetime=core.to_local(
                core.build_local_datetime(date(2026, 8, 4), (9, 0))
            ),
        )

        def build_provider(*_args, **_kwargs):
            provider = SimpleNamespace(
                occurrences=lambda: [occurrence],
                next_after=lambda after_local, **_kwargs: (
                    occurrence if occurrence.local_datetime > after_local else None
                ),
            )
            builders.append(provider)
            return provider

        due = core.build_local_datetime(date(2026, 8, 3), (9, 0))
        parent = _task(
            anchor_file="calendar.csv@t=09:00",
            anchor_mode="skip",
            anchor=None,
            cp=None,
            due=fmt_isoz(due),
            end=fmt_isoz(due + timedelta(hours=1)),
        )
        service = ChainGenerationService.from_core(core)

        with patch.object(anchor_inclusion, "_build_anchor_file_provider", build_provider):
            child_due, _metadata, _dnf = service.compute_anchor_child_due(parent)

        self.assertEqual(len(builders), 1)
        self.assertEqual(core.to_local(child_due).date(), date(2026, 8, 4))

    def test_hook_adapter_uses_shared_generation_service_without_legacy_helpers(self) -> None:
        class Hook:
            core = _Core()

            @staticmethod
            def legacy_compute_cp_child_due(_parent):
                raise AssertionError("modify helper must not be captured")

        service = ChainGenerationService.from_hook(Hook())
        due, metadata = service.compute_cp_child_due(_task(status="pending"))
        self.assertEqual(due, datetime(2026, 1, 3, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["basis"], "end+cp (preserve clock)")

    def test_cp_generation_rejects_malformed_sequence_and_missing_identity(self):
        with self.assertRaisesRegex(ValueError, "cp field"):
            self.service.compute_cp_child_due(_task(cp="not-a-duration"))
        with self.assertRaisesRegex(ValueError, "chainID"):
            _task(chainID="")

    def test_service_level_identity_guard_rejects_empty_chain_id(self):
        # A decoded task cannot normally contain an empty ChainID.  Exercise
        # the service boundary directly as a defensive check against a bad
        # adapter/projection, without weakening the domain model validator.
        task = object.__new__(NauticalTask)
        object.__setattr__(
            task,
            "identity",
            SimpleNamespace(chain_id=SimpleNamespace(value="")),
        )
        with self.assertRaisesRegex(ValueError, "chainID is required"):
            self.service._require_chain_id(task)
        with self.assertRaisesRegex(TypeError, "validated NauticalTask"):
            self.service._require_chain_id({"uuid": UUID, "link": 1})
        self.assertFalse(hasattr(self.service, "build_child_from_parent"))

    def test_anchor_omission_terminal_and_date_limit_evidence_are_not_invented(self):
        parent = _task(anchor="w:mon", cp=None, due="2026-01-02T09:00:00Z")
        omitted = SimpleNamespace(selected_occurrence=None)
        scheduler = SimpleNamespace(
            fingerprint="fixture",
            session=SimpleNamespace(evaluator=SimpleNamespace(anchor_dnf=None)),
            select_mode=lambda *_args, **_kwargs: omitted,
        )
        with patch.object(ChainGenerationService, "_task_scheduler", return_value=scheduler):
            with self.assertRaisesRegex(ValueError, "Could not compute next anchor"):
                self.service.compute_anchor_child_due(parent)

            terminal = OccurrenceSearchExhausted(
                "anchor occurrence", reference=datetime(9999, 12, 31), kind=OccurrenceSearchExhausted.DATE_LIMIT
            )
            scheduler.select_mode = lambda *_args, **_kwargs: (_ for _ in ()).throw(terminal)
            self.service._mode_cache.clear()
            with self.assertRaises(OccurrenceSearchExhausted) as raised:
                self.service.compute_anchor_child_due(parent)
        self.assertTrue(raised.exception.is_date_limit)

    def test_child_draft_preserves_chain_provenance_and_drops_native_fields(self):
        parent = _task(urgency=4.2, id=17)
        draft = self.service.build_child_draft(
            parent,
            datetime(2026, 1, 3, 10, tzinfo=timezone.utc),
            "due", 2, "11111111", "cp", 4, None,
        )
        payload = draft.to_mapping()
        self.assertEqual(payload["chainID"], "chain-a")
        self.assertEqual(payload["link"], 2)
        self.assertEqual(payload["prevLink"], "11111111")
        self.assertEqual(payload["chainMax"], 4)
        self.assertNotIn("urgency", payload)
        self.assertNotIn("id", payload)
        self.assertEqual(draft.target.value, datetime(2026, 1, 3, 10, tzinfo=timezone.utc))

    def test_child_draft_preserves_parent_business_calendar(self):
        parent = _task(anchor="w:sun", cp=None, bc="weekend")
        draft = self.service.build_child_draft(
            parent,
            datetime(2026, 1, 4, 9, tzinfo=timezone.utc),
            "due", 2, "11111111", "anchor", 0, None,
        )
        self.assertEqual(draft.to_mapping()["bc"], "weekend")

    def test_child_draft_reports_unrecoverable_relative_carry(self):
        parent = _task(wait="2026-01-02T08:00:00Z", due=None, scheduled=None)
        with self.assertRaisesRegex(RuntimeError, "wait carry failed"):
            self.service.build_child_draft(
                parent, datetime(2026, 1, 3, 10, tzinfo=timezone.utc),
                "due", 2, "11111111", "cp", 0, None,
            )


class IntegrityRecoveryContractTests(unittest.TestCase):
    def test_hookless_recovery_preserves_scheduled_and_wait_offsets(self):
        import nautical_core as core
        from nautical_core.chain_integrity_lifecycle import plan_recovery_decision
        from nautical_core.lifecycle.recovery_models import RecoveryPlanResult, RecoveryRefusal

        generation = ChainGenerationService.from_core(core)
        parent = _observation(
            status="completed",
            description="typed fixture task",
            cp="7d",
            due="2026-07-20T10:00:00Z",
            end="2026-07-20T11:00:00Z",
            until="2026-08-03T10:00:00Z",
            scheduled="2026-07-20T09:30:00Z",
            wait="2026-07-20T08:00:00Z",
        )
        plan = plan_recovery_decision(
            parent, existing_children=(), hook=None, generation=generation
        )
        self.assertIsInstance(plan, RecoveryPlanResult)
        assert isinstance(plan, RecoveryPlanResult)
        child = plan.plan.child_dict()
        child_due = datetime.fromisoformat(child["due"].replace("Z", "+00:00"))
        child_scheduled = datetime.fromisoformat(
            child["scheduled"].replace("Z", "+00:00")
        )
        child_wait = datetime.fromisoformat(child["wait"].replace("Z", "+00:00"))
        parent_due = datetime.fromisoformat("2026-07-20T10:00:00+00:00")
        parent_scheduled = datetime.fromisoformat("2026-07-20T09:30:00+00:00")
        parent_wait = datetime.fromisoformat("2026-07-20T08:00:00+00:00")
        self.assertEqual(child_scheduled - child_due, parent_scheduled - parent_due)
        self.assertEqual(child_wait - child_due, parent_wait - parent_due)

        for field in ("scheduled", "wait"):
            with self.subTest(field=field):
                invalid_fields = {
                    "status": "completed",
                    "description": "typed fixture task",
                    "cp": "7d",
                    "due": "2026-07-20T10:00:00Z",
                    "end": "2026-07-20T11:00:00Z",
                    "until": "2026-08-03T10:00:00Z",
                    "scheduled": "2026-07-20T09:30:00Z",
                    "wait": "2026-07-20T08:00:00Z",
                }
                invalid_fields[field] = "not-a-date"
                invalid = plan_recovery_decision(
                    _observation(**invalid_fields),
                    existing_children=(),
                    hook=None,
                    generation=generation,
                )
                self.assertIsInstance(invalid, RecoveryRefusal)
                assert isinstance(invalid, RecoveryRefusal)
                self.assertIn(field, invalid.reason)

    def test_existing_children_and_ambiguous_slots_are_fail_closed(self):
        child = _observation(uuid="22222222-2222-4222-8222-222222222222", link=2)
        service = IntegrityRecoveryService(child_lookup=lambda chain, link: child if (chain, link) == ("chain-a", 2) else None)
        self.assertEqual(service.existing_children(_observation()), (child,))
        other = _observation(uuid="33333333-3333-4333-8333-333333333333")
        self.assertIn(("chain-a", 1), service.ambiguous_candidate_slots((_observation(), other)))
        malformed = _observation(link="not-a-link", uuid="44444444-4444-4444-8444-444444444444")
        self.assertEqual(service.ambiguous_candidate_slots((malformed,)), {})
        with self.assertRaises(ValueError):
            IntegrityRecoveryService.native_until_request(malformed, "2026-01-04T09:00:00Z", mutation_epoch=1)

    def test_native_until_request_contains_guarded_identity_and_operation(self):
        request = IntegrityRecoveryService.native_until_request(
            _observation(), "2026-01-04T09:00:00Z", mutation_epoch=7
        )
        self.assertIs(request.operation, MutationOperation.NATIVE_UNTIL_REPAIR)
        self.assertEqual(request.guard.chain_id, "chain-a")
        self.assertEqual(request.guard.link, 1)
        self.assertEqual(request.guard.expected_mutation_epoch, 7)
        self.assertEqual(str(request.payload.expected_until), "2026-01-03 09:00:00+00:00")

    def test_audit_repairs_invalid_until_from_previous_link(self):
        previous = _observation(link=1, until="2026-01-03T09:00:00Z")
        current = _observation(
            uuid="22222222-2222-4222-8222-222222222222", link=2,
            due="2026-01-03T09:00:00Z", until="2026-01-02T09:00:00Z",
        )
        service = IntegrityRecoveryService()
        parse = lambda value: (_Core.parse_dt_any(value), None)
        audit = service.audit_native_until(
            (previous, current), predecessor=lambda _row: previous,
            safe_parse_datetime=parse, fmt_isoz=fmt_isoz,
            utc_to_local_naive=_Core.utc_to_local_naive,
            local_naive_to_utc=_Core.local_naive_to_utc,
        )
        self.assertEqual(audit.native_until.status, "invalid")
        self.assertEqual(audit.native_until.repairs[0]["action"], "repair_until")
        self.assertEqual(audit.native_until.repairs[0]["new_until"], "2026-01-04T09:00:00Z")
        self.assertEqual(len(audit.candidates), 1)

    def test_apply_classifies_lock_busy_without_mutation(self):
        service = IntegrityRecoveryService()
        row = _observation()
        item = {"action": "repair_until"}
        calls = []

        @contextmanager
        def lock(_value, *_args):
            yield False

        error = service.apply_native_until_candidate(
            row, None, item, repaired="2026-01-04T09:00:00Z", taskdata=object(),
            lease_held=False, mutation_lock=lock, parent_lock=lambda _uuid: lock(None),
            refresh_parent=lambda value: value, refresh_previous=lambda value: value,
            guard_error=lambda *_args: None, configuration=lambda: ("valid", ""),
            mutate=lambda *_args: calls.append("mutate"), verify=lambda *_args: True,
            on_lock_busy=lambda kind: calls.append(kind),
        )
        self.assertIn("reconcile", calls)
        self.assertEqual(item["action"], "repair_error")
        self.assertIn("already running", error)
        self.assertEqual(calls.count("mutate"), 0)

    def test_apply_classifies_parent_lock_and_mutation_failures(self):
        service = IntegrityRecoveryService()
        row = _observation()

        @contextmanager
        def reconcile_lock(_value, *_args):
            yield True

        @contextmanager
        def parent_busy(_value):
            yield False

        item = {"action": "repair_until"}
        service.apply_native_until_candidate(
            row, None, item, repaired="2026-01-04T09:00:00Z", taskdata=object(),
            lease_held=False, mutation_lock=reconcile_lock, parent_lock=parent_busy,
            refresh_parent=lambda value: value, refresh_previous=lambda value: value,
            guard_error=lambda *_args: None, configuration=lambda: ("valid", ""),
            mutate=lambda *_args: self.fail("busy parent must not mutate"), verify=lambda *_args: True,
            on_lock_busy=lambda kind: None,
        )
        self.assertEqual(item["action"], "repair_error")
        self.assertIn("lock busy", item["repair_error"])

        @contextmanager
        def parent_lock(_value):
            yield True

        item = {"action": "repair_until"}
        service.apply_native_until_candidate(
            row, None, item, repaired="2026-01-04T09:00:00Z", taskdata=object(),
            lease_held=False, mutation_lock=reconcile_lock, parent_lock=parent_lock,
            refresh_parent=lambda value: value, refresh_previous=lambda value: value,
            guard_error=lambda *_args: None, configuration=lambda: ("valid", ""),
            mutate=lambda *_args: (_ for _ in ()).throw(RuntimeError("persistence unavailable")),
            verify=lambda *_args: True, on_lock_busy=lambda _kind: None,
        )
        self.assertEqual(item["action"], "repair_error")
        self.assertEqual(item["repair_error"], "persistence unavailable")

    def test_native_until_audit_keeps_missing_and_malformed_predecessors_explicit(self):
        service = IntegrityRecoveryService()

        def parse(value):
            try:
                parsed = datetime.strptime(str(value), "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
                return parsed, None
            except (TypeError, ValueError) as exc:
                return None, str(exc)

        row = {
            "uuid": "00000000-0000-4000-8000-000000000701",
            "description": "fault recovery",
            "chain": "on",
            "chainID": "fault-recovery",
            "link": 2,
            "status": "pending",
            "due": "20260820T100000Z",
            "until": "20260820T090000Z",
        }
        unavailable = service.audit_native_until(
            (_observation(**row),),
            predecessor=lambda _row: None,
            safe_parse_datetime=parse,
            fmt_isoz=fmt_isoz,
            utc_to_local_naive=_Core.utc_to_local_naive,
            local_naive_to_utc=_Core.local_naive_to_utc,
        )
        self.assertEqual(unavailable.native_until.status, "invalid")
        self.assertEqual(unavailable.candidates[0].item.get("fallback"), "local 23:00")

        malformed = service.audit_native_until(
            (_observation(**{**row, "due": "not-a-date"}),),
            predecessor=lambda _row: None,
            safe_parse_datetime=lambda _value: (None, "malformed datetime"),
            fmt_isoz=fmt_isoz,
            utc_to_local_naive=_Core.utc_to_local_naive,
            local_naive_to_utc=_Core.local_naive_to_utc,
        )
        self.assertEqual(malformed.native_until.status, "invalid")
        self.assertEqual(malformed.native_until.repairs[0].get("action"), "manual_review")


if __name__ == "__main__":
    unittest.main()
