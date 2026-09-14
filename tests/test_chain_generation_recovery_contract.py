from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

from nautical_core.chain_generation import ChainGenerationService
from nautical_core.chain_integrity_recovery import IntegrityRecoveryService
from nautical_core.cp_parser import cp_sequence_interval_for_token, parse_cp_sequence_tokens
from nautical_core.integration_models import MutationOperation
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.task_models import NauticalTask
from nautical_core.timeutil import fmt_isoz


UUID = "11111111-1111-4111-8111-111111111111"


class _Core:
    DEFAULT_BUSINESS_CALENDAR = None
    ASTRONOMY_CONFIG = None
    ANCHOR_FILE_DIR = ""
    _LOCAL_TZ = timezone.utc

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
            from nautical_core import task_codec

            return task_codec
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
        self.service = ChainGenerationService.from_core(_Core())

    def test_cp_generation_uses_link_sequence_and_durable_metadata(self):
        parent = _task(link=1)
        due, metadata = self.service.compute_cp_child_due(parent)
        self.assertEqual(due, datetime(2026, 1, 3, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata["cp_sequence_len"], 2)
        self.assertEqual(metadata["cp_sequence_step"], 1)
        due2, metadata2 = self.service.compute_cp_child_due(_task(link=2))
        self.assertEqual(due2, datetime(2026, 1, 4, 9, tzinfo=timezone.utc))
        self.assertEqual(metadata2["cp_sequence_step"], 2)

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

    def test_child_draft_reports_unrecoverable_relative_carry(self):
        parent = _task(wait="2026-01-02T08:00:00Z", due=None, scheduled=None)
        with self.assertRaisesRegex(RuntimeError, "wait carry failed"):
            self.service.build_child_draft(
                parent, datetime(2026, 1, 3, 10, tzinfo=timezone.utc),
                "due", 2, "11111111", "cp", 0, None,
            )


class IntegrityRecoveryContractTests(unittest.TestCase):
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
