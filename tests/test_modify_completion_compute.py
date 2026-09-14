from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import nautical_core as core
from nautical_core import add_validation
from nautical_core.modify_completion_compute import completion_compute_child_due, completion_compute_next_and_limits
from nautical_core.modify_models import CompletionComputeServices, CompletionLifecycleResult
from nautical_core.scheduler_models import OccurrenceSearchExhausted
from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable
from nautical_core.modify_completion_preflight import completion_existing_next_or_fail


class CompletionComputeTerminalEvidenceTests(unittest.TestCase):
    def test_date_and_search_exhaustion_never_produce_child_tuples(self) -> None:
        from nautical_core import modify_completion_compute as compute

        terminal = OccurrenceSearchExhausted(
            "anchor scheduling", reference=date(9999, 12, 31), limit=2
        )
        search_limited = OccurrenceSearchExhausted(
            "anchor scheduling",
            reference=datetime(2026, 1, 1),
            limit=2,
            kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
        )
        panels = []
        task = {"chain": "on", "uuid": "terminal-task"}

        def scheduler(_task):
            raise outcomes.pop(0)

        outcomes = [terminal]
        result = compute.completion_compute_child_due(
            task,
            "anchor",
            compute_anchor_child_due=scheduler,
            compute_cp_child_due=scheduler,
            panel=lambda title, rows, **kwargs: panels.append((title, list(rows), kwargs)),
            print_task=lambda _value: None,
            on_terminal=lambda exc: task.update({"chain": "off", "terminal": exc.kind}),
        )
        self.assertIsNone(result)
        self.assertEqual(task, {"chain": "off", "uuid": "terminal-task", "terminal": "date_limit"})

        outcomes.append(search_limited)
        result = compute.completion_compute_child_due(
            {"chain": "on"},
            "anchor",
            compute_anchor_child_due=scheduler,
            compute_cp_child_due=scheduler,
            panel=lambda title, rows, **kwargs: panels.append((title, list(rows), kwargs)),
            print_task=lambda _value: None,
            on_terminal=lambda exc: panels.append(("terminal", [("kind", exc.kind)], {})),
        )
        self.assertIsNone(result)
        self.assertEqual(panels[-1], ("terminal", [("kind", "search_limit")], {}))

    def test_completion_caps_include_exact_boundary_and_stop_after_it(self) -> None:
        from nautical_core import modify_completion_compute as compute

        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        summaries = []
        printed = []
        task = {"chain": "on", "chainID": "cap-boundary", "uuid": "parent"}
        self.assertTrue(
            compute.completion_cap_guard_or_stop(
                task, 5, 5, now,
                end_chain_summary=lambda *args, **kwargs: summaries.append((args, kwargs)),
                print_task=lambda value: printed.append(dict(value)),
            )
        )
        self.assertEqual(task["chain"], "on")
        self.assertFalse(summaries)
        self.assertFalse(printed)
        self.assertFalse(
            compute.completion_cap_guard_or_stop(
                task, 6, 5, now,
                end_chain_summary=lambda *args, **kwargs: summaries.append((args, kwargs)),
                print_task=lambda value: printed.append(dict(value)),
            )
        )
        self.assertEqual(task["chain"], "off")
        self.assertIn("Reached cap #5", str(summaries[-1]))

        until = now + timedelta(days=2)
        task = {"chain": "on", "chainID": "cap-boundary", "uuid": "parent"}
        end = lambda *args, **kwargs: None
        self.assertTrue(
            compute.completion_until_guard_or_stop(
                task, until, until, now,
                end_chain_summary=end,
                print_task=lambda _value: None,
            )
        )
        self.assertFalse(
            compute.completion_until_guard_or_stop(
                task, until + timedelta(seconds=1), until, now,
                end_chain_summary=end,
                print_task=lambda _value: None,
            )
        )

    def test_completion_caps_choose_earliest_limit_without_dropping_estimates(self) -> None:
        from nautical_core import modify_completion_compute as compute

        child_due = datetime(2026, 1, 2, 9, tzinfo=timezone.utc)
        until = child_due + timedelta(days=10)

        result = compute.completion_caps(
            "cp",
            {"chainMax": 8, "chainUntil": core.fmt_isoz(until)},
            child_due,
            None,
            coerce_int=core.coerce_int,
            dtparse=core.parse_dt_any,
            estimate_cp_final_by_max=lambda *_args: child_due + timedelta(days=7),
            estimate_anchor_final_by_max=lambda *_args: None,
            cap_from_until_cp=lambda *_args: (5, child_due + timedelta(days=4)),
            cap_from_until_anchor=lambda *_args: (None, None),
        )
        cpmax, parsed_until, cap_no, finals, until_cap_no = result
        self.assertEqual(cpmax, 8)
        self.assertEqual(parsed_until, until)
        self.assertEqual((cap_no, until_cap_no), (5, 5))
        self.assertEqual([kind for kind, _value in finals], ["max", "until"])

        earlier_max = compute.completion_caps(
            "cp",
            {"chainMax": 3, "chainUntil": core.fmt_isoz(until)},
            child_due,
            None,
            coerce_int=core.coerce_int,
            dtparse=core.parse_dt_any,
            estimate_cp_final_by_max=lambda *_args: child_due + timedelta(days=2),
            estimate_anchor_final_by_max=lambda *_args: None,
            cap_from_until_cp=lambda *_args: (5, child_due + timedelta(days=4)),
            cap_from_until_anchor=lambda *_args: (None, None),
        )
        self.assertEqual((earlier_max[2], earlier_max[4]), (3, 5))

    def test_until_past_guard_orders_repeated_wall_times_by_instant(self) -> None:
        zone = ZoneInfo("Europe/Bucharest")
        now = datetime(2026, 10, 25, 3, 20, tzinfo=zone, fold=1)
        earlier = datetime(2026, 10, 25, 3, 18, tzinfo=zone, fold=0)
        panels = []

        result = core.modify_completion_compute.completion_until_or_fail(
            {"chainUntil": earlier},
            now,
            safe_parse_datetime=lambda _value: (earlier, None),
            validate_until_not_past=lambda until_dt, now_utc: add_validation.validate_until_not_past(
                until_dt, now_utc, core=core
            ),
            panel=lambda _title, rows, **_kwargs: panels.extend(rows),
            print_task=lambda _task: None,
        )

        self.assertFalse(result)
        reason = " ".join(str(value) for label, value in panels if label == "Reason")
        self.assertIn("in the past", reason)

    def test_unavailable_existing_next_lookup_stops_spawn_and_preserves_task(self) -> None:
        task = {"uuid": "parent", "link": 1}
        panels = []
        printed = []
        command = TaskCommand(("task", "export"), "next-link lookup", 1.0)
        unavailable = Unavailable(
            "chain:chain-1",
            FailureEvidence(
                command, CommandFailureKind.BUSY, 1, 1, 0.0, True, "lock busy"
            ),
        )

        allowed = completion_existing_next_or_fail(
            task,
            2,
            existing_next_lookup=lambda *_args: unavailable,
            short=lambda value: str(value)[:8],
            panel=lambda *args, **kwargs: panels.append((args, kwargs)),
            print_task=printed.append,
        )

        self.assertFalse(allowed)
        self.assertIn("unavailable", panels[0][0][0].lower())
        self.assertIn("lock busy", str(panels))
        self.assertEqual(printed, [task])
    def test_kind_guard_uses_preflight_ports(self) -> None:
        from nautical_core.modify_completion_effects import (
            CompletionPreflightPorts,
            kind_or_stop,
        )

        preflight = SimpleNamespace(
            completion_kind_or_stop=lambda task, now, **callbacks: (
                task["cp"], now, sorted(callbacks)
            )
        )
        ports = CompletionPreflightPorts(
            preflight=preflight,
            coerce_int=int,
            max_link_number=99,
            short_uuid=lambda value: value,
            panel=lambda *_args, **_kwargs: None,
            print_task=lambda *_args: None,
            end_chain_summary=lambda *_args, **_kwargs: None,
            existing_next_lookup=lambda *_args: None,
        )
        now = datetime(2026, 9, 11, 9, tzinfo=timezone.utc)

        result = kind_or_stop(ports, {"cp": "1d"}, now)

        self.assertEqual(result[0:2], ("1d", now))
        self.assertEqual(result[2], ["end_chain_summary", "panel", "print_task"])

    def test_effect_compute_orchestration_uses_explicit_ports(self) -> None:
        from nautical_core import modify_completion_compute as compute
        from nautical_core.modify_completion_effects import (
            CompletionComputePorts,
            CompletionLifecyclePlanPorts,
            compute_next_and_limits,
        )

        due = datetime(2026, 9, 12, 9, tzinfo=timezone.utc)
        ports = CompletionComputePorts(
            compute=compute,
            services_type=CompletionComputeServices,
            compute_child_due=lambda _task, _kind: (due, {"source": "cp"}, None),
            until_or_fail=lambda _task, _now: None,
            until_guard_or_stop=lambda *_args: True,
            require_child_due_or_fail=lambda *_args: True,
            warn_unreasonable_duration=lambda *_args: None,
            caps=lambda *_args: (None, None, None, [], None),
            cap_guard_or_stop=lambda *_args: True,
            lifecycle_result_type=CompletionLifecycleResult,
            lifecycle_plan=CompletionLifecyclePlanPorts(
                generation=None,
                scheduler_fingerprint=lambda: "",
                compare_datetimes=lambda _left, _right: 0,
                invalid_relative_carry_reason=lambda *_args: None,
                lifecycle_planner=None,
                lifecycle_models=None,
                modify_models=None,
                end_chain_summary=lambda *_args, **_kwargs: None,
                ensure_terminal_chain_off=lambda *_args, **_kwargs: None,
                panel=lambda *_args, **_kwargs: None,
                print_task=lambda *_args: None,
                diagnostic=lambda *_args: None,
            ),
        )

        result = compute_next_and_limits(
            ports, {"chain": "on"}, "cp", 2, SimpleNamespace()
        )

        self.assertEqual(result.child_due, due)
        self.assertEqual(result.meta, {"source": "cp"})

    def test_exhaustion_is_not_collapsed_after_terminal_presentation(self) -> None:
        observed: list[OccurrenceSearchExhausted] = []
        error = OccurrenceSearchExhausted(
            "monthly recurrence", reference="2026-01-01", limit=32,
            kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
        )

        def compute(_task: dict[str, object]):
            raise error

        with self.assertRaisesRegex(OccurrenceSearchExhausted, "monthly recurrence"):
            completion_compute_child_due(
                {"chain": "on", "chainID": "abcd1234", "link": 3},
                "anchor",
                compute_anchor_child_due=compute,
                compute_cp_child_due=compute,
                panel=lambda *_args, **_kwargs: None,
                print_task=lambda *_args: None,
                on_terminal=lambda exc: (observed.append(exc), True)[1],
            )
        self.assertIs(observed[0], error)

    def test_mutation_decision_retains_search_limit_kind(self) -> None:
        error = OccurrenceSearchExhausted(
            "monthly recurrence", reference="2026-01-01", limit=32,
            kind=OccurrenceSearchExhausted.SEARCH_LIMIT,
        )
        services = CompletionComputeServices(
            completion_compute_child_due=lambda *_args: (_ for _ in ()).throw(error),
            completion_until_or_fail=lambda *_args: None,
            completion_until_guard_or_stop=lambda *_args: True,
            completion_require_child_due_or_fail=lambda *_args: True,
            completion_warn_unreasonable_duration=lambda *_args: None,
            completion_caps=lambda *_args: (0, None, None, [], None),
            completion_cap_guard_or_stop=lambda *_args: True,
        )
        result = completion_compute_next_and_limits(
            {"chain": "on", "chainID": "abcd1234", "link": 3},
            "anchor",
            4,
            None,
            services=services,
        )
        self.assertIsInstance(result, CompletionLifecycleResult)
        assert isinstance(result, CompletionLifecycleResult)
        self.assertEqual(result.state, "retryable")
        self.assertEqual(result.diagnostic.failure_kind, "search_limit")


if __name__ == "__main__":
    unittest.main()
