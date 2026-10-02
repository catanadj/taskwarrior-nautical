"""Error contracts for the reconcile operator boundary."""

import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import unittest
from unittest.mock import patch

import nautical_core.tools.nautical_reconcile as reconcile
from nautical_core.task_models import TaskObservation


class ReconcileErrorContracts(unittest.TestCase):
    def test_configuration_verification_fails_closed_on_unexpected_fault(self) -> None:
        def broken_verifier() -> dict[str, bool]:
            raise RuntimeError("configuration snapshot unavailable")

        hook = SimpleNamespace(
            core=SimpleNamespace(configuration_drift=broken_verifier)
        )
        result = reconcile.configuration_verification(hook)

        self.assertEqual(result.status, "unavailable")
        self.assertIn("configuration snapshot unavailable", result.reason)

    def test_expiration_hop_limit_wraps_invalid_input_not_internal_faults(self) -> None:
        class BrokenIntegerConversion:
            def __int__(self) -> int:
                raise RuntimeError("integer conversion fault")

        with self.assertRaises(reconcile.argparse.ArgumentTypeError):
            reconcile._expiration_hop_limit("not-an-integer")
        with self.assertRaisesRegex(RuntimeError, "integer conversion fault"):
            reconcile._expiration_hop_limit(cast(str, BrokenIntegerConversion()))

    def test_failed_wave_preplan_is_diagnosed_and_candidate_retried_directly(self) -> None:
        taskdata = Path("/tmp/reconcile-error-contract")
        candidates = tuple(
            TaskObservation.from_mapping(
                {
                    "uuid": f"00000000-0000-4000-8000-00000000074{index}",
                    "status": "completed",
                    "chain": "on",
                    "chainID": f"error-contract-{index}",
                    "link": 1,
                },
                source_query="wave-fallback-contract",
            )
            for index in (0, 1)
        )
        lifecycle_service = SimpleNamespace(
            candidates=lambda: candidates, preflight_wave=lambda _rows: None
        )
        command_context = SimpleNamespace(command_budget=0)
        unit_of_work = SimpleNamespace(
            context=SimpleNamespace(
                taskdata=taskdata,
                command_prefix=("task",),
                configuration=SimpleNamespace(fingerprint="cfg", scheduler_fingerprint="sched"),
            ),
            commands=SimpleNamespace(
                calls=0,
                attempts=0,
                duration=0.0,
                failures=0,
                by_purpose={},
                context=command_context,
                budget_exceeded=False,
            ),
        )
        session = SimpleNamespace(
            snapshot=SimpleNamespace(_rows=None),
            control_plane=SimpleNamespace(drain_integrity=lambda *_args, **_kwargs: ()),
            mutation_gateway=object(),
            integrity_outbox=object(),
            lifecycle_service=lifecycle_service,
            lifecycle_application=object(),
            runtime_state=object(),
            audit_native_until=lambda *_args, **_kwargs: ([], [], "valid"),
        )
        calls: list[tuple[str, bool]] = []

        def candidate(*args: object, **kwargs: object) -> list[tuple[object, str]]:
            parent = args[2]
            assert isinstance(parent, dict)
            parent_uuid = str(parent["uuid"])
            applying = bool(kwargs["apply"])
            calls.append((parent_uuid, applying))
            if len(calls) == 1:
                raise RuntimeError("speculative planner failure")
            return []

        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}),
            patch.object(reconcile, "_build_reconcile_session", return_value=session),
            patch.object(reconcile, "_chain_generation_for_hook", return_value=object()),
            patch.object(reconcile, "_configuration_state", return_value=("valid", "")),
            patch.object(reconcile, "_reconcile_candidate", side_effect=candidate),
            patch.object(reconcile, "_repository", return_value=SimpleNamespace(metrics=lambda: {
                "calls": 0,
                "rows": 0,
                "seconds": 0.0,
                "slowest_seconds": 0.0,
            })),
            patch.object(reconcile, "_opportunistic_housekeeping", return_value={"status": "skipped"}),
            patch.object(reconcile, "render_result", return_value="{}"),
            contextlib.redirect_stderr(stderr),
        ):
            reconcile.main(
                ["--apply", "--json"],
                _apply_lease_held=True,
                _locked_taskdata=taskdata,
                _unit_of_work=unit_of_work,
            )

        self.assertIn((str(candidates[0].field("uuid").value), True), calls)
        self.assertIn("speculative planner failure", stderr.getvalue())

    def test_local_until_formatting_falls_back_to_raw_value(self) -> None:
        class BrokenParser:
            def parse(self, _value: object) -> tuple[None, str]:
                raise RuntimeError("parser unavailable")

        class ParsedValue:
            def parse(self, _value: object) -> tuple[object, None]:
                return object(), None

        raw = "2026-01-01T00:00:00Z"

        def fail_format(_value: object) -> str:
            raise RuntimeError("formatter unavailable")

        parser_failure = SimpleNamespace(
            datetime_parser=BrokenParser(), fmt_dt_local=lambda _value: "formatted"
        )
        formatter_failure = SimpleNamespace(
            datetime_parser=ParsedValue(), fmt_dt_local=fail_format
        )
        self.assertEqual(reconcile._format_local_until(parser_failure, raw), raw)
        self.assertEqual(reconcile._format_local_until(formatter_failure, raw), raw)

    def test_native_until_match_propagates_internal_parser_fault(self) -> None:
        class BrokenParser:
            def parse(self, _value: object) -> tuple[None, str]:
                raise RuntimeError("unexpected parser failure")

        task = TaskObservation.from_mapping(
            {"until": "2026-01-01T00:00:00Z"}, source_query="error-contract"
        )
        hook = SimpleNamespace(datetime_parser=BrokenParser())
        with self.assertRaisesRegex(RuntimeError, "unexpected parser failure"):
            reconcile._native_until_matches(task, "2026-01-02T00:00:00Z", hook)


if __name__ == "__main__":
    unittest.main()
