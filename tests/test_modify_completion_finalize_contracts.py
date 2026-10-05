from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import get_args
import unittest

import nautical_core.modify_completion_flow as flow
from nautical_core.modify_models import CompletionLifecycleResult, CompletionSpawnResult
from nautical_core.modify_runtime import ModifyRuntimeState


class ModifyCompletionFinalizeContractTests(unittest.TestCase):
    def test_modify_chain_state_callback_returns_the_named_runtime_state(self) -> None:
        from nautical_core.modify_models import CompletionRuntimeState, ModifyChainStateCallback

        self.assertIs(get_args(ModifyChainStateCallback)[1], CompletionRuntimeState)

    def test_lifecycle_presentation_diagnostic_failure_does_not_block_task_response(self) -> None:
        diagnostic_messages = []
        printed_tasks = []

        def failed_diagnostic(message):
            diagnostic_messages.append(message)
            raise OSError("stderr is unavailable")

        services = flow.CompletionFinalizeServices(
            build_and_spawn_child=lambda *_args, **_kwargs: None,
            seed_runtime_lookup_tasks=lambda *_args, **_kwargs: None,
            modify_chain_state=ModifyRuntimeState,
            lifecycle_read_service=None,
            chain_health_advice=lambda *_args, **_kwargs: None,
            chain_integrity_warnings=lambda *_args, **_kwargs: [],
            render_anchor_completion_feedback=lambda **_kwargs: None,
            render_cp_completion_feedback=lambda **_kwargs: None,
            render_lifecycle_result=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("optional lifecycle panel failed")
            ),
            print_task=printed_tasks.append,
            diag_summary=lambda: None,
            show_analytics=False,
            check_integrity=False,
            analytics_style="clinical",
            diagnostic=failed_diagnostic,
        )
        task = {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "status": "completed",
            "description": "completed task",
        }
        now = datetime(2026, 9, 14, tzinfo=timezone.utc)

        result = flow.finalize_completion_modify(
            new=task,
            ctx=SimpleNamespace(parent_short="00000000", base_no=1, next_no=2, kind="cp", chain_id=""),
            computed=SimpleNamespace(
                child_due=now,
                meta={},
                cpmax=0,
                until_dt=None,
            ),
            now_utc=now,
            need_chain=False,
            chain_snapshot_loaded=True,
            preloaded_chain=[],
            preloaded_chain_by_link={},
            preloaded_chain_by_short={},
            chain_id="",
            services=services,
        )

        self.assertEqual(result.state, "retryable")
        self.assertEqual(printed_tasks, [task])
        self.assertEqual(len(diagnostic_messages), 1)
        self.assertIn("completion lifecycle presentation failed", diagnostic_messages[0])

    def test_optional_completion_enrichment_and_diagnostic_failures_preserve_task_result(self) -> None:
        diagnostic_messages = []
        printed_tasks = []

        def failed_optional_diagnostic(message):
            diagnostic_messages.append(message)
            raise OSError("stderr is unavailable")

        services = flow.CompletionFinalizeServices(
            build_and_spawn_child=lambda *_args, **_kwargs: CompletionSpawnResult(
                child={"uuid": "00000000-0000-4000-8000-000000000222"},
                child_short="beeswax",
                stripped_attrs=[],
                verified=False,
                deferred_spawn=False,
                spawn_intent_id=None,
                outcome_state="applied",
            ),
            seed_runtime_lookup_tasks=lambda *_args, **_kwargs: None,
            modify_chain_state=ModifyRuntimeState,
            lifecycle_read_service=SimpleNamespace(
                get_chain_export=lambda _chain_id: (_ for _ in ()).throw(
                    RuntimeError("optional chain refresh failed")
                )
            ),
            chain_health_advice=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("optional analytics failed")
            ),
            chain_integrity_warnings=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("optional integrity summary failed")
            ),
            render_anchor_completion_feedback=lambda **_kwargs: None,
            render_cp_completion_feedback=lambda **_kwargs: None,
            render_lifecycle_result=lambda *_args, **_kwargs: None,
            print_task=printed_tasks.append,
            diag_summary=lambda: None,
            show_analytics=True,
            check_integrity=True,
            analytics_style="clinical",
            diagnostic=failed_optional_diagnostic,
        )
        now = datetime(2026, 9, 14, tzinfo=timezone.utc)
        task = {"anchor": "w:mon", "chainID": "abcd1234"}

        result = flow.finalize_completion_modify(
            new=task,
            ctx=SimpleNamespace(parent_short="00000000", base_no=1, next_no=2, kind="anchor", chain_id="abcd1234"),
            computed=SimpleNamespace(
                child_due=now,
                meta={"target_field": "due"},
                dnf=[[{"typ": "w", "spec": "mon", "mods": {}}]],
                until_dt=None,
                cpmax=0,
                cap_no=None,
                finals=[],
                until_cap_no=None,
            ),
            now_utc=now,
            need_chain=True,
            chain_snapshot_loaded=False,
            preloaded_chain=[{"uuid": "00000000-0000-4000-8000-000000000111"}],
            preloaded_chain_by_link={},
            preloaded_chain_by_short={},
            chain_id="abcd1234",
            services=services,
        )

        self.assertEqual(result.state, "applied")
        self.assertEqual(printed_tasks, [task])
        self.assertEqual(len(diagnostic_messages), 3)
        self.assertIn("completion chain refresh failed", diagnostic_messages[0])
        self.assertIn("completion analytics failed", diagnostic_messages[1])
        self.assertIn("completion integrity presentation failed", diagnostic_messages[2])

    def test_hidden_analytics_does_not_hide_integrity_checks_or_lifecycle_result(self) -> None:
        for make_result in (
            lambda: CompletionLifecycleResult("queued", deferred_spawn=True),
            lambda: CompletionLifecycleResult("applied", deferred_spawn=True),
        ):
            with self.subTest(factory=make_result), self.assertRaises(ValueError):
                make_result()

        captured = {}

        def render_anchor_feedback(**kwargs):
            request = kwargs["request"]
            captured["analytics_advice"] = request.analytics_advice
            captured["integrity_warnings"] = request.integrity_warnings
            captured["lifecycle_result"] = request.lifecycle_result

        services = flow.CompletionFinalizeServices(
            build_and_spawn_child=lambda *_args, **_kwargs: CompletionSpawnResult(
                child={"uuid": "00000000-0000-4000-8000-000000000222"},
                child_short="beeswax",
                stripped_attrs=[],
                verified=False,
                deferred_spawn=False,
                spawn_intent_id=None,
                outcome_state="applied",
            ),
            seed_runtime_lookup_tasks=lambda *_args, **_kwargs: None,
            modify_chain_state=ModifyRuntimeState,
            lifecycle_read_service=None,
            chain_health_advice=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("analytics must not be computed when hidden")
            ),
            chain_integrity_warnings=lambda *_args, **_kwargs: ["missing link"],
            render_anchor_completion_feedback=render_anchor_feedback,
            render_cp_completion_feedback=lambda *_args, **_kwargs: None,
            render_lifecycle_result=lambda *_args, **_kwargs: None,
            print_task=lambda *_args, **_kwargs: None,
            diag_summary=lambda *_args, **_kwargs: None,
            show_analytics=False,
            check_integrity=True,
            analytics_style="clinical",
        )
        now = datetime(2026, 9, 14, tzinfo=timezone.utc)
        result = flow.finalize_completion_modify(
            new={"anchor": "w:mon", "chainID": "abcd1234"},
            ctx=SimpleNamespace(parent_short="00000000", base_no=1, next_no=2, kind="anchor", chain_id=""),
            computed=SimpleNamespace(
                child_due=now,
                meta={"target_field": "due"},
                dnf=[[{"typ": "w", "spec": "mon", "mods": {}}]],
                until_dt=None,
                cpmax=0,
                cap_no=None,
                finals=[],
                until_cap_no=None,
            ),
            now_utc=now,
            need_chain=True,
            chain_snapshot_loaded=True,
            preloaded_chain=[{"uuid": "00000000-0000-4000-8000-000000000111"}],
            preloaded_chain_by_link=None,
            preloaded_chain_by_short=None,
            chain_id="",
            services=services,
        )

        self.assertIsNone(captured["analytics_advice"])
        self.assertEqual(captured["integrity_warnings"], ["missing link"])
        self.assertEqual(result.state, "applied")
        self.assertEqual(result.child_short, "beeswax")
        self.assertIsNotNone(result.diagnostic)
        self.assertEqual(result.diagnostic.stage, "finalize")
        self.assertEqual((result.diagnostic.parent_link, result.diagnostic.child_link), (1, 2))
        self.assertIs(captured["lifecycle_result"], result)


if __name__ == "__main__":
    unittest.main()
