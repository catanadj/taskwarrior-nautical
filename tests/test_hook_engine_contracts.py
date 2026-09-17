from __future__ import annotations

import unittest
from types import SimpleNamespace

from nautical_core.hook_engine import handle_on_modify
from nautical_core.modify_models import CompletionLifecycleResult


class HookEngineContractTests(unittest.TestCase):
    def test_delete_route_loads_core_only_for_nautical_tasks(self) -> None:
        calls = {"load": 0, "deleted": 0, "completion": 0, "non_completion": 0}

        class Services:
            def result(self, *, task, sanitize):
                return {"task": task, "sanitize": sanitize}

            def has_nautical_fields(self, task):
                return bool(task.get("anchor") or task.get("chainID"))

            def load_core(self):
                calls["load"] += 1

            def diag(self, _message):
                return None

            def fail_and_exit(self, *_args):
                raise AssertionError("delete must not fail")

            def handle_non_completion(self, *_args):
                calls["non_completion"] += 1

            def handle_completion(self, *_args):
                calls["completion"] += 1

            def handle_deleted(self, old, new, *_args):
                self_outer.assertEqual(old["status"], "pending")
                self_outer.assertEqual(new["status"], "deleted")
                calls["deleted"] += 1

        self_outer = self
        for task, expected_loads, expected_deletes in (
            ({"uuid": "00000000-0000-4000-8000-000000000301", "status": "pending"}, 0, 0),
            ({
                "uuid": "00000000-0000-4000-8000-000000000302",
                "status": "pending",
                "anchor": "w:mon",
                "chainID": "chain302",
            }, 1, 1),
        ):
            new = dict(task, status="deleted")
            request = SimpleNamespace(
                old=task,
                new=new,
                runtime=SimpleNamespace(uow=object(), lifecycle_result=None),
            )
            result = handle_on_modify(request, Services())
            self.assertEqual(result, {"task": new, "sanitize": False})
            self.assertIs(result["task"], new)
            self.assertEqual(calls["load"], expected_loads)
            self.assertEqual(calls["deleted"], expected_deletes)
            self.assertEqual(calls["completion"], 0)
            self.assertEqual(calls["non_completion"], 0)

    def test_completion_lifecycle_result_is_retained_without_alternate_output(self) -> None:
        lifecycle = CompletionLifecycleResult(
            state="retryable",
            reason="planner unavailable",
        )
        runtime = SimpleNamespace(lifecycle_result=None, uow=object())
        request = SimpleNamespace(
            old={
                "uuid": "00000000-0000-4000-8000-000000000303",
                "status": "pending",
                "chainID": "chain303",
            },
            new={
                "uuid": "00000000-0000-4000-8000-000000000303",
                "status": "completed",
                "chainID": "chain303",
            },
            runtime=runtime,
        )

        class Services:
            def result(self, *, task, sanitize):
                return {"task": task, "sanitize": sanitize}

            def has_nautical_fields(self, task):
                return bool(task.get("chainID"))

            def load_core(self):
                return None

            def diag(self, _message):
                return None

            def fail_and_exit(self, *_args):
                raise AssertionError("completion must not fail")

            def handle_completion(self, *_args):
                return lifecycle

            def handle_non_completion(self, *_args):
                raise AssertionError("non-completion route selected")

            def handle_deleted(self, *_args):
                raise AssertionError("delete route selected")

        result = handle_on_modify(request, Services())

        self.assertIsNone(result)
        self.assertIs(runtime.lifecycle_result, lifecycle)

    def test_scheduler_completion_failure_vetoes_taskwarrior_completion(self) -> None:
        lifecycle = SimpleNamespace(
            state="retryable",
            reason="These anchors joined with '+' don't share any possible date.",
            diagnostic=SimpleNamespace(failure_kind="scheduler_error"),
        )
        runtime = SimpleNamespace(lifecycle_result=None, uow=object())
        request = SimpleNamespace(
            old={"uuid": "00000000-0000-4000-8000-000000000304", "status": "pending", "chainID": "chain304"},
            new={"uuid": "00000000-0000-4000-8000-000000000304", "status": "completed", "chainID": "chain304"},
            runtime=runtime,
        )
        failures = []

        class Services:
            def result(self, *, task, sanitize):
                return {"task": task, "sanitize": sanitize}

            def has_nautical_fields(self, task):
                return bool(task.get("chainID"))

            def load_core(self):
                return None

            def diag(self, _message):
                return None

            def fail_and_exit(self, title, message):
                failures.append((title, message))
                raise RuntimeError("veto")

            def handle_completion(self, *_args):
                return lifecycle

            def handle_non_completion(self, *_args):
                raise AssertionError("non-completion route selected")

            def handle_deleted(self, *_args):
                raise AssertionError("delete route selected")

        with self.assertRaisesRegex(RuntimeError, "veto"):
            handle_on_modify(request, Services())

        self.assertEqual(
            failures,
            [("Completion blocked", "These anchors joined with '+' don't share any possible date.")],
        )


if __name__ == "__main__":
    unittest.main()
