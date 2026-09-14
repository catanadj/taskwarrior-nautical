from __future__ import annotations

import unittest
from types import SimpleNamespace

from nautical_core.hook_engine import handle_on_modify
from nautical_core.modify_models import CompletionLifecycleResult


class HookEngineContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
