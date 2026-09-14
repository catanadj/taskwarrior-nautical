"""In-process command contracts for the query tool."""

import contextlib
import io
import json
import unittest
from unittest.mock import patch
from unittest.mock import patch

from nautical_core.tools import nautical_query


class QueryCommandContractsTests(unittest.TestCase):
    def _run(self, arguments: list[str]) -> tuple[int, dict[str, object]]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            exit_code = nautical_query.main(arguments)
        return exit_code, json.loads(output.getvalue())

    def test_rejects_trailing_oversized_and_deep_requests(self) -> None:
        requests = (
            '{"selector": {"all_tasks": true}, "count": 1} trailing',
            "{" + '"x":{' * 70 + '"v":1' + "}" * 70 + "}",
            "{" + "\"padding\":\"" + ("x" * (1_048_576 + 1)) + "\"}",
        )
        for request in requests:
            with self.subTest(request_size=len(request)):
                exit_code, payload = self._run(["occurrences", "--request", request])
                self.assertEqual(exit_code, 2)
                self.assertEqual(payload["failure"]["code"], "invalid_request")

    def test_invalid_request_cli_emits_one_versioned_json_document(self) -> None:
        exit_code, payload = self._run(["occurrences", "--request", "{}"])
        self.assertEqual(exit_code, 2)
        self.assertEqual(payload["schema"], "nautical.query.occurrences")
        self.assertEqual(payload["status"], "invalid")
        self.assertEqual(payload["failure"]["code"], "invalid_request")

    def test_serializer_preserves_unicode_compactness_and_budget(self) -> None:
        from nautical_core.operator_models import OperatorV2Result, OperatorV2Status

        result = OperatorV2Result(
            schema="nautical.query.occurrences",
            operation="occurrences",
            status=OperatorV2Status.FOUND,
            payload={"description": "Méditation ⚓\nweekly"},
        )
        budget = type("Budget", (), {"report": lambda _self: {"commands": 2}})()
        output = io.StringIO()
        with patch("sys.stdout", output):
            exit_code = nautical_query._emit(result, budget=budget)

        encoded = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertEqual(len(encoded.splitlines()), 1)
        self.assertIn("Méditation ⚓", encoded)
        self.assertNotIn("\\u2693", encoded)
        payload = json.loads(encoded)
        self.assertEqual(payload["description"], "Méditation ⚓\nweekly")
        self.assertEqual(payload["budget"], {"commands": 2})

    def test_flags_reach_the_unavailable_read_boundary(self) -> None:
        def unavailable(**_kwargs: object) -> object:
            raise RuntimeError("test read boundary unavailable")

        with patch.object(nautical_query, "build_operator_uow", unavailable):
            exit_code, payload = self._run(
                [
                    "occurrences",
                    "--all",
                    "--after",
                    "2026-08-24",
                    "--count",
                    "2",
                    "--max-total-occurrences",
                    "3",
                ]
            )
        self.assertEqual(exit_code, 3)
        self.assertEqual(payload["failure"]["code"], "query_unavailable")

    def test_capabilities_are_taskwarrior_free_and_versioned(self) -> None:
        def unavailable(**_kwargs: object) -> object:
            raise AssertionError("capabilities must not build a Taskwarrior UoW")

        with patch.object(nautical_query, "build_operator_uow", unavailable):
            exit_code, payload = self._run(["capabilities"])
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema"], "nautical.query.capabilities")
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["operations"], ["occurrences", "next", "integrity"])
        limits = payload["limits"]
        self.assertGreaterEqual(limits["hard"]["occurrences"], limits["defaults"]["occurrences"])
        guide = payload.get("guide", {})
        self.assertIn("cp", guide.get("concepts", {}))
        self.assertIn("anchor", guide.get("concepts", {}))
        self.assertTrue(any("query occurrences" in item for item in guide.get("quick_start", [])))
        self.assertIn("task_range_rule", guide)


if __name__ == "__main__":
    unittest.main()
