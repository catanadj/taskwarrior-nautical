from __future__ import annotations

import io
import os
from types import SimpleNamespace
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import nautical_core.add_composition as add_composition


class AddCompositionContractTests(unittest.TestCase):
    @staticmethod
    def _host(core: object) -> SimpleNamespace:
        return SimpleNamespace(
            _INTEGRATION_CONTEXT=object(),
            _TASK_DATETIME_PARSER=object(),
            _CORE_READY=False,
            _MAX_JSON_BYTES=2048,
            _IMPORT_T0=0.0,
            _IMPORT_MS=None,
            time=SimpleNamespace(perf_counter=lambda: 1.0),
            core=core,
        )

    def test_invalid_max_json_size_keeps_safe_default(self) -> None:
        warnings = SimpleNamespace(warn_once_per_day_any=lambda *_args: None)

        class Core:
            MAX_JSON_BYTES = "not-an-integer"

            @staticmethod
            def _import_sibling(_name: str) -> SimpleNamespace:
                return warnings

        host = self._host(Core())

        add_composition.load_core(host)

        self.assertEqual(2048, host._MAX_JSON_BYTES)
        self.assertTrue(host._CORE_READY)

    def test_unexpected_max_json_size_failure_propagates(self) -> None:
        warnings = SimpleNamespace(warn_once_per_day_any=lambda *_args: None)

        class Core:
            @property
            def MAX_JSON_BYTES(self) -> int:
                raise RuntimeError("injected internal failure")

            @staticmethod
            def _import_sibling(_name: str) -> SimpleNamespace:
                return warnings

        host = self._host(Core())

        with self.assertRaisesRegex(RuntimeError, "injected internal failure"):
            add_composition.load_core(host)

    def test_core_loaded_warning_failure_is_reported_without_leaking_detail(self) -> None:
        class WarningSink:
            @staticmethod
            def warn_once_per_day_any(*_args: object) -> None:
                raise RuntimeError("sensitive warning detail")

        class Core:
            MAX_JSON_BYTES = 4096

            @staticmethod
            def _import_sibling(_name: str) -> WarningSink:
                return WarningSink()

        host = self._host(Core())
        stderr = io.StringIO()

        with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stderr(stderr):
            add_composition.load_core(host)

        self.assertTrue(host._CORE_READY)
        self.assertIn("on-add core-loaded warning failed", stderr.getvalue())
        self.assertIn("RuntimeError", stderr.getvalue())
        self.assertNotIn("sensitive warning detail", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
