from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

import nautical_core.diagnostic_warnings as diagnostic_warnings
import nautical_core.diagnostic_models as diagnostic_models
import nautical_core.runtime as runtime


class DiagnosticWarningsContractTests(unittest.TestCase):
    def test_structured_diagnostic_keeps_stdout_clean_and_record_stable(self) -> None:
        event = diagnostic_models.DiagnosticEvent(
            "chain.export_failed",
            "Taskwarrior lock active",
            hook="on-modify",
            level="warning",
            context={"chain_id": "abcd1234"},
        )
        record = event.to_log_record()
        self.assertEqual(record["code"], "chain.export_failed")
        self.assertEqual((record["level"], record["hook"]), ("warning", "on-modify"))
        self.assertEqual(record["context"], {"chain_id": "abcd1234"})

        stdout = StringIO()
        stderr = StringIO()
        with (
            patch.dict(
                os.environ,
                {"NAUTICAL_DIAG": "1", "NAUTICAL_DIAG_LOG": ""},
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            runtime.diag(event, "on-modify", "/tmp/nautical-diagnostic-contract")

        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Taskwarrior lock active", stderr.getvalue())

    def test_structured_diag_log_preserves_fields_and_redacts_sensitive_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stderr = StringIO()
            with (
                patch.dict(os.environ, {"NAUTICAL_DIAG": "0", "NAUTICAL_DIAG_LOG": "1"}),
                redirect_stderr(stderr),
            ):
                runtime.diag(
                    {"msg": "hello", "description": "private", "ok": "keep", "event": "test"},
                    data_dir=directory,
                )

            record = json.loads(
                (Path(directory) / ".nautical_diag.jsonl").read_text(encoding="utf-8").strip()
            )
            data = record["data"]
            self.assertEqual(data["description"], "[redacted]")
            self.assertEqual(data["ok"], "keep")
            self.assertEqual(data["msg"], "hello")
            self.assertEqual(data["event"], "test")
            self.assertIn("pid", record)
            self.assertIn("cwd", record)
            self.assertEqual(stderr.getvalue(), "")

    def test_diag_log_redacts_sensitive_legacy_json_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            message = json.dumps(
                {"description": "secret", "notes": "hidden", "ok": "keep"}
            )
            stderr = StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"NAUTICAL_DIAG": "0", "NAUTICAL_DIAG_LOG": "1"},
                ),
                redirect_stderr(stderr),
            ):
                runtime.diag(message, "on-modify", directory)

            path = Path(directory) / ".nautical_diag.jsonl"
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("secret", content)
            self.assertNotIn("hidden", content)
            self.assertIn("[redacted]", content)
            self.assertIn("keep", content)
            self.assertEqual(stderr.getvalue(), "")

    def test_diag_log_rotation_moves_oversized_file_and_writes_new_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".nautical_diag.jsonl"
            path.write_text("x" * 64, encoding="utf-8")
            with patch.dict(
                os.environ,
                {
                    "NAUTICAL_DIAG": "0",
                    "NAUTICAL_DIAG_LOG": "1",
                    "NAUTICAL_DIAG_LOG_MAX_BYTES": "20",
                },
            ):
                runtime.diag_log("rotate me", "on-modify", directory)

            overflow = list(Path(directory).glob(".nautical_diag.overflow.*.jsonl"))
            self.assertTrue(overflow)
            self.assertEqual(overflow[0].read_text(encoding="utf-8"), "x" * 64)
            self.assertIn("rotate me", path.read_text(encoding="utf-8"))

    def test_rate_limited_warning_emits_only_once_inside_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stderr = StringIO()
            with (
                patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}),
                redirect_stderr(stderr),
            ):
                diagnostic_warnings.warn_rate_limited_any(
                    "rate-limit-contract", "rate limit message", cache_dir=directory,
                    min_interval_s=3600,
                )
                diagnostic_warnings.warn_rate_limited_any(
                    "rate-limit-contract", "rate limit message", cache_dir=directory,
                    min_interval_s=3600,
                )

            self.assertEqual(stderr.getvalue().splitlines(), ["rate limit message"])

    def test_required_diagnostic_creates_daily_stamp_and_emits_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stderr = StringIO()
            with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stderr(stderr):
                diagnostic_warnings.warn_once_per_day(
                    "contract", "diagnostic message", cache_dir=directory, require_diag=True
                )

            stamp = Path(directory) / ".diag_contract.stamp"
            self.assertEqual(stamp.read_text(encoding="utf-8"), date.today().isoformat())
            self.assertEqual(stderr.getvalue(), "diagnostic message\n")

    def test_required_diagnostic_is_silent_and_does_not_stamp_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stderr = StringIO()
            with patch.dict(os.environ):
                os.environ.pop("NAUTICAL_DIAG", None)
                with redirect_stderr(stderr):
                    diagnostic_warnings.warn_once_per_day(
                        "contract", "diagnostic message", cache_dir=directory, require_diag=True
                    )

            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(stderr.getvalue(), "")

    def test_optional_diagnostic_stamps_silently_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stderr = StringIO()
            with patch.dict(os.environ):
                os.environ.pop("NAUTICAL_DIAG", None)
                with redirect_stderr(stderr):
                    diagnostic_warnings.warn_once_per_day(
                        "contract", "diagnostic message", cache_dir=directory, require_diag=False
                    )

            self.assertTrue((Path(directory) / ".diag_contract.stamp").is_file())
            self.assertEqual(stderr.getvalue(), "")

    def test_daily_warning_does_not_hide_unexpected_filesystem_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                diagnostic_warnings.os,
                "makedirs",
                side_effect=RuntimeError("warning filesystem implementation failed"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "warning filesystem implementation failed"
                ):
                    diagnostic_warnings.warn_once_per_day(
                        "contract", "diagnostic message", cache_dir=directory, require_diag=False
                    )

    def test_rate_limited_warning_does_not_hide_unexpected_filesystem_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                diagnostic_warnings.os,
                "makedirs",
                side_effect=RuntimeError("warning filesystem implementation failed"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "warning filesystem implementation failed"
                ):
                    diagnostic_warnings.warn_rate_limited_any(
                        "contract", "diagnostic message", cache_dir=directory
                    )

    def test_warning_filesystem_unavailability_remains_best_effort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            warning_calls = (
                lambda: diagnostic_warnings.warn_once_per_day(
                    "contract", "diagnostic message", cache_dir=directory, require_diag=False
                ),
                lambda: diagnostic_warnings.warn_rate_limited_any(
                    "contract", "diagnostic message", cache_dir=directory
                ),
            )
            for warn in warning_calls:
                with patch.object(
                    diagnostic_warnings.os,
                    "makedirs",
                    side_effect=OSError("disk unavailable"),
                ):
                    warn()


if __name__ == "__main__":
    unittest.main()
