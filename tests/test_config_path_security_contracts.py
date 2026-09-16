from __future__ import annotations

import os
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from nautical_core import config_support, runtime


class ConfigPathSecurityContractTests(unittest.TestCase):
    def test_diagnostic_search_order_is_emitted_once_per_signature(self) -> None:
        with patch.dict(os.environ, {"NAUTICAL_DIAG": "1"}), redirect_stderr(io.StringIO()) as stream:
            previous = config_support._LAST_DIAG_SEARCH_ORDER
            config_support._LAST_DIAG_SEARCH_ORDER = None
            try:
                for _ in range(3):
                    config_support.config_paths(warn_env_config_missing=lambda _path: None)
            finally:
                config_support._LAST_DIAG_SEARCH_ORDER = previous
        self.assertEqual(stream.getvalue().count("Config search order:"), 1)

    def test_taskdata_context_prefers_argv_then_environment_then_fallback(self) -> None:
        argv_path, argv_rc, argv_source = runtime.resolve_task_data_context(
            argv=["api:2", "command:modify", "data:/tmp/nautical_core_arg_test"],
            env={"TASKDATA": "/tmp/nautical_core_env_test"},
            tw_dir="/tmp/nautical_core_fallback_test",
        )
        self.assertEqual(argv_path, "/tmp/nautical_core_arg_test")
        self.assertTrue(argv_rc)
        self.assertEqual(argv_source, "argv")

        env_path, env_rc, env_source = runtime.resolve_task_data_context(
            argv=["api:2", "command:modify"],
            env={"TASKDATA": "/tmp/nautical_core_env_test"},
            tw_dir="/tmp/nautical_core_fallback_test",
        )
        self.assertEqual(env_path, "/tmp/nautical_core_env_test")
        self.assertTrue(env_rc)
        self.assertEqual(env_source, "env")

        fallback, fallback_rc, fallback_source = runtime.resolve_task_data_context(
            argv=["api:2", "command:modify"],
            env={},
            tw_dir="/tmp/nautical_core_fallback_test",
        )
        self.assertEqual(fallback, "/tmp/nautical_core_fallback_test")
        self.assertFalse(fallback_rc)
        self.assertEqual(fallback_source, "fallback")

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are required")
    def test_world_writable_taskdata_falls_back_unless_override_is_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unsafe = os.path.join(directory, "unsafe-data-dir")
            os.mkdir(unsafe)
            os.chmod(unsafe, 0o777)
            fallback = os.path.join(directory, "safe-fallback")

            rejected = runtime.resolve_task_data_context(
                argv=["api:2", "command:modify", f"data:{unsafe}"],
                env={},
                tw_dir=fallback,
            )
            self.assertEqual(rejected, (fallback, False, "fallback"))

            trusted = runtime.resolve_task_data_context(
                argv=["api:2", "command:modify", f"data:{unsafe}"],
                env={"NAUTICAL_TRUST_TASKDATA_PATH": "1"},
                tw_dir=fallback,
            )
            self.assertEqual(trusted, (unsafe, True, "argv"))

    def test_taskdata_parent_traversal_falls_back_without_rc_override(self) -> None:
        taskdata, use_rc_location, source = runtime.resolve_task_data_context(
            argv=["api:2", "command:modify", "data:../nautical_bad_dir"],
            env={},
            tw_dir="/tmp/nautical_core_fallback_test",
        )

        self.assertEqual(source, "fallback")
        self.assertFalse(use_rc_location)
        self.assertEqual(taskdata, "/tmp/nautical_core_fallback_test")

    def test_untrusted_config_override_rejects_parent_traversal(self) -> None:
        with patch.dict(os.environ, {"NAUTICAL_CONFIG": "../nautical.toml"}):
            os.environ.pop("NAUTICAL_TRUST_CONFIG_PATH", None)
            paths = config_support.config_paths(
                warn_env_config_missing=lambda _path: None
            )

        self.assertEqual(paths, [])

    def test_trusted_config_override_allows_parent_traversal(self) -> None:
        with patch.dict(
            os.environ,
            {
                "NAUTICAL_CONFIG": "../nautical.toml",
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
            },
        ):
            path = os.path.abspath("../nautical.toml")
            paths = config_support.config_paths(
                warn_env_config_missing=lambda _path: None
            )

        self.assertEqual(paths, [path])


if __name__ == "__main__":
    unittest.main()
