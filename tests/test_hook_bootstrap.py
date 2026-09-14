from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from nautical_core import hook_bootstrap
from nautical_core.hook_runtime import HookModuleAccess


class HookBootstrapTrustTests(unittest.TestCase):
    def test_optional_and_required_module_failures_preserve_import_detail(self) -> None:
        access = HookModuleAccess(
            {},
            {
                "broken": (
                    "_broken", "_broken_failed", "broken.py",
                    "nautical_core.no_such_module_for_test",
                )
            },
        )

        self.assertIsNone(access.module("broken", required=False))
        self.assertIn("ModuleNotFoundError", access.errors.get("broken", ""))
        with self.assertRaisesRegex(RuntimeError, "ModuleNotFoundError"):
            access.module("broken")

    def test_numeric_environment_values_fall_back_and_clamp_to_bounds(self) -> None:
        self.assertEqual(hook_bootstrap.env_int("VALUE", 5, env={"VALUE": "bad"}), 5)
        self.assertEqual(
            hook_bootstrap.env_int(
                "VALUE", 5, env={"VALUE": "-99"}, min_value=0, max_value=10
            ),
            0,
        )
        self.assertEqual(
            hook_bootstrap.env_int(
                "VALUE", 5, env={"VALUE": "999"}, min_value=0, max_value=10
            ),
            10,
        )
        self.assertEqual(
            hook_bootstrap.env_float(
                "VALUE", 1.5, env={"VALUE": "nan"}, min_value=0.1, max_value=10.0
            ),
            1.5,
        )
        self.assertEqual(
            hook_bootstrap.env_float(
                "VALUE", 1.5, env={"VALUE": "-4"}, min_value=0.1, max_value=10.0
            ),
            0.1,
        )

    def test_untrusted_override_is_not_a_bootstrap_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            hook_dir = root / "hooks"
            tw_dir = root / "taskwarrior"
            override = root / "override"
            (hook_dir / "nautical_core").mkdir(parents=True)
            (tw_dir / "nautical_core").mkdir(parents=True)
            (override / "nautical_core").mkdir(parents=True)
            (override / "hook_bootstrap.py").write_text("raise RuntimeError('untrusted')")
            (override / "nautical_core" / "hook_bootstrap.py").write_text("raise RuntimeError('untrusted')")
            override.chmod(0o777)
            candidates = hook_bootstrap.bootstrap_candidates(
                hook_dir,
                tw_dir,
                env={"NAUTICAL_CORE_PATH": str(override)},
            )
            self.assertNotIn(override / "hook_bootstrap.py", candidates)
            self.assertNotIn(override / "nautical_core" / "hook_bootstrap.py", candidates)

    def test_explicitly_trusted_override_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            override = root / "override"
            override.mkdir()
            candidates = hook_bootstrap.bootstrap_candidates(
                root / "hooks", root / "taskwarrior",
                env={
                    "NAUTICAL_CORE_PATH": str(override),
                    "NAUTICAL_TRUST_CORE_PATH": "1",
                },
            )
            self.assertIn(override / "hook_bootstrap.py", candidates)


if __name__ == "__main__":
    unittest.main()
