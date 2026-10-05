from __future__ import annotations

from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import nautical_core.hook_bootstrap as hook_bootstrap
from nautical_core.hook_runtime import HookModuleAccess


class HookBootstrapTrustTests(unittest.TestCase):
    def test_core_import_identity_probe_failure_reloads_selected_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("# package\n", encoding="utf-8")
            existing = ModuleType("nautical_core")

            def broken_identity(name: str) -> None:
                if name == "__file__":
                    raise RuntimeError("existing module identity unavailable")

            existing.__getattr__ = broken_identity
            replacement = ModuleType("nautical_core")
            with (
                patch.dict(hook_bootstrap.sys.modules, {"nautical_core": existing}),
                patch.object(hook_bootstrap.sys, "path", list(hook_bootstrap.sys.path)),
                patch.object(hook_bootstrap.importlib, "import_module", return_value=replacement) as importer,
            ):
                module, target, error = hook_bootstrap.import_core_package(root)

        self.assertIs(module, replacement)
        self.assertEqual(target, package / "__init__.py")
        self.assertIsNone(error)
        importer.assert_called_once_with("nautical_core")

    def test_core_import_failure_is_returned_with_original_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("# package\n", encoding="utf-8")
            failure = RuntimeError("core import defect")
            with (
                patch.dict(hook_bootstrap.sys.modules, {"nautical_core": None}),
                patch.object(hook_bootstrap.sys, "path", list(hook_bootstrap.sys.path)),
                patch.object(hook_bootstrap.importlib, "import_module", side_effect=failure),
            ):
                module, target, error = hook_bootstrap.import_core_package(root)

        self.assertIsNone(module)
        self.assertEqual(target, package / "__init__.py")
        self.assertIs(error, failure)

    def test_helper_module_import_failure_is_returned_with_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper = root / "helper.py"
            helper.write_text(
                "raise RuntimeError('helper import defect')\n",
                encoding="utf-8",
            )

            module, helper_path, error = hook_bootstrap.load_core_helper_module(
                root,
                "helper.py",
                "nautical_test_helper_import_failure",
            )

        self.assertIsNone(module)
        self.assertEqual(helper_path, helper)
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "helper import defect")

    def test_core_override_resolution_contains_expected_path_failures(self) -> None:
        default_base = Path("/default-core")
        candidate = Path("/configured-core")
        with patch.object(type(candidate), "resolve", side_effect=RuntimeError("symlink loop")):
            selected = hook_bootstrap.trusted_core_base(
                default_base,
                env={"NAUTICAL_CORE_PATH": "/configured-core"},
            )

        self.assertEqual(selected, default_base)

    def test_core_override_resolution_does_not_hide_internal_failure(self) -> None:
        candidate = Path("/configured-core")
        with patch.object(type(candidate), "resolve", side_effect=KeyError("resolver defect")):
            with self.assertRaisesRegex(KeyError, "resolver defect"):
                hook_bootstrap.trusted_core_base(
                    Path("/default-core"),
                    env={"NAUTICAL_CORE_PATH": "/configured-core"},
                )

    def test_core_override_security_probe_contains_oserror(self) -> None:
        default_base = Path("/default-core")
        candidate = Path("/configured-core")
        with (
            patch.object(type(candidate), "resolve", return_value=candidate),
            patch.object(hook_bootstrap.os, "stat", side_effect=OSError("stat failed")),
        ):
            selected = hook_bootstrap.trusted_core_base(
                default_base,
                env={"NAUTICAL_CORE_PATH": "/configured-core"},
            )

        self.assertEqual(selected, default_base)

    def test_core_override_security_probe_does_not_hide_internal_failure(self) -> None:
        candidate = Path("/configured-core")
        with patch.object(
            type(candidate), "resolve", return_value=candidate
        ):
            with patch.object(
                hook_bootstrap.os,
                "stat",
                side_effect=RuntimeError("stat adapter defect"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stat adapter defect"):
                    hook_bootstrap.trusted_core_base(
                        Path("/default-core"),
                        env={"NAUTICAL_CORE_PATH": "/configured-core"},
                    )

    def test_unsafe_override_fallback_survives_diagnostic_failure(self) -> None:
        default_base = Path("/default-core")
        candidate = Path("/configured-core")
        with (
            patch.object(type(candidate), "resolve", return_value=candidate),
            patch.object(hook_bootstrap.os, "stat", side_effect=OSError("unsafe path")),
            patch.object(
                hook_bootstrap.sys.stderr,
                "write",
                side_effect=RuntimeError("diagnostic stream failed"),
            ),
        ):
            selected = hook_bootstrap.trusted_core_base(
                default_base,
                env={"NAUTICAL_CORE_PATH": "/configured-core"},
                diag_enabled=True,
            )

        self.assertEqual(selected, default_base)

    def test_core_target_probe_contains_expected_oserror(self) -> None:
        base = Path("/unavailable-hook-base")
        with patch.object(type(base), "is_file", side_effect=OSError("unavailable")):
            self.assertIsNone(hook_bootstrap.core_target_from_base(base))

    def test_core_target_probe_does_not_hide_unexpected_failure(self) -> None:
        base = Path("/unavailable-hook-base")
        with patch.object(
            type(base), "is_file", side_effect=RuntimeError("path adapter defect")
        ):
            with self.assertRaisesRegex(RuntimeError, "path adapter defect"):
                hook_bootstrap.core_target_from_base(base)

    def test_helper_path_probe_contains_expected_oserror(self) -> None:
        base = Path("/unavailable-hook-base")
        with patch.object(type(base), "is_file", side_effect=OSError("unavailable")):
            module, helper_path, error = hook_bootstrap.load_core_helper_module(
                base,
                "optional_helper.py",
                "nautical_test_optional_helper",
            )

        self.assertIsNone(module)
        self.assertIsNone(helper_path)
        self.assertIsNone(error)

    def test_helper_path_probe_does_not_hide_unexpected_failure(self) -> None:
        base = Path("/unavailable-hook-base")
        with patch.object(
            type(base), "is_file", side_effect=RuntimeError("path adapter defect")
        ):
            with self.assertRaisesRegex(RuntimeError, "path adapter defect"):
                hook_bootstrap.load_core_helper_module(
                    base,
                    "optional_helper.py",
                    "nautical_test_optional_helper",
                )

    def test_core_target_requires_package_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "nautical_core"
            package.mkdir()
            package_init = package / "__init__.py"
            package_init.write_text("# package core\n", encoding="utf-8")
            legacy_module = root / "nautical_core.py"
            legacy_module.write_text("# legacy core\n", encoding="utf-8")

            self.assertEqual(hook_bootstrap.core_target_from_base(root), package_init)
            self.assertIsNone(hook_bootstrap.core_target_from_base(legacy_module))

    def test_light_taskdata_resolution_matches_hook_precedence(self) -> None:
        import nautical_core.config_support as config_support

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_dir = root / "env-data"
            arg_dir = root / "arg-data"
            env_dir.mkdir()
            arg_dir.mkdir()
            env = {"TASKDATA": str(env_dir)}
            path_support = config_support

            from_env = hook_bootstrap.resolve_task_data_context_light(
                path_support=path_support,
                argv=[],
                env=env,
                tw_dir=str(root),
            )
            from_argv = hook_bootstrap.resolve_task_data_context_light(
                path_support=path_support,
                argv=[f"data.location:{arg_dir}"],
                env=env,
                tw_dir=str(root),
            )
            fallback = hook_bootstrap.resolve_task_data_context_light(
                path_support=path_support,
                argv=[],
                env={},
                tw_dir=str(root),
            )

        self.assertEqual(from_env, (str(env_dir), True, "env"))
        self.assertEqual(from_argv, (str(arg_dir), True, "argv"))
        self.assertEqual(fallback, (str(root), False, "fallback"))

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

    def test_module_loader_does_not_hide_module_initialization_defects(self) -> None:
        access = HookModuleAccess(
            {},
            {"broken": ("_broken", "_broken_failed", "broken.py", "broken_module")},
        )
        with patch(
            "nautical_core.hook_runtime.importlib.import_module",
            side_effect=RuntimeError("module initialization invariant failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "module initialization invariant failed"):
                access.module("broken", required=False)

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
            self.assertIn(tw_dir / "nautical_core" / "hook_bootstrap.py", candidates)

    def test_candidate_resolution_failure_uses_only_builtin_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook_dir = root / "hooks"
            tw_dir = root / "taskwarrior"
            candidate = root / "override"
            with patch.object(type(candidate), "resolve", side_effect=OSError("resolve failed")):
                paths = hook_bootstrap.bootstrap_candidates(
                    hook_dir,
                    tw_dir,
                    env={"NAUTICAL_CORE_PATH": str(candidate)},
                )

        self.assertEqual(
            paths,
            (
                hook_dir / "nautical_core" / "hook_bootstrap.py",
                tw_dir / "nautical_core" / "hook_bootstrap.py",
            ),
        )

    def test_candidate_resolution_does_not_hide_internal_failure(self) -> None:
        candidate = Path("/override")
        with patch.object(type(candidate), "resolve", side_effect=KeyError("resolve defect")):
            with self.assertRaisesRegex(KeyError, "resolve defect"):
                hook_bootstrap.bootstrap_candidates(
                    Path("/hooks"),
                    Path("/taskwarrior"),
                    env={"NAUTICAL_CORE_PATH": "/override"},
                )

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
