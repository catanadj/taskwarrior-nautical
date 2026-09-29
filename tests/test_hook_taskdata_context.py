"""Process-level contracts for on-add Taskdata argument resolution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
_CONTEXT_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys
    from pathlib import Path

    root = Path(sys.argv[1])
    hook_name = sys.argv[2]
    hook_args = sys.argv[3:]
    sys.path.insert(0, str(root))
    implementation = {
        "on-add": "add_impl.py",
        "on-modify": "modify_impl.py",
        "on-exit": "exit_impl.py",
    }[hook_name]
    sys.argv = [f"{hook_name}.nautical", *hook_args]
    source = root / "nautical_core" / "hooks" / implementation
    spec = importlib.util.spec_from_file_location("_nautical_taskdata_context", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load on-add implementation: {source}")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    hook._load_core()
    command_prefix = getattr(hook, "_task_cmd_prefix", None)
    if callable(command_prefix):
        command_prefix = command_prefix()
    else:
        command_prefix = hook._INTEGRATION_CONTEXT.command_prefix
    print(json.dumps({
        "taskdata": str(hook.TW_DATA_DIR),
        "tw_dir": str(hook.TW_DIR),
        "uses_rc_data_location": bool(hook._USE_RC_DATA_LOCATION),
        "command_prefix": command_prefix,
    }))
    """
)


class OnAddTaskdataContextTests(unittest.TestCase):
    def _resolve(
        self,
        hook_name: str,
        arguments: tuple[str, ...],
        *,
        taskdata_env: str | None = None,
    ) -> dict:
        environment = os.environ.copy()
        environment["NAUTICAL_CORE_PATH"] = str(ROOT)
        environment["NAUTICAL_TRUST_CORE_PATH"] = "1"
        if taskdata_env is None:
            environment.pop("TASKDATA", None)
        else:
            environment["TASKDATA"] = taskdata_env
        result = subprocess.run(
            [sys.executable, "-c", _CONTEXT_SCRIPT, str(ROOT), hook_name, *arguments],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_on_add_no_explicit_taskdata_skips_rc_data_location(self) -> None:
        result = self._resolve("on-add", ())

        self.assertFalse(result["uses_rc_data_location"])
        self.assertFalse(
            any(str(part).startswith("rc.data.location=") for part in result["command_prefix"])
        )

    def test_on_add_reads_data_arg_from_hook_argv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_data_arg_add_") as data_dir:
            result = self._resolve("on-add", ("api:2", "command:add", f"data:{data_dir}"))

        self.assertEqual(Path(result["taskdata"]), Path(data_dir))
        self.assertTrue(result["uses_rc_data_location"])
        self.assertIn(f"rc.data.location={data_dir}", result["command_prefix"])

    def test_on_add_data_arg_overrides_taskdata_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_env_add_") as env_dir:
            with tempfile.TemporaryDirectory(prefix="nautical_arg_add_") as arg_dir:
                result = self._resolve(
                    "on-add",
                    ("api:2", "command:add", f"data:{arg_dir}"),
                    taskdata_env=env_dir,
                )

        self.assertEqual(Path(result["taskdata"]), Path(arg_dir))
        self.assertIn(f"rc.data.location={arg_dir}", result["command_prefix"])

    def test_on_modify_no_explicit_taskdata_skips_rc_data_location(self) -> None:
        result = self._resolve("on-modify", ())

        self.assertFalse(result["uses_rc_data_location"])
        self.assertFalse(
            any(str(part).startswith("rc.data.location=") for part in result["command_prefix"])
        )

    def test_on_modify_reads_data_arg_from_hook_argv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_data_arg_modify_") as data_dir:
            result = self._resolve(
                "on-modify",
                ("api:2", "command:modify", f"data:{data_dir}"),
            )

        self.assertEqual(Path(result["taskdata"]), Path(data_dir))
        self.assertTrue(result["uses_rc_data_location"])
        self.assertIn(f"rc.data.location={data_dir}", result["command_prefix"])

    def test_on_modify_data_arg_overrides_taskdata_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_env_modify_") as env_dir:
            with tempfile.TemporaryDirectory(prefix="nautical_arg_modify_") as arg_dir:
                result = self._resolve(
                    "on-modify",
                    ("api:2", "command:modify", f"data:{arg_dir}"),
                    taskdata_env=env_dir,
                )

        self.assertEqual(Path(result["taskdata"]), Path(arg_dir))
        self.assertIn(f"rc.data.location={arg_dir}", result["command_prefix"])

    def test_on_exit_reads_data_arg_from_hook_argv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_data_arg_exit_") as data_dir:
            result = self._resolve(
                "on-exit",
                ("api:2", "command:modify", f"data:{data_dir}"),
            )

        self.assertEqual(Path(result["taskdata"]), Path(data_dir))
        self.assertTrue(result["uses_rc_data_location"])
        self.assertIn(f"rc.data.location={data_dir}", result["command_prefix"])

    def test_on_exit_data_arg_overrides_taskdata_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_env_exit_") as env_dir:
            with tempfile.TemporaryDirectory(prefix="nautical_arg_exit_") as arg_dir:
                result = self._resolve(
                    "on-exit",
                    ("api:2", "command:modify", f"data:{arg_dir}"),
                    taskdata_env=env_dir,
                )

        self.assertEqual(Path(result["taskdata"]), Path(arg_dir))
        self.assertIn(f"rc.data.location={arg_dir}", result["command_prefix"])

    def test_on_modify_missing_taskdata_uses_tw_dir(self) -> None:
        result = self._resolve("on-modify", ())

        self.assertEqual(result["taskdata"], result["tw_dir"])

    def test_on_modify_ignores_unsafe_core_path_override(self) -> None:
        import importlib.util
        from unittest.mock import patch

        source = ROOT / "nautical_core" / "hooks" / "modify_impl.py"
        spec = importlib.util.spec_from_file_location("_nautical_modify_unsafe_path_contract", source)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        hook = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hook)

        with tempfile.TemporaryDirectory() as unsafe_path:
            Path(unsafe_path).chmod(0o777)
            with patch.dict(os.environ, {"NAUTICAL_CORE_PATH": unsafe_path}):
                os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
                resolved = hook._trusted_core_base(Path(hook.TW_DIR))

        self.assertEqual(Path(resolved).resolve(), Path(hook.TW_DIR).resolve())
