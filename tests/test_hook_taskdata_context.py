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
    hook_args = sys.argv[2:]
    sys.path.insert(0, str(root))
    sys.argv = ["on-add.nautical", *hook_args]
    source = root / "nautical_core" / "hooks" / "add_impl.py"
    spec = importlib.util.spec_from_file_location("_nautical_add_taskdata_context", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load on-add implementation: {source}")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    hook._load_core()
    print(json.dumps({
        "taskdata": str(hook.TW_DATA_DIR),
        "uses_rc_data_location": bool(hook._USE_RC_DATA_LOCATION),
        "command_prefix": hook._task_cmd_prefix(),
    }))
    """
)


class OnAddTaskdataContextTests(unittest.TestCase):
    def _resolve(self, arguments: tuple[str, ...], *, taskdata_env: str | None = None) -> dict:
        environment = os.environ.copy()
        environment["NAUTICAL_CORE_PATH"] = str(ROOT)
        environment["NAUTICAL_TRUST_CORE_PATH"] = "1"
        if taskdata_env is None:
            environment.pop("TASKDATA", None)
        else:
            environment["TASKDATA"] = taskdata_env
        result = subprocess.run(
            [sys.executable, "-c", _CONTEXT_SCRIPT, str(ROOT), *arguments],
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
        result = self._resolve(())

        self.assertFalse(result["uses_rc_data_location"])
        self.assertFalse(
            any(str(part).startswith("rc.data.location=") for part in result["command_prefix"])
        )

    def test_on_add_reads_data_arg_from_hook_argv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_data_arg_add_") as data_dir:
            result = self._resolve(("api:2", "command:add", f"data:{data_dir}"))

        self.assertEqual(Path(result["taskdata"]), Path(data_dir))
        self.assertTrue(result["uses_rc_data_location"])
        self.assertIn(f"rc.data.location={data_dir}", result["command_prefix"])

