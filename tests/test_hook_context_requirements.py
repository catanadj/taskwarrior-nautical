"""Fail-closed context requirements for modify and exit hook startup."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
_MISSING_CONTEXT_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import json
    import os
    import sys
    from pathlib import Path

    root = Path(sys.argv[1])
    hook_name = sys.argv[2]
    fake_core = Path(sys.argv[3])
    os.environ["NAUTICAL_CORE_PATH"] = str(fake_core)
    os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
    sys.path.insert(0, str(root))
    sys.argv = [f"{hook_name}.nautical"]
    source_name = {"on-modify": "modify_impl.py", "on-exit": "exit_impl.py"}[hook_name]
    source = root / "nautical_core" / "hooks" / source_name
    spec = importlib.util.spec_from_file_location("_nautical_missing_context_contract", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {hook_name} implementation")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    try:
        hook._load_core()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
    else:
        raise SystemExit("hook unexpectedly started without its integration context module")
    """
)


class HookContextRequirementTests(unittest.TestCase):
    def _assert_context_helper_required(self, hook_name: str) -> None:
        with tempfile.TemporaryDirectory() as td:
            fake_core = Path(td) / "nautical_core"
            fake_core.mkdir()
            (fake_core / "__init__.py").write_text(
                "def _warn_once_per_day_any(*_args, **_kwargs):\n    return None\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-c", _MISSING_CONTEXT_SCRIPT, str(ROOT), hook_name, td],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertIn("integration_context.py is required", payload["error"])

    def test_on_modify_requires_integration_context_helper(self) -> None:
        self._assert_context_helper_required("on-modify")

    def test_on_exit_requires_integration_context_helper(self) -> None:
        self._assert_context_helper_required("on-exit")

