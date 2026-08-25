"""Static ownership checks for the workflow effect boundary."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PURE_WORKFLOW_MODULES = (
    "nautical_core/lifecycle_planner.py",
    "nautical_core/chain_repair_planner.py",
    "nautical_core/modify_feedback.py",
    "nautical_core/panel_diagnostics.py",
    "nautical_core/panel_colours.py",
)
FORBIDDEN_IMPORTS = {
    "subprocess",
    "nautical_core.lifecycle_outbox",
    "nautical_core.runtime_command",
    "nautical_core.taskwarrior_mutations",
    "nautical_core.task_command",
}


class EffectBoundaryTests(unittest.TestCase):
    def test_planners_and_presenters_have_no_external_effect_imports(self) -> None:
        for relative in PURE_WORKFLOW_MODULES:
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"), filename=relative)
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            forbidden = sorted(
                name
                for name in imported
                if name in FORBIDDEN_IMPORTS
                or any(name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORTS if prefix != "subprocess")
            )
            self.assertEqual(forbidden, [], f"{relative} imports external effect owners: {forbidden}")


if __name__ == "__main__":
    unittest.main()
