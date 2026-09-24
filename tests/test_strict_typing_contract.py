from pathlib import Path
import unittest


class StrictTypingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parents[1]

    def test_global_mypy_configuration_keeps_package_strict(self) -> None:
        config = (self.root / "mypy.ini").read_text(encoding="utf-8")
        global_section = config.split("[mypy-", 1)[0]
        self.assertIn("disallow_untyped_defs = True", global_section)
        self.assertIn("disallow_incomplete_defs = True", global_section)
        for code in ("union-attr", "attr-defined", "assignment", "arg-type", "return-value", "operator"):
            self.assertIn(code, global_section)

    def test_ci_full_package_gate_keeps_strict_function_checks_explicit(self) -> None:
        workflow = (self.root / ".github" / "workflows" / "type-check.yml").read_text(encoding="utf-8")
        gate = workflow.split("- name: Run full strict package mypy", 1)[1]
        self.assertIn("--disallow-untyped-defs", gate)
        self.assertIn("--disallow-incomplete-defs", gate)
