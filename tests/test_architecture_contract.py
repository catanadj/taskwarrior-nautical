from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dev_tools import nautical_deploy_sanity
from nautical_core import architecture_contract


class ArchitectureContractTests(unittest.TestCase):
    def test_module_map_assigns_explicit_layers(self) -> None:
        layers = architecture_contract.module_layer_map(Path(__file__).parents[1])
        self.assertEqual(layers["nautical_core/task_models.py"], architecture_contract.DOMAIN)
        self.assertEqual(layers["nautical_core/scheduler_service.py"], architecture_contract.RECURRENCE)
        self.assertEqual(layers["nautical_core/taskwarrior_client.py"], architecture_contract.INTEGRATION)
        self.assertEqual(layers["nautical_core/tools/nautical_query.py"], architecture_contract.ENTRYPOINT)
        self.assertEqual(layers["nautical_core/compat_api.py"], architecture_contract.COMPATIBILITY)

    def test_invalid_fixture_reports_file_dependency_and_layer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "task_models.py").write_text("import sqlite3\n", encoding="utf-8")

            violations = architecture_contract.validate(root)

        self.assertEqual(len(violations), 1)
        violation = violations[0]
        self.assertEqual(violation.importing_file, "nautical_core/task_models.py")
        self.assertEqual(violation.dependency, "sqlite3")
        self.assertEqual(violation.layer, architecture_contract.DOMAIN)
        self.assertIn("task_models.py", violation.as_dict()["message"])
        self.assertIn("sqlite3", violation.as_dict()["message"])
        self.assertIn(architecture_contract.DOMAIN, violation.as_dict()["message"])

    def test_invalid_root_facade_import_is_rejected_for_internal_owner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "task_models.py").write_text("import nautical_core\n", encoding="utf-8")

            result = architecture_contract.check(root)[0]

        self.assertFalse(result["ok"])
        self.assertEqual(result["violations"][0]["dependency"], "nautical_core")

    def test_deployment_sanity_runs_the_candidate_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "task_models.py").write_text("from rich.console import Console\n", encoding="utf-8")
            (package / "architecture_contract.py").write_text(
                (Path(__file__).parents[1] / "nautical_core" / "architecture_contract.py").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            result = nautical_deploy_sanity._check_architecture_contract(root)

        self.assertFalse(result[0]["ok"])
        self.assertIn("task_models.py", result[0]["message"])
        self.assertIn("rich.console", result[0]["message"])


if __name__ == "__main__":
    unittest.main()
