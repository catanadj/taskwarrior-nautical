from __future__ import annotations

import tempfile
import unittest
import re
import inspect
from dataclasses import fields
from pathlib import Path

from dev_tools import nautical_deploy_sanity
from nautical_core import architecture_contract
from nautical_core.modify_timeline import TimelineFormattingServices, TimelineProjectionServices
from nautical_core.add_anchor_preview import (
    AnchorExpressionPreviewServices,
    AnchorFilePreviewServices,
    handle_anchor_file_preview_on_add,
    handle_anchor_preview_on_add,
)


class ArchitectureContractTests(unittest.TestCase):
    def test_operator_presentation_has_no_mutation_dependencies(self) -> None:
        source = (
            Path(__file__).parents[1] / "nautical_core" / "operator_presentation.py"
        ).read_text(encoding="utf-8").lower()
        forbidden = ("taskwarrior", "sqlite", "lifecycle_application", "task_command")
        self.assertFalse(any(token in source for token in forbidden))

    def test_runtime_manifest_covers_panel_colours_for_each_hook(self) -> None:
        from nautical_core.runtime_manifest import HOOK_RUNTIME_FILES

        for event in ("on-add", "on-modify", "on-exit"):
            with self.subTest(event=event):
                self.assertIn("panel_colours.py", HOOK_RUNTIME_FILES.get(event, ()))

    def test_removed_exit_flow_modules_are_absent_from_runtime_tree(self) -> None:
        from nautical_core.runtime_manifest import HOOK_RUNTIME_FILES

        legacy = (
            "exit_drain_flow.py",
            "exit_entry_flow.py",
            "exit_models.py",
            "exit_side_effects.py",
        )
        core_directory = Path(__file__).parents[1] / "nautical_core"
        for name in legacy:
            with self.subTest(module=name):
                self.assertFalse((core_directory / name).exists())
                self.assertTrue(
                    all(name not in files for files in HOOK_RUNTIME_FILES.values())
                )

    def test_repository_modify_effect_operations_do_not_receive_hook_host(self) -> None:
        violations = (
            violation.as_dict()
            for violation in architecture_contract.validate(Path(__file__).parents[1])
            if violation.dependency == "hook-host"
        )
        self.assertEqual(list(violations), [])

    def test_primary_bound_apis_do_not_reintroduce_facade_lookups(self) -> None:
        root = Path(__file__).parents[1]
        pattern = re.compile(r"\bcore\s*(?:\[|\.get\s*\()")
        for relative in (
            "nautical_core/parser_api.py",
            "nautical_core/scheduler_api.py",
            "nautical_core/cache_api.py",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            self.assertIsNone(pattern.search(source), relative)

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

    def test_primary_owner_cannot_import_compatibility_implementation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "task_models.py").write_text(
                "from nautical_core.compat_api import legacy_export\n",
                encoding="utf-8",
            )
            (package / "compat_api.py").write_text("legacy_export = object()\n", encoding="utf-8")

            violations = architecture_contract.validate(root)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].dependency, "nautical_core.compat_api")
        self.assertIn("compatibility", violations[0].rule)

    def test_modify_effect_operation_cannot_receive_hook_host(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "modify_example_effects.py").write_text(
                "def apply_change(host, task):\n"
                "    return host._module('task_codec').decode_row(task)\n",
                encoding="utf-8",
            )

            violations = architecture_contract.validate(root)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].dependency, "hook-host")
        self.assertIn("apply_change", violations[0].rule)

    def test_modify_effect_operation_cannot_reach_global_hook_host(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "modify_example_effects.py").write_text(
                "def apply_change(task):\n"
                "    return host._module('task_codec').decode_row(task)\n",
                encoding="utf-8",
            )

            violations = architecture_contract.validate(root)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].dependency, "hook-host")
        self.assertIn("dynamic hook-host lookup", violations[0].rule)

    def test_named_modify_composition_adapter_may_receive_hook_host(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "modify_example_effects.py").write_text(
                "def example_port_for(host):\n"
                "    return host._module('task_codec')\n\n"
                "def example_ports_for(host):\n"
                "    return host._module('task_models')\n\n"
                "def example_services_for(host):\n"
                "    return host._module('lifecycle_read_service')\n\n"
                "def operation_for_host(host):\n"
                "    return host.core.parser\n",
                encoding="utf-8",
            )

            violations = architecture_contract.validate(root)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].dependency, "hook-host")
        self.assertIn("operation_for_host", violations[0].rule)

    def test_presentation_contexts_exclude_host_and_module_loaders(self) -> None:
        for context in (
            AnchorExpressionPreviewServices,
            AnchorFilePreviewServices,
            TimelineProjectionServices,
            TimelineFormattingServices,
        ):
            with self.subTest(context=context.__name__):
                names = {field.name for field in fields(context)}
                self.assertTrue(names.isdisjoint({"core", "host", "module_loader"}))

        projection_fields = {field.name for field in fields(TimelineProjectionServices)}
        formatting_fields = {field.name for field in fields(TimelineFormattingServices)}
        self.assertIn("scheduler_service_for_task", projection_fields)
        self.assertIn("recurrence_evaluator_for_task", projection_fields)
        self.assertIn("fmtlocal", formatting_fields)
        self.assertIn("short", formatting_fields)
        self.assertTrue({"scheduler_service_for_task", "omit_description_for_task_date"}.isdisjoint(formatting_fields))
        self.assertTrue({"fmtlocal", "fmt_dt_local", "short"}.isdisjoint(projection_fields))

        file_fields = {field.name for field in fields(AnchorFilePreviewServices)}
        self.assertTrue({"validate_anchor_syntax_strict", "validate_native_until_after_target"}.isdisjoint(file_fields))
        self.assertNotIn("core", inspect.signature(handle_anchor_preview_on_add).parameters)
        self.assertNotIn("core", inspect.signature(handle_anchor_file_preview_on_add).parameters)

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
