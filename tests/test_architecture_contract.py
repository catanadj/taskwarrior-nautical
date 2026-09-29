from __future__ import annotations

import ast
import tempfile
import unittest
import re
import inspect
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from dev_tools import nautical_deploy_sanity
import nautical_core.architecture_contract as architecture_contract
from nautical_core.compat_api import PUBLIC_EXPORTS
from nautical_core.modify_timeline import TimelineFormattingServices, TimelineProjectionServices
from nautical_core.add_anchor_preview import (
    AnchorExpressionPreviewServices,
    AnchorFilePreviewServices,
    handle_anchor_file_preview_on_add,
    handle_anchor_preview_on_add,
)


class ArchitectureContractTests(unittest.TestCase):
    def test_owner_contracts_do_not_access_private_facade_exports(self) -> None:
        tests = Path(__file__).parent
        private_reads = []
        for filename in (
            "test_parser_owner_api_contracts.py",
            "test_scheduler_api_contract.py",
            "test_runtime_config_contracts.py",
            "recurrence/test_yearly_token_migration.py",
            "recurrence/test_parser_fuzz_contracts.py",
            "recurrence/test_cp_sequence_contracts.py",
            "recurrence/test_scheduler_cross_path_conformance.py",
            "test_astronomy_contracts.py",
        ):
            path = tests / filename
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            facade_aliases = {
                alias.asname or alias.name.split(".", 1)[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
                if alias.name == "nautical_core"
            }
            private_reads.extend(
                f"{path.name}:{node.lineno}: {node.value.id}.{node.attr}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Attribute)
                and node.attr.startswith("_")
                and isinstance(node.value, ast.Name)
                and node.value.id in facade_aliases
            )
        self.assertEqual(private_reads, [], "owner contracts must use owner APIs directly")

    def test_repository_consumers_import_internal_modules_from_their_owners(self) -> None:
        root = Path(__file__).parents[1]
        violations: list[str] = []
        for directory in ("nautical_core", "tests", "dev_tools"):
            for path in sorted((root / directory).rglob("*.py")):
                if path.name == "__init__.py" and directory == "nautical_core":
                    continue
                if path.relative_to(root).as_posix() == "tests/test_navigator_view_models.py":
                    continue  # Navigator is excluded from this refactor scope.
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except (OSError, SyntaxError) as exc:
                    self.fail(f"could not inspect {path.relative_to(root)}: {exc}")
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom) or node.module != "nautical_core":
                        continue
                    for alias in node.names:
                        if alias.name not in PUBLIC_EXPORTS:
                            violations.append(
                                f"{path.relative_to(root)}:{node.lineno}: {alias.name}"
                            )
        self.assertEqual(violations, [], "internal modules must be imported from their owners")

    def test_outbox_sql_and_connection_ownership_stay_in_their_modules(self) -> None:
        core = Path(__file__).parents[1] / "nautical_core"
        repository_source = (core / "lifecycle_outbox.py").read_text(encoding="utf-8")
        query_source = (core / "lifecycle_outbox_queries.py").read_text(encoding="utf-8")
        maintenance_source = (core / "lifecycle_outbox_maintenance.py").read_text(encoding="utf-8")

        for query in (
            "SELECT processing_state, COUNT(*) FROM lifecycle_outbox",
            "SELECT * FROM lifecycle_outbox ORDER BY intent_id ASC",
        ):
            self.assertIn(query, query_source)
        for query in (
            "SELECT intent_id FROM lifecycle_outbox",
            "DELETE FROM lifecycle_outbox",
            "CREATE TABLE IF NOT EXISTS lifecycle_maintenance",
            "PRAGMA wal_checkpoint(PASSIVE)",
        ):
            self.assertIn(query, maintenance_source)

        tree = ast.parse(repository_source)
        repository_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_LifecycleOutboxRepository"
        )
        for method_name in ("status", "snapshot_records", "prune_acknowledged", "opportunistic_housekeeping"):
            method = next(
                node for node in repository_class.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name
            )
            sql_literals = [
                node.value for node in ast.walk(method)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.lstrip().upper().startswith(("SELECT", "UPDATE", "DELETE", "CREATE TABLE"))
            ]
            self.assertEqual(sql_literals, [], method_name)

        self.assertIn("def _connect(", repository_source)
        self.assertIn("def _transaction(", repository_source)
        self.assertIn("_secure_state_files()", repository_source)
        for owner_source in (query_source, maintenance_source):
            self.assertNotIn("_secure_state_files", owner_source)
            self.assertNotIn("BEGIN IMMEDIATE", owner_source)

    def test_outbox_repository_delegates_each_read_and_maintenance_owner_once(self) -> None:
        from nautical_core.lifecycle_outbox import LifecycleOutboxRepository
        import nautical_core.lifecycle_outbox_maintenance as lifecycle_outbox_maintenance
        import nautical_core.lifecycle_outbox_queries as lifecycle_outbox_queries

        with tempfile.TemporaryDirectory(prefix="nautical-outbox-owner-contract-") as td:
            repository = LifecycleOutboxRepository(Path(td))
            self.assertTrue(repository.open().ok)

            with patch.object(
                lifecycle_outbox_queries, "status_summary", wraps=lifecycle_outbox_queries.status_summary
            ) as status_query:
                status_result, payload = repository.status()
            self.assertTrue(status_result.ok)
            self.assertIn("retention", payload)
            self.assertIn("records", payload)
            status_query.assert_called_once()

            with patch.object(
                lifecycle_outbox_queries, "snapshot_rows", wraps=lifecycle_outbox_queries.snapshot_rows
            ) as snapshot_query:
                snapshot_result, snapshot = repository.snapshot_records()
            self.assertTrue(snapshot_result.ok)
            self.assertEqual(snapshot, ())
            snapshot_query.assert_called_once()

            with patch.object(
                lifecycle_outbox_maintenance, "prune_acknowledged_rows",
                wraps=lifecycle_outbox_maintenance.prune_acknowledged_rows,
            ) as prune_query:
                prune_result = repository.prune_acknowledged()
            self.assertTrue(prune_result.ok)
            self.assertEqual(prune_result.removed, 0)
            prune_query.assert_called_once()

            with patch.object(
                lifecycle_outbox_maintenance, "housekeeping_rows",
                wraps=lifecycle_outbox_maintenance.housekeeping_rows,
            ) as housekeeping_query:
                housekeeping_result = repository.opportunistic_housekeeping(
                    size_threshold_bytes=2**31
                )
            self.assertTrue(housekeeping_result.ok)
            self.assertTrue(housekeeping_result.skipped)
            self.assertEqual(housekeeping_result.reason, "no_work")
            housekeeping_query.assert_called_once()

    def test_hooks_do_not_call_subprocess_run_outside_task_execution(self) -> None:
        root = Path(__file__).parents[1]

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.functions: list[str] = []
                self.violations: list[tuple[int, str]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.functions.append(node.name)
                self.generic_visit(node)
                self.functions.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.functions.append(node.name)
                self.generic_visit(node)
                self.functions.pop()

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "run"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "subprocess"
                ):
                    current = self.functions[-1] if self.functions else "<module>"
                    if current != "_run_task":
                        self.violations.append((node.lineno, current))
                self.generic_visit(node)

        for hook_name in ("on-add.nautical", "on-modify.nautical"):
            with self.subTest(hook=hook_name):
                path = root / hook_name
                visitor = Visitor()
                visitor.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
                self.assertEqual(visitor.violations, [], hook_name)

    def test_extracted_model_modules_have_explicit_domain_ownership(self) -> None:
        self.assertEqual(architecture_contract.module_layer("common.py"), architecture_contract.DOMAIN)
        self.assertEqual(architecture_contract.module_layer("hint_models.py"), architecture_contract.DOMAIN)
        self.assertEqual(architecture_contract.module_layer("configuration_facade.py"), architecture_contract.COMPATIBILITY)
        self.assertEqual(architecture_contract.module_layer("timezone_facade.py"), architecture_contract.COMPATIBILITY)

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

    def test_repository_bound_owners_do_not_read_facade_namespaces(self) -> None:
        violations = [
            violation.as_dict()
            for violation in architecture_contract.validate(Path(__file__).parents[1])
            if violation.dependency == "module-namespace"
        ]
        self.assertEqual(violations, [])

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

    def test_owner_namespace_reads_are_rejected_except_at_composition_adapters(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical-architecture-contract-") as td:
            root = Path(td)
            package = root / "nautical_core"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "scheduler_api.py").write_text(
                "def owner(module):\n"
                "    return module.next_after_expr\n\n"
                "def for_core(module):\n"
                "    return module.composition_input\n\n"
                "def compat_adapter(core):\n"
                "    return core.public_call\n\n"
                "def owner_alias(module):\n"
                "    namespace = module\n"
                "    return namespace.next_after_expr\n",
                encoding="utf-8",
            )

            violations = architecture_contract.validate(root)

        self.assertEqual(len(violations), 2)
        violation = violations[0]
        self.assertEqual(violation.importing_file, "nautical_core/scheduler_api.py")
        self.assertEqual(violation.dependency, "module-namespace")
        self.assertEqual(violation.line, 2)
        self.assertIn("explicit dependencies", violation.rule)
        self.assertEqual(violations[1].line, 12)
        self.assertIn("owner_alias", violations[1].rule)

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
