from __future__ import annotations

from datetime import datetime, timezone
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

from nautical_core.integration_context import (
    IntegrationAccess,
    IntegrationContextError,
    IntegrationRuntime,
    SilentDiagnostics,
    build_integration_context,
    build_operator_context,
)


class IntegrationRuntimePortTests(unittest.TestCase):
    def test_invocation_context_freezes_one_read_only_runtime_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            taskdata = Path(td).resolve()
            facade = ModuleType("integration_context_contract_core")
            calls = {"resolve": 0, "reload": 0, "snapshot": 0}

            def resolve_task_data_context(**_kwargs):
                calls["resolve"] += 1
                return str(taskdata), True, "test"

            def reload_taskdata_config(selected):
                calls["reload"] += 1
                self.assertEqual(Path(selected), taskdata)
                return {"ok": True, "scheduler_fingerprint": "scheduler-fp"}

            def effective_config_snapshot():
                calls["snapshot"] += 1
                return {
                    "source": str(taskdata / "nautical.toml"),
                    "fingerprint": "config-fp",
                    "values": {"tz": "UTC"},
                }

            facade.resolve_task_data_context = resolve_task_data_context
            facade.reload_taskdata_config = reload_taskdata_config
            facade.effective_config_snapshot = effective_config_snapshot
            facade.scheduling_configuration_error = lambda: ""
            facade.LOCAL_TZ_NAME = "UTC"
            facade._LOCAL_TZ = timezone.utc

            class FixedClock:
                def now_utc(self):
                    return datetime(2026, 8, 13, 9, 0, tzinfo=timezone.utc)

            context = build_integration_context(
                runtime=IntegrationRuntime.from_compatibility_facade(facade),
                argv=(f"data:{taskdata}",),
                env={"PATH": os.environ.get("PATH", "")},
                tw_dir=str(taskdata),
                task_binary=sys.executable,
                access=IntegrationAccess.READ_ONLY,
                command_budget=17,
                diagnostics=SilentDiagnostics(),
                clock=FixedClock(),
                invocation_id="invocation-1",
            )

            self.assertEqual(context.taskdata, taskdata)
            self.assertEqual(context.command_prefix[0], str(Path(sys.executable).resolve()))
            self.assertEqual(context.command_prefix[1], f"rc.data.location={taskdata}")
            self.assertEqual(context.configuration.fingerprint, "config-fp")
            self.assertIs(context.local_timezone, timezone.utc)
            self.assertFalse(context.mutation_capable)
            self.assertEqual(calls, {"resolve": 1, "reload": 1, "snapshot": 1})

    def test_operator_context_discovers_taskdata_once_and_preserves_failure_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nautical_operator_context_") as td:
            root = Path(td)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            task_binary = root / "task"
            task_binary.write_text(
                f"#!/bin/sh\nprintf '%s\\n' '{taskdata}'\n", encoding="utf-8"
            )
            task_binary.chmod(0o700)
            facade = ModuleType("operator_context_test_core")
            calls = {"resolve": 0, "reload": 0}

            def resolve_task_data_context(**kwargs: object) -> tuple[str, bool, str]:
                calls["resolve"] += 1
                return str(kwargs["tw_dir"]), False, "fallback"

            def reload_taskdata_config(selected: str | os.PathLike[str]) -> dict[str, str | bool]:
                calls["reload"] += 1
                self.assertEqual(Path(selected), taskdata)
                return {"ok": True, "scheduler_fingerprint": "operator-scheduler-fp"}

            facade.resolve_task_data_context = resolve_task_data_context
            facade.reload_taskdata_config = reload_taskdata_config
            facade.scheduling_configuration_error = lambda: ""
            facade.effective_config_snapshot = lambda: {
                "source": str(taskdata / "nautical.toml"),
                "fingerprint": "operator-config-fp",
                "values": {"tz": "UTC"},
            }
            facade.LOCAL_TZ_NAME = "UTC"
            facade._LOCAL_TZ = timezone.utc

            context = build_operator_context(
                runtime=IntegrationRuntime.from_compatibility_facade(facade),
                task_binary=str(task_binary),
                env={"PATH": os.environ.get("PATH", "")},
                access=IntegrationAccess.MUTATION,
            )
            self.assertEqual(context.taskdata, taskdata)
            self.assertEqual(context.taskdata_source, "taskwarrior")
            self.assertTrue(context.mutation_capable)
            self.assertEqual(calls, {"resolve": 1, "reload": 1})

            facade.reload_taskdata_config = lambda _selected: (_ for _ in ()).throw(
                RuntimeError("malformed operator config")
            )
            with self.assertRaises(IntegrationContextError) as raised:
                build_operator_context(
                    runtime=IntegrationRuntime.from_compatibility_facade(facade),
                    task_binary=str(task_binary),
                    env={"PATH": os.environ.get("PATH", "")},
                )
            self.assertEqual(raised.exception.stage, "configuration")
            self.assertEqual(raised.exception.taskdata, taskdata)

    def test_compatibility_adapter_binds_callbacks_and_reads_timezone_after_reload(self) -> None:
        core = SimpleNamespace(
            resolve_task_data_context=lambda **_kwargs: ("/tmp/taskdata", False, "test"),
            reload_taskdata_config=lambda _path: {"ok": True},
            effective_config_snapshot=lambda: {"values": {}},
            scheduling_configuration_error=lambda: "",
            LOCAL_TZ_NAME="UTC",
            _LOCAL_TZ=timezone.utc,
        )
        runtime = IntegrationRuntime.from_compatibility_facade(core)

        self.assertEqual(
            runtime.resolve_task_data_context(argv=[], env={}, tw_dir="~/.task"),
            ("/tmp/taskdata", False, "test"),
        )
        self.assertEqual(runtime.reload_taskdata_config("/tmp/taskdata"), {"ok": True})
        self.assertEqual(runtime.effective_config_snapshot(), {"values": {}})
        self.assertEqual(runtime.current_timezone(), (timezone.utc, "UTC"))

        core.LOCAL_TZ_NAME = "Etc/GMT-3"
        self.assertEqual(runtime.current_timezone(), (timezone.utc, "Etc/GMT-3"))


if __name__ == "__main__":
    unittest.main()
