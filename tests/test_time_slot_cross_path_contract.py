from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TimeSlotCrossPathContractTests(unittest.TestCase):
    def test_symbolic_time_slots_match_add_and_modify_hooks(self) -> None:
        script = """
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core.time_slots as time_slots
from nautical_core.hooks import add_impl, modify_impl

hook_core = SimpleNamespace(
    _import_sibling=lambda _name: time_slots,
    ASTRONOMY_CONFIG={},
    to_local=lambda value: value,
)
expected_event = datetime(2026, 7, 6, 18, 0, tzinfo=timezone.utc)
value = {"t": "sunset", "time_offset_minutes": 45}
with (
    patch.object(add_impl, "core", hook_core),
    patch.object(modify_impl, "core", hook_core),
    patch.object(time_slots.astronomy, "resolve_event", return_value=expected_event),
):
    result = {
        "shared": time_slots.resolve_time_slots(value, date(2026, 7, 6), to_local=hook_core.to_local),
        "add": add_impl._resolve_time_slots(value, date(2026, 7, 6)),
    }
    modify_time = modify_impl._module("modify_time_effects")
    result["modify"] = modify_time.normalize_hhmm_list(
        modify_time.time_slot_ports_for(modify_impl), value, date(2026, 7, 6)
    )
print(json.dumps(result))
"""
        with tempfile.TemporaryDirectory() as taskdata:
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONPATH": str(ROOT),
                    "TASKDATA": taskdata,
                    "TASKRC": str(Path(taskdata) / "taskrc"),
                }
            )
            process = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                text=True,
                capture_output=True,
                env=environment,
                timeout=15,
                check=False,
            )

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertEqual(
            json.loads(process.stdout),
            {"shared": [[18, 45]], "add": [[18, 45]], "modify": [[18, 45]]},
        )


if __name__ == "__main__":
    unittest.main()
