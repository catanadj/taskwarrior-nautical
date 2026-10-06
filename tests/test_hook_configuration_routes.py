from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.support.hook_process import HookSubprocessFixture


class HookConfigurationRouteTests(HookSubprocessFixture):
    def test_on_modify_rejects_unknown_business_calendar_with_choices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config-nautical.toml"
            config.write_text(
                '[business_calendar.work]\nanchor = "w:mon..fri"\n',
                encoding="utf-8",
            )
            old = {
                "uuid": "00000000-0000-4000-8000-000000000123",
                "description": "change business calendar",
                "status": "pending",
                "anchor": "w:mon",
                "bc": "work",
            }
            new = {**old, "bc": "missing"}

            process = self.run_hook(
                "on-modify.nautical",
                json.dumps(old) + "\n" + json.dumps(new) + "\n",
                extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
            )

        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "")
        self.assertIn("Invalid business calendar", process.stderr)
        self.assertIn("configured calendars:", process.stderr)
        self.assertIn("work.", process.stderr)

    def test_on_modify_invalid_timezone_blocks_recurrence_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config-nautical.toml"
            config.write_text('tz = "Invalid/Timezone"\n', encoding="utf-8")
            old = {
                "uuid": "00000000-0000-4000-8000-000000000128",
                "description": "invalid timezone modify",
                "status": "pending",
                "anchor": "w:mon",
            }
            new = {
                **old,
                "status": "completed",
                "end": "20260808T120000Z",
                "modified": "20260808T120000Z",
            }

            process = self.run_hook(
                "on-modify.nautical",
                json.dumps(old) + "\n" + json.dumps(new) + "\n",
                extra_environment={"NAUTICAL_CONFIG": str(config), "NO_COLOR": "1"},
            )

        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "")
        self.assertIn("Invalid Nautical configuration", process.stderr)
        self.assertIn("timezone", process.stderr.lower())


if __name__ == "__main__":
    unittest.main()
