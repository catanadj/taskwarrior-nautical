from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
BOOTSTRAP = ROOT / "bootstrap.sh"


class BootstrapContractTests(unittest.TestCase):
    """Exercise bootstrap's local, side-effect-free command contract."""

    def run_bootstrap(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(BOOTSTRAP), *args],
            cwd=ROOT,
            env={**os.environ, "NO_COLOR": "1"},
            capture_output=True,
            text=True,
            check=False,
        )

    def test_script_passes_shell_syntax_check(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(BOOTSTRAP)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_help_describes_safe_and_installation_options(self) -> None:
        result = self.run_bootstrap("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--version",
            "--taskdata",
            "--launcher-path",
            "--hooks-dir",
            "--dry-run",
            "--install-deps",
            "--keep-checkout",
        ):
            self.assertIn(option, result.stdout)

    def test_missing_option_values_fail_before_external_commands(self) -> None:
        for option in ("--version", "--taskdata", "--launcher-path", "--hooks-dir"):
            with self.subTest(option=option):
                result = self.run_bootstrap(option)
                self.assertEqual(result.returncode, 2)
                self.assertIn("requires", result.stderr)

    def test_unknown_option_fails_before_external_commands(self) -> None:
        result = self.run_bootstrap("--not-a-bootstrap-option")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown option", result.stderr)

    def test_release_contract_keeps_optional_astronomy_separate(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn('requirements_file="$CHECKOUT/requirements.txt"', source)
        self.assertIn('astronomy_requirements_file="$CHECKOUT/requirements-astronomy.txt"', source)
        self.assertIn('if astronomy_configured;', source)
        self.assertIn('pip_requirement_args+=(-r "$astronomy_requirements_file")', source)

    def test_requirement_gate_rejects_installed_but_incompatible_versions(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn("from packaging.requirements import Requirement", source)
        self.assertIn("requirement.specifier.contains(", source)
        self.assertIn("installed_version, prereleases=True", source)
        self.assertIn("operator_match = re.fullmatch", source)
        self.assertIn("installed_key < expected_key", source)

    def test_verification_is_required_after_non_dry_install(self) -> None:
        source = BOOTSTRAP.read_text(encoding="utf-8")
        self.assertIn('if (( ! DRY_RUN )); then', source)
        self.assertIn('doctor_report="$CHECKOUT/doctor-installation.json"', source)
        self.assertIn('nautical_install_verify.py', source)
        self.assertIn('if ((verification_status == 2)); then', source)
        self.assertIn('Manual action required: installed launcher is not executable', source)


if __name__ == "__main__":
    unittest.main()
