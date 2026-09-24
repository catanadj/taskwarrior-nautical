"""Recurrence-focused golden tests extracted from the legacy runner."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from datetime import date


ROOT = Path(__file__).resolve().parents[2]
CORE_PATH = ROOT / "nautical_core" / "__init__.py"


def _load_core_module(path: Path, module_name: str, config_path: str):
    previous = os.environ.get("NAUTICAL_CONFIG")
    os.environ["NAUTICAL_CONFIG"] = config_path
    try:
        spec = importlib.util.spec_from_file_location(
            module_name,
            path,
            submodule_search_locations=[str(path.parent)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create package spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        refresh = getattr(module, "_refresh_facade_config_exports", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        return module
    finally:
        if previous is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = previous


def test_random_salt_namespaces_draws():
    """wrand_salt should remain an explicit namespace for deterministic draws."""
    start = date(2026, 6, 7)

    def sequence(module) -> list[date]:
        dnf = module.parse_anchor_expr_to_dnf_cached("w:rand")
        current = start
        output = []
        for _ in range(12):
            current, _metadata = module.next_after_expr(
                dnf,
                current,
                default_seed=start,
                seed_base="salt-test-chain",
            )
            output.append(current)
        return output

    with tempfile.TemporaryDirectory() as temporary:
        first_config = Path(temporary) / "salt-a.toml"
        second_config = Path(temporary) / "salt-b.toml"
        first_config.write_text('wrand_salt = "salt-a"\n', encoding="utf-8")
        second_config.write_text('wrand_salt = "salt-b"\n', encoding="utf-8")
        first = _load_core_module(CORE_PATH, "_nautical_core_salt_a_test", str(first_config))
        second = _load_core_module(CORE_PATH, "_nautical_core_salt_b_test", str(second_config))
        first_sequence = sequence(first)
        if first_sequence != sequence(first):
            raise AssertionError("the same random salt must replay identically")
        if first_sequence == sequence(second):
            raise AssertionError("changing wrand_salt should change the random sequence")


TESTS = (test_random_salt_namespaces_draws,)
