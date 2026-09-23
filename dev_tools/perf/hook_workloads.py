"""Hook-startup benchmark workloads."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable


def fast_paths(
    cfg: dict,
    *,
    panel_mode: str,
    root: Path,
    install_runtime: Any,
    python_subprocess_env: Callable[[], dict[str, str]],
    panel_mode_config: Callable[[str], str],
    measure_hook_fast_path: Callable[..., dict],
    measure_managed_hook_latency: Callable[..., dict],
    measure_staged_hook_latency: Callable[..., dict],
) -> dict[str, dict]:
    hook_cfg = cfg.get("hook_fast_path")
    if not isinstance(hook_cfg, dict) or not hook_cfg.get("enabled", True):
        return {}
    repeats = max(1, int(hook_cfg.get("repeats", 7)))
    max_ratios = hook_cfg.get("max_ratio") if isinstance(hook_cfg.get("max_ratio"), dict) else {}
    managed_max_ratio = float(hook_cfg.get("managed_layout_max_ratio", 1.5))
    staged_max_ratio = float(hook_cfg.get("staged_layout_max_ratio", 1.5))
    plain = {
        "uuid": "11111111-1111-1111-1111-111111111111", "description": "plain hook latency",
        "status": "pending", "entry": "20260101T000000Z", "modified": "20260101T000000Z",
    }
    modified = dict(plain, modified="20260101T000001Z")
    with tempfile.TemporaryDirectory(prefix="nautical-hook-perf-") as td:
        temp_root = Path(td)
        config_path = temp_root / "config-nautical.toml"
        config_path.write_text(f'tz = "UTC"\npanel_mode = "{panel_mode_config(panel_mode)}"\n', encoding="utf-8")
        base_env = python_subprocess_env()
        base_env.update({
            "NAUTICAL_CONFIG": str(config_path), "NAUTICAL_CORE_PATH": str(root),
            "NAUTICAL_TRUST_CONFIG_PATH": "1", "NAUTICAL_TRUST_CORE_PATH": "1", "TZ": "UTC",
        })
        for key in ("NAUTICAL_DIAG", "NAUTICAL_DIAG_LOG", "NAUTICAL_PROFILE"):
            base_env.pop(key, None)
        cases = []
        add_data = temp_root / "add-data"; add_data.mkdir()
        cases.append(("hook_plain_add", root / "on-add.nautical", json.dumps(plain, ensure_ascii=False), plain, add_data))
        modify_data = temp_root / "modify-data"; modify_data.mkdir()
        cases.append(("hook_plain_modify", root / "on-modify.nautical", json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False), modified, modify_data))
        nautical_old = dict(plain, cp="P1D", chain="on", chainID="abcd1234", link=3, due="20270101T090000Z")
        nautical_modified = dict(nautical_old, description="ordinary Nautical edit", modified="20260101T000001Z")
        cases.append(("hook_nautical_ordinary_modify", root / "on-modify.nautical", json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(nautical_modified, ensure_ascii=False), nautical_modified, modify_data))
        exit_data = temp_root / "exit-data"; exit_data.mkdir()
        cases.append(("hook_empty_exit", root / "on-exit.nautical", "", None, exit_data))
        results = {}
        for name, hook_path, input_text, expected_task, taskdata in cases:
            env = dict(base_env); env["TASKDATA"] = str(taskdata)
            results[name] = measure_hook_fast_path(name, hook_path, input_text=input_text, expected_task=expected_task, base_env=env, repeats=repeats, max_ratio=float(max_ratios.get(name, 0.8)))
        managed_data = temp_root / "managed-data"
        install_runtime.install_release(source=root, taskdata=managed_data, release_id="perf-managed", smoke=False)
        managed_env = dict(base_env); managed_env["TASKDATA"] = str(managed_data)
        for name, source_hook, input_text, expected_task, _taskdata in cases:
            managed_name = f"managed_{name}"
            results[managed_name] = measure_managed_hook_latency(managed_name, managed_data / "hooks" / source_hook.name, input_text=input_text, expected_task=expected_task, base_env=managed_env, repeats=repeats, baseline_median_s=float(results[name]["median_s"]), max_ratio=managed_max_ratio)
        staged_root = temp_root / "staged-source"
        shutil.copytree(root, staged_root, ignore=shutil.ignore_patterns(".git", ".nautical-cache", "__pycache__", "backups", "benchmarks", ".director"))
        staged_env = dict(base_env); staged_env["NAUTICAL_CORE_PATH"] = str(staged_root); staged_env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        for name, source_hook, input_text, expected_task, _taskdata in cases:
            staged_name = f"staged_{name}"
            results[staged_name] = measure_staged_hook_latency(staged_name, staged_root / source_hook.relative_to(root), input_text=input_text, expected_task=expected_task, base_env=staged_env, repeats=repeats, baseline_median_s=float(results[name]["median_s"]), max_ratio=staged_max_ratio)
        return results
