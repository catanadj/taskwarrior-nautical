from __future__ import annotations

from typing import Any


CONFIG_SPECS: dict[str, dict[str, Any]] = {
    "wrand_salt": {"type": "string", "default": "nautical|wrand|v4"},
    "tz": {"type": "string", "default": "Europe/Bucharest"},
    "anchor_file_dir": {"type": "string", "default": ""},
    "omit_file_dir": {"type": "string", "default": ""},
    "enable_anchor_cache": {"type": "bool", "default": False},
    "anchor_cache_dir": {"type": "string", "default": ""},
    "anchor_cache_ttl": {"type": "int", "default": 0, "min": 0},
    "chain_color_per_chain": {"type": "bool", "default": False},
    "show_timeline_gaps": {"type": "bool", "default": True},
    "show_analytics": {"type": "bool", "default": True},
    "analytics_style": {
        "type": "choice",
        "default": "clinical",
        "choices": {"coach": "coach", "clinical": "clinical"},
    },
    "analytics_ontime_tol_secs": {"type": "int", "default": 14400, "min": 0},
    "check_chain_integrity": {"type": "bool", "default": False},
    "debug_wait_sched": {"type": "bool", "default": False},
    "recurrence_update_udas": {"type": "string_list", "default": []},
    "panel_mode": {
        "type": "choice",
        "default": "rich",
        "choices": {
            "rich": "rich",
            "live": "live",
            "fast": "fast",
            "plain": "fast",
            "line": "line",
            "minimal": "line",
            "text": "text",
            "quiet": "text",
            "compact": "rich",
        },
    },
    "live_panel_duration_ms": {"type": "int", "default": 160, "min": 0, "max": 1000},
    "exit_progress": {"type": "bool", "default": True},
    "fast_color": {"type": "bool", "default": True},
    "spawn_queue_max_bytes": {"type": "int", "default": 524288, "min": 0},
    "spawn_queue_drain_max_items": {"type": "int", "default": 200, "min": 1, "max": 100000},
    "max_chain_walk": {"type": "int", "default": 500, "min": 1},
    "max_anchor_iterations": {"type": "int", "default": 128, "min": 32, "max": 1024},
    "max_link_number": {"type": "int", "default": 10000, "min": 1},
    "sanitize_uda": {"type": "bool", "default": False},
    "sanitize_uda_max_len": {"type": "int", "default": 1024, "min": 64, "max": 4096},
    "max_json_bytes": {
        "type": "int",
        "default": 10 * 1024 * 1024,
        "min": 1024,
        "max": 100 * 1024 * 1024,
    },
    "cache_ttl_secs": {"type": "int", "default": 3600, "min": 0},
    "cache_load_mem_max": {"type": "int", "default": 128, "min": 16, "max": 4096},
    "cache_load_mem_ttl": {"type": "int", "default": 300, "min": 0, "max": 86400},
    "anchor_presets": {"type": "table", "default": {}},
    "omit_presets": {"type": "table", "default": {}},
    "business_calendar": {"type": "table", "default": {}},
    "recurrence": {"type": "table", "default": {}},
    "recurrence.update_udas": {"type": "string_list", "default": []},
}

DEPRECATED_CONFIG_KEYS = {
    "holiday_region": "Holiday behavior is defined with [business_calendar.<name>]; holiday_region has no effect.",
    "verify_import": "Child imports are always verified; verify_import no longer changes behavior.",
}


def spec_default(key: str) -> Any:
    return CONFIG_SPECS[key]["default"]


def normalized_choice(key: str, value: Any) -> str:
    spec = CONFIG_SPECS[key]
    text = str(value or "").strip().lower()
    return spec["choices"].get(text, str(spec["default"]))


def _effective_int(spec: dict[str, Any], value: Any) -> int:
    try:
        effective = int(str(value).strip())
    except Exception:
        effective = int(spec["default"])
    if "min" in spec:
        effective = max(int(spec["min"]), effective)
    if "max" in spec:
        effective = min(int(spec["max"]), effective)
    return effective


def _matches_type(kind: str, value: Any) -> bool:
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in {"string", "choice"}:
        return isinstance(value, str)
    if kind == "string_list":
        return isinstance(value, str) or (
            isinstance(value, list) and all(isinstance(item, str) for item in value)
        )
    if kind == "table":
        return isinstance(value, dict)
    return True


def validate_config(data: dict[str, Any], *, skip_keys: set[str] | None = None) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    skipped = skip_keys or set()
    normalized = {str(key).strip().lower(): value for key, value in (data or {}).items()}

    for key, value in normalized.items():
        if key in skipped:
            continue
        if key in DEPRECATED_CONFIG_KEYS:
            issues.append(
                {
                    "kind": "deprecated",
                    "key": key,
                    "configured": value,
                    "message": DEPRECATED_CONFIG_KEYS[key],
                }
            )
            continue
        spec = CONFIG_SPECS.get(key)
        if spec is None:
            issues.append({"kind": "unknown", "key": key, "configured": value})
            continue

        kind = str(spec["type"])
        effective: Any = value
        if kind == "int":
            effective = _effective_int(spec, value)
        elif kind == "choice":
            effective = normalized_choice(key, value)

        if not _matches_type(kind, value):
            issues.append(
                {
                    "kind": "type",
                    "key": key,
                    "configured": value,
                    "effective": effective if kind in {"int", "choice"} else spec["default"],
                    "expected": kind,
                }
            )
            continue

        if kind == "int" and effective != value:
            issues.append(
                {
                    "kind": "range",
                    "key": key,
                    "configured": value,
                    "effective": effective,
                    "min": spec.get("min"),
                    "max": spec.get("max"),
                }
            )
        elif kind == "choice" and str(value).strip().lower() not in spec["choices"]:
            issues.append(
                {
                    "kind": "choice",
                    "key": key,
                    "configured": value,
                    "effective": effective,
                    "choices": sorted(spec["choices"]),
                }
            )

    recurrence = normalized.get("recurrence")
    if isinstance(recurrence, dict):
        for nested_key, nested_value in recurrence.items():
            nested_name = str(nested_key).strip().lower()
            if nested_name != "update_udas":
                issues.append(
                    {
                        "kind": "unknown",
                        "key": f"recurrence.{nested_key}",
                        "configured": nested_value,
                    }
                )
            elif not _matches_type("string_list", nested_value):
                issues.append(
                    {
                        "kind": "type",
                        "key": "recurrence.update_udas",
                        "configured": nested_value,
                        "effective": [],
                        "expected": "string_list",
                    }
                )
    return issues
