from __future__ import annotations

import os
from typing import Any


def recurrence_timezone_warning(core: Any, task: dict[str, Any]) -> str:
    if getattr(core, "_LOCAL_TZ", None) is not None:
        return ""
    if not _has_recurrence_source(task):
        return ""
    return "Timezone data unavailable; using UTC fallback. Run nautical doctor."


def config_warnings() -> list[str]:
    env_path = str(os.environ.get("NAUTICAL_CONFIG") or "").strip()
    if not env_path:
        return []
    expanded = os.path.abspath(os.path.expanduser(env_path))
    if os.path.isfile(expanded):
        return []
    return [f"NAUTICAL_CONFIG points to a missing file; built-in defaults are active ({env_path})."]


def file_source_warnings(core: Any, task: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    anchor_file = str(task.get("anchor_file") or "").strip()
    omit_file = str(task.get("omit_file") or "").strip()
    if anchor_file:
        try:
            anchor_files = core._import_sibling("anchor_files")
            dates = anchor_files.load_anchor_file_dates(anchor_file, getattr(core, "ANCHOR_FILE_DIR", ""))
            unmatched = anchor_files.unmatched_anchor_file_patterns(
                anchor_file,
                getattr(core, "ANCHOR_FILE_DIR", ""),
            )
            for pattern in unmatched:
                warnings.append(f"anchor_file pattern '{pattern}' matched no files.")
            if not dates and not unmatched:
                warnings.append(f"anchor_file '{_file_label(anchor_file)}' has no usable dates.")
        except Exception:
            pass
    if omit_file:
        try:
            omit_files = core._import_sibling("omit_files")
            dates = omit_files.load_omit_file_dates(omit_file, getattr(core, "OMIT_FILE_DIR", ""))
            unmatched = omit_files.unmatched_omit_file_patterns(
                omit_file,
                getattr(core, "OMIT_FILE_DIR", ""),
            )
            for pattern in unmatched:
                warnings.append(f"omit_file pattern '{pattern}' matched no files.")
            if not dates and not unmatched:
                warnings.append(f"omit_file '{_file_label(omit_file)}' has no usable dates.")
        except Exception:
            pass
    return warnings


def panel_warnings(core: Any, task: dict[str, Any], *, include_files: bool = True) -> list[str]:
    warnings: list[str] = []
    tz_warning = recurrence_timezone_warning(core, task)
    if tz_warning:
        warnings.append(tz_warning)
    warnings.extend(config_warnings())
    if include_files:
        warnings.extend(file_source_warnings(core, task))
    return _dedup(warnings)


def _has_recurrence_source(task: dict[str, Any]) -> bool:
    return any(str(task.get(key) or "").strip() for key in ("anchor", "anchor_file", "cp"))


def _file_label(expr: str) -> str:
    return str(expr or "").split("@", 1)[0].strip() or str(expr or "").strip()


def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
