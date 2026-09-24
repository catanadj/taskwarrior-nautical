"""Pure report-shaping helpers for performance benchmark results."""

from __future__ import annotations

from typing import Any


def attach_timing_breakdown(
    result: dict[str, Any],
    wall_samples: list[float],
    timing_samples: list[dict[str, float]],
) -> None:
    """Attach command, startup, presentation, and derived Python timing."""
    result["timing_stats"] = timing_samples
    result["timing_breakdown"] = [
        {
            "wall_seconds": round(float(wall), 6),
            "taskwarrior_seconds": round(float(timing.get("run_task_seconds", 0.0)), 6),
            "startup_seconds": round(float(timing.get("startup_total_ms", 0.0)) / 1000.0, 6),
            "drain_seconds": round(float(timing.get("drain_ms", 0.0)) / 1000.0, 6),
            "presentation_seconds": round(float(timing.get("presentation_ms", 0.0)) / 1000.0, 6),
            "non_taskwarrior_seconds": round(
                max(0.0, float(wall) - float(timing.get("run_task_seconds", 0.0))), 6
            ),
        }
        for wall, timing in zip(wall_samples, timing_samples)
    ]


def merge_task_timing_stats(*stats: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in stats:
        for key, value in item.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def compact_reconcile_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the performance-relevant fields from one reconcile report."""
    return {
        key: report.get(key)
        for key in (
            "status",
            "mode",
            "stage_seconds",
            "export_calls",
            "export_rows",
            "export_seconds",
            "slowest_export_seconds",
            "task_command_calls",
            "task_command_attempts",
            "task_command_duration",
            "task_command_by_purpose",
            "task_command_budget_exceeded",
            "integrity_seconds",
            "integrity_application_seconds",
            "lock_contention",
        )
        if key in report
    }


def attach_reconcile_reports(result: dict[str, Any], reports: list[dict[str, Any]]) -> None:
    """Attach reconcile command, export, and phase metrics to a result."""
    result["reconcile_reports"] = reports
