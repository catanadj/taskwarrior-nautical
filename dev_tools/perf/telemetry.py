"""Timing helpers shared by performance workloads and report assembly."""

from __future__ import annotations

import statistics
import time
import tracemalloc
from collections.abc import Callable
from typing import Any


def measure(
    name: str,
    fn: Callable[[], float],
    repeats: int,
    *,
    trace_memory: bool = False,
) -> dict[str, Any]:
    """Measure a workload while retaining its reported wall-time contract."""
    samples: list[float] = []
    cpu_samples: list[float] = []
    wall_samples: list[float] = []
    peak_memory_samples: list[int] = []

    # Warmup once for interpreter/cache stabilization.
    fn()
    for _ in range(max(1, repeats)):
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        started_tracing = tracemalloc.is_tracing()
        if trace_memory and not started_tracing:
            tracemalloc.start()
        if trace_memory:
            tracemalloc.reset_peak()
        reported = float(fn())
        elapsed_wall = time.perf_counter() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        if trace_memory:
            _current, peak_memory = tracemalloc.get_traced_memory()
        else:
            peak_memory = 0
        if trace_memory and not started_tracing:
            tracemalloc.stop()
        samples.append(reported)
        cpu_samples.append(max(0.0, elapsed_cpu))
        wall_samples.append(max(0.0, elapsed_wall))
        peak_memory_samples.append(max(0, int(peak_memory)))

    samples.sort()
    cpu_samples.sort()
    wall_samples.sort()
    peak_memory_samples.sort()
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": statistics.median(samples),
        "max_s": samples[-1],
        "cpu_samples_s": cpu_samples,
        "cpu_median_s": statistics.median(cpu_samples),
        "measured_wall_median_s": statistics.median(wall_samples),
        "peak_memory_samples_bytes": peak_memory_samples,
        "peak_memory_median_bytes": statistics.median(peak_memory_samples),
        "memory_tracing": trace_memory,
    }


def measure_workflow(name: str, samples: list[float], budget: float) -> dict[str, Any]:
    """Summarize workflow samples against the existing budget semantics."""
    ordered = sorted(float(value) for value in samples)
    median = float(statistics.median(ordered))
    return {
        "name": name,
        "samples_s": ordered,
        "min_s": ordered[0],
        "median_s": median,
        "max_s": ordered[-1],
        "budget_s": float(budget),
        "pass": float(budget) <= 0.0 or median <= float(budget),
    }
