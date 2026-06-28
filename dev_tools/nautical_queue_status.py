#!/usr/bin/env python3
"""Compatibility wrapper for nautical_core/tools/nautical_queue_status.py."""

from __future__ import annotations

import runpy
from pathlib import Path


TARGET = Path(__file__).resolve().parent.parent / "nautical_core" / "tools" / "nautical_queue_status.py"
runpy.run_path(str(TARGET), run_name="__main__")
