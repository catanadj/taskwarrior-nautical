from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nautical_core.modify_models import AnchorFeedbackServices, CpFeedbackServices


@dataclass(slots=True)
class ModifyRuntimeServices:
    core: Any
    debug_wait_sched: bool
    last_wait_sched_debug: Any
    diag_enabled: bool
    format_root_and_age: Any
    append_next_wait_sched_rows: Any
    timeline_lines: Any
    show_timeline_gaps: bool
    root_uuid_from: Any
    short: Any
    format_next_anchor_rows: Any
    format_next_cp_rows: Any
    format_line_preview: Any
    panel_line: Any
    panel: Any
    chain_color_per_chain: bool
    chain_colour_for_task: Any
    strip_quotes: Any
    human_delta: Any
    export_uuid_short_cached: Any


def build_anchor_feedback_services(runtime: ModifyRuntimeServices) -> AnchorFeedbackServices:
    return AnchorFeedbackServices(
        core=runtime.core,
        debug_wait_sched=runtime.debug_wait_sched,
        last_wait_sched_debug=runtime.last_wait_sched_debug,
        diag_enabled=runtime.diag_enabled,
        format_root_and_age=runtime.format_root_and_age,
        append_next_wait_sched_rows=runtime.append_next_wait_sched_rows,
        timeline_lines=lambda kind, task, child_due, child_short, dnf, *, next_count, cap_no, cur_no, show_gaps: runtime.timeline_lines(
            kind,
            task,
            child_due,
            child_short,
            dnf,
            next_count=next_count,
            cap_no=cap_no,
            cur_no=cur_no,
            show_gaps=show_gaps,
            round_anchor_gaps=True,
        ),
        show_timeline_gaps=runtime.show_timeline_gaps,
        root_uuid_from=runtime.root_uuid_from,
        short=runtime.short,
        format_next_anchor_rows=runtime.format_next_anchor_rows,
        format_line_preview=runtime.format_line_preview,
        panel_line=runtime.panel_line,
        panel=runtime.panel,
        chain_color_per_chain=runtime.chain_color_per_chain,
        chain_colour_for_task=runtime.chain_colour_for_task,
        strip_quotes=runtime.strip_quotes,
        human_delta=runtime.human_delta,
    )


def build_cp_feedback_services(runtime: ModifyRuntimeServices) -> CpFeedbackServices:
    return CpFeedbackServices(
        core=runtime.core,
        diag_enabled=runtime.diag_enabled,
        format_root_and_age=runtime.format_root_and_age,
        append_next_wait_sched_rows=runtime.append_next_wait_sched_rows,
        timeline_lines=lambda kind, task, child_due, child_short, dnf, *, next_count, cap_no, cur_no, show_gaps: runtime.timeline_lines(
            kind,
            task,
            child_due,
            child_short,
            dnf,
            next_count=next_count,
            cap_no=cap_no,
            cur_no=cur_no,
            show_gaps=show_gaps,
            round_anchor_gaps=False,
        ),
        show_timeline_gaps=runtime.show_timeline_gaps,
        format_next_cp_rows=runtime.format_next_cp_rows,
        format_line_preview=runtime.format_line_preview,
        panel_line=runtime.panel_line,
        panel=runtime.panel,
        chain_color_per_chain=runtime.chain_color_per_chain,
        chain_colour_for_task=runtime.chain_colour_for_task,
        human_delta=runtime.human_delta,
        export_uuid_short_cached=runtime.export_uuid_short_cached,
    )
