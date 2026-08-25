"""Typed contract vocabulary for the next-generation hook workflow engine.

This module is deliberately dependency-light.  It defines only closed
classification and result vocabularies; current hook composition roots are
migrated to these types in later passes.
"""

from __future__ import annotations

from enum import Enum


class HookKind(str, Enum):
    """Taskwarrior hook entry points owned by the workflow engine."""

    ADD = "add"
    MODIFY = "modify"
    EXIT = "exit"


class WorkflowRoute(str, Enum):
    """Explicit route families; no route may fall through implicitly."""

    ORDINARY = "ordinary"
    CP_ACTIVATION = "cp_activation"
    ANCHOR_ACTIVATION = "anchor_activation"
    ANCHOR_FILE_ACTIVATION = "anchor_file_activation"
    RECURRING_EDIT = "recurring_edit"
    COMPLETION = "completion"
    DELETION = "deletion"
    CHAIN_DISABLE = "chain_disable"
    MANUAL_CHAIN_OFF = "manual_chain_off"
    RECURRENCE_REMOVAL = "recurrence_removal"
    RESUME = "resume"
    TERMINAL_STOP = "terminal_stop"
    EXIT_DRAIN = "exit_drain"


class WorkflowOutcomeKind(str, Enum):
    """Closed outcome set shared by add, modify, and exit workflows."""

    PASSTHROUGH = "passthrough"
    ACCEPTED_PATCH = "accepted_patch"
    LIFECYCLE_APPLICATION = "lifecycle_application"
    TERMINAL_TRANSITION = "terminal_transition"
    REJECTED_INPUT = "rejected_input"
    RETRYABLE_UNAVAILABLE = "retryable_unavailable"
    INTERNAL_FAILURE = "internal_failure"


class WorkflowFailureCategory(str, Enum):
    """Operational failure categories exposed by the workflow contract."""

    INVALID_INPUT = "invalid_input"
    INVALID_CONFIGURATION = "invalid_configuration"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EVIDENCE_UNAVAILABLE = "evidence_unavailable"
    SCHEDULER_EXHAUSTED = "scheduler_exhausted"
    LIFECYCLE_CONFLICT = "lifecycle_conflict"
    MANUAL_REVIEW = "manual_review"
    PROGRAMMING_ERROR = "programming_error"


__all__ = [
    "HookKind",
    "WorkflowFailureCategory",
    "WorkflowOutcomeKind",
    "WorkflowRoute",
]
