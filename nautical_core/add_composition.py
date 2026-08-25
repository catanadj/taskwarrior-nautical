"""Installed-layout composition services for the typed on-add workflow."""

from __future__ import annotations

from typing import Any


class AddCompositionServices:
    """Bind add workflow infrastructure without owning recurrence decisions."""

    def __init__(self, host: Any, result_cls: Any) -> None:
        self._host = host
        self._result_cls = result_cls
        core = host.core
        workflow = core._import_sibling("add_workflow")
        self._workflow_application = workflow.AddWorkflowApplication(
            record_schedule_fn=self.record_schedule,
            record_limits_fn=self.record_limits,
            record_preview_fn=self.record_preview,
            build_context_fn=self.build_context,
            stamp_chain_id_fn=self.stamp_chain_id,
            render_anchor_preview_fn=self.render_anchor_preview,
            render_cp_preview_fn=self.render_cp_preview,
        )

    def workflow_application(self) -> Any:
        return self._workflow_application

    def result(self, task, *, sanitize: bool, prof):
        return self._result_cls(task=task, sanitize=sanitize, prof=prof)

    def has_nautical_fields(self, task) -> bool:
        return self._host._task_has_nautical_fields(task)

    def load_core(self) -> None:
        self._host._load_core()

    def core(self) -> Any:
        return self._host.core

    def diag(self, message: str) -> None:
        self._host._diag(message)

    def fail_and_exit(self, title: str, message: str) -> None:
        self._host._fail_and_exit(title, message)

    def build_context(self, task, now_utc, now_local, *, observation=None, prof):
        return self._host._build_on_add_context(
            task, now_utc, now_local, observation=observation, prof=prof
        )

    def record_schedule(self, plan, task, target_field):
        core = self._host.core
        workflow = core._import_sibling("add_workflow")
        raw = task.get(target_field)
        try:
            value = core.parse_dt_any(raw)
            timestamp = core._import_sibling("task_models").TaskTimestamp
            return workflow.record_schedule(plan, first_occurrence=timestamp(value))
        except Exception as exc:
            self._host._fail_and_exit(
                "Scheduler unavailable", f"Could not record first {target_field}: {exc}"
            )
            raise

    def record_preview(self, plan):
        core = self._host.core
        workflow = core._import_sibling("add_workflow")
        policy = workflow.preview_policy(
            panel_mode=getattr(core, "PANEL_MODE", "rich"),
            requested_limit=self._host.UPCOMING_PREVIEW,
            hard_cap=self._host._PREVIEW_HARD_CAP,
        )
        return workflow.record_preview(plan, policy)

    def record_limits(self, plan, task, context):
        core = self._host.core
        workflow = core._import_sibling("add_workflow")
        timestamp = core._import_sibling("task_models").TaskTimestamp

        def as_timestamp(field):
            raw = task.get(field)
            return None if not raw else timestamp(core.parse_dt_any(raw))

        chain_max = core.coerce_int(task.get("chainMax"), 0)
        limits = workflow.AddScheduleLimits(
            native_until=as_timestamp("until"),
            chain_until=as_timestamp("chainUntil"),
            chain_max=chain_max if chain_max > 0 else None,
            wait=as_timestamp("wait"),
            scheduled=as_timestamp("scheduled"),
        )
        return workflow.record_limits(plan, limits)

    def stamp_chain_id(self, task) -> None:
        self._host._stamp_chain_id_on_add(task)

    def render_anchor_preview(self, context, *, prof) -> None:
        self._host._handle_anchor_preview_on_add_context(context, prof=prof)

    def render_cp_preview(self, context, *, prof) -> None:
        self._host._handle_cp_preview_on_add_context(context, prof=prof)


__all__ = ("AddCompositionServices",)
