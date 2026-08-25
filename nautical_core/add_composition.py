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

    def validate_task(self, task):
        """Validate and classify the add request before workflow construction."""
        return validate_task(self._host, task)


    def build_context(self, task, now_utc, now_local, *, observation=None, prof):
        host = self._host
        core = host.core
        hook_context = host._module("hook_context")
        started = host.time.perf_counter()
        try:
            ctx = hook_context.build_on_add_context(
                task,
                now_utc,
                now_local,
                validate_kind_not_conflicting=core._import_sibling(
                    "hook_validation_pipeline"
                ).recurrence_kind_conflict,
                kind_and_defaults_on_add=host._kind_and_defaults_on_add,
                validate_chain_limits_on_add=host._validate_chain_limits_on_add,
                due_context_on_add=host._due_context_on_add,
                observation=observation,
            )
            omit_expr = host._strip_quotes(str(task.get("omit") or "").strip())
            anchor_file = host._strip_quotes(str(task.get("anchor_file") or "").strip())
            omit_file = host._strip_quotes(str(task.get("omit_file") or "").strip())
            if omit_expr:
                task["omit"] = omit_expr
            if anchor_file:
                task["anchor_file"] = anchor_file
            if omit_file:
                task["omit_file"] = omit_file
            pipeline = core._import_sibling("hook_validation_pipeline")
            findings = pipeline.validate_recurrence_files(
                ctx.anchor_str,
                anchor_file,
                omit_expr,
                omit_file,
                load_anchor_file=host._load_anchor_file_dates,
                load_omit_file=host._load_omit_file_dates,
            )
            if findings:
                finding = findings[0]
                host._error_and_exit([(f"Invalid {finding.field}", finding.reason)])
            return ctx
        except ValueError as exc:
            host._error_and_exit([("Invalid chain config", str(exc))])
            raise
        finally:
            if prof is not None:
                prof.add_ms("validate:cp_vs_anchor", (host.time.perf_counter() - started) * 1000.0)

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
        self._host._module("add_preview_composition").render_anchor(
            self._host, task=context.task, anchor_str=context.anchor_str,
            anchor_file_str=context.anchor_file_str, ch=context.chain_state,
            now_utc=context.now_utc, now_local=context.now_local,
            user_provided_due=context.user_provided_due,
            recurrence_field=context.recurrence_field, due_dt=context.due_dt,
            due_day=context.due_day, due_hhmm=context.due_hhmm,
            until_dt=context.until_dt, past_due_warning=context.past_due_warning,
            prof=prof,
        )

    def render_cp_preview(self, context, *, prof) -> None:
        self._host._module("add_preview_composition").render_cp(
            self._host, context.task, context.cp_str, context.chain_state,
            context.now_utc, context.user_provided_due,
            context.recurrence_field, context.due_dt, context.until_dt, prof=prof,
        )


def validate_task(host: Any, task):
    """Validate and classify an add request without constructing services."""
    core = host.core
    validation = core._import_sibling("hook_validation_pipeline")
    route = (
        validation.WorkflowRoute.CP_ACTIVATION
        if str(task.get("cp") or "").strip()
        else validation.WorkflowRoute.ANCHOR_FILE_ACTIVATION
        if str(task.get("anchor_file") or "").strip()
        else validation.WorkflowRoute.ANCHOR_ACTIVATION
        if str(task.get("anchor") or "").strip()
        else validation.WorkflowRoute.ORDINARY
    )
    observation, report = validation.validate_task_mapping(
        task, route=route, source_query="on-add validation"
    )
    if report.status is not validation.ValidationStatus.VALID:
        finding = report.findings[0]
        title = "Invalid chainMax" if finding.code == "chain_max_invalid" else "Invalid Nautical task"
        host._error_and_exit([(title, f"{finding.reason} {finding.correction}")])
    return observation


__all__ = ("AddCompositionServices", "validate_task")


def build_on_add_context(host: Any, task, now_utc, now_local, *, observation=None, prof=None):
    """Build recurrence context through the installed composition boundary."""
    return AddCompositionServices(host, object()).build_context(
        task, now_utc, now_local, observation=observation, prof=prof
    )


def render_anchor_preview(host: Any, context, *, prof) -> None:
    AddCompositionServices(host, object()).render_anchor_preview(context, prof=prof)


def render_cp_preview(host: Any, context, *, prof) -> None:
    AddCompositionServices(host, object()).render_cp_preview(context, prof=prof)
