"""Execution of tasks and validation of their completion."""

from swarm_memorizer.delegation import redelegate_task_executor
from swarm_memorizer.event import Event, TaskValidation
from swarm_memorizer.id_generation import generate_id
from swarm_memorizer.schema import EventId, TaskWorkStatus
from swarm_memorizer.task import (
    ExecutionReport,
    Task,
    change_status,
    generate_artifact,
    validate_task_completion,
)


async def execute_and_validate(task: Task) -> ExecutionReport:
    """Execute and validate a task until a stopping point, and update the task's status. This bridges the gap between an executor's `execute` and the usage of the method in an orchestrator."""
    task.start_timer()
    assert task.executor
    report = await task.executor.execute()
    assert isinstance(
        report.task_completed, bool
    ), "Task completion must be determined after execution."
    if not report.task_completed:
        status_update_event = change_status(
            task, TaskWorkStatus.BLOCKED, "Task is blocked until reply to message."
        )
        report.new_parent_events.append(status_update_event)
        return report

    validation_status_event = change_status(  # MUTATION
        task, TaskWorkStatus.IN_VALIDATION, "Validation has begun for task."
    )
    validation_result = validate_task_completion(task, report)
    if not validation_result.valid and task.validation_failures:
        _ = redelegate_task_executor(task.executor)
        raise NotImplementedError("TODO")
        # task.reset_progress()
        # task.change_executor(new_executor)  # MUTATION
    if not validation_result.valid:
        new_status = TaskWorkStatus.BLOCKED
        reason = "Failed completion validation."
        task.add_validation_failure()  # MUTATION
        report.task_completed = False  # MUTATION
    else:
        if report.artifacts is None:
            report.artifacts = [generate_artifact(task)]
        assert (
            report.artifacts
        ), "Artifact(s) must be present when wrapping up execution."
        new_status = TaskWorkStatus.COMPLETED
        reason = "Validated as complete."
        task.output_artifacts = report.artifacts  # MUTATION
        task.wrap_execution(success=True)  # MUTATION
        # validation_result = WorkValidationResult(valid=True, feedback="")
    # generated_artifact = (
    #     self.generate_artifact(bot_reply)
    #     if (
    #         task_completed
    #         and not self.reports_artifacts
    #         and not bot_reply.artifacts
    #     )
    #     else None
    # )
    # if generated_artifact:
    #     bot_reply.artifacts = [generated_artifact]
    #     bot_reply.report.reply = "Task completed. See artifacts below for details."

    validation_result_event = Event(
        data=TaskValidation(
            validator_id=task.validator.id,
            task_id=task.id,
            validation_result=validation_result,
        ),
        generating_task_id=task.id,
        id=generate_id(EventId, task.id_generator),
    )
    status_update_event = change_status(task, new_status, reason)  # MUTATION
    report.validation = validation_result  # MUTATION
    report.new_parent_events.extend(  # MUTATION
        [validation_status_event, validation_result_event, status_update_event]
    )
    return report
