"""Execution of tasks and validation of their completion."""

from swarm_memorizer.artifact import ArtifactValidationError, ArtifactValidationMessage
from swarm_memorizer.delegation import redelegate_task_executor
from swarm_memorizer.event import Event, TaskValidation
from swarm_memorizer.id_generation import generate_id
from swarm_memorizer.schema import EventId, TaskWorkStatus, WorkValidationResult
from swarm_memorizer.task import (
    ExecutionReport,
    Task,
    change_status,
    generate_artifact,
    validate_task_completion,
)


def translate_artifact_validation_error(error: ArtifactValidationError) -> str:
    """Translate an artifact validation error to a failure reason."""
    if error.message == ArtifactValidationMessage.MISSING_LOCATION:
        return "Executor did not provide a location for the output of the task, or the provided location was not precise."
    raise NotImplementedError("TODO") from error


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
    if not validation_result.valid and task.validation_fail_count >= 2:
        new_executor = redelegate_task_executor(task.executor)
        raise NotImplementedError("TODO")
        task.reset_progress()
        task.change_executor(new_executor)  # MUTATION
    if not validation_result.valid:
        new_status = TaskWorkStatus.BLOCKED
        reason = "Failed completion validation."
        task.increment_fail_count()  # MUTATION
        report.task_completed = False  # MUTATION
    elif not report.artifacts:
        try:
            artifact = generate_artifact(task)
        except ArtifactValidationError as error:
            new_status = TaskWorkStatus.BLOCKED
            reason = translate_artifact_validation_error(error)
            validation_result = WorkValidationResult(
                valid=False, feedback=reason
            )
            task.increment_fail_count()  # MUTATION
            report.task_completed = False  # MUTATION
        else:
            report.artifacts = [artifact]
            new_status = TaskWorkStatus.COMPLETED
            reason = "Validated as complete."
            task.output_artifacts = report.artifacts  # MUTATION
            task.wrap_execution(success=True)  # MUTATION
    else:
        # assert (
        #     report.artifacts
        # ), "Artifact(s) must be present when wrapping up execution."
        new_status = TaskWorkStatus.COMPLETED
        reason = "Validated as complete."
        task.output_artifacts = report.artifacts  # MUTATION
        task.wrap_execution(success=True)  # MUTATION
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
