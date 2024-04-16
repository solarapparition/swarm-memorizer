"""Execution of tasks and validation of their completion."""

from swarm_memorizer.artifact import ArtifactValidationError, ArtifactValidationMessage
from swarm_memorizer.bot import Bot
from swarm_memorizer.delegation import Delegator
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


def mutate_executor(
    task: Task,
    delegator: Delegator,
    recent_events_size: int,
    auto_await: bool,
    executor_selection_reasoning: str,
    executor_memory: str | None,
) -> None:
    """Upgrade the executor of a task."""
    raise NotImplementedError("TODO")
    # > when changing executors, need to add event to parent that describes the fact that executor has changed > move to upgrade_executor
    assert task.executor, "Task must have an executor for it to be upgraded."
    if isinstance(task.executor, Bot):
        # bots can't be upgraded, so we just assign to the next best agent
        delegator.assign_executor(
            task=task,
            recent_events_size=recent_events_size,
            auto_await=auto_await,
            executor_selection_reasoning=executor_selection_reasoning,
            executor_memory=executor_memory,
            excluded_executors=task.failed_executors,
        )
        return

    raise NotImplementedError("TODO")
    # > lazy regeneration: set flag on specific parts of reasoning that need to be regenerated, and only regenerate those parts when needed # avoids removing parts of reasoning that aren't related to the task
    # > when regenerating reasoning, subtask identification needs to have its own more granular signal
    # > when updating reasoning, must make sure to include knowledge
    # > TODO: agent regeneration: if agent fails task, first time is just a message; new version of agent probably should only have its knowledge updated on second fail; on third fail, whole agent is regenerated; on next fail, the next best agent is chosen, and the process repeats again; if the next best agent still can't solve the task, the task is auto-cancelled since it's likely too difficult (manual cancellation by orchestrator is still possible) > when regenerating agent components, include specific information from old agent > if agent is bot, skip update and regeneration and just message/choose next best agent
    # > mutation > update: unify mutation with generation: mutation is same as re-generating each component of agent, including knowledge > blueprint: model parameter # explain that cheaper model costs less but may reduce accuracy > blueprint: novelty parameter: likelihood of choosing unproven subagent > blueprint: temperature parameter > when mutating agent, either update knowledge, or tweak a single parameter > when mutating agent, use component optimization of other best agents (that have actual trajectories) > new mutation has a provisional rating based on the rating of the agent it was mutated from; but doesn't appear in optimization list until it has a trajectory > only mutate when agent fails at some task > add success record to reasoning processes > retrieve previous reasoning for tasks similar to current task


async def execute_and_validate(
    task: Task,
    delegator: Delegator,
    recent_events_size: int,
    auto_await: bool,
    executor_selection_reasoning: str,
    executor_memory: str | None,
) -> ExecutionReport:
    """Execute and validate a task until a stopping point, and update the task's status. This bridges the gap between an executor's `execute` and the usage of the method in an orchestrator."""
    task.start_timer()
    if task.validation_fail_count >= 2:
        task.reset_rank_limit()  # MUTATION
        mutate_executor(
            task,
            delegator=delegator,
            recent_events_size=recent_events_size,
            auto_await=auto_await,
            executor_selection_reasoning=executor_selection_reasoning,
            executor_memory=executor_memory,
        )
        task.reset_fail_count()  # MUTATION
        task.reset_event_log()  # MUTATION
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
            validation_result = WorkValidationResult(valid=False, feedback=reason)
            task.increment_fail_count()  # MUTATION
            report.task_completed = False  # MUTATION
        else:
            report.artifacts = [artifact]
            new_status = TaskWorkStatus.COMPLETED
            reason = "Validated as complete."
            task.output_artifacts = report.artifacts  # MUTATION
            task.wrap_execution(success=True)  # MUTATION
    else:
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
