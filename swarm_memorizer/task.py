"""Swarm tasks."""

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from textwrap import indent
import time
from typing import Any, Iterator, Literal, Protocol, Self, runtime_checkable

from langchain.schema import SystemMessage

from swarm_memorizer.artifact import (
    Artifact,
    artifacts_printout,
    write_file_artifact,
)
from swarm_memorizer.blueprint import Blueprint
from swarm_memorizer.config import SWARM_COLOR, VERBOSE
from swarm_memorizer.event import (
    Event,
    EventLog,
    Message,
    TaskStatusChange,
)
from swarm_memorizer.id_generation import generate_id
from swarm_memorizer.schema import (
    NONE,
    REASONING_OUTPUT_INSTRUCTIONS,
    ArtifactType,
    Concept,
    EventId,
    ExecutionOutcome,
    IdGenerator,
    RuntimeId,
    TaskId,
    TaskWorkStatus,
    WorkValidationResult,
    WorkValidator,
)
from swarm_memorizer.task_data import ExecutionHistory, TaskData, TaskDescription
from swarm_memorizer.toolkit.models import format_messages, query_model, PRECISE_MODEL
from swarm_memorizer.toolkit.text import dedent_and_strip, extract_and_unpack
from swarm_memorizer.toolkit.yaml_tools import format_as_yaml_str, DEFAULT_YAML


@dataclass
class TaskList:
    """A list of tasks and their managment functionality."""

    items: list["Task"] = field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the task list."""
        # if we're printing out the whole task list, assume these are subtasks
        return "\n".join([task.as_subtask_printout for task in self.items]) or NONE

    def __iter__(self) -> Iterator["Task"]:
        """Iterate over the task list."""
        return iter(self.items)

    def __bool__(self) -> bool:
        """Whether the task list is empty."""
        return bool(self.items)

    def __len__(self) -> int:
        """Length of the task list."""
        return len(self.items)

    def filter_by_status(self, status: TaskWorkStatus) -> "TaskList":
        """Filter the task list by status."""
        return TaskList(
            items=[task for task in self.items if task.work_status == status]
        )


@dataclass
class ExecutionReport:
    """Report from an executor."""

    reply: str
    task_completed: bool | None = None
    validation: WorkValidationResult | None = None
    new_parent_events: list[Event[Any]] = field(default_factory=list)
    artifacts: list[Artifact] | None = None
    """Artifacts produced by the executor. `None` means that the executor does not report whether any artifacts are produced, while an empty list means that the executor explicitly reported that no artifacts were produced."""

    def __str__(self) -> str:
        """String representation of the execution report."""
        external_artifacts = (
            [
                artifact
                for artifact in self.artifacts
                if artifact.type != ArtifactType.INLINE
            ]
            if self.artifacts
            else []
        )
        template = """
        {reply}

        Artifacts:
        {artifacts}
        """
        return dedent_and_strip(template).format(
            reply=self.reply,
            artifacts=artifacts_printout(external_artifacts),
        )


@runtime_checkable
class Executor(Protocol):
    """An agent responsible for executing a task. Normally either an orchestrator or a bot."""

    @property
    def blueprint(self) -> Blueprint:
        """Blueprint of the executor."""
        raise NotImplementedError

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the executor."""
        raise NotImplementedError

    @property
    def rank(self) -> int | None:
        """Rank of the executor."""
        raise NotImplementedError

    @property
    def output_dir(self) -> Path:
        """Directory for the executor's output."""
        raise NotImplementedError

    def accepts(self, task: "Task") -> bool:
        """Decides whether the executor accepts a task."""
        raise NotImplementedError

    def save_blueprint(self) -> None:
        """Save the executor blueprint."""
        raise NotImplementedError

    async def execute(self) -> ExecutionReport:
        """Execute the task. Adds own message to the event log at the end of execution."""
        raise NotImplementedError


@dataclass
class Task:
    """Holds information about a task."""

    data: TaskData
    validator: WorkValidator
    id_generator: IdGenerator
    task_records_dir: Path
    executor: Executor | None = None
    work_status: TaskWorkStatus = TaskWorkStatus.IDENTIFIED

    def __post_init__(self) -> None:
        """Post init."""
        self.task_records_dir.mkdir(parents=True, exist_ok=True)
        self.data.id = self.data.id or generate_id(TaskId, self.id_generator)

    @property
    def id(self) -> TaskId:
        """Id of the task."""
        assert self.data.id
        return self.data.id

    @property
    def description(self) -> TaskDescription:
        """Description of the task."""
        return self.data.description

    @description.setter
    def description(self, value: TaskDescription) -> None:
        """Set description of the task."""
        self.data.description = value

    @property
    def owner_id(self) -> RuntimeId:
        """Id of the task's owner."""
        return self.data.owner_id

    @owner_id.setter
    def owner_id(self, value: RuntimeId) -> None:
        """Set id of the task's owner."""
        self.data.owner_id = value

    @property
    def rank_limit(self) -> int | None:
        """Rank limit for the task."""
        return self.data.rank_limit

    @rank_limit.setter
    def rank_limit(self, value: int | None) -> None:
        """Set rank limit for the task."""
        self.data.rank_limit = value

    @property
    def output_dir(self) -> Path:
        """Directory for the task's output."""
        assert self.executor
        return self.executor.output_dir

    @property
    def name(self) -> str | None:
        """Name of the task."""
        return self.data.name

    @name.setter
    def name(self, value: str | None) -> None:
        """Set name of the task."""
        self.data.name = value

    @property
    def execution_history(self) -> ExecutionHistory:
        """Execution history for the task."""
        return self.data.execution_history

    @execution_history.setter
    def execution_history(self, value: ExecutionHistory) -> None:
        """Set execution history for the task."""
        self.data.execution_history = value

    @property
    def input_artifacts(self) -> list[Artifact]:
        """Artifacts for the task."""
        return self.data.input_artifacts

    @input_artifacts.setter
    def input_artifacts(self, value: list[Artifact]) -> None:
        """Set artifacts for the task."""
        self.data.input_artifacts = value

    @property
    def output_artifacts(self) -> list[Artifact]:
        """Artifacts for the task."""
        return self.data.output_artifacts

    @output_artifacts.setter
    def output_artifacts(self, value: list[Artifact]) -> None:
        """Set artifacts for the task."""
        self.data.output_artifacts = value

    @property
    def definition_of_done(self) -> str | None:
        """Definition of done for the task."""
        return self.description.definition_of_done

    @property
    def information(self) -> str:
        """Information on the task."""
        return self.description.information

    @property
    def information_with_artifacts(self) -> str:
        """Information on the task with artifacts."""
        return self.data.information_with_artifacts

    @property
    def initial_information(self) -> str:
        """Initial information on the task."""
        return self.data.initial_information

    @cached_property
    def event_log(self) -> EventLog:
        """Event log for the task."""
        return EventLog()

    @cached_property
    def subtasks(self) -> TaskList:
        """Subtasks of the task."""
        return TaskList()

    @property
    def messages(self) -> EventLog:
        """Messages for the task."""
        return self.event_log.messages

    @property
    def as_main_task_printout(self) -> str:
        """String representation of the task as it would appear as a main task."""
        return self.description.full

    def __str__(self) -> str:
        """String representation of task status."""
        return self.as_main_task_printout

    @property
    def executor_id(self) -> RuntimeId:
        """Id of the task's executor."""
        assert self.executor
        return self.executor.id

    @property
    def last_executor_rank(self) -> int | None:
        """Rank of the last executor."""
        return self.data.last_executor_rank

    @property
    def last_executor_id(self) -> RuntimeId:
        """Id of the last executor."""
        return self.data.last_executor_id

    @property
    def input_artifacts_printout(self) -> str:
        """String representation of the artifacts."""
        return self.data.input_artifacts_printout

    @property
    def output_artifacts_printout(self) -> str:
        """String representation of the artifacts."""
        return self.data.output_artifacts_printout

    @property
    def as_subtask_printout(self) -> str:
        """String representation of task as it would appear as a subtask."""
        assert self.name
        if self.work_status == TaskWorkStatus.COMPLETED:
            template = """
            - Id: {id}
              Name: {name}
              Artifacts:
            {artifacts}
            """
            return dedent_and_strip(template).format(
                id=self.id,
                name=self.name,
                artifacts=indent(self.output_artifacts_printout, "  "),
            )

        if self.work_status in {TaskWorkStatus.COMPLETED, TaskWorkStatus.CANCELLED}:
            template = """
            - Id: {id}
              Name: {name}
            """
            return dedent_and_strip(template).format(
                id=self.id,
                name=self.name,
            )
        template = """
        - Id: {id}
          Name: {name}
          Work Status: {work_status}
        """
        return dedent_and_strip(template).format(
            id=self.id,
            name=self.name,
            work_status=self.work_status.value,
        )

    @property
    def current_execution_outcome(self) -> ExecutionOutcome:
        """Add an execution outcome to the task."""
        assert self.executor
        return ExecutionOutcome(
            executor_id=self.executor.id,
            blueprint_id=self.executor.blueprint.id,
            executor_rank=self.executor.rank,
            validator_id=self.validator.id,
            validator_name=self.validator.name,
            success=False,
        )

    @property
    def closed(self) -> bool:
        """Whether the task is closed."""
        return self.work_status in {TaskWorkStatus.COMPLETED, TaskWorkStatus.CANCELLED}

    @property
    def context(self) -> str | None:
        """Information for the context of the task."""
        return self.data.context

    @property
    def completion_time(self) -> float | None:
        """How long it took to complete the task."""
        return (
            self.data.end_timestamp - self.data.start_timestamp
            if self.data.start_timestamp and self.data.end_timestamp
            else None
        )

    @property
    def validation_fail_count(self) -> int:
        """Number of times the task has failed validation."""
        return self.data.validation_failures

    def increment_fail_count(self) -> None:
        """Increment the number of validation failures."""
        self.data.validation_failures += 1

    def reformat_event_log(
        self,
        event_log: EventLog,
        pov: Literal[Concept.EXECUTOR, Concept.MAIN_TASK_OWNER, Concept.OBJECTIVE_POV],
    ) -> str:
        """Format an event log."""
        # use case for executor pov is for printing recent events log
        if pov == Concept.EXECUTOR:
            task_id_replacement = {
                self.id: Concept.MAIN_TASK.value,
            }
            subtask_executor_replacement = {
                subtask.executor_id: f"{Concept.EXECUTOR.value} for subtask {subtask.id}"
                for subtask in self.subtasks
                if subtask.executor
            }
            return event_log.to_str_with_pov(
                pov_id=self.executor_id,
                other_id=self.owner_id,
                other_name=Concept.MAIN_TASK_OWNER.value,
                task_id_replacement=task_id_replacement,
                subtask_executor_replacement=subtask_executor_replacement,
            )

        # use case for main task owner pov is for printing subtask discussion log when focused on a subtask
        if pov == Concept.MAIN_TASK_OWNER:
            assert (
                self.name is not None
            ), "If POV in a task discussion is from the main task owner, the task must have a name."
            task_id_replacement = None
            return event_log.to_str_with_pov(
                pov_id=self.owner_id,
                other_id=self.executor_id,
                other_name=Concept.EXECUTOR.value,
            )

        # use case for objective pov is for printing out to a 3rd party, such as a validator
        return event_log.to_str_with_objective_pov(
            task_owner_id=self.owner_id, executor_id=self.executor_id
        )

    def discussion(
        self,
        pov: Literal[
            Concept.EXECUTOR, Concept.MAIN_TASK_OWNER, Concept.OBJECTIVE_POV
        ] = Concept.OBJECTIVE_POV,
    ) -> str:
        """Discussion of a task in the event log."""
        return self.reformat_event_log(self.event_log.messages, pov)

    def execution_reply_message(self, reply: str) -> Event[Message]:
        """Create events for updating the status of the task upon execution."""
        assert self.executor
        return Event(
            data=Message(
                sender=self.executor_id,
                recipient=self.owner_id,
                content=reply,
            ),
            generating_task_id=self.id,
            id=generate_id(EventId, self.id_generator),
        )

    def add_execution_reply(self, reply: str) -> None:
        """Add an execution reply to the event log."""
        self.event_log.add(self.execution_reply_message(reply=reply))

    @property
    def serialization_location(self) -> Path:
        """Location for serializing the task."""
        return self.task_records_dir / f"{self.id}.yaml"

    def save(self) -> None:
        """Save the task."""
        serialized_data = self.data.serialize()
        self.serialization_location.write_text(format_as_yaml_str(serialized_data))

    def add_current_execution_outcome_to_history(self) -> None:
        """Add an execution outcome to the task."""
        self.execution_history.add(self.current_execution_outcome)

    def change_executor(self, executor: Executor) -> None:
        """Update the executor of the task."""
        # we assume that the executor is being changed because the previous executor failed, so we don't save the blueprint
        self.executor = executor
        self.add_current_execution_outcome_to_history()

    def start_timer(self) -> None:
        """Start the task timer."""
        self.data.start_timestamp = self.data.start_timestamp or time.time()

    def end_timer(self) -> None:
        """End the task timer."""
        assert (
            self.data.start_timestamp
        ), "Task timer must be started before it can be ended."
        assert not self.data.end_timestamp, "Task timer must not be ended twice."
        self.data.end_timestamp = time.time()

    def wrap_execution(self, success: bool) -> None:
        """Wrap up execution of the task."""
        self.end_timer()
        assert self.execution_history and self.executor
        assert self.executor.rank is not None
        self.execution_history.last_entry.success = success
        self.execution_history.last_entry.executor_rank = self.executor.rank
        self.executor.save_blueprint()
        self.executor = None
        self.save()

    @classmethod
    def from_serialized_data(
        cls,
        data: dict[str, Any],
        id_generator: IdGenerator,
        task_records_dir: Path,
    ) -> Self:
        """Deserialize the task."""
        raise NotImplementedError
        # TODO: this may never be needed
        # unchanged_fields = {"id", "description", "owner_id", "rank_limit", "name"}
        # modified_fields = {"validator", "work_status"}
        # excluded_fields = {"executor", "id_generator", "task_records_dir"}
        # assert (modified_fields | unchanged_fields) == (
        #     field_names := set(data.keys())
        # ), f"Field names don't match expected fields:\n{field_names=}\n{excluded_fields=}\n{modified_fields=}\n{unchanged_fields=}"

        # unchanged_data = {
        #     field.name: field.type(**data[field.name])
        #     for field in fields(cls)
        #     if field.name in unchanged_fields
        # }
        # modified_data = {
        #     "validator": Human(),
        #     "work_status": TaskWorkStatus(data["work_status"]),
        # }
        # return cls(
        #     **unchanged_data,
        #     **modified_data,
        #     id_generator=id_generator,
        #     task_records_dir=task_records_dir,
        # )


def validate_task_completion(
    task: Task, report: ExecutionReport
) -> WorkValidationResult:
    """Validate a task."""
    assert report.task_completed is True, "Task must be completed to be validated."
    context = """
    The following task is being executed by a task executor:
    ```start_of_task_specification
    {task_specification}
    ```end_of_task_specification

    The executor has reported that the task is complete. Here is the conversation for the task:
    ```start_of_task_conversation
    {task_conversation}
    ```end_of_task_conversation
    """
    context = dedent_and_strip(context).format(
        task_specification=task.as_main_task_printout,
        task_conversation=task.discussion(),
    )
    return task.validator.validate(context)


def change_status(
    task: Task, new_status: TaskWorkStatus, reason: str
) -> Event[TaskStatusChange]:
    """Change the status of a task."""
    assert (
        task.work_status != new_status
    ), "New status must be different from old status."
    status_update_event = Event(
        data=TaskStatusChange(
            changing_agent=task.validator.id,
            task_id=task.id,
            old_status=task.work_status,
            new_status=new_status,
            reason=reason,
        ),
        generating_task_id=task.id,
        id=generate_id(EventId, task.id_generator),
    )
    task.work_status = new_status  # MUTATION
    return status_update_event


def generate_artifact(task: Task) -> Artifact:
    """Generate artifacts."""
    context = """
    ## MISSION:
    You are a reviewer for a task, looking over the information for a completed task. You are now figuring out what {ARTIFACT}(s) must be generated for the task, and then writing the specifications for those {ARTIFACT}(s).

    ## CONCEPTS:
    These are the concepts you should be familiar with:
    - {EXECUTOR}: the agent that is responsible for executing a task.
    - {MAIN_TASK_OWNER}: the agent that owns the task and is responsible for providing information to the executor to help it execute the task.
    - {ARTIFACT}: the output of a task, in the form of a file, message, or a resource like a webpage. An {ARTIFACT} represents the final deliverable of a task.
    - {ARTIFACT} SPEC: a specification for an {ARTIFACT}, which includes all information needed to locate the {ARTIFACT}, or create it if it doesn't exist yet.

    ## TASK SPECIFICATION:
    Here is the task specification:
    ```start_of_task_specification
    {task_description}
    ```end_of_task_specification

    ## TASK CONVERSATION:
    Here is the conversation for the task:
    ```start_of_task_conversation
    {task_discussion}
    ```end_of_task_conversation

    ## {ARTIFACT} TYPES:
    There are several cases for what kind of {ARTIFACT}(s) you might need to generate for the task.

    ### ARTIFACT TYPE A: INLINE {ARTIFACT}
    An "inline" {ARTIFACT} is something that can be easily communicated in a sentence or two—something that can be quickly said in a live conversation. This {ARTIFACT} does not need to be stored and referenced in a separate resource, which is why it's called "inline".

    Format (YAML):
    ```start_of_inline_artifact_spec_format
    type: inline
    description: {{inline_artifact_description_str}}
    must_be_created: true
    content: |-
      {{full_artifact_content_str}}
    location: null
    ```end_of_inline_artifact_spec_format

    ### ARTIFACT TYPE B: FILE {ARTIFACT}
    A "file" {ARTIFACT} is information saved in a local file.

    Format (YAML):
    ```start_of_file_artifact_spec_format
    type: file
    description: {{file_description_str}}
    # `location` of `null` means we don't know the location yet
    must_be_created: {{true_or_false}}
    content: |-
      {{full_artifact_content_str_or_empty}}
    location: {{file_path_or_null}}
    ```end_of_file_artifact_spec_format

    ### ARTIFACT TYPE C: REMOTE RESOURCE {ARTIFACT}
    A "remote resource" {ARTIFACT} is information saved in a resource that is not local to the system, such as a webpage or a file stored in a cloud storage service. It can be any resource that is identified by a URI.

    Format (YAML):
    ```start_of_remote_resource_artifact_spec_format
    type: remote_resource
    description: {{resource_description}}
    # `location` of `null` means we don't know the location yet
    # We always assume that remote resource artifacts have already been created.
    must_be_created: false
    # Since the resource has already been created, the `content` field is `null`.
    content: null
    location: {{resource_uri_or_null}}
    ```end_of_remote_resource_artifact_spec_format
    """
    context = dedent_and_strip(context).format(
        ARTIFACT=Concept.ARTIFACT.value,
        MAIN_TASK_OWNER=Concept.MAIN_TASK_OWNER.value,
        EXECUTOR=Concept.EXECUTOR.value,
        task_description=task.description,
        task_discussion=task.discussion(pov=Concept.OBJECTIVE_POV),
    )
    reasoning_process = """
    determine_required_artifact_type_phase:
    - description: Review the TASK SPECIFICATION to determine if the task explicitly requires a particular {ARTIFACT} TYPE (A, B, or C).
      case_1:
        description: TASK SPECIFICATION clearly requires particular {ARTIFACT} TYPE.
        action: you know what {ARTIFACT} TYPE is needed, so skip the rest of the steps in this phase and go to `gather_spec_info_phase`.
      case_2:
        description: TASK SPECIFICATION does not clearly require particular {ARTIFACT} TYPE.
        action: move to the next step.
    - description: Assuming the task does not explicitly define what {ARTIFACT} TYPE is required, then check whether the latest message(s) in the TASK CONVERSATION from the {EXECUTOR} already have references to a specific {ARTIFACT} generated.
      case_1a:
        description: there are references to more than one {ARTIFACT}s.
        action: choose a single {ARTIFACT} that represents the output of the task, and go on to case_1b, the case for a single {ARTIFACT}.
      case_1b:
        description: there is a reference to a generated {ARTIFACT}.
        action:
          description: the {ARTIFACT} TYPE for each generated {ARTIFACT} is either B or C. Determine what kind of {ARTIFACT} TYPE is appropriate based on the TASK CONVERSATION, by checking whether the {ARTIFACT} has file references or remote resource references.
          case_1_1: the {ARTIFACT} has file references. That means it's of {ARTIFACT} TYPE B.
          case_1_2: the {ARTIFACT} has remote resource references. That means it's of {ARTIFACT} TYPE C.
          followup: you now know the required {ARTIFACT} TYPE, so skip the rest of the steps in this phase and go to `gather_spec_info_phase`.
      case_2:
        description: there are no references to a generated {ARTIFACT}.
        action: move to the next step.
    - description: Assuming there are no references to a generated {ARTIFACT}, check if the output of the task is something that can be easily communicated in a sentence or two—something that can be quickly said in a live conversation. In this case, we call the output "simple".
      case_1:
        description: the output is "simple".
        action: you now know the {ARTIFACT} TYPE is A.
      case_2:
        description: the output is not "simple".
        action: you now know the {ARTIFACT} TYPE is B.
    gather_spec_info_phase:
      case_A:
        description: the {ARTIFACT} TYPE is A, an inline {ARTIFACT}.
        required_info: TYPE A {ARTIFACT}s require gathering info about the `description` and `content` fields. The other fields are preset for this type.
        steps:
        - Reproduce the YAML spec format for the {ARTIFACT} TYPE as a reminder of what it should be.
        - Determine the value of the `description` field. For TYPE A, the `description` field is a description of what the `content` is for, to provide context for someone who reads the `content` without access to the original conversation.
        - Determine the value of the `content` field. For TYPE A, the `content` field is the exact message from the {EXECUTOR} that provides the answer to the task.
      case_B:
        description: the {ARTIFACT} TYPE is B, a local file {ARTIFACT}.
        required_info: TYPE B {ARTIFACT}s require gathering info about the `description`, `location`, `must_be_created`, and `content` fields. The other fields are preset for this type.
        steps:
        - Reproduce the YAML spec format for the {ARTIFACT} TYPE as a reminder of what it should be.
        - Determine the value of the `description` field. For TYPE B, the `description` is a concise summary of what's in the {ARTIFACT}.
        - Determine the value of the `must_be_created` field, by checking the TASK CONVERSATION to understand if the file for the {ARTIFACT} is supposed to already have been created. If it has, set the `must_be_created` field to `false`; otherwise, set it to `true`.
        - Determine the value of the `content` field. If `must_be_created` is `true`, then this would be the full output required by the task. Otherwise, the value would be the empty string.
        - Determine the value of the `location` field, by checking whether the TASK CONVERSATION contains an absolute path for the {ARTIFACT}. If it does, then set the `location` field to the absolute path to the file; if no path is present, or the path is relative, set it to `null`.
      case_C:
        description: the {ARTIFACT} TYPE is C, a remote resource {ARTIFACT}.
        required_info: TYPE C {ARTIFACT}s require gathering info in the `description`, and `location` fields. The other fields are preset for this type.
        steps:
        - Reproduce the YAML spec format for the {ARTIFACT} TYPE as a reminder of what it should be.
        - Determine the value of the `description` field. For TYPE C, the `description` is a concise description of what's in the {ARTIFACT}.
        - Determine the value of the `location` field, by checking whether the TASK CONVERSATION contains a full URI for the {ARTIFACT}. If it does, then set the `location` field to the full URI; otherwise, set it to `null`.
    """
    reasoning_process = dedent_and_strip(reasoning_process).format(
        ARTIFACT=Concept.ARTIFACT.value,
        EXECUTOR=Concept.EXECUTOR.value,
    )
    request = """
    ## REQUEST FOR YOU:
    Use the following reasoning process to gather the information for specifying {ARTIFACT}s for the task.

    ```start_of_reasoning_process
    {reasoning_process}
    ```end_of_reasoning_process

    {output_instructions}
    Remember to go through both the `determine_required_artifact_type_phase` and the `gather_spec_info_phase` phases of the reasoning process.

    After this block, you must output the final {ARTIFACT} SPEC in this format:
    ```start_of_artifact_spec_output
    {{artifact_spec_yaml}}
    ```end_of_artifact_spec_output
    Any additional comments or thoughts can be added as commented text in the yaml.
    """
    request = (
        dedent_and_strip(request)
        .replace("{output_instructions}", REASONING_OUTPUT_INSTRUCTIONS)
        .format(
            ARTIFACT=Concept.ARTIFACT.value,
            EXECUTOR=Concept.EXECUTOR.value,
            reasoning_process=reasoning_process,
        )
    )
    messages = [
        SystemMessage(content=context),
        SystemMessage(content=request),
    ]
    result = query_model(
        model=PRECISE_MODEL,
        messages=messages,
        preamble=f"Generating artifact specifications for task {task.id}...\n{format_messages(messages)}",
        printout=VERBOSE,
        color=SWARM_COLOR,
    )
    artifact_spec = extract_and_unpack(
        result, "start_of_artifact_spec_output", "end_of_artifact_spec_output"
    )
    artifact = Artifact.from_serialized_data(DEFAULT_YAML.load(artifact_spec))
    artifact.validate()
    # post-validation, we can assume that for file artifacts, if `content` exists
    if artifact.must_be_created and artifact.type == ArtifactType.FILE:
        artifact = write_file_artifact(artifact, task.output_dir)
    return artifact


def create_task_message(
    task: Task, message: str, sender_id: RuntimeId, event_id: EventId
) -> Event[Message]:
    """Create a message event for a task."""
    return Event(
        data=Message(
            sender=sender_id,
            recipient=task.executor_id,
            content=message,
        ),
        generating_task_id=task.id,
        id=event_id,
    )


def send_subtask_message(
    subtask: Task, message_event: Event[Message], initial: bool
) -> list[Event[TaskStatusChange]]:
    """Send a message to the executor of a subtask."""
    subtask.event_log.add(message_event)
    report_status_change = (
        not initial and subtask.work_status != TaskWorkStatus.IN_PROGRESS
    )
    status_change_event = change_status(
        subtask,
        TaskWorkStatus.IN_PROGRESS,
        f"Sent message to {Concept.EXECUTOR.value} unblock subtask.",
    )
    return [status_change_event] if report_status_change else []
