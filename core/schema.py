"""Base level types for the swarm."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, NewType, Protocol, Sequence, TypeVar
from uuid import UUID

from langchain.schema import AIMessage, HumanMessage

NONE = "None"
BlueprintId = NewType("BlueprintId", str)
TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)
DelegatorId = NewType("DelegatorId", str)
RuntimeId = NewType("RuntimeId", str)
TaskHistory = list[TaskId]
IdGenerator = Callable[[], UUID]
IdTypeT = TypeVar("IdTypeT", BlueprintId, TaskId, EventId, DelegatorId)
ConversationHistory = Sequence[HumanMessage | AIMessage]
NoneStr = Literal["None"]
PauseExecution = NewType("PauseExecution", bool)
DelegationSuccessful = NewType("DelegationSuccessful", bool)


class Concept(Enum):
    """Concepts for swarm agents."""

    MAIN_TASK = "MAIN TASK"
    MAIN_TASK_OWNER = f"{MAIN_TASK} OWNER"
    ORCHESTRATOR = "ORCHESTRATOR"
    ORCHESTRATOR_ACTIONS = f"{ORCHESTRATOR} ACTIONS"
    EXECUTOR = "EXECUTOR"
    RECENT_EVENTS_LOG = "RECENT EVENTS LOG"
    ORCHESTRATOR_INFORMATION_SECTIONS = "ORCHESTRATOR INFORMATION SECTIONS"
    SUBTASK = "SUBTASK"
    SUBTASK_STATUS = f"{SUBTASK} STATUS"
    SUBTASK_EXECUTOR = f"{SUBTASK} {EXECUTOR}"
    FOCUSED_SUBTASK = f"FOCUSED {SUBTASK}"
    FOCUSED_SUBTASK_DISCUSSION_LOG = f"{FOCUSED_SUBTASK} DISCUSSION LOG"
    MAIN_TASK_DESCRIPTION = f"{MAIN_TASK} DESCRIPTION"
    MAIN_TASK_INFORMATION = f"{MAIN_TASK} INFORMATION"
    MAIN_TASK_DEFINITION_OF_DONE = f"{MAIN_TASK} DEFINITION OF DONE"
    TASK_MESSAGES = "TASK MESSAGES"
    LAST_READ_MAIN_TASK_OWNER_MESSAGE = f"LAST READ {MAIN_TASK_OWNER} MESSAGE"
    OBJECTIVE_POV = "OBJECTIVE POV"
    ARTIFACT = "ARTIFACT"
    CONTEXT = "CONTEXT"
    DELEGATOR = "DELEGATOR"
    DELEGATOR_INFORMATION_SECTIONS = "DELEGATOR INFORMATION SECTIONS"
    RECIPE = "SUBTASK_RECIPE"


CONCEPT_DEFINITIONS = {
    Concept.RECIPE: "a sequential set of {SUBTASK}s for completing a particular {MAIN_TASK}. Each {SUBTASK} in the recipe contains both a high-level description of what must be done, as well as things to consider when executing the {SUBTASK}.".format(
        **{key.name: key.value for key in Concept}
    )
}


@dataclass
class WorkValidationResult:
    """Validation of work done by agent."""

    valid: bool
    feedback: str


class Role(Enum):
    """Role of an agent."""

    ORCHESTRATOR = "orchestrator"
    BOT = "bot"


class ArtifactType(Enum):
    """Types of artifacts."""

    INLINE = "inline"
    FILE = "file"
    REMOTE_RESOURCE = "remote_resource"

    def __str__(self) -> str:
        """String representation of the artifact type."""
        return self.value


class TaskWorkStatus(Enum):
    """Status of the work for a task."""

    IDENTIFIED = "IDENTIFIED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    IN_VALIDATION = "IN_VALIDATION"


class WorkValidator(Protocol):
    """A validator of a task."""

    @property
    def name(self) -> str:
        """Name of the validator."""
        raise NotImplementedError

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the validator."""
        raise NotImplementedError

    def validate(self, context: str) -> WorkValidationResult:
        """Validate the work done by an executor for a task."""
        raise NotImplementedError


class ActionName(Enum):
    """Names of actions available to the orchestrator."""

    IDENTIFY_NEW_SUBTASK = "IDENTIFY_NEW_SUBTASK"
    START_DISCUSSION_FOR_SUBTASK = "START_DISCUSSION_FOR_SUBTASK"
    MESSAGE_TASK_OWNER = "MESSAGE_TASK_OWNER"
    REPORT_MAIN_TASK_COMPLETE = "REPORT_MAIN_TASK_COMPLETE"
    WAIT = "WAIT"
    MESSAGE_SUBTASK_EXECUTOR = "MESSAGE_SUBTASK_EXECUTOR"
    PAUSE_SUBTASK_DISCUSSION = "PAUSE_SUBTASK_DISCUSSION"
    CANCEL_SUBTASK = "CANCEL_SUBTASK"


class ReasoningGenerationNotes(Enum):
    """Template notes for reasoning generation."""

    OVERVIEW = "Provide a nested, robust reasoning structure in YAML format for the {role} to sequentially think through the information it has access to so that it has the appropriate mental context for deciding what to do next. Provide the internal thinking that an intelligent agent must go through so that they have all the relevant information on top of mind. Some things to note:"
    INFORMATION_RESTRICTIONS = "Assume that the {role} has access to what's described in {INFORMATION_SECTIONS} above, but no other information, except for general world knowledge that is available to a standard LLM like GPT-3."
    TERM_REFERENCES = """The {role} requires precise references to information it's been given, and it may need a reminder to check for specific parts; it's best to be explicit and use the _exact_ capitalized terminology to refer to concepts or information sections (e.g. "{example_section_1}" or "{example_section_2}"); however, only use capitalization to refer to specific terms—don't use capitalization as emphasis, as that could be confusing to the {role}."""
    STEPS_RESTRICTIONS = "The reasoning process should be written in second person, in YAML format, and be around 5-7 overall parts, though they can be nested arbitrarily deep as needed."
    PROCEDURAL_SCRIPTING = "The reasoning process can refer to the results of previous parts of the process, and it may be effective to build up the {role}'s mental context step by step, starting from examining basic facts, to more advanced compositional analysis, similar to writing a procedural script for a program but in natural language instead of code."


REASONING_PROCESS_OUTPUT_INSTRUCTIONS = """
Provide the reasoning process in the following YAML format in this block:
```start_of_reasoning_process
{{reasoning_process}}
```end_of_reasoning_process
You may add comments or thoughts before or after the reasoning process, but the reasoning process block itself must only contain the reasoning structure. Remember, the block must start with "```start_of_reasoning_process" and end with "```end_of_reasoning_process".
""".strip()


REASONING_OUTPUT_INSTRUCTIONS = """
In your reply, you must include output from _all_ parts of the reasoning process, in this block format:
```start_of_reasoning_output
{{reasoning_output}}
```end_of_reasoning_output
""".strip()


@dataclass
class ExecutionOutcome:
    """Outcome of an execution attempt."""

    executor_id: RuntimeId
    blueprint_id: BlueprintId
    executor_rank: int | None
    validator_id: RuntimeId
    validator_name: str
    success: bool


@dataclass
class ExecutionError(Exception):
    """Error when executing a task."""

    message: str
