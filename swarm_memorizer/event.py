"""Classes for events."""

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, TypeVar, Generic

from swarm_memorizer.schema import (
    NONE,
    Concept,
    EventId,
    RuntimeId,
    TaskId,
    TaskWorkStatus,
    WorkValidationResult,
)
from swarm_memorizer.id_generation import utc_timestamp


@dataclass(frozen=True)
class Message:
    """Data for a message."""

    sender: RuntimeId
    recipient: RuntimeId
    content: str

    def __str__(self) -> str:
        return f"{self.sender} (to {self.recipient}): {self.content}"


@dataclass(frozen=True)
class SubtaskIdentification:
    """Data for identifying a new subtask."""

    owner_id: RuntimeId
    subtask: str
    subtask_id: TaskId
    validation_result: WorkValidationResult

    def __str__(self) -> str:
        if self.validation_result.valid:
            return f"{self.owner_id}: Successfully identified subtask: `{self.subtask}`; assigned subtask id: `{self.subtask_id}`."
        return f'{self.owner_id}: Attempted to identify subtask `{self.subtask}`, but validation of the subtask failed, with the following feedback: "{self.validation_result.feedback}"'


@dataclass(frozen=True)
class TaskStatusChange:
    """Data for changing the status of a subtask."""

    changing_agent: RuntimeId
    task_id: TaskId
    old_status: TaskWorkStatus
    new_status: TaskWorkStatus
    reason: str

    def __str__(self) -> str:
        return f"System: Status of task {self.task_id} has been updated from {self.old_status.value} to {self.new_status.value}. Reason: {self.reason.rstrip('.')}."


@dataclass(frozen=True)
class StartedSubtaskDiscussion:
    """Data for changing the focus of a main task owner."""

    owner_id: RuntimeId
    subtask_id: TaskId

    def __str__(self) -> str:
        return (
            f"{self.owner_id}: I've started discussion for subtask {self.subtask_id}."
        )


@dataclass(frozen=True)
class PausedSubtaskDiscussion:
    """Data for unfocusing from a subtask."""

    owner_id: RuntimeId
    subtask_id: TaskId

    def __str__(self) -> str:
        return f"{self.owner_id}: I've paused discussion for subtask {self.subtask_id}."


@dataclass(frozen=True)
class TaskDescriptionUpdate:
    """Data for updating the description of a task."""

    changing_agent: RuntimeId
    task_id: TaskId
    old_description: str
    new_description: str
    reason: str

    def __str__(self) -> str:
        return f"{self.changing_agent}: I've updated the description of task {self.task_id}. Reason: {self.reason.rstrip('.')}."


@dataclass(frozen=True)
class Thought:
    """Data for a thought."""

    agent_id: RuntimeId
    content: str

    def __str__(self) -> str:
        return f"{self.agent_id} (Thought): {self.content}"


@dataclass(frozen=True)
class TaskValidation:
    """Data for validating a task."""

    validator_id: RuntimeId
    task_id: TaskId
    validation_result: WorkValidationResult

    def __str__(self) -> str:
        if self.validation_result.valid:
            return (
                f"System: Task {self.task_id} was completed and has passed validation."
            )
        return f"System: Task {self.task_id} was reported as complete by executor, but failed validation, with the following feedback: {self.validation_result.feedback}."


def replace_agent_id(
    text_to_replace: str, replace_with: str, agent_id: RuntimeId
) -> str:
    """Replace agent id with some other string."""
    return (
        text_to_replace.replace(f"agent {agent_id}", replace_with)
        .replace(f"Agent {agent_id}", replace_with.title())
        .replace(agent_id, replace_with)
    )


def replace_task_id(text_to_replace: str, task_id: TaskId, replacement: str) -> str:
    """Replace task id with some other string."""
    return text_to_replace.replace(task_id, f"`{replacement}`")


EventData = TypeVar(
    "EventData",
    Message,
    SubtaskIdentification,
    TaskStatusChange,
    StartedSubtaskDiscussion,
    PausedSubtaskDiscussion,
    TaskDescriptionUpdate,
    Thought,
    TaskValidation,
)


@dataclass
class Event(Generic[EventData]):
    """An event in the event log."""

    data: EventData
    generating_task_id: TaskId
    """Id of the task that generated the event."""
    id: EventId
    timestamp: str = field(default_factory=utc_timestamp)

    def __str__(self) -> str:
        return f"{self.data}"

    def to_str_with_pov(
        self,
        pov_id: RuntimeId,
        other_id: RuntimeId,
        other_name: str,
        task_id_replacement: dict[TaskId, str] | None = None,
        subtask_executor_replacement: dict[RuntimeId, str] | None = None,
    ) -> str:
        """String representation of the event with a point of view from a certain executor."""
        event_printout = replace_agent_id(str(self), "You", pov_id)
        event_printout = replace_agent_id(event_printout, other_name, other_id)
        if not task_id_replacement:
            return event_printout
        for task_id, replacement in task_id_replacement.items():
            event_printout = replace_task_id(event_printout, task_id, replacement)
        if not subtask_executor_replacement:
            return event_printout
        for executor_id, replacement in subtask_executor_replacement.items():
            event_printout = replace_agent_id(event_printout, replacement, executor_id)
        return event_printout

    def to_str_with_objective_pov(
        self, task_owner_id: RuntimeId, executor_id: RuntimeId
    ) -> str:
        """String representation of the event with an objective point of view."""
        event_printout = replace_agent_id(
            str(self), Concept.MAIN_TASK_OWNER.value, task_owner_id
        )
        event_printout = replace_agent_id(
            event_printout, Concept.EXECUTOR.value, executor_id
        )
        return event_printout

    def serialize(self) -> dict[str, Any]:
        """Serialize the event."""
        return asdict(self)

    def __repr__(self) -> str:
        """String representation of the event."""
        return str(self.serialize())


@dataclass
class EventLog:
    """A log of events within a task."""

    events: list[Event[Any]] = field(default_factory=list)

    @property
    def last_event(self) -> Event[Any] | None:
        """Last event in the event log."""
        return self.events[-1] if self.events else None

    @property
    def messages(self) -> "EventLog":
        """Messages in the event log."""
        return EventLog(
            events=[event for event in self.events if isinstance(event.data, Message)]
        )

    def to_str_with_pov(
        self,
        pov_id: RuntimeId,
        other_id: RuntimeId,
        other_name: str,
        task_id_replacement: dict[TaskId, str] | None = None,
        subtask_executor_replacement: dict[RuntimeId, str] | None = None,
    ) -> str:
        """String representation of the event log with a point of view from a certain executor."""
        return (
            "\n".join(
                [
                    event.to_str_with_pov(
                        pov_id,
                        other_id,
                        other_name,
                        task_id_replacement,
                        subtask_executor_replacement,
                    )
                    for event in self.events
                ]
            )
            if self.events
            else NONE
        )

    def to_str_with_objective_pov(
        self, task_owner_id: RuntimeId, executor_id: RuntimeId
    ) -> str:
        """String representation of the event log with an objective point of view."""
        return (
            "\n".join(
                [
                    event.to_str_with_objective_pov(task_owner_id, executor_id)
                    for event in self.events
                ]
            )
            if self.events
            else NONE
        )

    def recent(self, num_recent: int) -> "EventLog":
        """Recent events."""
        return EventLog(events=self.events[-num_recent:])

    def add(self, *events: Event[Any]) -> None:
        """Add events to the event log."""
        self.events.extend(events)

    def __str__(self) -> str:
        """String representation of the event log."""
        return "\n".join([str(event) for event in self.events]) if self.events else NONE

    def __bool__(self) -> bool:
        """Whether the event log is empty."""
        return bool(self.events)

    def __iter__(self) -> Iterator[Event[Any]]:
        """Iterate over the event log."""
        return iter(self.events)
