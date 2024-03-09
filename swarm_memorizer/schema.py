"""Base level types for the swarm."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, NewType, Sequence, TypeVar
from uuid import UUID

from langchain.schema import AIMessage, HumanMessage

BlueprintId = NewType("BlueprintId", str)
TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)
DelegatorId = NewType("DelegatorId", str)
RuntimeId = NewType("RuntimeId", str)
TaskHistory = list[TaskId]
IdGenerator = Callable[[], UUID]
IdTypeT = TypeVar("IdTypeT", BlueprintId, TaskId, EventId, DelegatorId)
ConversationHistory = Sequence[HumanMessage | AIMessage]

NONE = "None"
NoneStr = Literal["None"]


@dataclass
class WorkValidationResult:
    """Validation of work done by agent."""

    valid: bool
    feedback: str


class Role(Enum):
    """Role of an agent."""

    ORCHESTRATOR = "orchestrator"
    BOT = "bot"
