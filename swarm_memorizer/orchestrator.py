"""Orchestrator agent."""

from dataclasses import asdict, dataclass
from typing import Any, Self

from swarm_memorizer.schema import BlueprintId, Role


@dataclass
class Reasoning:
    """Reasoning instructions for an agent."""

    default_action_choice: str | None = None
    subtask_action_choice: str | None = None
    subtask_extraction: str | None = None
    executor_selection: str | None = None
    learning: str | None = None


@dataclass
class Knowledge:
    """Knowledge for the orchestrator."""

    executor_learnings: str
    other_learnings: str

    def __str__(self) -> str:
        """String representation of the orchestrator's knowledge."""
        return f"{self.executor_learnings}\n{self.other_learnings}"


@dataclass
class Blueprint:
    """A blueprint for an orchestrator."""

    id: BlueprintId
    name: str
    description: str | None
    rank: int | None
    reasoning: Reasoning
    knowledge: Knowledge | None
    recent_events_size: int
    auto_wait: bool

    @property
    def role(self) -> Role:
        """Role of the agent."""
        return Role.ORCHESTRATOR

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any]) -> Self:
        """Deserialize the blueprint from a JSON-compatible dictionary."""
        data = data.copy()
        data["id"] = BlueprintId(data["id"])
        data["reasoning"] = Reasoning(**data["reasoning"])
        data["knowledge"] = Knowledge(**data["knowledge"])
        del data["role"]
        return cls(**data)

    def serialize(self) -> dict[str, Any]:
        """Serialize the blueprint to a JSON-compatible dictionary."""
        data = asdict(self)
        data["role"] = self.role.value
        return data
