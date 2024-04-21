"""Blueprints."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Self, Sequence
from core.schema import BlueprintId, Role
from core.toolkit.text import dedent_and_strip
from core.toolkit.yaml_tools import DEFAULT_YAML, format_as_yaml_str


@dataclass
class BotBlueprint:
    """A blueprint for a bot."""

    id: BlueprintId
    name: str
    description: str | None

    @property
    def role(self) -> Role:
        """Role of the agent."""
        return Role.BOT

    @property
    def rank(self) -> int:
        """Rank of the bot, which is always 0."""
        return 0

    def serialize(self) -> dict[Any, Any]:
        """Serialize the blueprint to a JSON-compatible dictionary."""
        assert self.description
        return asdict(self)

    @classmethod
    def from_serialized_data(cls, data: dict[Any, Any]) -> Self:
        """Deserialize the blueprint from a JSON-compatible dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            id=BlueprintId(data["id"]),
        )


@dataclass
class Reasoning:
    """Reasoning instructions for an agent."""

    default_action_choice: str | None = None
    subtask_action_choice: str | None = None
    subtask_extraction: str | None = None
    executor_selection: str | None = None
    learning: str | None = None


@dataclass
class TaskRecipe:
    """Recipe for a task."""

    task: str
    subtask_sequence: Sequence[str]

    def __str__(self) -> str:
        """String representation of the task recipe."""
        return format_as_yaml_str(asdict(self))


@dataclass
class ExecutionMemory:
    """Execution attempt of a subtask."""

    subtask_info: str
    successful: bool

    def __str__(self) -> str:
        """String representation of the execution memory."""
        template = """
        Subtask: {subtask_info}
        Successful: {successful}
        """
        return dedent_and_strip(template).format(
            subtask_info=self.subtask_info, successful=self.successful
        )


@dataclass
class ExecutorMemory:
    """Memory of an executor."""

    blueprint_id: BlueprintId
    task_execution_attempts: list[ExecutionMemory]

    def __str__(self) -> str:
        """String representation of the executor memory."""
        execution_attempts = format_as_yaml_str(
            [str(attempt) for attempt in self.task_execution_attempts]
        )
        template = """
        Executor ID: {blueprint_id}
        Task execution attempts:
        {execution_attempts}
        """
        return dedent_and_strip(template).format(
            blueprint_id=self.blueprint_id, execution_attempts=execution_attempts
        )


@dataclass
class Knowledge:
    """Knowledge for the orchestrator."""

    task_recipe: TaskRecipe
    executor_memories: list[ExecutorMemory]

    @property
    def executor_memories_text(self) -> str:
        """String representation of the executor memories."""
        return format_as_yaml_str(
            [str(executor_memory) for executor_memory in self.executor_memories]
        )


@dataclass
class OrchestratorBlueprint:
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


Blueprint = BotBlueprint | OrchestratorBlueprint


def load_blueprint(blueprint_path: Path) -> Blueprint:
    """Load a blueprint from a file."""
    blueprint_data = DEFAULT_YAML.load(blueprint_path)
    try:
        role = Role(blueprint_data["role"])
    except ValueError as error:
        raise ValueError(
            f"Invalid role for blueprint: {blueprint_data['role']}"
        ) from error

    BlueprintConstructor = BotBlueprint if role == Role.BOT else OrchestratorBlueprint
    blueprint = BlueprintConstructor.from_serialized_data(blueprint_data)
    assert blueprint.description, "Blueprint description cannot be empty."
    return blueprint


def load_blueprints(executors_dir: Path) -> Iterable[Blueprint]:
    """Load blueprints from the executors directory."""
    dirs = (
        executor_dir
        for executor_dir in executors_dir.iterdir()
        if executor_dir.is_dir()
    )
    return (
        load_blueprint(path)
        for executor_dir in dirs
        if (path := executor_dir / "blueprint.yaml").exists()
    )


def is_bot(blueprint: Blueprint) -> bool:
    """Check if a blueprint is a bot."""
    return blueprint.rank == 0
