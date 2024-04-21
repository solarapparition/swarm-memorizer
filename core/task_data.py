"""Data types for tasks."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import time
from typing import Any, Self, Sequence

from llama_index.schema import TextNode
from llama_index import VectorStoreIndex

from core.artifact import Artifact, artifacts_printout, abbreviated_artifacts_printout
from core.schema import BlueprintId, ExecutionOutcome, RuntimeId, TaskId
from core.toolkit.text import dedent_and_strip
from core.toolkit.yaml_tools import DEFAULT_YAML


@dataclass
class ExecutionHistory:
    """History of execution attempts for a task."""

    history: list[ExecutionOutcome] = field(default_factory=list)

    def add(self, execution_outcome: ExecutionOutcome) -> None:
        """Add an execution attempt to the history."""
        self.history.append(execution_outcome)

    @property
    def last_entry(self) -> ExecutionOutcome:
        """Current entry in the history."""
        return self.history[-1]

    def __bool__(self) -> bool:
        """Whether the execution history is empty."""
        return bool(self.history)


@dataclass
class TaskDescription:
    """Description of a task."""

    information: str
    definition_of_done: str | None = None
    initial_information: str = field(init=False)

    def __post_init__(self) -> None:
        """Post init."""
        self.initial_information = self.information

    @property
    def full(self) -> str:
        """Full description of the task."""
        template = """
        Information:
        {information}

        Definition of Done:
        {definition_of_done}
        """
        return dedent_and_strip(template).format(
            information=self.information, definition_of_done=self.definition_of_done
        )

    def __str__(self) -> str:
        """String representation of the task description."""
        return self.full if self.definition_of_done else self.information


@dataclass
class TaskData:
    """Data for a task."""

    description: TaskDescription
    owner_id: RuntimeId
    rank_limit: int | None
    input_artifacts: list[Artifact]
    id: TaskId | None = None
    name: str | None = None
    context: str | None = None
    execution_history: ExecutionHistory = field(default_factory=ExecutionHistory)
    output_artifacts: list[Artifact] = field(default_factory=list)
    start_timestamp: float | None = field(default=None, init=False)
    end_timestamp: float | None = field(default=None, init=False)
    validation_failures: int = field(default=0, init=False)
    parent_rank_limit: int | None = None

    def serialize(self) -> dict[str, Any]:
        """Serialize the task."""
        serialized_data = asdict(self)
        serialized_data["input_artifacts"] = [
            artifact.serialize() for artifact in self.input_artifacts
        ]
        serialized_data["output_artifacts"] = [
            artifact.serialize() for artifact in self.output_artifacts
        ]
        return serialized_data

    @classmethod
    def from_serialized_data(cls, data: dict[str, Any]) -> Self:
        """Deserialize the task from a JSON-compatible dictionary."""
        return cls(
            id=TaskId(data["id"]),
            description=TaskDescription(
                information=data["description"]["information"],
                definition_of_done=data["description"]["definition_of_done"],
            ),
            owner_id=RuntimeId(data["owner_id"]),
            rank_limit=data["rank_limit"],
            input_artifacts=data["input_artifacts"],
            name=data["name"],
            execution_history=ExecutionHistory(
                history=[
                    ExecutionOutcome(
                        executor_id=RuntimeId(execution["executor_id"]),
                        blueprint_id=BlueprintId(execution["blueprint_id"]),
                        executor_rank=execution["executor_rank"],
                        validator_id=RuntimeId(execution["validator_id"]),
                        validator_name=execution["validator_name"],
                        success=execution["success"],
                    )
                    for execution in data["execution_history"]["history"]
                ]
            ),
        )

    @property
    def last_executor_blueprint_id(self) -> BlueprintId | None:
        """Id of the last executor blueprint."""
        assert self.execution_history
        return self.execution_history.last_entry.blueprint_id

    @property
    def last_executor_rank(self) -> int | None:
        """Rank of the last executor."""
        assert self.execution_history
        return self.execution_history.last_entry.executor_rank

    @property
    def last_executor_id(self) -> RuntimeId:
        """Id of the last executor."""
        assert self.execution_history
        return self.execution_history.last_entry.executor_id

    @property
    def execution_successful(self) -> bool:
        """Whether the last execution was successful."""
        assert self.execution_history
        return self.execution_history.last_entry.success

    @property
    def executor_blueprint_ids(self) -> list[BlueprintId]:
        """Ids of all executor blueprints."""
        return [outcome.blueprint_id for outcome in self.execution_history.history]

    @property
    def input_artifacts_printout(self) -> str:
        """String representation of the artifacts."""
        # return artifacts_printout(self.input_artifacts)
        return abbreviated_artifacts_printout(self.input_artifacts)

    @property
    def output_artifacts_printout(self) -> str:
        """String representation of the artifacts."""
        return abbreviated_artifacts_printout(self.output_artifacts)

    @property
    def information_with_artifacts(self) -> str:
        """Information on the task with artifacts."""
        return f"Information:\n{self.description.information}\n\nInput Artifacts:\n{self.input_artifacts_printout}"

    @property
    def initial_information(self) -> str:
        """Initial information on the task."""
        return self.description.initial_information


def search_task_records_in_paths(
    task_info: str, task_record_files: Sequence[Path]
) -> list[TaskData]:
    """Search for similar tasks in the task records."""
    start_time = time.time()
    nodes: list[TextNode] = []
    task_data_list: list[TaskData] = []
    for task_record_file in task_record_files:
        old_task_data_dict: dict[str, Any] = DEFAULT_YAML.load(task_record_file)
        old_task_data = TaskData.from_serialized_data(old_task_data_dict)
        node = TextNode(
            text=old_task_data.initial_information,
            metadata=old_task_data_dict,  # type: ignore
            excluded_embed_metadata_keys=list(old_task_data_dict.keys()),
        )
        nodes.append(node)
        task_data_list.append(old_task_data)
    # print(time.time())
    index = VectorStoreIndex(nodes)
    # print(time.time())
    results = (
        index.as_query_engine(similarity_top_k=1000, response_mode="no_text")
        .query(task_info)
        .source_nodes
    )
    similarity_cutoff = 0.8
    results = [
        node.metadata["id"]
        for node in results
        if node.score and node.score > similarity_cutoff
    ]
    if time.time() - start_time > 10:
        raise NotImplementedError("TODO")
        # > TODO: need more efficient system to retrieve tasks; probably save `index`

    return [task_data for task_data in task_data_list if str(task_data.id) in results]


def search_task_records(task_info: str, task_records_dir: Path) -> list[TaskData]:
    """Search for similar tasks in the task records."""
    if not (task_record_files := tuple(task_records_dir.iterdir())):
        return []
    return search_task_records_in_paths(task_info, task_record_files)


def find_successful_tasks(
    blueprint_id: BlueprintId, task_info: str, task_records_dir: Path
) -> list[TaskData]:
    """List of tasks that have been successfully completed by the orchestrator, related to a particular task."""
    return [
        task_data
        for task_data in search_task_records(
            task_info, task_records_dir=task_records_dir
        )
        if task_data.last_executor_blueprint_id == blueprint_id
        and task_data.execution_successful
    ]


def find_failed_tasks(
    blueprint_id: BlueprintId, task_info: str, task_records_dir: Path
) -> list[TaskData]:
    """List of tasks that have failed to be completed by the orchestrator, related to a particular task."""
    return [
        task_data
        for task_data in search_task_records(
            task_info, task_records_dir=task_records_dir
        )
        if blueprint_id in task_data.executor_blueprint_ids
        and not task_data.execution_successful
    ]

