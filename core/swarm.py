"""Structure for swarm agents."""

import asyncio
from dataclasses import InitVar, dataclass, field
from functools import cached_property
from pathlib import Path
import shelve
from typing import Protocol
from uuid import UUID

from .config import configure_langchain_cache
from .delegation import Delegator
from .execution import execute_and_validate
from .event import (
    Message,
    Event,
)
from .human import Human
from .id_generation import IdGenerator as DefaultIdGenerator, generate_id
from .schema import (
    EventId,
    ExecutionError,
    RuntimeId,
    IdGenerator,
    TaskWorkStatus,
    WorkValidator,
)
from .task import ExecutionReport, Task
from .task_data import TaskDescription, TaskData
from .toolkit.files import make_if_not_exist
from .toolkit.text import dedent_and_strip


class Director(Protocol):
    """Core of a swarm."""

    async def direct(self, task: Task, report: ExecutionReport) -> str:
        """Direct the task."""
        raise NotImplementedError


@dataclass
class DummyDirector:
    """Dummy core that just reflects back the validation error, or redirects to a human."""

    human: Human

    async def direct(self, task: Task, report: ExecutionReport) -> str:
        """Sends back the validation error for the task."""
        if report.validation and not report.validation.valid:
            return (
                f"The task was not successfully executed. {report.validation.feedback}"
            )
        return self.human.advise(report.reply)


@dataclass(frozen=True)
class Swarm:
    """Main interfacing class for the swarm."""

    task_description: str
    """Initial task description."""
    files_dir: Path
    """Directory for files related to the swarm."""
    core: Director = field(
        default_factory=lambda: DummyDirector(human=Human(name="Human Director"))
    )
    """Core of the swarm that makes top-level decisions."""
    validator: WorkValidator = field(
        default_factory=lambda: Human(name="Human Validator")
    )
    """Agent that approves or rejects work."""
    recent_events_size: int = 15
    """Number of recent events to display in orchestrators' event logs."""
    auto_wait: bool = True
    """Whether orchestrators will automatically wait for their executors. If disabled, orchestrators may perform other actions in parallel while an executor works on a subtask."""
    task_search_rerank_threshold: int = 100
    """When searching for similar past tasks, run a reranker if there are more than this many tasks."""
    id_generator: IdGenerator = field(default_factory=DefaultIdGenerator)
    """Generator for ids of entities in the system."""
    llm_cache_enabled: InitVar[bool] = field(default=True)
    """Whether to enable the LLM cache for identical calls to models."""

    def __post_init__(self, llm_cache_enabled: bool) -> None:
        """Post-initialization hook."""
        if llm_cache_enabled:
            configure_langchain_cache()

    @cached_property
    def id(self) -> RuntimeId:
        """Runtime id of the agent."""
        return RuntimeId(str(self.id_generator()))

    @property
    def name(self) -> str:
        """Name of the swarm."""
        return f"swarm_{self.id}"

    @property
    def cache_dir(self) -> Path:
        """Directory for the LLM cache."""
        return make_if_not_exist(self.files_dir / ".cache")

    @property
    def executors_dir(self):
        """Directory for executors."""
        return make_if_not_exist(self.files_dir / "executors")

    @property
    def task_records_dir(self):
        """Directory for task records."""
        if not (task_records_dir := self.files_dir / "task_records").exists():
            task_records_dir.mkdir(parents=True, exist_ok=True)
        return task_records_dir

    @property
    def executor_selection_reasoning(self) -> str:
        """Reasoning for selecting an executor."""
        reasoning = """
        1. Review the TASK INFORMATION to understand the nature and requirements of the TASK. Take note of any specific skills, expertise, or resources that are mentioned as being necessary to complete the TASK successfully.

        2. Examine the EXECUTOR CANDIDATES list to familiarize yourself with the potential executors' capabilities:
          a. Analyze each executor candidate’s DESCRIPTION to assess their theoretical ability to handle the TASK, focusing on any standout strengths or weaknesses relative to the TASK requirements.
          b. Identify if any of the candidates are NEW EXECUTOR, which signifies a lack of historical data on their TASK PERFORMANCE.

        3. For all non-NEW EXECUTOR candidates, evaluate their historical TASK PERFORMANCE:
          a. Consider the SUCCESS RATE to understand how consistently each executor has completed similar tasks in the past.
          b. Examine the COMPLETION TIME to gauge how efficiently each executor has completed similar tasks previously.

        4. Consider the importance of TASK PERFORMANCE relative to the TASK at hand:
          a. If the TASK is complex or has high-stakes outcomes, lean towards candidates with a higher SUCCESS RATE.
          b. If the TASK is time-sensitive, prioritize candidates with a lower COMPLETION TIME.

        5. Decide if any non-NEW EXECUTOR candidates are a suitable match based on the TASK INFORMATION and their TASK PERFORMANCE:
          a. If one or more non-NEW EXECUTOR candidates seem well-suited for the TASK, prepare to make a selection from among them in the final step.
          b. If no non-NEW EXECUTOR candidates are suitable, or if the TASK is one where exploration could yield better long-term results (e.g., low stakes or an opportunity to develop newer executors), consider a NEW EXECUTOR candidate.

        6. If considering a NEW EXECUTOR, evaluate the risk versus the potential of investing in the development of this executor:
          a. Appraise the potential benefits of allowing a NEW EXECUTOR to gain experience and possibly become a reliable option for future tasks.
          b. Balance the risk by reflecting on the criticality of the TASK, the theoretical capability of the NEW EXECUTOR, and the willingness to tolerate potential setbacks in TASK completion.

        7. Finalize the selection process by comparing executors:
          a. If a non-NEW EXECUTOR is deemed suitable based on their proven TASK PERFORMANCE and aptitude for the TASK, choose the best-fit candidate.
          b. If a NEW EXECUTOR is being considered for the reasons outlined in step 6 and their DESCRIPTION aligns well with the TASK, select one of them to balance the immediate needs with long-term strategic development.
          c. If neither non-NEW EXECUTOR candidates nor NEW EXECUTOR candidates are adequately matched to the TASK, opt not to delegate the TASK to any executor and reassess the required capabilities for the TASK.
        """
        return dedent_and_strip(reasoning)

    @cached_property
    def task(self) -> Task:
        """Task the swarm is keyed on."""
        return Task(
            data=TaskData(
                description=TaskDescription(information=self.task_description),
                owner_id=self.id,
                rank_limit=None,
                parent_rank_limit=None,
                input_artifacts=[],
            ),
            validator=self.validator,
            id_generator=self.id_generator,
            task_records_dir=self.task_records_dir,
        )

    @cached_property
    def delegator(self) -> Delegator:
        """Delegator for assigning tasks to executors."""
        return Delegator(
            executors_dir=self.executors_dir,
            task_records_dir=self.task_records_dir,
            task_search_rerank_threshold=self.task_search_rerank_threshold,
            id_generator=self.id_generator,
        )

    def receive(self, message: str) -> None:
        """Receive a message."""
        assert (
            self.task.executor is not None
        ), "Task executor must exist in order to be executed."
        message_event = Event(
            data=Message(
                sender=self.id,
                recipient=self.task.executor.id,
                content=message,
            ),
            generating_task_id=self.task.id,
            id=generate_id(EventId, self.id_generator),
        )
        self.task.work_status = TaskWorkStatus.IN_PROGRESS
        self.task.event_log.add(message_event)

    async def execute(self) -> ExecutionReport:
        """Execute the task the swarm is keyed on."""
        if self.task.work_status == TaskWorkStatus.COMPLETED:
            raise ExecutionError("Task has already been completed.")
        if not self.task.executor:
            self.delegator.assign_executor(
                self.task,
                self.recent_events_size,
                self.auto_wait,
                self.executor_selection_reasoning,
                executor_memory=None,
            )
            assert self.task.executor, "Task executor assignment failed."
        self.task.work_status = TaskWorkStatus.IN_PROGRESS
        report = await execute_and_validate(
            self.task,
            delegator=self.delegator,
            recent_events_size=self.recent_events_size,
            auto_await=self.auto_wait,
            executor_selection_reasoning=self.executor_selection_reasoning,
            executor_memory=None,
        )
        while True:
            if report.task_completed:
                return report
            directive = await self.core.direct(self.task, report)
            self.receive(directive)
            report = await execute_and_validate(
                self.task,
                delegator=self.delegator,
                recent_events_size=self.recent_events_size,
                auto_await=self.auto_wait,
                executor_selection_reasoning=self.executor_selection_reasoning,
                executor_memory=None,
            )

    async def receive_and_execute(self, message: str) -> ExecutionReport:
        """Receive and execute a task."""
        self.receive(message)
        return await self.execute()


TEST_DIR = Path(".data/test/agents")


def test_human_cache_response():
    """Test human response."""

    def ask_questions():
        with shelve.open(str(cache_path), writeback=True) as cache:
            human = Human(reply_cache=cache)
            human.advise("What is your name?")
            human.advise("What is your age?")

    cache_path = Path(".data/test/test_human_reply_cache")
    cache_path.unlink(missing_ok=True)
    ask_questions()
    ask_questions()
    cache_path.unlink(missing_ok=True)


@dataclass
class TestTask:
    """Test task."""

    task: str
    id_namespace: str
    purpose: str | None = None


async def run_test_task(test_task: TestTask) -> None:
    """Run a test task."""
    with shelve.open(".data/cache/human_reply", writeback=True) as cache:
        human_tester = Human(reply_cache=cache)
        files_dir = Path(f"tests/swarms/{test_task.id_namespace}")
        swarm = Swarm(
            core=DummyDirector(human=human_tester),
            task_description=test_task.task,
            files_dir=files_dir,
            validator=human_tester,
            id_generator=DefaultIdGenerator(
                namespace=UUID(test_task.id_namespace), seed="test"
            ),
        )
        report = await swarm.execute()
        while not report.task_completed:
            message = human_tester.advise(report.reply)
            report = await swarm.receive_and_execute(message)

    for file in (files_dir / "task_records").iterdir():
        if file.is_file():
            file.unlink()

CURRICULUM = [
    # TestTask(
    #     task="Write 'Hello, World!' to a file.",
    #     id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df4",
    # ),
    TestTask(
        task="Write 'Hello, World!' to a file.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e03",
    ),
    TestTask(
        task="Calculate 3 + 4 * 5.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df5",
        purpose="Tests a simple task that requires orchestration.",
    ),
    TestTask(
        task="Perform a quick search of the internet to figure out who 'Vivy' is, in an anime context.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df9",
        purpose="Tests perplexity base swarm bot.",
    ),
    TestTask(
        task="What are the colors of the rainbow?",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e04",
        purpose="Tests LLM assistant bot.",
    ),
    TestTask(
        task="Write 'Hello, World!' to a file.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e00",
        purpose="Tests core executor bot.",
    ),
    TestTask(
        task="Learn what the contents of the file at `tests/data/example_report.md` are.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e05",
        purpose="Test document reader bot.",
    ),

    # > basic coding task case: 20 lines or less of base python > coding bot will be equipped with function it wrote
    # > basic search task case: search for basic info about a concept
    # > basic file reading/writing task case
    # > basic browser task case
    # > learn how to create langchain agent
    # > full flow of learning how to perform some skill from a tutorial
    # > create an oai assistant agent using only documentation # need to set up virtual environment for it
    # > buy something from amazon

    # TestTask(
    #     task="Create a mock timestamp generator that advances by 1 second each time it is called.",
    #     id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df6",
    #     purpose="Tests function writing."
    # ),
    # TestTask(
    #     task="Find the first 10 prime numbers.",
    #     id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df7",
    #     purpose="Tests open interpreter bot."
    # ),
    # TestTask(
    #     task="Research Inflection 2.5, write a description of it and count the number of characters in the description.",
    #     id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e01",
    #     purpose="This tests integration of 2 base swarm bots: open interpreter and perplexity.",
    # ),
]


MINOR_CASES = [
    # TestTask(
    #     task="Write 'Hello, World!' to a file.",
    #     id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df8",
    # ),
    # TestTask(
    #     task="Research Inflection 2.5, write a description of it and count the number of characters in the description.",
    #     id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e02",
    #     purpose="This tests what happens when a bot fails a task but might not trigger validation flow.",
    # ),
]


def test() -> None:
    """Run tests."""
    configure_langchain_cache()
    asyncio.run(run_test_task(CURRICULUM[-1]))


if __name__ == "__main__":
    test()
