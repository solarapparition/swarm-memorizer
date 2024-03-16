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

    def direct(self, task: Task, report: ExecutionReport) -> str:
        """Direct the task."""
        raise NotImplementedError


@dataclass
class DummyDirector:
    """Dummy core that just reflects back the validation error."""

    def direct(self, task: Task, report: ExecutionReport) -> str:
        """Sends back the validation error for the task."""
        assert report.validation and not report.validation.valid
        return f"The task was not successfully executed. {report.validation.feedback}"


@dataclass(frozen=True)
class Swarm:
    """Main interfacing class for the swarm."""

    task_description: str
    """Initial task description."""
    files_dir: Path
    """Directory for files related to the swarm."""
    core: Director = field(default_factory=DummyDirector)
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
        report = await execute_and_validate(self.task)
        while True:
            if report.task_completed:
                return report
            directive = self.core.direct(self.task, report)
            self.receive(directive)
            report = await execute_and_validate(self.task)

    async def receive_and_execute(self, task_description: str) -> ExecutionReport:
        """Receive and execute a task."""
        self.receive(task_description)
        return await self.execute()


# test director system
# ....
# > update open interpreter: "programmatic task" to make it clear this is a coding agent
# check to see if learning works (when open interpreter fails)
# > access to and integration with devin # https://www.cognition-labs.com/blog
# test integration to base swarm
# ....
# > factor out tests
# > add generic typing to event
# > decouple __repr__ artifact from __str__ artifact > bypass step of having to have llm pass around artifact info # may require creation of artifact_id, to allow for orchestrator to reference artifact without seeing its data structure # would also be helpful for REPORT_TASK_AS_COMPLETE
# > add placeholder system for a different mode for each orchestrator action, to allow for reasoning within that mode
# > investigate having blueprints for swarm
# > add printout for open interpreter output
# integrate function writer into base swarm # make clear this is repeatable
# > (next_curriculum_task:integration)
# > (next_curriculum_task:basic) # reminder: system only meant to handle static, repeatable tasks; okay for it to not be able to do dynamic, state-based tasks
# add ability for bots to (optionally) save, which creates a specialized version of them
# bot: document chat > embedchain # has save() which creates embedchain wrapper around document
# > bot: image interpreter model > can this be done with open interpreter?
# > bot: script writer: update script writer to create wrapper bot around output script on save()
# bot: web browser > autogen web surfer agent
# bot: script runner: wrapper around a script that can run it # maybe open interpreter or autogen # has access to interactive python environment # need to be able to adapt it > possible: convert function to script using python fire lib > possible: use fire lib help function > when calling, try to determine missing arguments first; if any are missing, ask for them
# > (next_not_implemented)
# write readme > artifact system > demo > maybe video
# ---MVP---
# > bot: gpt-researcher
# > bot: openai assistant
# > bot: add pure, offline language frontier model assistants # really bad at math > gpt-4 > gemini > claude 3
# > bot: huggingface agent
# > bot: web browser > webvoyager > autogen web surfer agent
# > bot: document chat > embedchain > gemini pro 1.5
# > bot: search agent > exaai > tavily > perplexity
# > auto-integration of new bots > phase: convert any demo page to a script > phase: test any script based on spec from demo page > phase: convert any script to conversational interface (if needed) > phase: convert any conversational interface to a bot > phase: integrate bot into swarm > given an api page, convert to a bot for the api > bot creation: try generating command external agent interface using python fire lib
# > improve validation printout
# > bot: generalist agents > multion > cognosys > os-copilot https://github.com/OS-Copilot/FRIDAY > self-operating computer
# > soul engine https://github.com/opensouls/soul-engine
# > move off of langchain primitives to use litellm
# > main system dynamically populate hierarchical identity tree using memories
# > bot: alphacodium
# > bot: chat with github repo
# > investigate using campfire as backend
# > upgrade to new embedding model
# > explore chain-of-code # https://arxiv.org/abs/2312.04474
# > bot: slack agent: https://python.langchain.com/docs/integrations/toolkits/slack
# > bot creation: robocorp (langchain thing)
# > autonomous goal: learn to do more tasks
# > bot: code chain: https://huggingface.co/papers/2310.08992
# > llm based rerankers https://x.com/rohanpaul_ai/status/1753888043369726137?s=46&t=R6mLA3s_DNKUEwup7QWyCA
# > use any tool api retriever
# > try agent learning algorithm # agent learning paper: https://x.com/rohanpaul_ai/status/1754837097951666434?s=46&t=R6mLA3s_DNKUEwup7QWyCA
# > generate name as part of description generation
# > when assigning new executor, need to add event describing the switch # nice to have # not displayed to executor
# > replace messaging with instructor.patch
# > factor out validations into separate variable
# > unify validator functionality; validator protocol should hold specific functions to validate specific aspects of a task
# > estimate rank of task based on previous successful tasks
# > incorporate bot generation via autogen's builder
# > bot: amazon mturk
# turn printout into configurable parameter for swarm


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
        swarm = Swarm(
            task_description=test_task.task,
            files_dir=Path(f"test/swarms/{test_task.id_namespace}"),
            validator=human_tester,
            id_generator=DefaultIdGenerator(
                namespace=UUID(test_task.id_namespace), seed="test"
            ),
        )
        report = await swarm.execute()
        while not report.task_completed:
            message = human_tester.advise(report.reply)
            report = await swarm.receive_and_execute(message)


MAIN_CURRICULUM = [
    TestTask(
        task="Write 'Hello, World!' to a file.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df4",
    ),
    TestTask(
        task="Calculate 3 + 4 * 5.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df5",
        purpose="Tests a simple task that requires orchestration.",
    ),
    TestTask(
        task="Create a mock timestamp generator that advances by 1 second each time it is called.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df6",
    ),
    TestTask(
        task="Find the first 10 prime numbers.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df7",
    ),
    TestTask(
        task="Tell me about Inflection 2.5.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df9",
    ),
    TestTask(
        task="Write 'Hello, World!' to a file.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e00",
        purpose="This tests what happens when a bot fails a task.",
    ),
    TestTask(
        task="Write a description of Inflection 2.5 to a file called 'inflection_description.txt'.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108e00",
        purpose="This tests what happens when a bot fails a task.",
    ),
    # "Create a mock timestamp generator that advances by 1 second each time it is called, and run it 5 times.",
    # > basic coding task case: 20 lines or less of base python > coding bot will be equipped with function it wrote
    # > basic search task case: search for basic info about a concept
    # > basic file reading/writing task case
    # > basic browser task case
    # > learn how to create langchain agent
    # > full flow of learning how to perform some skill from a tutorial
    # > create an oai assistant agent using only documentation # need to set up virtual environment for it
    # > buy something from amazon
]
# CURRICULUM_TEST_TASKS = [
#     "Write 'Hello, World!' to a file.",
#     "Calculate 3 + 4 * 5.",
#     "Create a mock timestamp generator that advances by 1 second each time it is called.",
#     "Find the first 10 prime numbers.",
#     "Tell me about Inflection 2.5.", # recently released as of test creation
#     "Write a description of Inflection 2.5 to a file called 'inflection_description.txt'.",
# ]


MINOR_TASKS = [
    TestTask(
        task="Write 'Hello, World!' to a file.",
        id_namespace="6bcf7dd4-8e29-58f6-bf5f-7566d4108df8",
    ),
]

# MINOR_TASKS = [
#     "Write 'Hello, World!' to a file.",
# ]


def test() -> None:
    """Run tests."""
    configure_langchain_cache()
    asyncio.run(run_test_task(MAIN_CURRICULUM[5]))


if __name__ == "__main__":
    test()
