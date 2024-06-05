"""Bot functionality for the swarm."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Protocol

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from swarm_memorizer.artifact import artifacts_printout
from swarm_memorizer.blueprint import BotBlueprint
from swarm_memorizer.config import SWARM_COLOR
from swarm_memorizer.acceptance import Acceptor, decide_acceptance
from swarm_memorizer.event import Event, Message
from swarm_memorizer.task import ExecutionReport, Executor, Task
from swarm_memorizer.schema import ConversationHistory, RuntimeId
from swarm_memorizer.task_data import TaskDescription
from swarm_memorizer.toolkit.files import make_if_not_exist
from swarm_memorizer.toolkit.models import PRECISE_MODEL, format_messages, query_model
from swarm_memorizer.toolkit.text import dedent_and_strip, extract_and_unpack
from swarm_memorizer.toolkit.yaml_tools import DEFAULT_YAML


class Runner(Protocol):
    """Core of a bot."""

    def __call__(
        self,
        task_description: TaskDescription,
        message_history: ConversationHistory,
        output_dir: Path,
    ) -> ExecutionReport:
        """Reply to a message."""
        raise NotImplementedError


class Serializer(Protocol):
    """Saving protocol for a bot."""

    def __call__(self, bot: "Bot") -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class BotCore:
    """Core of a bot."""

    runner: Runner
    acceptor: Acceptor | None = None
    serializer: Serializer | None = None


class CoreLoader(Protocol):
    """A loader of the core of a bot."""

    def __call__(
        self, blueprint: BotBlueprint, task: Task, files_dir: Path
    ) -> BotCore | Executor:
        """Load the core of a bot."""
        raise NotImplementedError


def extract_bot_core_loader(loader_location: Path) -> CoreLoader:
    """Extract a bot loader from a loader file."""
    assert loader_location.exists(), f"Loader not found: {loader_location}"
    module_name = loader_location.stem
    spec = importlib.util.spec_from_file_location(module_name, loader_location)
    assert spec and spec.loader, f"Could not load loader: {loader_location}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    assert hasattr(module, "load_bot"), "'load_bot' not found in the module"
    return getattr(module, "load_bot")


@dataclass(frozen=True)
class Bot:
    """Bot wrapper around a runner."""

    blueprint: BotBlueprint
    task: Task
    files_parent_dir: Path
    core: BotCore
    # runner: BotRunner
    # acceptor: Acceptor = decide_acceptance

    # @classmethod
    # def from_core(
    #     cls,
    #     blueprint: BotBlueprint,
    #     task: Task,
    #     files_parent_dir: Path,
    #     core: BotCore,
    # ) -> Self:
    #     """Create a bot from a core."""
    #     assert not isinstance(core, Executor)
    #     if not core.acceptor:
    #         return cls(blueprint, task, files_parent_dir, core.runner)
    #     return cls(blueprint, task, files_parent_dir, core.runner, core.acceptor)

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the bot."""
        return RuntimeId(f"{self.blueprint.id}_{self.task.id}")

    @property
    def rank(self) -> int:
        """Rank of the bot."""
        return 0

    def accepts(self, task: Task) -> bool:
        """Decides whether the bot accepts a task."""
        return (
            self.core.acceptor(task, self)
            if self.core.acceptor
            else decide_acceptance(task, self)
        )
        # return self.acceptor(task, self)

    def serialize(self) -> dict[str, Any] | None:
        """Serialize the bot."""
        return self.core.serializer(self) if self.core.serializer else None

    def save_blueprint(self) -> None:
        """Save the blueprint of the bot."""
        if not (blueprint := self.serialize()):
            return
        DEFAULT_YAML.dump(blueprint, self.serialization_location)

    @property
    def message_history(self) -> ConversationHistory:
        """Messages from the discussion log."""
        task_messages = self.task.event_log.messages

        def to_bot_message(event: Event[Message]) -> HumanMessage | AIMessage:
            """Convert an event to a message."""
            assert isinstance(event.data, Message)
            assert event.data.sender in {self.id, self.task.owner_id}
            message_constructor = (
                AIMessage if event.data.sender == self.id else HumanMessage
            )
            return message_constructor(content=event.data.content)

        message_history = [to_bot_message(event) for event in task_messages]
        if message_history:
            assert isinstance(
                message_history[-1], HumanMessage
            ), "Last message must be from the task owner"
        return message_history

    @property
    def formatted_message_history(self) -> str | None:
        """Formatted message history."""

        def sender(message: AIMessage | HumanMessage) -> str:
            """Sender of the message."""
            return "You" if isinstance(message, AIMessage) else "Task Owner"

        return (
            "\n".join(
                f"{sender(message)}: {message.content}"  # type: ignore
                for message in self.message_history
            )
            or None
        )

    @property
    def files_dir(self) -> Path:
        """Directory for the bot."""
        return make_if_not_exist(
            self.files_parent_dir / self.blueprint.id / self.task.id
        )

    @property
    def output_dir(self) -> Path:
        """Output directory for the bot."""
        return make_if_not_exist(self.files_dir / "output")

    @property
    def serialization_location(self) -> Path:
        """Location where the bot should be serialized."""
        return self.files_dir / "blueprint.yaml"

    @staticmethod
    def format_reply_message(report: ExecutionReport) -> str:
        """Format the reply message."""
        reply_message_with_artifacts = """
        {reply}

        Artifacts:
        {artifacts}
        """
        return (
            dedent_and_strip(reply_message_with_artifacts).format(
                reply=report.reply,
                artifacts=artifacts_printout(report.artifacts),
            )
            if report.artifacts
            else report.reply
        )

    def task_messages(self, report: ExecutionReport, executor_name: str = "You") -> str:
        """Get the task messages. Assume that `bot_reply` hasn't been added to the task's messages yet."""
        reply = f"{executor_name}: {report.reply}"
        return "\n".join(
            [self.formatted_message_history, reply]
            if self.formatted_message_history
            else [reply]
        )

    def generate_completion_status(self, report: ExecutionReport) -> bool:
        """Examine whether the bot has completed the task."""
        if report.artifacts:
            return True

        context = """
        # MISSION:
        You are a reviewer reviewing the status of a task executor.

        ## TASK INFORMATION:
        Here is the information about the task:
        ```start_of_task_info
        {task_information}
        ```end_of_task_info

        ## TASK MESSAGES:
        Here are the messages with the task owner:
        ```start_of_task_messages
        {task_messages}
        ```end_of_task_messages
        """
        context = dedent_and_strip(context).format(
            task_information=self.task.description,
            task_messages=self.task_messages(report, executor_name="Executor"),
        )
        request = """
        ## REQUEST FOR YOU:
        Based on the information above, determine if the executor seems to be waiting for an answer or confirmation from the task owner. Post your output in the following block format:
        ```start_of_confirmation_status
        comment: |-
          {{comment}}
        waiting_for_reply: !!bool |-
          {{waiting_for_reply}}
        ```end_of_confirmation_status
        {{waiting_for_reply}} must be `true` if the executor seems to be waiting for a reply from the task owner, and `false` if the executor does not seem to be waiting for a reply.
        """
        request = dedent_and_strip(request)
        messages = [
            SystemMessage(content=context),
            SystemMessage(content=request),
        ]
        result = query_model(
            model=PRECISE_MODEL,
            messages=messages,
            printout=True,
            preamble=f"Checking bot completion status...\n{format_messages(messages)}",
            color=SWARM_COLOR,
        )
        extracted = extract_and_unpack(result, "start_of_confirmation_status")
        extracted = DEFAULT_YAML.load(extracted)
        in_progress = extracted["waiting_for_reply"]
        assert isinstance(
            in_progress, bool
        ), f"Expected bool for `completed`, got: {in_progress}"
        return not in_progress

    def run(
        self,
        task_description: TaskDescription,
        message_history: ConversationHistory,
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        return self.core.runner(task_description, message_history, output_dir)

    async def execute(self) -> ExecutionReport:
        """Execute the task. Adds own message to the event log at the end of execution."""
        report = self.run(
            self.task.description, self.message_history, output_dir=self.output_dir
        )
        report.task_completed = (
            report.task_completed
            if isinstance(report.task_completed, bool)
            else self.generate_completion_status(report)
        )
        self.task.add_execution_reply(self.format_reply_message(report))
        return report
