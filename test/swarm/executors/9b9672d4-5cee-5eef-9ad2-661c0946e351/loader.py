"""Loader for human fallback executor."""

from dataclasses import dataclass, field
from pathlib import Path

from colorama import Fore
from langchain.schema import SystemMessage, AIMessage
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate

from swarm_memorizer.swarm import (
    Blueprint,
    Task,
    Executor,
    RuntimeId,
    get_choice,
    dedent_and_strip,
    ExecutorReport,
    as_printable,
)
from swarm_memorizer.toolkit.models import query_model, precise_model

AGENT_COLOR = Fore.GREEN


@dataclass(frozen=True)
class AcceptAdvisor:
    """Advisor for accepting a task."""

    messages: list[SystemMessage | AIMessage] = field(default_factory=list)

    def advise(self, prompt: str) -> str:
        """Get advice about whether to accept the task or not."""
        self.messages.append(SystemMessage(content=prompt))
        result = query_model(
            model=precise_model,
            messages=self.messages,
            preamble=as_printable(self.messages),
            color=AGENT_COLOR,
            printout=True,
        )
        self.messages.append(AIMessage(content=dedent_and_strip(result)))
        return result


@dataclass(frozen=True)
class TextWriter:
    """Writes code and saves it."""

    blueprint: Blueprint
    task: Task
    files_dir: Path

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the orchestrator."""
        return RuntimeId(f"{self.blueprint.id}_{self.task.id}")

    @property
    def rank(self) -> int:
        """Return rank."""
        return 0

    def accepts(self, task: Task) -> bool:
        """Check if task is accepted by executor."""
        prompt = """
        Determine whether the following request is a request to write text to a file:
        ```
        {task_information}
        ```

        Reply with either "y" or "n", and no other text.
        """
        prompt = dedent_and_strip(prompt).format(task_information=task.information)
        return (
            get_choice(
                prompt,
                allowed_choices={"y", "n"},
                advisor=AcceptAdvisor(),
            )
            == "y"
        )

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        return self.files_dir / "output"

    async def execute(self, message: str | None = None) -> ExecutorReport:
        """Execute the subtask. Adds a message to the task's event log if provided, and adds own message to the event log at the end of execution."""
        assert not message, "No message should be provided to the TextWriter."

        @tool
        def write_text(text: str, file_name: str) -> None:
            """Write text to a file."""
            (self.output_dir / file_name).write_text(text)

        template = """
        

        # ....
        # > determine if params are there
        # > determine if params are valid
        # > if params are there and valid, write text to file
        # > if params are not there or not valid, return error message
        # ....

        {input}"""

        # https://python.langchain.com/docs/expression_language/cookbook/tools
        # ....
        prompt = ChatPromptTemplate.from_template(template)

        # write using lcel > define custom tools
        # ....
        breakpoint()
        raise NotImplementedError


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> Executor:
    """Load bot."""
    return TextWriter(blueprint, task, files_dir)
