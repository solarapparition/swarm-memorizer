"""Bot for writing simple Python functions based on user requirements."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import sys

from colorama import Fore
from langchain.schema import AIMessage, HumanMessage

from core.config import PROMPT_COLOR
from core.toolkit.script_runner import create_script_runner
from core.task import ExecutionReport
from core.task_data import TaskDescription
from core.bot import BotCore

AGENT_COLOR = Fore.GREEN


def create_message(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> str:
    """Create message to send to Open Interpreter."""
    if not message_history:
        return str(task_description)
    if len(message_history) == 1:
        return f"{task_description}\n\n{message_history[0].content}"  # type: ignore
    assert isinstance(message_history[-1], HumanMessage), (
        f"Expected last message to be a HumanMessage, but got: {type(message_history[-1])}.\n"
        "Message history:\n"
        f"{message_history}"
    )
    return str(message_history[-1].content)  # type: ignore


def run_open_interpreter(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
    interpreter_proxy: Callable[[str], str],
) -> ExecutionReport:
    """Run Open Interpreter."""
    message = create_message(task_description, message_history)
    print(f"{PROMPT_COLOR}{message}{Fore.RESET}")
    reply = interpreter_proxy(message)
    print(f"{AGENT_COLOR}{reply}{Fore.RESET}")
    return ExecutionReport(reply)


@dataclass
class OpenInterpreterProxy:
    """Proxy for the Open Interpreter."""

    runner: Callable[[str], str] | None = None

    def __call__(self, message: str) -> str:
        """Send a message to the Open Interpreter."""
        assert self.runner, "Runner not set."
        return self.runner(message)


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    interpreter_proxy = OpenInterpreterProxy()

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        if not interpreter_proxy.runner:
            # very hacky, but Open Interpreter has a habit of taking over the rest of the stack that I don't understand very well so it must be isolated
            interpreter_proxy.runner = create_script_runner(
                script=Path(__file__).resolve().parent / "run_interpreter.py",
                input_pattern="Send message: ",
                output_pattern="Open Interpreter Reply:\r\n",  # yes this is correct on UNIX, see  https://pexpect.readthedocs.io/en/stable/overview.html#find-the-end-of-line-cr-lf-conventions
                interpreter=Path(sys.executable),
                cwd=output_dir,
            )
        return run_open_interpreter(
            task_description=task_description,
            message_history=message_history,
            interpreter_proxy=interpreter_proxy,
        )

    return BotCore(run, None)
