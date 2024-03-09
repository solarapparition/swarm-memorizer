"""Bot for writing simple Python functions based on user requirements."""

from pathlib import Path
from typing import Callable, Sequence
import sys

import pexpect
from colorama import Fore
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from interpreter import OpenInterpreter

from swarm_memorizer.swarm import (
    Artifact,
    BotCore,
    TaskDescription,
    dedent_and_strip,
    ExecutorReport,
    format_messages,
    Concept,
    SWARM_COLOR,
)
from swarm_memorizer.toolkit.models import query_model, precise_model
from swarm_memorizer.toolkit.script_runner import create_script_runner
from swarm_memorizer.toolkit.text import extract_and_unpack

AGENT_COLOR = Fore.GREEN


def create_message(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> str:
    """Create message to send to Open Interpreter."""
    if not message_history:
        return str(task_description)
    assert isinstance(message_history[-1], HumanMessage), (
        f"Expected last message to be a HumanMessage, but got: {type(message_history[-1])}.\n"
        "Message history:\n"
        f"{message_history}"
    )
    return str(message_history[-1].content)  # type: ignore


def run_open_interpreter(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
    output_dir: Path,
    send_message: Callable[[str], str],
) -> ExecutorReport:
    """Run Open Interpreter."""
    message = create_message(task_description, message_history)
    print(message)
    reply = send_message(message)
    print(f"{AGENT_COLOR}{reply}{Fore.RESET}")
    return ExecutorReport(reply)


def create_message_sender() -> Callable[[str], str]:
    """Create a message sender for the Open Interpreter."""
    python_interpreter = Path(sys.executable)
    script_path = Path(__file__).resolve().parent / "run_interpreter.py"
    input_pattern = "Send message: "
    return create_script_runner(
        script_path,
        input_pattern=input_pattern,
        output_pattern=None,
        interpreter=python_interpreter,
    )


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    send_to_open_interpreter = create_message_sender()

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutorReport:
        return run_open_interpreter(
            task_description=task_description,
            message_history=message_history,
            output_dir=output_dir,
            send_message=send_to_open_interpreter,
        )

    return run, None
