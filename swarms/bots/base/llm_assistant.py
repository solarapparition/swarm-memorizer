"""Offline general-purpose LLM assistant."""

from pathlib import Path
from typing import Sequence
from colorama import Fore

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from swarm_memorizer.bot import BotCore
from swarm_memorizer.task import ExecutionReport
from swarm_memorizer.task_data import TaskDescription
from swarm_memorizer.toolkit.models import CREATIVE_MODEL, format_messages, query_model


AGENT_COLOR = Fore.GREEN

SYSTEM_MESSAGE = """
You are a helpful assistant.
Do your best to fulfill the user's request.
Remember that you cannot access the internet, nor any local files.
""".strip()


def create_conversation(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> Sequence[HumanMessage | AIMessage]:
    """Create messages for the bot."""

    def merge_messages(a: str, b: str) -> str:
        return f"{a}\n\n{b}"

    if message_history and isinstance(message_history[0], HumanMessage):
        first_message = merge_messages(
            str(task_description), str(message_history[0].content)
        )
        return [
            HumanMessage(content=first_message),
            *message_history[1:],
        ]
    return [
        HumanMessage(content=str(task_description)),
        *message_history,
    ]


def load_bot(*_) -> BotCore:
    """Load the bot core."""

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        system_message = SYSTEM_MESSAGE.format(task_description=task_description)
        conversation = create_conversation(task_description, message_history)
        messages = [SystemMessage(content=system_message), *conversation]
        output = query_model(
            model=CREATIVE_MODEL,
            messages=messages,
            color=AGENT_COLOR,
            preamble=format_messages(messages),
            printout=True,
        )
        return ExecutionReport(reply=output)

    return BotCore(run)
