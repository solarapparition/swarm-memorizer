"""Interface with Perplexity's online model."""

from pathlib import Path
from typing import Sequence
from colorama import Fore

from langchain.schema import AIMessage, HumanMessage

from core.bot import BotCore
from core.task import ExecutionReport
from core.task_data import TaskDescription
from core.toolkit.models import SEARCH_MODEL, format_messages, query_model


AGENT_COLOR = Fore.GREEN


def load_bot(*_) -> BotCore:
    """Load the bot core."""

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,  # pylint: disable=unused-argument
    ) -> ExecutionReport:
        """Run the bot."""
        if message_history and isinstance(message_history[0], HumanMessage):
            combined_message = "\n".join(
                [str(task_description), str(message_history[0].content)]
            )
            messages = [
                HumanMessage(content=combined_message),
                *message_history[1:],
            ]
        else:
            messages = [
                HumanMessage(content=str(task_description)),
                *message_history,
            ]
        result = query_model(
            model=SEARCH_MODEL,
            messages=messages,
            preamble=format_messages(messages),
            color=AGENT_COLOR,
            printout=True,
        )
        return ExecutionReport(reply=result)

    return BotCore(run, None)
