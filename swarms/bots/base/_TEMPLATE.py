"""Interface with Perplexity's online model."""

from pathlib import Path
from typing import Sequence
from colorama import Fore

from langchain.schema import AIMessage, HumanMessage

from core.bot import BotCore
from core.task import ExecutionReport
from core.task_data import TaskDescription


AGENT_COLOR = Fore.GREEN

def load_bot(*_) -> BotCore:
    """Load the bot core."""

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        return ExecutionReport(
            'The code executed successfully, and "Hello, World!" has been written to a file named `hello.txt`.'
        )

    return BotCore(run)
