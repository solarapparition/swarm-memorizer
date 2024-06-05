"""Interface with Perplexity's online model."""

from colorama import Fore
from pathlib import Path
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage

from swarm_memorizer.bot import BotCore
from swarm_memorizer.task import ExecutionReport
from swarm_memorizer.task_data import TaskDescription


AGENT_COLOR = Fore.GREEN

def load_bot(*_) -> BotCore:
    """Load the bot core."""

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        raise NotImplementedError
        return ExecutionReport(
            'The code executed successfully, and "Hello, World!" has been written to a file named `hello.txt`.'
        )

    return BotCore(run)
