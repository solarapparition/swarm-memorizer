"""Dummy bot core for testing purposes."""

from pathlib import Path
from typing import Sequence

from langchain.schema import AIMessage, HumanMessage

from swarm_memorizer.bot import BotCore
from swarm_memorizer.task import ExecutionReport
from swarm_memorizer.task_data import TaskDescription


def load_bot(*_) -> BotCore:
    """Load the bot core."""

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        return ExecutionReport(
            'The description of Inflection 2.5 has been written to a file named "inflection_description.txt"',
            task_completed=True,
        )

    return run, None
