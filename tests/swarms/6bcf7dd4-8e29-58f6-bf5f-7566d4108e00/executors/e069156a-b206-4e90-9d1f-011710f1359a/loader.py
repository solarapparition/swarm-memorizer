"""Dummy bot core for testing purposes."""

from pathlib import Path
from typing import Sequence

from langchain.schema import AIMessage, HumanMessage

from core.bot import BotCore
from core.task import ExecutionReport
from core.task_data import TaskDescription


def load_bot(*_) -> BotCore:
    """Load the bot core."""

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        breakpoint()
        return ExecutionReport(
            "'Hello World!' was written to 'output.txt' successfully.",
            task_completed=True,
        )

    return BotCore(run, None)
    # return run, None
