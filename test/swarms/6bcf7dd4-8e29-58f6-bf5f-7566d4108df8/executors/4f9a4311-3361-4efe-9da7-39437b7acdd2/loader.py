"""Bot for writing simple Python functions based on user requirements."""

from pathlib import Path
from typing import Sequence

from colorama import Fore
from langchain.schema import AIMessage, HumanMessage

from core.task import ExecutionReport
from core.task_data import TaskDescription
from core.bot import BotCore

AGENT_COLOR = Fore.GREEN

from swarms.expanded_bots.open_interpreter.loader import load_bot

# def load_bot(*_) -> BotCore:
#     """Load the bot core."""

#     def run(
#         task_description: TaskDescription,
#         message_history: Sequence[HumanMessage | AIMessage],
#         output_dir: Path,
#     ) -> ExecutionReport:
#         """Run the bot."""
#         return ExecutionReport(
#             'The code executed successfully, and "Hello, World!" has been written to a file named `hello.txt`.'
#         )

#     return run, None
