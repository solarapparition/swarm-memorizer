"""Loader for simple script writer executor."""

from pathlib import Path
from functools import partial


from swarm_memorizer.swarm import (
    Blueprint,
    Task,
    BotCore,
)
from swarm_memorizer.core_bots.function_writer import run_function_writer


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> BotCore:
    """Load the bot core."""
    runner = partial(
        run_function_writer,
        task_description=task.description,
        output_dir=files_dir / "output",
    )
    return runner, None
