"""Loader for simple script writer executor."""

from swarm_memorizer.swarm import BotCore
from swarm_memorizer.core_bots.function_writer import run_function_writer


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    return run_function_writer, None
