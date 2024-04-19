"""Loader for simple script writer executor."""

from swarm_memorizer.bot import BotCore
from base_swarm.bots.function_writer import run_function_writer


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    return BotCore(run_function_writer, None)
    # return run_function_writer, None
