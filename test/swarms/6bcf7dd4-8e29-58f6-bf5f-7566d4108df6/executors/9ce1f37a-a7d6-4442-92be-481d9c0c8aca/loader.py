"""Loader for simple script writer executor."""

from core.bot import BotCore
from swarms.expanded_bots.component_crafter.function_writer import run_function_writer


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    return BotCore(run_function_writer, None)
    # return run_function_writer, None
