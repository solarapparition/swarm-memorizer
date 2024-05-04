"""Test AutoGen core executor."""

# pylint:disable=redefined-outer-name

from pathlib import Path

from langchain.schema import HumanMessage, AIMessage
import pytest

from core.bot import BotCore
from core.task_data import TaskDescription
from swarms.bots.base.core_executor import load_bot
from tests.helpers.llm_evaluator import llm_evaluate


@pytest.fixture
def prime_task() -> str:
    """Return a task description."""
    return "Tell me the first 20 prime numbers."


@pytest.fixture
def bot_core() -> BotCore:
    """Return a bot core."""
    return load_bot()


def test_bot_no_initial_message(prime_task: str, bot_core: BotCore):
    """Test bot function when there's no initial message."""
    result = bot_core.runner(
        task_description=TaskDescription(prime_task),
        message_history=[],
        output_dir=Path("test/output"),
    )
    assert llm_evaluate(
        result.reply,
        condition="The reply states that the following are the first 20 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
    )


def test_bot_with_initial_message(prime_task: str, bot_core: BotCore):
    """Test bot function when there's an initial message."""
    result = bot_core.runner(
        task_description=TaskDescription(prime_task),
        message_history=[
            HumanMessage(
                content="Please ask me any questions about the task if you have any."
            )
        ],
        output_dir=Path("test/output"),
    )
    assert llm_evaluate(
        result.reply,
        condition="The VALUE states that the following are the first 20 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
    )


def test_bot_with_multiple_messages(prime_task: str, bot_core: BotCore):
    """Test bot function when there are multiple messages."""
    result_1 = bot_core.runner(
        task_description=TaskDescription("What are prime numbers?"),
        message_history=[],
        output_dir=Path("test/output"),
    )
    result = bot_core.runner(
        task_description=TaskDescription("What are prime numbers?"),
        message_history=[
            AIMessage(content=result_1.reply),
            HumanMessage(content=prime_task),
        ],
        output_dir=Path("test/output"),
    )
    assert llm_evaluate(
        result.reply,
        condition="The VALUE states that the following are the first 20 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
    )
