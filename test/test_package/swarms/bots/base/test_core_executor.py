"""Test AutoGen core executor."""

# pylint:disable=redefined-outer-name

from pathlib import Path
from typing import Any

from langchain.schema import SystemMessage
import pytest

from core.bot import BotCore
from core.task_data import TaskDescription
from core.toolkit.models import query_model, FAST_MODEL
from core.toolkit.text import dedent_and_strip, extract_and_unpack
from core.toolkit.yaml_tools import DEFAULT_YAML
from swarms.bots.base.core_executor import AutoGenRunner, load_bot


def test_set_agents():
    """Test set_agents method."""
    runner = AutoGenRunner()
    runner.set_agents(Path("test/output"))
    assert runner.assistant
    assert runner.user_proxy


def llm_evaluate(
    value: str, condition: str, query_kwargs: dict[str, Any] | None = None
) -> bool:
    """Evaluate whether a value meets a condition using an LLM."""
    query_kwargs = query_kwargs or {}
    instructions = """
    # MISSION
    You are an evaluator for the text output of some process. You will be given a CONDITION and a VALUE, and you must determine whether the VALUE meets the CONDITION.

    ## CONDITION
    The following is the condition you must evaluate the VALUE against:
    ```start_of_condition
    {condition}
    ```end_of_condition

    ## VALUE
    The following is the VALUE you must evaluate:
    ```start_of_value
    {value}
    ```end_of_value

    ## OUTPUT
    Output your evaluation in the following YAML format:
    ```start_of_output_yaml
    thoughts: |-
        {{thoughts}}
    condition_met: !!bool {{condition_met}}
    ```end_of_output_yaml

    Make sure to output the ``` delimiters.
    """
    instructions = dedent_and_strip(instructions).format(
        condition=condition, value=value
    )
    messages = [
        SystemMessage(content=instructions),
    ]
    output = query_model(
        model=FAST_MODEL,
        messages=messages,
        preamble=instructions,
        **query_kwargs,
    )
    output = extract_and_unpack(output, start_block_type="start_of_output_yaml")
    output = DEFAULT_YAML.load(output)
    return output["condition_met"]


@pytest.fixture
def task_description() -> TaskDescription:
    """Return a task description."""
    return TaskDescription("Tell me the first 20 prime numbers.")


@pytest.fixture
def bot_core() -> BotCore:
    """Return a bot core."""
    return load_bot()


def test_bot_no_initial_message(task_description: TaskDescription, bot_core: BotCore):
    """Test bot function when there's no initial message."""
    result = bot_core.runner(
        task_description=task_description,
        message_history=[],
        output_dir=Path("test/output"),
    )
    assert llm_evaluate(
        result.reply,
        condition="The reply states that the following are the first 20 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
    )


def test_bot_with_initial_message(task_description: TaskDescription, bot_core: BotCore):
    """Test bot function when there's an initial message."""
    result = bot_core.runner(
        task_description=task_description,
        message_history=[
            SystemMessage(
                content="Please ask me any questions about the task if you have any."
            )
        ],
        output_dir=Path("test/output"),
    )
    assert llm_evaluate(
        result.reply,
        condition="The reply states that the following are the first 20 primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71",
    )


# case: multiple messages
