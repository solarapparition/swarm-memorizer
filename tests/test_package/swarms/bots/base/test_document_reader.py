"""Tests for document reader."""

# pylint:disable=redefined-outer-name

from pathlib import Path
from typing import Sequence

import pytest
from langchain_core.messages import HumanMessage

from core.artifact import Artifact, abbreviated_artifacts_printout
from core.bot import BotCore
from core.config import configure_langchain_cache
from core.task_data import TaskDescription
from core.toolkit.text import dedent_and_strip
from core.schema import ArtifactType
from swarms.bots.base.document_reader import load_bot
from tests.helpers.llm_evaluator import llm_evaluate

# from swarms.bots.base.document_reader import chat_with_resource

configure_langchain_cache(Path("tests/.data/.langchain_cache.db"))


@pytest.fixture
def report_location() -> Path:
    """The location of the example report."""
    return Path("tests/data/example_report.md")


@pytest.fixture
def input_artifacts(report_location: Path) -> Sequence[Artifact]:
    """The input artifacts."""
    return [
        Artifact(
            type=ArtifactType.FILE,
            description="A report.",
            location=str(report_location),
            must_be_created=False,
            content=None,
        )
    ]


@pytest.fixture
def bot_core() -> BotCore:
    """Return a bot core."""
    return load_bot()


@pytest.fixture
def task_description() -> TaskDescription:
    """The task description."""
    return TaskDescription(information="What is this report about?")


@pytest.fixture
def initial_message(input_artifacts: Sequence[Artifact]) -> str:
    """The request to the bot."""
    message = """
    Please feel free to ask me any questions about the context of this task—I've only given you a brief description to start with, but I can provide more information if you need it.

    Here are some existing artifacts that may be relevant for the task:
    {artifacts_printout}
    """
    artifacts_printout = abbreviated_artifacts_printout(input_artifacts)
    return dedent_and_strip(message).format(artifacts_printout=artifacts_printout)


def test_bot(
    initial_message: str,
    task_description: TaskDescription,
    bot_core: BotCore,
    tmp_path: Path,
):
    """Test the bot."""
    report = bot_core.runner(
        task_description=task_description,
        message_history=[HumanMessage(content=initial_message)],
        output_dir=tmp_path / "output",
    )
    assert llm_evaluate(
        report.reply,
        condition="The VALUE states that the document is about 3D processor technology.",
    )
