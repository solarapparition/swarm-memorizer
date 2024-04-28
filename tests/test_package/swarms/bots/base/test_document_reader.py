"""Tests for document reader."""

# pylint:disable=redefined-outer-name

from pathlib import Path

from core.config import configure_langchain_cache
# from swarms.bots.base.document_reader import chat_with_resource
from tests.helpers.llm_evaluator import llm_evaluate

configure_langchain_cache(Path("tests/.data/.langchain_cache.db"))

def test_chat_with_resource(tmp_path: Path):
    """Test chat_with_resource function."""
    db_dir = tmp_path / "chat_with_resource_db"
    answer = chat_with_resource(
        query="What is the net worth of Elon Musk?",
        source="https://www.forbes.com/profile/elon-musk",
        db_dir=db_dir,
    )
    assert llm_evaluate(
        answer,
        condition="The VALUE states that the net worth of Elon Musk is $191.1 billion.",
    )
