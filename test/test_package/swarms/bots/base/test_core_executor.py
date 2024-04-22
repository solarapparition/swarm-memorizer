"""Test AutoGen core executor."""

from pathlib import Path

from swarms.bots.base.core_executor import AutoGenRunner


def test_set_agents():
    """Test set_agents method."""
    runner = AutoGenRunner()
    runner.set_agents(Path("test/output"))
    assert runner.assistant
    assert runner.user_proxy
