"""Agent loading."""

from pathlib import Path

from swarm_memorizer.blueprint import Blueprint, BotBlueprint, OrchestratorBlueprint
from swarm_memorizer.bot import Bot, extract_bot_core_loader
from swarm_memorizer.orchestration import Orchestrator
from swarm_memorizer.schema import Role
from swarm_memorizer.delegation import Delegator
from swarm_memorizer.task import Executor, Task


def load_executor(
    blueprint: Blueprint, task: Task, files_dir: Path, delegator: Delegator
) -> Executor:
    """Factory function for loading an executor from a blueprint."""
    if blueprint.role == Role.BOT:
        assert isinstance(blueprint, BotBlueprint)
        loader_location = files_dir / "loader.py"
        load_bot_core = extract_bot_core_loader(loader_location)
        bot_core_or_executor = load_bot_core(blueprint, task, files_dir)
        if isinstance(bot_core_or_executor, Executor):
            return bot_core_or_executor
        return Bot.from_core(
            blueprint=blueprint,
            task=task,
            files_parent_dir=files_dir.parent,
            core=bot_core_or_executor,
        )
    assert isinstance(blueprint, OrchestratorBlueprint)
    return Orchestrator(
        blueprint=blueprint,
        task=task,
        files_parent_dir=files_dir.parent,
        delegator=delegator,
    )
