"""Loader for simple script writer executor."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Sequence

from colorama import Fore
from langchain.schema import SystemMessage, BaseMessage, AIMessage, HumanMessage

from swarm_memorizer.swarm import (
    Blueprint,
    BotReply,
    EventLog,
    Task,
    Executor,
    RuntimeId,
    TaskDescription,
    get_choice,
    dedent_and_strip,
    ExecutorReport,
    format_messages,
    BotCore,
)
from swarm_memorizer.toolkit.models import query_model, precise_model
from swarm_memorizer.config import autogen_config_list

AGENT_COLOR = Fore.GREEN


def run_script_writer(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage]
) -> BotReply:
    """Run the script writer."""




    breakpoint()
    # > generate this using system message generator
    raise NotImplementedError("TODO")
    # > move this into the main package as a core bot


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> BotCore:
    """Load the bot core."""
    return run_script_writer, None
