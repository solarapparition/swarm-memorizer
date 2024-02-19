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


def run_function_writer(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> BotReply:
    """Run the function writer."""
    request = """
    ## REQUEST
    Use the following reasoning process to respond to the user:
    {
        "system_instruction": "Create a simple Python function based on user requirements, ensuring clarity and effectiveness.",
        "task": "Write a simple Python function for a specific user-defined programming task.",
        "objective": [
            "Functions must only use base Python packages.",
            "The function should be straightforward to execute without needing testing.",
            "Refuse any tasks that are complex or require external packages."
        ],
        "steps": [
            "Assess the complexity and requirements of the user-defined task.",
            "Draft a function if the task is simple and requires only base Python packages.",
            "Ask for clarifications if the task description is ambiguous.",
            "Refuse the task if it requires external packages or is too complex."
        ],
        "parameters": {
            "user_defined_task": "The task as described by the user.",
            "discussion": "Discussion with the user to clarify the task requirements."
        },
        "output": {
            "options": {
                "function": "A Python function that meets the task requirements or a message explaining why the task cannot be fulfilled.",
                "clarification_needed": "A message asking for more details if the user's requirements are unclear.",
                "refusal": "A message explaining why the task cannot be fulfilled."
            }
            "format": {
                "main_output": {
                    block_delimiters: "The function or message itself is wrapped in ```start_of_main_output and ```end_of_main_output blocks. Must only contain the function or message.",
                    usage_examples: "Any usage examples must be part of the function's docstring."
                },
                "additional_comments": "Any additional comments must be outside of the main output block."
            }
        },
        "feedback": {
            "clarification_needed": "If the user's requirements are unclear, ask for more details.",
            "revision_request": "Allow for revisions based on user feedback."
        }
    }
    """
    request = dedent_and_strip(request)
    messages = [
        HumanMessage(content=str(task_description)),
        *message_history,
        SystemMessage(content=request),
    ]
    result = query_model(
        model=precise_model,
        messages=messages,
        preamble=f"Running Script Writer...\n{format_messages(messages)}",
        printout=True,
        color=AGENT_COLOR,
    )

    # generate this using system message generator
    breakpoint()
    raise NotImplementedError("TODO")
    # > move this into the main package as a core bot


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> BotCore:
    """Load the bot core."""
    return run_function_writer, None
