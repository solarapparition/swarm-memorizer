"""Loader for simple script writer executor."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Sequence

from colorama import Fore
from langchain.schema import SystemMessage, BaseMessage, AIMessage, HumanMessage

from swarm_memorizer.swarm import (
    Artifact,
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
from swarm_memorizer.toolkit.text import extract_and_unpack

AGENT_COLOR = Fore.GREEN


def generate_messages(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> Sequence[HumanMessage | AIMessage]:
    """Generate messages for the model."""
    return [
        HumanMessage(content=str(task_description)),
        *message_history,
    ]


def generate_script(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> str:
    """Generate a simple Python script."""
    request = """
    ## REQUEST
    Use the following reasoning process to respond to the user:
    {
        "system_instruction": "Create a simple Python function based on user requirements, ensuring clarity and effectiveness.",
        "task": "Execute reasoning process to write a simple Python function for a specific user-defined programming task.",
        "objective": [
            "Functions must only use base Python packages.",
            "The function should be straightforward to execute without needing testing.",
            "Refuse any tasks that are complex or require external packages."
        ],
        "reasoning_process": [
            "Assess the complexity and requirements of the user-defined task.",
            {
                "action_determination": [
                {
                    "if": "The task is simple and requires only base Python packages",
                    "then": "Draft a function that meets the task requirements."
                },
                {
                    "if": "The task description is ambiguous",
                    "then": "Ask for clarifications."
                },
                {
                    "if": "The task requires external packages or is too complex",
                    "then": "Refuse the task."
                },
                ]
            }
        ],
        "parameters": {
            "user_defined_task": "The task as described by the user.",
            "discussion": "Discussion with the user to clarify the task requirements."
        },
        "output": {
            "format": {
                "reasoning_output": {
                    "block_delimiters": "The reasoning output is wrapped in ```start_of_reasoning_output and ```end_of_reasoning_output blocks.",
                    "output_scope": "All parts of the reasoning process must be included in the output."
                }
                "main_output": {
                    "block_delimiters": "The function or message itself is wrapped in ```{start_of_main_output} and ```{end_of_main_output} blocks. Must only contain the function or message.",
                    "usage_examples": "Any usage examples must be commented out to avoid accidental execution."
                    "additional_comments": "Any additional comments must be outside of the main output block."
                },
            }
            "main_output_options": {
                "function": "A Python function that meets the task requirements or a message explaining why the task cannot be fulfilled.",
                "clarification_needed": "A message asking for more details if the user's requirements are unclear.",
                "refusal": "A message explaining why the task cannot be fulfilled."
            }
        },
        "feedback": {
            "clarification_needed": "If the user's requirements are unclear, ask for more details.",
            "revision_request": "Allow for revisions based on user feedback."
        }
    }
    """
    start_delimiter = "start_of_main_output"
    end_delimiter = "end_of_main_output"
    request = (
        dedent_and_strip(request)
        .replace("{start_of_main_output}", start_delimiter)
        .replace("{end_of_main_output}", end_delimiter)
    )
    messages = [
        *generate_messages(
            task_description=task_description, message_history=message_history
        ),
        SystemMessage(content=request),
    ]
    result = query_model(
        model=precise_model,
        messages=messages,
        preamble=f"Running Script Writer...\n{format_messages(messages)}",
        printout=True,
        color=AGENT_COLOR,
    )
    return extract_and_unpack(
        text=result, start_block_type=start_delimiter, end_block_type=end_delimiter
    )


def determine_task_completion(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
    task_result: str,
) -> bool:
    """Determine if the task was completed."""
    request = """
    ## REQUEST
    For saving a record of this request, please output the following:
    - "y" if you have generated a Python function based on the user's requirements.
    - "n" otherwise

    Output the answer in the following block:
    ```start_of_y_or_n_output
    {y_or_n}
    ```end_of_y_or_n_output
    """
    request = dedent_and_strip(request)
    messages = [
        *generate_messages(
            task_description=task_description, message_history=message_history
        ),
        AIMessage(content=task_result),
        SystemMessage(content=request),
    ]
    result = query_model(
        model=precise_model,
        messages=messages,
        preamble=f"Checking task completion...\n{format_messages(messages)}",
        printout=True,
        color=AGENT_COLOR,
    )
    task_completed = extract_and_unpack(
        text=result,
        start_block_type="start_of_y_or_n_output",
        end_block_type="end_of_y_or_n_output",
    )
    assert task_completed in {
        "y",
        "n",
    }, f"Invalid task completion: {task_completed}. Must be 'y' or 'n'."
    return task_completed == "y"


def save_artifact(result: str, output_dir: Path) -> Path:
    """Save the artifact."""
    output_location = output_dir / "script.py"
    output_location.write_text(result, encoding="utf-8")
    return output_location


def run_function_writer(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
    output_dir: Path,
) -> BotReply:
    """Run the function writer."""
    result = generate_script(task_description, message_history)
    task_completed = determine_task_completion(
        task_description, message_history, result
    )
    if task_completed:
        output_location = save_artifact(result, output_dir)
        reply = "Function has been successfully written."
        artifacts = [
            Artifact(
                location=str(output_location),
                description=f"Python function written for the following task: {task_description}",
            )
        ]
    else:
        reply = result
        artifacts = []
    report = ExecutorReport(
        reply=reply,
        task_completed=task_completed,
    )
    return BotReply(
        report=report,
        artifacts=artifacts,
    )


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> BotCore:
    """Load the bot core."""
    return run_function_writer, None
