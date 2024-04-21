"""Toolkit for generating various descriptions."""

from langchain.schema import SystemMessage
from core.config import SWARM_COLOR, VERBOSE

from core.toolkit.models import query_model, PRECISE_MODEL, format_messages
from core.toolkit.text import dedent_and_strip, extract_blocks


def generate_agent_description(task_information: str) -> str:
    """Generate a description for an agent based on a task it has done."""
    context = """
    system_instruction: Develop Agent Capability Descriptions,
    task: Infer agent capabilities descriptions based on example task that agent has performed.
    objective: To transform a given example task description into a detailed and actionable description of an agent's full capabilities.
    steps:
    - Given a task description, infer potential capabilities the agent would be expected to have in order to solve the task.
    - Create a comprehensive description that outlines these capabilities, focusing on clarity and actionability.
    parameters:
      input: Task description,
      output: Agent capability description
    constraints:
    - Do not create the actual agent
    - Ensure the description is specific and actionable
    final_output:
      content: A clear, concise, and actionable description of what the agent can do, based on the provided task description.
      delimiters: The output should be wrapped in specific blocks indicated by "```start_of_agent_description_output" and "```end_of_agent_description_output" blocks.
    """
    context = dedent_and_strip(context)
    request = """
    ```start_of_task_description
    {task_information}
    ```end_of_task_description
    """
    request = dedent_and_strip(request).format(task_information=task_information)
    messages = [
        SystemMessage(content=context),
        SystemMessage(content=request),
    ]
    output = query_model(
        model=PRECISE_MODEL,
        messages=messages,
        preamble=f"Generating agent description from task description...\n{format_messages(messages)}",
        printout=VERBOSE,
        color=SWARM_COLOR,
    )
    output = extract_blocks(output, "start_of_agent_description_output")
    assert (
        output and len(output) == 1
    ), "Exactly one agent description output is expected."
    return output[0]
