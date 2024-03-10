"""Task acceptance decisionmaking."""

from typing import Protocol

from langchain.schema import SystemMessage

from swarm_memorizer.config import SWARM_COLOR, VERBOSE
from swarm_memorizer.task_data import find_failed_tasks, find_successful_tasks
from swarm_memorizer.task import Executor, Task
from swarm_memorizer.toolkit.models import PRECISE_MODEL, format_messages, query_model
from swarm_memorizer.toolkit.text import dedent_and_strip, extract_blocks
from swarm_memorizer.toolkit.yaml_tools import DEFAULT_YAML, format_as_yaml_str


class Acceptor(Protocol):
    """Decides whether an executor accepts a task."""

    def __call__(self, task: Task, executor: Executor) -> bool:
        """Decide whether an executor accepts a task."""
        raise NotImplementedError


def decide_acceptance(task: Task, executor: Executor) -> bool:
    """Decide whether an executor accepts a task."""
    successful_tasks = find_successful_tasks(
        executor.blueprint.id, task.initial_information, task.task_records_dir
    )
    failed_tasks = find_failed_tasks(
        executor.blueprint.id, task.initial_information, task.task_records_dir
    )
    context = {
        "context": "The task history of an agent.",
        "success_list": [
            task_data.initial_information for task_data in successful_tasks
        ],
        "failure_list": [task.initial_information for task in failed_tasks],
        "description": executor.blueprint.description,
        "new_task": task.initial_information,
    }
    # context = dedent_and_strip(json.dumps(context, indent=4))
    context = dedent_and_strip(format_as_yaml_str(context))
    request = """
    system_instruction: Analyze and predict an agent's capability to perform a new task, based on its history of successes and failures.
    task: Prediction of success for a new task
    objective: Utilize agent description and historical performance data to assess the likelihood of an agent's success in a new task.
    analysis_steps:
    - description: Collect and review the agent's past successful and failed tasks.
      details:
        failure_tasks: List of tasks the agent has failed to complete in the past.
        success_tasks: List of tasks the agent has successfully completed in the past.
    - description: Analyze the new task in the context of the agent's historical performance.
      details:
        comparison_criteria: Similarity to past tasks, required skills, and task complexity.
        task_to_assess: Detailed description of the new task.
    - description: Validate that the task falls within the agent's theoretical capabilities, based on its description.
      details:
        agent_description: Description of the agent's theoretical capabilities and limitations.
    - description: Predict the agent's likelihood of success on the new task, based on the analysis.
      details:
        evaluation_method: Assessment of similarities, required skills, and complexity compared to past performances.
        prediction: Determination of success likelihood, clearly marked within specified blocks for emphasis.
    parameters:
      input_data:
        description: Description of the agent's capabilities and limitations
        failure_list: Tasks the agent has failed at
        new_task: The task to be assessed for likelihood of success
        success_list: Tasks successfully completed by the agent
      output_format:
        analysis_steps_output: Output of reasoning process for analysis steps
        analysis_steps_output_format: |-
          ```start_of_analysis_steps_output
          {analysis_steps_output}
          ```end_of_analysis_steps_output
        prediction: The agent's predicted success, highlighted within answer block.
        prediction_block_format: |-
          ```start_of_prediction_output
          comment: {comment}
          prediction: {prediction}
          ```end_of_prediction_output
        prediction_enums:
        - y
        - n
    feedback: Ensure clarity and precision in the analysis steps for effective understanding.
    """
    request = dedent_and_strip(request)
    messages = [
        SystemMessage(content=context),
        SystemMessage(content=request),
    ]
    output = query_model(
        model=PRECISE_MODEL,
        messages=messages,
        preamble=f"Deciding whether to accept new task...\n{format_messages(messages)}",
        printout=VERBOSE,
        color=SWARM_COLOR,
    )
    output = extract_blocks(output, "start_of_prediction_output")
    assert output and len(output) == 1, "Exactly one prediction output is expected."
    prediction = DEFAULT_YAML.load(output[0])["prediction"]
    assert prediction in {
        "y",
        "n",
    }, f"Prediction output must be 'y' or 'n'. Found:\n'{prediction}'."
    return prediction == "y"
