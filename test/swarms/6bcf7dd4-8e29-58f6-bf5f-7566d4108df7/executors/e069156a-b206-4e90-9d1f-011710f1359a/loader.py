"""Bot for writing simple Python functions based on user requirements."""

from pathlib import Path
from typing import Sequence

from colorama import Fore
from langchain.schema import SystemMessage, AIMessage, HumanMessage
from interpreter import OpenInterpreter

from swarm_memorizer.swarm import (
    Artifact,
    BotCore,
    TaskDescription,
    dedent_and_strip,
    ExecutorReport,
    format_messages,
    Concept,
)
from swarm_memorizer.toolkit.models import query_model, precise_model
from swarm_memorizer.toolkit.text import extract_and_unpack

AGENT_COLOR = Fore.GREEN


def create_message(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
) -> str:
    """Create message to send to Open Interpreter."""
    if not message_history:
        return str(task_description)
    assert isinstance(message_history[-1], HumanMessage), (
        f"Expected last message to be a HumanMessage, but got: {type(message_history[-1])}.\n"
        "Message history:\n"
        f"{message_history}"
    )
    return str(message_history[-1].content)  # type: ignore


def setup_interpreter() -> OpenInterpreter:
    """Set up the Open Interpreter."""
    interpreter = OpenInterpreter()
    interpreter.llm.model = "gpt-4-turbo-preview"
    interpreter.safe_mode = "auto"
    interpreter.auto_run = True
    interpreter.llm.context_window = 128_000
    interpreter.llm.max_tokens = 4096
    return interpreter


DUMMY_REPLY = ExecutorReport(
    reply="Here are the first 100 prime numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541",
)


def run_open_interpreter(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
    output_dir: Path,
) -> ExecutorReport:
    """Run Open Interpreter."""

    message = create_message(task_description, message_history)
    interpreter = setup_interpreter()
    



    breakpoint()
    artifacts_instructions = """
    Definition: {ARTIFACT}: some information at a location that is relevant to your task. Usually {ARTIFACT}(s) are files, but they can also be URLs, database queries, or other precise references to information.
    - You may be given {ARTIFACT}(s) when a task is given to you.
    - You may need to generate {ARTIFACT}(s) as output(s) for your task.

    ### Artifact Generation Process:
    Use the following process to generate output {ARTIFACT}(s):
    1. Determine whether the task is complete. If the task is not complete, then you do **not** need to generate any output {ARTIFACT}(s). THE REMAINING STEPS ASSUME THAT THE TASK IS COMPLETE.
    2. When a task is complete, check whether the user explicitly asked you to generate any output {ARTIFACT}(s). If the user did ask, then generate the {ARTIFACT}(s), even if they contain extremely simple output, such as a single word.
    3. If the user did not explicitly request an artifact, then check if the output of the task is something that would be simple enough to communicate in a mobile text message, usually a sentence or two—for example, a quick answer to a simple question. If it is, then you do not need to generate any output {ARTIFACT}(s); otherwise, generate the {ARTIFACT}(s).
    4. Generate the {ARTIFACT}(s) as output(s) for your task:
    - Most {ARTIFACT}(s) you generate will be files.
    - When generating file {ARTIFACT}(s), save them to {output_dir}.
    - File {ARTIFACT}(s) should be named in a way that is clear and descriptive of their contents, with extensions that make it clear what type of file they are.
    5. Correct any writing errors in the {ARTIFACT}(s) you generated.
    6. Send a final message to the user that includes the locations of the {ARTIFACT}(s) you generated.
    - Your final message to the user must include the FULL PATH to specific locations of the {ARTIFACT}(s) you generated. Assume the reader will only have access to your final message and the {ARTIFACT}(s) you generated.
    - Example: "I have generated the following {ARTIFACT}(s) for you: {output_dir}/{{file_1_name}}, which has {{some_description}}, and {output_dir}/{{file_2_name}}, which has {{some_other_description}}."
    """
    artifact_instructions = dedent_and_strip(artifacts_instructions).format(
        ARTIFACT=Concept.ARTIFACT.value, output_dir=output_dir
    )
    interpreter.custom_instructions = artifact_instructions
    reply_message = interpreter.chat(message=message)
    reply_message = interpreter.chat(
        message="Please check whether you've followed the Artifact Generation Process in ADDITIONAL INSTRUCTIONS; if not, go through it step-by-step"
    )

    # define artifact creation rules
    breakpoint()

    raise NotImplementedError

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
    # > open interpreter bug: not sure why max_tokens gets system tokens deducted twice # .venv/lib/python3.11/site-packages/tokentrim/tokentrim.py
    # > move off of langchain primitives to use litellm
    # > classifier to determine whether a task is complete or not
    # > update artifact validation to add exception for simple answers # basically, if the answer is something that can be communicated in a mobile text message, then don't need artifact # needs an artifact if reply implies that something was done


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    return run_open_interpreter, None


# def generate_messages(
#     task_description: TaskDescription,
#     message_history: Sequence[HumanMessage | AIMessage],
# ) -> Sequence[HumanMessage | AIMessage]:
#     """Generate messages for the model."""
#     return [
#         HumanMessage(content=str(task_description)),
#         *message_history,
#     ]


# def generate_script(
#     task_description: TaskDescription,
#     message_history: Sequence[HumanMessage | AIMessage],
# ) -> str:
#     """Generate a simple Python script."""
#     request = """
#     ## REQUEST
#     Use the following reasoning process to respond to the user:
#     {
#         "system_instruction": "Create a simple Python function based on user requirements, ensuring clarity and effectiveness.",
#         "task": "Execute reasoning process to write a simple Python function for a specific user-defined programming task.",
#         "objective": [
#             "Functions must only use base Python packages.",
#             "The function should be straightforward to execute without needing testing.",
#             "Refuse any tasks that are complex or require external packages."
#         ],
#         "reasoning_process": [
#             "Assess the complexity and requirements of the user-defined task.",
#             {
#                 "action_determination": [
#                 {
#                     "if": "The task is simple and requires only base Python packages",
#                     "then": "Draft a function that meets the task requirements."
#                 },
#                 {
#                     "if": "The task description is ambiguous",
#                     "then": "Ask for clarifications."
#                 },
#                 {
#                     "if": "The task requires external packages or is too complex",
#                     "then": "Refuse the task."
#                 },
#                 ]
#             }
#         ],
#         "parameters": {
#             "user_defined_task": "The task as described by the user.",
#             "discussion": "Discussion with the user to clarify the task requirements."
#         },
#         "output": {
#             "format": {
#                 "reasoning_output": {
#                     "block_delimiters": "The reasoning output is wrapped in ```start_of_reasoning_output and ```end_of_reasoning_output blocks.",
#                     "output_scope": "All parts of the reasoning process must be included in the output."
#                 }
#                 "main_output": {
#                     "block_delimiters": "The function or message itself is wrapped in ```{start_of_main_output} and ```{end_of_main_output} blocks. Must only contain the function or message.",
#                     "usage_examples": "Any usage examples must be commented out to avoid accidental execution."
#                     "additional_comments": "Any additional comments must be outside of the main output block."
#                 },
#             }
#             "main_output_options": {
#                 "function": "A Python function that meets the task requirements or a message explaining why the task cannot be fulfilled.",
#                 "clarification_needed": "A message asking for more details if the user's requirements are unclear.",
#                 "refusal": "A message explaining why the task cannot be fulfilled."
#             }
#         },
#         "feedback": {
#             "clarification_needed": "If the user's requirements are unclear, ask for more details.",
#             "revision_request": "Allow for revisions based on user feedback."
#         }
#     }
#     """
#     start_delimiter = "start_of_main_output"
#     end_delimiter = "end_of_main_output"
#     request = (
#         dedent_and_strip(request)
#         .replace("{start_of_main_output}", start_delimiter)
#         .replace("{end_of_main_output}", end_delimiter)
#     )
#     messages = [
#         *generate_messages(
#             task_description=task_description, message_history=message_history
#         ),
#         SystemMessage(content=request),
#     ]
#     result = query_model(
#         model=precise_model,
#         messages=messages,
#         preamble=f"Running Script Writer...\n{format_messages(messages)}",
#         printout=True,
#         color=AGENT_COLOR,
#     )
#     return extract_and_unpack(
#         text=result, start_block_type=start_delimiter, end_block_type=end_delimiter
#     )


# def determine_task_completion(
#     task_description: TaskDescription,
#     message_history: Sequence[HumanMessage | AIMessage],
#     task_result: str,
# ) -> bool:
#     """Determine if the task was completed."""
#     request = """
#     ## REQUEST
#     For saving a record of this request, please output the following:
#     - "y" if you have generated a Python function based on the user's requirements.
#     - "n" otherwise

#     Output the answer in the following block:
#     ```start_of_y_or_n_output
#     {y_or_n}
#     ```end_of_y_or_n_output
#     """
#     request = dedent_and_strip(request)
#     messages = [
#         *generate_messages(
#             task_description=task_description, message_history=message_history
#         ),
#         AIMessage(content=task_result),
#         SystemMessage(content=request),
#     ]
#     result = query_model(
#         model=precise_model,
#         messages=messages,
#         preamble=f"Checking task completion...\n{format_messages(messages)}",
#         printout=True,
#         color=AGENT_COLOR,
#     )
#     task_completed = extract_and_unpack(
#         text=result,
#         start_block_type="start_of_y_or_n_output",
#         end_block_type="end_of_y_or_n_output",
#     )
#     assert task_completed in {
#         "y",
#         "n",
#     }, f"Invalid task completion: {task_completed}. Must be 'y' or 'n'."
#     return task_completed == "y"


# def save_artifact(result: str, output_dir: Path) -> Path:
#     """Save the artifact."""
#     output_location = output_dir / "script.py"
#     output_location.write_text(result, encoding="utf-8")
#     return output_location


# def run_function_writer(
#     task_description: TaskDescription,
#     message_history: Sequence[HumanMessage | AIMessage],
#     output_dir: Path,
# ) -> BotReply:
#     """Run the function writer."""
#     result = generate_script(task_description, message_history)
#     task_completed = determine_task_completion(
#         task_description, message_history, result
#     )
#     if task_completed:
#         output_location = save_artifact(result, output_dir)
#         reply = "Function has been successfully written."
#         artifacts = [
#             Artifact(
#                 location=str(output_location),
#                 description=f"Python function written for the following task: {task_description}",
#             )
#         ]
#     else:
#         reply = result
#         artifacts = []
#     report = ExecutorReport(
#         reply=reply,
#         task_completed=task_completed,
#     )
#     return BotReply(
#         report=report,
#         artifacts=artifacts,
#     )


# def load_bot(*_) -> BotCore:
#     """Load the bot core."""
#     return run_function_writer, None
