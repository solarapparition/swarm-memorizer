"""Loader for human fallback executor."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from colorama import Fore
from langchain.schema import SystemMessage, AIMessage
from autogen import AssistantAgent, UserProxyAgent  # type: ignore

from swarm_memorizer.event import EventLog
from swarm_memorizer.schema import RuntimeId
from swarm_memorizer.toolkit.models import query_model, PRECISE_MODEL, format_messages
from swarm_memorizer.config import AUTOGEN_CONFIG_LIST
from swarm_memorizer.task import ExecutionReport, Executor, Task
from swarm_memorizer.toolkit.advisor import get_choice
from swarm_memorizer.toolkit.text import dedent_and_strip
from swarm_memorizer.blueprint import BotBlueprint as Blueprint

AGENT_COLOR = Fore.GREEN


OaiMessage = Annotated[dict[str, str], "OpenAI message"]


@dataclass(frozen=True)
class AcceptAdvisor:
    """Advisor for accepting a task."""

    messages: list[SystemMessage | AIMessage] = field(default_factory=list)

    def advise(self, prompt: str) -> str:
        """Get advice about whether to accept the task or not."""
        self.messages.append(SystemMessage(content=prompt))
        result = query_model(
            model=PRECISE_MODEL,
            messages=self.messages,
            preamble=format_messages(self.messages),
            color=AGENT_COLOR,
            printout=True,
        )
        self.messages.append(AIMessage(content=dedent_and_strip(result)))
        return result


@dataclass(frozen=True)
class Multiplier:
    """Write text to a file."""

    blueprint: Blueprint
    task: Task
    files_dir: Path

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the orchestrator."""
        return RuntimeId(f"{self.blueprint.id}_{self.task.id}")

    @property
    def rank(self) -> int:
        """Return rank."""
        return 0

    def accepts(self, task: Task) -> bool:
        """Check if task is accepted by executor."""
        prompt = """
        Determine whether the following request is a request to multiply two numbers:
        ```
        {task_information}
        ```

        Reply with either "y" or "n", and no other text.
        """
        prompt = dedent_and_strip(prompt).format(task_information=task.information)
        return (
            get_choice(
                prompt,
                allowed_choices={"y", "n"},
                advisor=AcceptAdvisor(),
            )
            == "y"
        )

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        return self.files_dir / "output"

    @property
    def role_prompt(self) -> str:
        """Return role prompt."""
        role = """
        # MISSION
        You are a bot responsible for using a multiplication function to add two numbers together. You will be given the conversation with the user so far below to determine what to multiply together.
        """
        return dedent_and_strip(role)

    @property
    def task_prompt(self) -> str:
        """Return task prompt."""
        task = """
        # TASK
        Use the following reasoning process to determine what action to take:
        
        ## CASE 1: the user has provided all required arguments of your primary function
        - ACTION: pass the arguments to the primary function

        ## CASE 2: the user has not provided all required arguments of your function, or the arguments are invalid for what the function expects
        - ACTION: call the error function to create an error message to the user
        """
        return dedent_and_strip(task)

    @property
    def discussion_messages(self) -> EventLog:
        """Return messages from the discussion."""
        return self.task.event_log.messages

    def save_blueprint(self) -> None:
        """Not implemented."""

    async def execute(self) -> ExecutionReport:
        """Execute the subtask. Add a message to the task's event log if provided, and adds own message to the event log at the end of execution."""

        def assistant_termination(message: OaiMessage) -> bool:
            """Condition for assistant to terminate the conversation."""
            try:
                json.loads(message["content"].strip())
            except json.JSONDecodeError:
                return False
            return message["role"] == "tool"

        assistant = AssistantAgent(
            "assistant",
            llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            system_message=self.role_prompt,
            is_termination_msg=assistant_termination,
        )
        user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": "coding"},
            llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            human_input_mode="NEVER",
        )

        @user_proxy.register_for_execution()  # type: ignore
        @assistant.register_for_llm(description="Create an error message.")  # type: ignore
        def error_message(  # type: ignore
            message: Annotated[str, "The error message to return back to the user."]
        ) -> Annotated[dict[str, Any], "The object containing the error message."]:
            return {
                "artifact": None,
                "successful": False,
                "message": message,
            }

        @user_proxy.register_for_execution()  # type: ignore
        @assistant.register_for_llm(description="Write text to a file.")  # type: ignore
        def multiply(  # type: ignore
            num_1: Annotated[float | int, "The first number to multiply."],
            num_2: Annotated[float | int, "The second number to multiply."],
        ) -> Annotated[dict[str, Any], "The result of multiplying the numbers."]:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                num_mult = num_1 * num_2
                file_name = f"mult_{num_1}_{num_2}.txt"
                (artifact_path := (self.output_dir / file_name)).write_text(
                    str(num_mult), encoding="utf-8"
                )
                # NOTE: should probably be a decorator around function  # to_artifact_func
                return {
                    "artifact_path": str(artifact_path),
                    "successful": True,
                    "message": "Execution successful. See artifact for result.",
                }
            except Exception as error:  # pylint:disable=broad-except
                return error_message(f"Execution failed with error:\n{error}")

        user_proxy.send(  # type: ignore
            self.task.information, assistant, request_reply=False, silent=True
        )
        for _ in self.discussion_messages.events[1:]:
            raise NotImplementedError
            # TODO: depending on originating party of message, send to assistant or user_proxy
        assistant.chat_messages[user_proxy].append(  # type: ignore
            {"role": "system", "content": self.task_prompt}
        )

        # autogen needs a message to be sent to run, so we remove the last message and send it
        assert (
            last_conversation_message := assistant.chat_messages[user_proxy].pop(-2)  # type: ignore
        )["role"] == "user"
        last_conversation_message_2 = user_proxy.chat_messages[assistant].pop(-1)  # type: ignore
        assert (
            last_conversation_message_2["content"]
            == last_conversation_message["content"]
        )
        user_proxy.initiate_chat(  # type: ignore
            assistant,
            clear_history=False,
            message=last_conversation_message["content"],  # type: ignore
            silent=True,
        )
        reply = str(user_proxy.last_message()["content"])
        successful = json.loads(reply)["successful"]
        self.task.event_log.add(self.task.execution_reply_message(reply=reply))
        return ExecutionReport(reply=reply, task_completed=successful)


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> Executor:
    """Load bot."""
    return Multiplier(blueprint, task, files_dir)
