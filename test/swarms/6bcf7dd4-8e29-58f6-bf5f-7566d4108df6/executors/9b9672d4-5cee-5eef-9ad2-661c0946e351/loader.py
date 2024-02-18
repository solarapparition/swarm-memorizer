"""Loader for human fallback executor."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from colorama import Fore
from langchain.schema import SystemMessage, AIMessage
from autogen import AssistantAgent, UserProxyAgent  # type: ignore

from swarm_memorizer.swarm import (
    Blueprint,
    TaskWorkStatus,
    EventLog,
    Task,
    Executor,
    RuntimeId,
    get_choice,
    dedent_and_strip,
    ExecutorReport,
    format_messages,
)
from swarm_memorizer.toolkit.models import query_model, precise_model
from swarm_memorizer.config import autogen_config_list

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
            model=precise_model,
            messages=self.messages,
            preamble=format_messages(self.messages),
            color=AGENT_COLOR,
            printout=True,
        )
        self.messages.append(AIMessage(content=dedent_and_strip(result)))
        return result


@dataclass(frozen=True)
class TextWriter:
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
        Determine whether the following request is a request to write text to a file:
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
        You are a bot responsible for helping a user write text to a file. You will be given the conversation with the user so far below to determine what file to write, and what text to write to it.
        """
        return dedent_and_strip(role)

    @property
    def task_prompt(self) -> str:
        """Return task prompt."""
        task = """
        # TASK
        Use the following reasoning process to determine what action to take:
        
        ## CASE 1: the user has provided a file name and text to write to the file
        - ACTION: call a function to write the text to the file with that name

        ## CASE 2: the user has provided the text to write to the file, but not the file name
        - ACTION: call a function to write the text to reasonable file name given the text

        ## CASE 3: the user has not provided text to write to the file
        - ACTION: call a function to create an error message to the user
        """
        return dedent_and_strip(task)

    @property
    def discussion_messages(self) -> EventLog:
        """Return messages from the discussion."""
        return self.task.event_log.messages

    def save_blueprint(self) -> None:
        """Not implemented."""

    async def execute(self) -> ExecutorReport:
        """Execute the subtask. Adds a message to the task's event log if provided, and adds own message to the event log at the end of execution."""

        def assistant_termination(message: OaiMessage) -> bool:
            """Condition for assistant to terminate the conversation."""
            try:
                json.loads(message["content"].strip())
            except json.JSONDecodeError:
                return False
            return message["role"] == "function"

        assistant = AssistantAgent(
            "assistant",
            llm_config={"config_list": autogen_config_list},
            system_message=self.role_prompt,
            is_termination_msg=assistant_termination,
        )
        user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": "coding"},
            llm_config={"config_list": autogen_config_list},
            human_input_mode="NEVER",
        )

        @user_proxy.register_for_execution()  # type: ignore
        @assistant.register_for_llm(description="Write text to a file.")  # type: ignore
        def write_text(  # type: ignore
            text: Annotated[str, "The text to write to the file."],
            file_name: Annotated[str, "The name of the file to write to."],
        ) -> Annotated[dict[str, Any], "The result of writing the text to the file."]:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (full_file_path := (self.output_dir / file_name)).write_text(
                    text, encoding="utf-8"
                )
                return {
                    "full_file_path": str(full_file_path),
                    "successful": True,
                    "error": None,
                }
            except Exception as error:  # pylint:disable=broad-except
                return {
                    "full_file_path": None,
                    "successful": False,
                    "error": str(error),
                }

        @user_proxy.register_for_execution()  # type: ignore
        @assistant.register_for_llm(description="Create an error message.")  # type: ignore
        def error_message(  # type: ignore
            message: Annotated[str, "The error message to return back to the user."]
        ) -> Annotated[dict[str, Any], "The object containing the error message."]:
            return {
                "successful": False,
                "error": message,
                "need_additional_information": True,
            }

        user_proxy.send(  # type: ignore
            self.task.information, assistant, request_reply=False, silent=True
        )
        for _ in self.discussion_messages:
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
        return ExecutorReport(reply=reply, task_completed=successful)


def load_bot(blueprint: Blueprint, task: Task, files_dir: Path) -> Executor:
    """Load bot."""
    return TextWriter(blueprint, task, files_dir)
