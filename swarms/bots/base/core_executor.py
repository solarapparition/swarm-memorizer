"""Interface with core AutoGen executor."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Annotated

from autogen import AssistantAgent, UserProxyAgent  # type: ignore
from colorama import Fore
from langchain.schema import AIMessage, HumanMessage

from core.bot import BotCore
from core.config import AUTOGEN_CONFIG_LIST
from core.task import ExecutionReport
from core.task_data import TaskDescription

AGENT_COLOR = Fore.GREEN
OaiMessage = Annotated[dict[str, str], "OpenAI message"]
Conversation = Sequence[HumanMessage | AIMessage]


@dataclass
class AutoGenRunner:
    """Proxy for AutoGen system."""

    assistant: AssistantAgent | None = None
    user_proxy: UserProxyAgent | None = None

    def __call__(
        self,
        user_message: str,
    ) -> str:
        assert self.assistant, "Cannot run AutoGen Runner: assistant not set."
        assert self.user_proxy, "Cannot run AutoGen Runner: user proxy not set."

        self.user_proxy.initiate_chat(  # type: ignore
            self.assistant,
            clear_history=False,
            message=user_message,
        )
        if not self.user_proxy.last_message()["content"]:
            # this means that the assistant has terminated the conversation due to user sending an empty message
            self.user_proxy.chat_messages[self.assistant].pop()
        return (
            str(self.assistant.last_message()["content"])
            .removesuffix("TERMINATE")
            .strip()
        )

    def set_agents(self, output_dir: Path) -> None:
        """Set agents for the system."""
        if self.assistant:
            assert self.user_proxy, "If assistant is set, user proxy must be set."
            return
        assert not self.user_proxy

        def assistant_termination(message: OaiMessage) -> bool:
            return not message["content"]

        self.assistant = AssistantAgent(
            "assistant",
            llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            is_termination_msg=assistant_termination,
        )

        def user_termination(message: OaiMessage) -> bool:
            return message["content"][-9:].strip().upper() == "TERMINATE"

        self.user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": str(output_dir)},
            # llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            human_input_mode="NEVER",
            is_termination_msg=user_termination,
        )


def create_user_message(
    task_description: TaskDescription, message_history: Conversation, initial_run: bool
) -> str:
    """Create a user message for the bot."""
    if not initial_run:
        assert isinstance(last_message := message_history[-1], HumanMessage)
        return str(last_message.content)  # type: ignore

    # now we know we're running for the first time
    if not message_history:
        return str(task_description)

    # now we know there's an additional message to add to the initial message
    assert len(message_history) == 1, "Expected only one initial message from the user"
    return f"{task_description}\n\n{message_history[0].content}"  # type: ignore


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    runner = AutoGenRunner()

    def run(
        task_description: TaskDescription,
        message_history: Conversation,
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        initial_run = not bool(runner.assistant)
        runner.set_agents(output_dir)
        user_message = create_user_message(task_description, message_history, initial_run)
        return ExecutionReport(runner(user_message))

    return BotCore(run, None)
