"""Interface with Perplexity's online model."""

from dataclasses import dataclass
import json
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
            # silent=True,
        )
        return (
            str(self.assistant.last_message()["content"])
            .removesuffix("TERMINATE")
            .strip()
        )

        # self.user_proxy.send(  # type: ignore
        #     user_message, self.assistant, request_reply=False, silent=True
        # )
        # for _ in self.discussion_messages:
        #     raise NotImplementedError
        #     # TODO: depending on originating party of message, send to assistant or user_proxy
        # assistant.chat_messages[user_proxy].append(  # type: ignore
        #     {"role": "system", "content": self.task_prompt}
        # )

        # # autogen needs a message to be sent to run, so we remove the last message and send it
        # assert (
        #     last_conversation_message := assistant.chat_messages[user_proxy].pop(-2)  # type: ignore
        # )["role"] == "user"
        # last_conversation_message_2 = user_proxy.chat_messages[assistant].pop(-1)  # type: ignore
        # assert (
        #     last_conversation_message_2["content"]
        #     == last_conversation_message["content"]
        # )
        # user_proxy.initiate_chat(  # type: ignore
        #     assistant,
        #     clear_history=False,
        #     message=last_conversation_message["content"],  # type: ignore
        #     silent=True,
        # )
        # reply = str(user_proxy.last_message()["content"])
        # successful = json.loads(reply)["successful"]
        # self.task.event_log.add(self.task.execution_reply_message(reply=reply))
        # return ExecutionReport(reply=reply, task_completed=successful)

    def set_agents(self, output_dir: Path) -> None:
        """Set agents for the system."""
        if self.assistant:
            assert self.user_proxy, "If assistant is set, user proxy must be set."
            return
        assert not self.user_proxy

        # def assistant_termination(message: OaiMessage) -> bool:
        #     """Condition for assistant to terminate the conversation."""
        #     try:
        #         json.loads(message["content"].strip())
        #     except json.JSONDecodeError:
        #         return False
        #     return message["role"] == "tool"

        def is_termination_msg(message: OaiMessage) -> bool:
            return message["content"][-9:].strip().upper() == "TERMINATE"

        self.assistant = AssistantAgent(
            "assistant",
            llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            # is_termination_msg=assistant_termination,
        )
        self.user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": str(output_dir)},
            # llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            human_input_mode="NEVER",
            is_termination_msg=is_termination_msg,
        )


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    runner = AutoGenRunner()

    def run(
        task_description: TaskDescription,
        message_history: Sequence[HumanMessage | AIMessage],
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""

        def create_user_message(initial_run: bool) -> str:
            if not initial_run:
                assert isinstance(last_message := message_history[-1], HumanMessage)
                breakpoint()
                return str(last_message.content)  # type: ignore

            # now we know we're running for the first time
            if not message_history:
                return task_description.information

            # now we know there's an additional message to add to the initial message
            assert (
                len(message_history) == 1
            ), "Expected only one initial message from the user"
            breakpoint()
            return f"{task_description.information}\n\n{message_history[0].content}"  # type: ignore

        initial_run = not bool(runner.assistant)
        runner.set_agents(output_dir)
        user_message = create_user_message(initial_run)
        return ExecutionReport(runner(user_message))

    return BotCore(run, None)
