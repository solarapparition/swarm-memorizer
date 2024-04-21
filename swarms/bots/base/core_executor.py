"""Interface with Perplexity's online model."""

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


def run(
    task_description: TaskDescription,
    message_history: Sequence[HumanMessage | AIMessage],
    output_dir: Path,
) -> ExecutionReport:
    """Run the bot."""

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
        is_termination_msg=assistant_termination,
    )
    user_proxy = UserProxyAgent(
        "user_proxy",
        code_execution_config={"work_dir": "coding"},
        llm_config={"config_list": AUTOGEN_CONFIG_LIST},
        human_input_mode="NEVER",
    )

    # synchronize message history with autogen message history # shouldn't include all details
    breakpoint()
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
        last_conversation_message_2["content"] == last_conversation_message["content"]
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


def load_bot(*_) -> BotCore:
    """Load the bot core."""

    return BotCore(run, None)
