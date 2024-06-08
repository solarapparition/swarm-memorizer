"""Interface with core AutoGen executor."""

from pathlib import Path

from autogen import AssistantAgent, ConversableAgent, UserProxyAgent  # type: ignore

from swarm_memorizer.bot import BotCore
from swarm_memorizer.config import AUTOGEN_CONFIG_LIST
from swarm_memorizer.task import ExecutionReport
from swarm_memorizer.task_data import TaskDescription
from swarms.bots.toolkit.autogen_runner import AutoGenRunner
from swarms.bots.toolkit.messages import (
    Conversation,
    OaiMessage,
    create_next_user_message,
)


def assistant_termination(message: OaiMessage) -> bool:
    """Check if the assistant needs to terminate the conversation."""
    return not message["content"]


def user_termination(message: OaiMessage) -> bool:
    """Check if the user needs to terminate the conversation."""
    return message["content"].strip()[-9:].upper() == "TERMINATE"


def run_autogen_pair(
    user_proxy: ConversableAgent, assistant: ConversableAgent, user_message: str
) -> str:
    """Run the bot."""
    user_proxy.initiate_chat(  # type: ignore
        assistant,
        clear_history=False,
        message=user_message,
    )
    if not user_proxy.last_message()["content"]:
        # this means that the assistant has terminated the conversation due to user sending an empty message
        user_proxy.chat_messages[assistant].pop()
    return str(assistant.last_message()["content"]).removesuffix("TERMINATE").strip()


def load_bot(*_) -> BotCore:
    """Load the bot core."""
    runner = AutoGenRunner(run_autogen_pair)

    def run(
        task_description: TaskDescription,
        message_history: Conversation,
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        assistant = AssistantAgent(
            "assistant",
            llm_config={"config_list": AUTOGEN_CONFIG_LIST},
            is_termination_msg=assistant_termination,
        )
        assistant.update_system_message(
            f"{assistant.system_message}\n\nAfter a task is complete, always tell the user what files were created or modified, and always use the full path to the file. If you are not sure about the full path of a file, simply prefix {output_dir} to the filename when reporting to the user."
        )
        user_proxy = UserProxyAgent(
            "user_proxy",
            code_execution_config={"work_dir": str(output_dir)},
            human_input_mode="NEVER",
            is_termination_msg=user_termination,
        )
        initial_run = not bool(runner.assistant)
        runner.set_agents(assistant, user_proxy)
        user_message = create_next_user_message(
            task_description, message_history, initial_run
        )
        return ExecutionReport(runner(user_message))

    return BotCore(run, None)
