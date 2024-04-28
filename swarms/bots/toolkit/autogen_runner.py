"""AutoGen Runner."""

from dataclasses import dataclass
from typing import Callable, Annotated, Sequence

from autogen import ConversableAgent  # type: ignore
from langchain.schema import AIMessage, HumanMessage

from core.task_data import TaskDescription

OaiMessage = Annotated[dict[str, str], "OpenAI message"]
TerminationCondition = Callable[[OaiMessage], bool]
Conversation = Sequence[HumanMessage | AIMessage]


@dataclass
class AutoGenRunner:
    """Proxy for AutoGen system."""

    run: Callable[[ConversableAgent, ConversableAgent, str], str]
    assistant: ConversableAgent | None = None
    user_proxy: ConversableAgent | None = None

    def __call__(
        self,
        user_message: str,
    ) -> str:
        assert self.assistant, "Cannot run AutoGen Runner: assistant not set."
        assert self.user_proxy, "Cannot run AutoGen Runner: user proxy not set."
        return self.run(self.assistant, self.user_proxy, user_message)

    def set_agents(
        self,
        assistant: ConversableAgent,
        user_proxy: ConversableAgent,
    ) -> None:
        """Set agents for the system."""
        if self.assistant:
            assert self.user_proxy, "If assistant is set, user proxy must be set."
            return
        assert (
            not self.user_proxy
        ), "If assistant is not set, user proxy must not be set."
        self.assistant = assistant
        self.user_proxy = user_proxy


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
