"""Functions for message management for bots."""

from typing import Annotated, Sequence

from langchain.schema import AIMessage, HumanMessage

from core.task_data import TaskDescription

OaiMessage = Annotated[dict[str, str], "OpenAI message"]
Conversation = Sequence[HumanMessage | AIMessage]

def create_next_user_message(
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
