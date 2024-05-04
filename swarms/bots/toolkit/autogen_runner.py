"""AutoGen Runner."""

from dataclasses import dataclass
from typing import Callable

from autogen import ConversableAgent  # type: ignore


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
        return self.run(self.user_proxy, self.assistant, user_message)

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
