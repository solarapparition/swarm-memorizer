"""For getting a choice from some agent."""

from typing import Any, Protocol


class Advisor(Protocol):
    """A single-reply advisor for some issue."""

    def advise(self, prompt: str) -> str:
        """Advise on some issue."""
        raise NotImplementedError


def get_choice(prompt: str, allowed_choices: set[Any], advisor: Advisor) -> Any:
    """Get a choice from the advisor."""
    while True:
        if (choice := advisor.advise(prompt)) in allowed_choices:
            return choice
        prompt = f"Invalid input. Valid choices: {allowed_choices}."
