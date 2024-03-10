"""Handling of human parts of the swarm."""

from dataclasses import dataclass, field
from typing import MutableMapping

from swarm_memorizer.schema import RuntimeId, WorkValidationResult


@dataclass
class Human:
    """A human agent. Can be slotted into various roles for tasks that the agent can't yet handle autonomously."""

    name: str = "Human"
    thread: list[str] = field(default_factory=list)
    _reply_cache: MutableMapping[str, str] | None = None

    @property
    def id(self) -> RuntimeId:
        """Runtime id of the human."""
        return RuntimeId(self.name)

    def respond_manually(self) -> str:
        """Get manual response from the human."""
        return input("Enter your response: ").strip()

    def respond_using_cache(self, reply_cache: MutableMapping[str, str]) -> str:
        """Get cached reply based on thread."""
        if reply := reply_cache.get(str(self.thread)):
            print(f"Cached reply found: {reply}")
            return reply
        if reply := self.respond_manually():
            reply_cache.update({str(self.thread): reply})
        return reply

    def advise(self, prompt: str) -> str:
        """Get input from the human."""
        print(prompt)
        self.thread.append(prompt)
        self.thread.append(
            reply := (
                self.respond_using_cache(self._reply_cache)
                if self._reply_cache is not None
                else self.respond_manually()
            )
        )
        return reply

    def validate(self, context: str) -> WorkValidationResult:
        """Validate some work done."""
        prompt = f"{context}\n\nPlease validate the work as described above (y/n): "
        while True:
            validation_input: str = self.advise(prompt).strip().lower()
            if validation_input in {"y", "n"}:
                valid: bool = validation_input == "y"
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        feedback: str = "" if valid else self.advise("Provide feedback: ")
        return WorkValidationResult(valid, feedback)
