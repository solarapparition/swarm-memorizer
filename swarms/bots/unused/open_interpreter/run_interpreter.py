"""Bot for writing simple Python functions based on user requirements."""

from colorama import Fore
from interpreter import OpenInterpreter

AGENT_COLOR = Fore.GREEN
DUMMY_REPLY = "Here are the first 10 prime numbers: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29"


def setup_interpreter() -> OpenInterpreter:
    """Set up the Open Interpreter."""
    interpreter = OpenInterpreter()
    interpreter.llm.model = "gpt-4-turbo-preview"
    interpreter.safe_mode = "auto"
    interpreter.auto_run = True
    interpreter.llm.context_window = 128_000
    interpreter.llm.max_tokens = 4096
    return interpreter


def main() -> None:
    """Run the bot."""
    interpreter = setup_interpreter()
    while True:
        message = input("Send message: ")
        interpreter.chat(message=message, display=False, stream=False)
        print("Open Interpreter Reply:")
        print(interpreter.messages[-1]["content"])
        # print(DUMMY_REPLY)


if __name__ == "__main__":
    main()
