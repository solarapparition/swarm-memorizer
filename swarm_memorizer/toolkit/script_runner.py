"""Run external Python scripts."""

from pathlib import Path
from typing import Callable, Literal
import pexpect


# kind of a hacky way to run an interactive script—use with caution; probably doesn't work with Windows
def create_script_runner(
    script: Path,
    input_pattern: str,
    output_pattern: str | None = None,
    interpreter: Path = Path("python"),
    cwd: Path = Path("."),
) -> Callable[[str], str]:
    """
    Run an interactive Python script. Returns a function to send further messages.

    Args:
    - script: The path to the script to run.
    - input_pattern: The pattern the script will output to indicate it's ready for input.
    - output_pattern: The pattern the script uses to indicate the start of its output. By default, uses the input message itself (this assumes that the input message is printed out).
    - interpreter: The Python interpreter to use. Defaults to "python".
    """
    child = pexpect.spawn(f"{interpreter} {script}", cwd=str(cwd), encoding="utf-8")

    def send_message(message: str) -> str:
        """Send a message to interactive script and return the reply."""
        child.expect(input_pattern)
        child.sendline(message)
        child.expect(output_pattern or message, timeout=None)
        child.expect(input_pattern)
        return str(child.before).strip()

    return send_message
