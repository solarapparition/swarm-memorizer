"""Extract blocks from text."""

from typing import Sequence
from dataclasses import dataclass
import re
from textwrap import dedent, indent

from langchain.schema import BaseMessage


def dedent_and_strip(text: str) -> str:
    """Dedent and strip text."""
    return dedent(text).strip()


@dataclass(frozen=True)
class ExtractionError(Exception):
    """Raised when an extraction fails."""

    problem: str
    text: str | None = None
    start_block_type: str | None = None
    end_block_type: str | None = None

    @property
    def message(self) -> str:
        """Get the error message."""
        template = """
        Failed to extract a block:
        - Start block type: {start_block_type}
        - End block type: {end_block_type}
        - Problem: {problem}
        - Text:
        {text}
        """
        text = indent(str(self.text or "N/A"), "  ")
        output = template.format(
            start_block_type=self.start_block_type or "N/A",
            end_block_type=self.end_block_type or "N/A",
            problem=self.problem or "N/A",
            text=text,
        )
        return dedent_and_strip(output)


def extract_block(text: str, block_type: str) -> str | None:
    """Extract a code block from the text."""
    pattern = (
        r"```{block_type}\n(.*?)```".format(  # pylint:disable=consider-using-f-string
            block_type=block_type
        )
    )
    match = re.search(pattern, text, re.DOTALL)
    return match[1].strip() if match else None


def extract_blocks(
    text: str, start_block_type: str, end_block_type: str = ""
) -> list[str] | None:
    """Extracts specially formatted blocks of text from the LLM's output. `block_type` corresponds to a label for a markdown code block such as `yaml` or `python`."""
    pattern = r"```{start_block_type}\n(.*?)```{end_block_type}".format(  # pylint:disable=consider-using-f-string
        start_block_type=start_block_type,
        end_block_type=end_block_type,
    )
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches] if matches else None


def unpack_block(
    text: str,
    extracted_result: list[str] | None,
    start_block_type: str,
    end_block_type: str,
) -> str:
    """Validate and unpack the extracted block."""
    assert extracted_result and len(extracted_result) == 1, ExtractionError(
        text=text,
        start_block_type=start_block_type,
        end_block_type=end_block_type,
        problem="Expected exactly one block.",
    )
    (block,) = extracted_result
    return block


def extract_and_unpack(
    text: str, start_block_type: str, end_block_type: str = ""
) -> str:
    """Extract and unpack a block."""
    extracted_result = extract_blocks(text, start_block_type, end_block_type)
    return unpack_block(text, extracted_result, start_block_type, end_block_type)
