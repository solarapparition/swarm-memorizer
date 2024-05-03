"""Load and return the content of a markdown file."""

from pathlib import Path
from typing import Any

class MarkdownLoadError(Exception):
    """Class for errors raised when loading markdown."""
    def __init__(self, message: str, *args: Any):
        self.message = message
        super().__init__(self.message, *args)

def load_markdown(file_location: Path) -> str:
    """
    Load and return the content of a markdown file.
    Validates file type, existence, and content before reading.
    """
    if not file_location.name.endswith('.md'):
        raise MarkdownLoadError(f"File at {file_location} must be a markdown file with '.md' extension.")
    if not file_location.exists():
        raise MarkdownLoadError(f"File at {file_location} does not exist.")
    try:
        with open(file_location, 'r', encoding="utf-8") as file:
            content = file.read()
    except UnicodeDecodeError as e:
        raise MarkdownLoadError(f"File at {file_location} is not a text file.") from e
    if not content:
        raise MarkdownLoadError(f"File at {file_location} is empty.")
    return content
