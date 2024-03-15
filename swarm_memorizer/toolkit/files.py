"""File-related utilities."""

from pathlib import Path
from slugify import slugify


def make_if_not_exist(path: Path) -> Path:
    """Make a directory if it does not exist, and returns the directory."""
    if not path.exists():
        path.mkdir(parents=True)
    return path


def sanitize_filename(description: str, max_length: int = 255) -> str:
    """Sanitize a description to be used as a filename."""
    return slugify(description, max_length=max_length)
