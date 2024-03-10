"""File-related utilities."""

from pathlib import Path


def make_if_not_exist(path: Path) -> Path:
    """Make a directory if it does not exist, and returns the directory."""
    if not path.exists():
        path.mkdir(parents=True)
    return path


def sanitize_filename(
    description: str, max_length: int = 255, invalid_characters: set[str] | None = None
) -> str:
    """Sanitize a description to be used as a filename."""
    if invalid_characters is None:
        invalid_characters = set('<>:"/\\|?!*\0' + "".join(chr(i) for i in range(32)))
    pre_sanitized = description.replace(" ", "_")
    sanitized = "".join(
        c if c not in invalid_characters else "_" for c in pre_sanitized
    )
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized
