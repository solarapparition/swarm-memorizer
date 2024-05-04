"""Tests for loading markdown."""

from pathlib import Path
import pytest

from swarms.bots.toolkit.load_markdown import MarkdownLoadError, load_markdown


def test_valid_file(tmp_path: Path) -> None:
    """Test loading a valid markdown file."""
    test_file = tmp_path / "valid.md"
    test_file.write_text("# Valid Markdown File")
    assert load_markdown(test_file) == "# Valid Markdown File"


def test_invalid_extension(tmp_path: Path) -> None:
    """Test loading a file with an invalid extension."""
    test_file = tmp_path / "invalid.txt"
    test_file.write_text("# Invalid Extension")
    with pytest.raises(MarkdownLoadError):
        load_markdown(test_file)


def test_non_existent_file(tmp_path: Path) -> None:
    """Test loading a non-existent file."""
    test_file = tmp_path / "nonexistent.md"
    with pytest.raises(MarkdownLoadError):
        load_markdown(test_file)


def test_non_text_file(tmp_path: Path) -> None:
    """Test loading a non-text file."""
    test_file = tmp_path / "image.md"
    test_file.write_bytes(b"\x00\xFF\xFF\x00")
    with pytest.raises(MarkdownLoadError):
        load_markdown(test_file)


def test_empty_file(tmp_path: Path) -> None:
    """Test loading an empty file."""
    test_file = tmp_path / "empty.md"
    test_file.write_text("")
    with pytest.raises(MarkdownLoadError):
        load_markdown(test_file)
