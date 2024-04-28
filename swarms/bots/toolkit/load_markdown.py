"""Load and return the content of a markdown file."""

from pathlib import Path

def load_markdown(file_location: Path) -> str:
    """
    Load and return the content of a markdown file.
    Validates file type, existence, and content before reading.
    """
    if not file_location.name.endswith('.md'):
        raise ValueError("File must be a markdown file with '.md' extension.")
    if not file_location.exists():
        raise FileNotFoundError("File does not exist.")
    try:
        with open(file_location, 'r', encoding="utf-8") as file:
            content = file.read()
    except UnicodeDecodeError as e:
        raise TypeError("File is not a text file.") from e
    if not content:
        raise ValueError("File is empty.")
    # mime_type, _ = mimetypes.guess_type(file_location)
    # if mime_type is None or not mime_type.startswith('text'):
    #     raise TypeError("File is not a text file.")

    # content = file_location.read_text()
    # if len(content) < 1:
    #     raise ValueError("File is empty.")
    return content
