"""YAML tools for Hivemind."""

import os
from pathlib import Path
from typing import Mapping, Any, Sequence
from ruamel.yaml import YAML
from ruamel.yaml.compat import StringIO

yaml_safe = YAML(typ="safe")
yaml_safe.default_flow_style = False
yaml_safe.default_style = "|"  # type: ignore
yaml_safe.allow_unicode = True

DEFAULT_YAML = YAML()
DEFAULT_YAML.default_flow_style = False
DEFAULT_YAML.default_style = "|"  # type: ignore
DEFAULT_YAML.allow_unicode = True


def save_yaml(data: Mapping[str, Any], location: Path) -> None:
    """Save YAML to a file, making sure the directory exists."""
    if not location.exists():
        os.makedirs(location.parent, exist_ok=True)
    DEFAULT_YAML.dump(data, location)


def format_as_yaml_str(
    data: Mapping[str, Any] | Sequence[Any], yaml: YAML = DEFAULT_YAML
) -> str:
    """Dump yaml as a string."""
    yaml.dump(data, stream := StringIO())
    return stream.getvalue().strip()
