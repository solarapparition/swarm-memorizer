"""Hashing utilities."""

from typing import Any, Callable, Hashable
import hashlib
import json


def stable_hash(
    value: Hashable, json_default: Callable[[Any], Any] | None = None
) -> str:
    """
    Return a stable hash for a value.

    >>> stable_hash({'a': 1, 'b': [2, 3, 4], 'c': {'d': 5}})
    'a6c0d6d8e6c0d6d8e6c0d6d8e6c0d6d8'

    Adapted from https://death.andgravity.com/stable-hashing
    """
    json_str = json.dumps(
        value,
        default=json_default,
        ensure_ascii=False,
        sort_keys=True,
        indent=None,
        separators=(",", ":"),
    )
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()
