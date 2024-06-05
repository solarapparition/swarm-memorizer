"""Configuration process for Hivemind."""

import os
from pathlib import Path
from os import makedirs

from colorama import Fore
import langchain.globals
from langchain.cache import SQLiteCache

DATA_DIR = Path(".data")
CACHE_DIR = DATA_DIR / "cache"

SWARM_COLOR = Fore.MAGENTA
PROMPT_COLOR = Fore.BLUE
VERBOSE = True

GPT_4_TURBO = "gpt-4-turbo-2024-04-09"
GPT_4O = "gpt-4o-2024-05-13"
AUTOGEN_CONFIG_LIST = [{"model": GPT_4O, "api_key": os.getenv("OPENAI_API_KEY")}]


def configure_langchain_cache(
    database_path: Path = CACHE_DIR / ".langchain.db",
) -> None:
    """Configure the LLM cache."""
    # makedirs(CACHE_DIR, exist_ok=True)
    makedirs(database_path.parent, exist_ok=True)
    if not langchain.globals.get_llm_cache():
        langchain.globals.set_llm_cache(SQLiteCache(database_path=str(database_path)))
