"""Configuration process for Hivemind."""

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

AUTOGEN_CONFIG_LIST = [{'model': 'gpt-4-1106-preview'}]

def configure_langchain_cache() -> None:
    """Configure the LLM cache."""
    makedirs(CACHE_DIR, exist_ok=True)
    if not langchain.globals.get_llm_cache():
        langchain.globals.set_llm_cache(
            SQLiteCache(database_path=str(CACHE_DIR / ".langchain.db"))
        )
