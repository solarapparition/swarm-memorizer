"""Main entry point to running swarms."""

from argparse import ArgumentParser, Namespace
import asyncio

from colorama import Fore

from swarm_memorizer.config import SWARM_COLOR
from swarm_memorizer.swarm import Swarm


def get_args() -> Namespace:
    """Get the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "task",
        help="A task to be executed.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the main function."""
    swarm = Swarm(task_description=get_args().task, llm_cache_enabled=True)
    report = asyncio.run(swarm.execute())
    print(f"{SWARM_COLOR}{report}{Fore.RESET}")


if __name__ == "__main__":
    main()
