"""Main entry point to running swarms."""

from argparse import ArgumentParser, Namespace
import asyncio
from pathlib import Path

from colorama import Fore

from core.config import SWARM_COLOR
from core.swarm import Swarm


def get_args() -> Namespace:
    """Get the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "task",
        type=str,
        help="A task to be executed.",
    )
    parser.add_argument(
        "--data-dir",
        default="swarms/data",
        type=str,
        help="Path to the directory containing data files needed for the swarm.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the main function."""
    args = get_args()
    swarm = Swarm(
        task_description=args.task,
        llm_cache_enabled=True,
        files_dir=Path(args.data_dir),
    )
    report = asyncio.run(swarm.execute())
    print(f"{SWARM_COLOR}{report}{Fore.RESET}")


if __name__ == "__main__":
    main()
