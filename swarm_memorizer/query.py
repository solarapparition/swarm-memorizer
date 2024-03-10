"""Query and extraction utilities for the swarm."""

from typing import Sequence

from langchain.schema import SystemMessage
from swarm_memorizer.config import SWARM_COLOR

from swarm_memorizer.toolkit.models import query_model, SUPER_CREATIVE_MODEL
from swarm_memorizer.toolkit.text import ExtractionError, extract_blocks


def query_and_extract_reasoning(
    messages: Sequence[SystemMessage], preamble: str, printout: bool
) -> str:
    """Query the model and extract the reasoning process."""
    if printout:
        result = query_model(
            model=SUPER_CREATIVE_MODEL,
            messages=messages,
            preamble=preamble,
            color=SWARM_COLOR,
            printout=printout,
        )
    else:
        result = query_model(
            model=SUPER_CREATIVE_MODEL,
            messages=messages,
            printout=printout,
        )
    if not (extracted_result := extract_blocks(result, "start_of_reasoning_process")):
        raise ExtractionError("Could not extract reasoning process from the result.")
    return extracted_result[0]

