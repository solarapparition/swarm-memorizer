"""Craft a component from parts."""

from dataclasses import dataclass
from typing import Sequence

from swarm_memorizer.toolkit.text import dedent_and_strip


@dataclass
class Definition:
    """Concepts for an agent."""

    name: str
    description: str


def format_definitions(concepts: Sequence[Definition]) -> str:
    """Format a list of definitions."""
    if not concepts:
        return "N/A"
    raise NotImplementedError


def create_text_section(
    text: str, heading: str, level: int = 2, preamble: str | None = None
) -> str:
    """Create a section of text."""
    assert level > 0, "Heading level must be greater than 0."
    with_preamble_template = """
    {heading}
    {preamble}
    {text}
    """
    without_preamble_template = """
    {heading}
    {text}
    """
    heading = f"{'#' * level} {heading}"
    return (
        dedent_and_strip(with_preamble_template).format(
            heading=heading,
            preamble=preamble,
            text=text,
        )
        if preamble
        else dedent_and_strip(without_preamble_template).format(
            heading=heading,
            text=text,
        )
    )


def create_definition_section(
    definitions: Sequence[Definition], preamble: str | None = None
) -> str:
    """Format a section of definitions."""
    text = format_definitions(definitions)
    return create_text_section(text, heading="CONCEPTS", level=2, preamble=preamble)


def merge_text_sections(sections: Sequence[str]) -> str:
    """Merge multiple text sections."""
    return "\n\n".join(sections)


def create_background(
    agent_name: str,
    mission: str,
    concepts: Sequence[Definition],
    information_sections: Sequence[Definition],
) -> str:
    """Create an informational context for an agent."""
    mission_section = create_text_section(mission, heading="MISSION", level=1)
    concept_preamble = f"Here are the concepts that {agent_name} will be familiar with:"
    concept_section = create_definition_section(concepts, preamble=concept_preamble)
    information_preamble = f"{agent_name} has access to several sections of information that is relevant to its decisionmaking:"
    information_section = create_definition_section(
        information_sections, preamble=information_preamble
    )
    return merge_text_sections([mission_section, concept_section, information_section])


def specify_function(
    context: Sequence[str], requirements: Sequence[str]
) -> Sequence[str]:
    """Write specifications for the function."""
    agent_name = "TEST_WRITER"
    mission = f"You are the instructor for an advanced AI agent (referred to as {agent_name} from now on) for writing unit tests for a function based on context and requirement. Your purpose is to provide a reasoning structure for the agent to think through how to generate the tests."
    concepts: list[Definition] = []
    information_sections: list[Definition] = []
    background_sections = create_background(
        agent_name=agent_name,
        mission=mission,
        concepts=concepts,
        information_sections=information_sections,
    )
    

    # commit
    breakpoint()

    def create_bulleted_section(
        items: Sequence[str], preamble: str | None = None
    ) -> str:
        bullet_points = "\n".join(f"- {item}" for item in items)

    requirements_sections = create_bulleted_section(
        requirements, preamble="The function must adhere to the following requirements:"
    )

    breakpoint()

    request = """
    ## REQUEST FOR YOU:
    {OVERVIEW}
    - {INFORMATION_RESTRICTIONS}
    - {TERM_REFERENCES}
    - The {DELEGATOR} does _not_ have to select _any_ of the candidates, if it deems none of them to be suitable for the task.
    - The {DELEGATOR} must understand the difference between the actual requirements of the TASK and the {CONTEXT} that the TASK is being executed in. {EXECUTOR} candidates only need to be able to fulfill the actual requirements of the TASK itself—the {CONTEXT} is for background information only.
    - {STEPS_RESTRICTIONS}
    - {PROCEDURAL_SCRIPTING}
    - The final decision of which {EXECUTOR} CANDIDATE to use (or to not use any at all) must be done on the last step only, after considering all the information available from the previous steps.

    {output_instructions}
    """
    request = (
        dedent_and_strip(request)
        .replace("{output_instructions}", REASONING_PROCESS_OUTPUT_INSTRUCTIONS)
        .format(
            DELEGATOR=Definition.DELEGATOR.value,
            OVERVIEW=DelegatorReasoningGenerationNotes.OVERVIEW.value,
            INFORMATION_RESTRICTIONS=DelegatorReasoningGenerationNotes.INFORMATION_RESTRICTIONS.value,
            TERM_REFERENCES=DelegatorReasoningGenerationNotes.TERM_REFERENCES.value,
            EXECUTOR=Definition.EXECUTOR.value,
            CONTEXT=Definition.CONTEXT.value,
            STEPS_RESTRICTIONS=DelegatorReasoningGenerationNotes.STEPS_RESTRICTIONS.value,
            PROCEDURAL_SCRIPTING=DelegatorReasoningGenerationNotes.PROCEDURAL_SCRIPTING.value,
        )
    )
    messages = [
        SystemMessage(content=background_sections),
        SystemMessage(content=request),
    ]
    return query_and_extract_reasoning(
        messages,
        preamble="Generating reasoning for executor selection...\n"
        f"{format_messages(messages)}",
        printout=VERBOSE,
    )

    # generate reasoning
    # > include "semantic tests" for the code that checks whether it adheres to requirements > immutable > strongly typed
    # > there's a default test that it runs at all
    breakpoint()
    background_sections = """
    ## MISSION:
    You are a senior software engineer who is an expert in Python. 

    ## 
    """

    request = """
    ## REQUEST FOR YOU:
    Use the following reasoning process to write a {RECIPE} for completing tasks similar to the {MAIN_TASK}:
    ```start_of_reasoning_process
    {reasoning_process}
    Remember to refer to specific {EXECUTOR} IDs when discussing them, as they may show up again in future tasks.
    ```end_of_reasoning_process

    {reasoning_output_instructions}

    After this process, output the final {RECIPE} in the following YAML format:
    ```start_of_final_output_yaml
    - |-
        {{subtask_1}}
    - |-
        {{subtask_2}}
    - |-
        [... etc.]
    ```end_of_final_output_yaml
    """

    # write tests
    breakpoint()
    """
    - write function signature
    - write step-by-step of what needs to happen
    - go to steps checking flow
    """


# function specification flow
# ....
# > ---0.2.2---
# > (commit)
# steps checking flow
# function creation flow
# debug flow


"""
function specification flow:
- create context for function
- write tests
- write function signature
- write step-by-step of what needs to happen
- go to steps checking flow
"""

"""
steps checking flow:
- get list of all functions we already have > include import code
- for each step, check if we have a function that does that, or if it can be done via a few lines of base python
  - if yes: go to function creation flow
  - if no:
    - go to function specification flow for subfunction, with steps of parent function as context
    - reset and restart steps checking flow
"""

"""
function creation flow:
- write function > needs to be typed
- run tests
- if tests pass, commit function
- if tests fail, go to debug flow
"""

"""
debug flow:
- try to get a passing version of each failed test
- take the best version out of the pool, and try to combine it with the ones that passed tests that it failed
"""
