"""Craft a component from parts."""

from dataclasses import dataclass
from typing import Sequence

from swarm_memorizer.toolkit.text import dedent_and_strip


@dataclass
class Concept:
    """Concepts for an agent."""

    name: str
    description: str


def format_concepts(concepts: Sequence[Concept]) -> str:
    """Format concepts into a string."""
    if not concepts:
        return "N/A"
    raise NotImplementedError


def specify_function(context: str, requirements: str) -> Sequence[str]:
    """Write specifications for the function."""
    context = """
    ## MISSION:
    You are the instructor for an AI agent (referred to as {agent} from now on) for writing unit tests for a function based on a description of it. Your purpose is to provide a reasoning structure for the agent to think through how to generate the tests.

    ## CONCEPTS:
    {concepts}

    ## AGENT INFORMATION SECTIONS:
    The agent has access to several sections of information that is relevant to its decisionmaking.
    {information_sections}
    """
    agent = "TEST_WRITER"
    concepts: list[Concept] = []
    information_sections: list[Concept] = []
    context = dedent_and_strip(context).format(
        agent=agent,
        concepts=format_concepts(concepts),
        information_sections=format_concepts(information_sections),
    )





    # commit
    request = """
    ## REQUEST FOR YOU:
    {OVERVIEW}
    - {INFORMATION_RESTRICTIONS}
    - {TERM_REFERENCES}
    - As an initial part of the reasoning, the delegator must figure out whether to lean towards exploration using NEW {EXECUTOR} candidates or exploitation using non-NEW {EXECUTOR} candidates. This of course depends on how good the non-NEW {EXECUTOR} candidates are.
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
            DELEGATOR=Concept.DELEGATOR.value,
            OVERVIEW=DelegatorReasoningGenerationNotes.OVERVIEW.value,
            INFORMATION_RESTRICTIONS=DelegatorReasoningGenerationNotes.INFORMATION_RESTRICTIONS.value,
            TERM_REFERENCES=DelegatorReasoningGenerationNotes.TERM_REFERENCES.value,
            EXECUTOR=Concept.EXECUTOR.value,
            CONTEXT=Concept.CONTEXT.value,
            STEPS_RESTRICTIONS=DelegatorReasoningGenerationNotes.STEPS_RESTRICTIONS.value,
            PROCEDURAL_SCRIPTING=DelegatorReasoningGenerationNotes.PROCEDURAL_SCRIPTING.value,
        )
    )
    messages = [
        SystemMessage(content=context),
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
    context = """
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
