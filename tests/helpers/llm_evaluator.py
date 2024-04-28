"""Evaluations using LLMs."""

from typing import Any

from langchain.schema import SystemMessage

from core.toolkit.models import query_model, FAST_MODEL
from core.toolkit.text import dedent_and_strip, extract_and_unpack
from core.toolkit.yaml_tools import DEFAULT_YAML



def llm_evaluate(
    value: str, condition: str, query_kwargs: dict[str, Any] | None = None
) -> bool:
    """Evaluate whether a value meets a condition using an LLM."""
    query_kwargs = query_kwargs or {}
    instructions = """
    # MISSION
    You are an evaluator for the text output of some process. You will be given a CONDITION and a VALUE, and you must determine whether the VALUE meets the CONDITION.

    ## CONDITION
    The following is the condition you must evaluate the VALUE against:
    ```start_of_condition
    {condition}
    ```end_of_condition

    ## VALUE
    The following is the VALUE you must evaluate:
    ```start_of_value
    {value}
    ```end_of_value

    ## OUTPUT
    Output your evaluation in the following YAML format:
    ```start_of_output_yaml
    thoughts: |-
        {{thoughts}}
    condition_met: !!bool {{condition_met}}
    ```end_of_output_yaml

    Make sure to output the ```start_of_output_yaml and ```end_of_output_yaml delimiters.
    """
    instructions = dedent_and_strip(instructions).format(
        condition=condition, value=value
    )
    messages = [
        SystemMessage(content=instructions),
    ]
    output = query_model(
        model=FAST_MODEL,
        messages=messages,
        preamble=instructions,
        **query_kwargs,
    )
    output = extract_and_unpack(output, start_block_type="start_of_output_yaml")
    output = DEFAULT_YAML.load(output)
    return output["condition_met"]
