"""Bot for querying documents."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from colorama import Fore
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from core.bot import BotCore
from core.task import ExecutionReport
from core.task_data import TaskDescription
from core.toolkit.models import (
    BROAD_MODEL,
    FAST_MODEL,
    format_messages,
    query_model,
)
from core.toolkit.text import dedent_and_strip, extract_and_unpack
from core.toolkit.yaml_tools import DEFAULT_YAML as YAML
from swarms.bots.base.core_executor import Conversation
from swarms.bots.toolkit.load_markdown import MarkdownLoadError, load_markdown


AGENT_COLOR = Fore.GREEN


@dataclass
class DocumentProxy:
    """A proxy for the bot."""

    file_location: Path | None = None
    text: str | None = None
    # initial_run: bool = True


def format_message(message: HumanMessage | AIMessage) -> str:
    """Format a message."""
    sender = "USER" if isinstance(message, HumanMessage) else "ASSISTANT"
    return f"{sender}: {message.content}"


def format_conversation(
    task_description: TaskDescription, message_history: Conversation
) -> str:
    """Format the conversation."""
    initial_message = f"USER: {task_description}"
    messages = [format_message(message) for message in message_history]
    return "\n".join([initial_message] + messages)


class LocationNotFoundError(FileNotFoundError):
    """Raised when a file location was not successfully extracted."""

    class ReasonCode(Enum):
        """Reason codes for the error."""

        LOCATION_NOT_MENTIONED = "LOCATION_NOT_MENTIONED"
        LOCATION_NOT_LOCAL = "LOCATION_NOT_LOCAL"
        NO_FILE_AT_LOCATION = "NO_FILE_AT_LOCATION"

        def __str__(self) -> str:
            return self.value

    def __init__(self, reason_code: ReasonCode, *args: Any):
        super().__init__(*args)
        self.reason_code = reason_code


def extract_file_location(
    task_description: TaskDescription, message_history: Conversation
) -> Path:
    """Extract file location from task description or message history."""
    conversation = format_conversation(task_description, message_history)
    context = """
    # MISSION
    Your task is to determine if the user in a conversation given below has given the location of a local file for an agent to answer a question about, and extract it if needed.

    ## CONVERSATION
    In the conversation below, a USER is asking an ASSISTANT a query about some resource.
    <conversation>
    {conversation}
    </conversation>
    You are not the ASSISTANT, but you are helping the ASSISTANT determine whether there is a local file it can look at to answer the USER's query.

    ## REASONING_PROCESS
    This is the reasoning process you must follow to extract the file location of the document:
    <reasoning-process-yaml>
    - initial_check:
        action: look through the CONVERSATION to determine if the USER mentions a location for what they're asking questions about.
        cases:
            location_not_mentioned: exit REASONING_PROCESS with output flag "LOCATION_NOT_MENTIONED".
            location_mentioned: continue to next step of REASONING_PROCESS.
    - location_type:
        action: check if location appears to refers to a local file.
        cases:
            location_not_local: exit REASONING_PROCESS with output flag "LOCATION_NOT_LOCAL".
            location_local: continue to next step of REASONING_PROCESS.
    - location_extraction:
        action: extract the location of the local file, which should be a path string.
    </reasoning-process-yaml>
    """
    context = dedent_and_strip(context).format(conversation=conversation)

    request = """
    # REQUEST
    Your task is to follow the REASONING_PROCESS above to determine the OUTPUT_FLAG, which can then be used to inform the ASSISTANT of its next actions.

    ## REASONING_OUTPUT
    You must output the intermediate results of the REASONING_PROCESS in the following format:
    <reasoning-output-yaml>
    {reasoning_output}
    </reasoning-output-yaml>
    The `reasoning-output-yaml` should have a similar structure to the `reasoning-process-yaml`, except with the results of each part of the process.

    ## FINAL_OUTPUT
    After you have posted the REASONING_OUTPUT, output the extracted location and OUTPUT_FLAG that you have determined from the REASONING_PROCESS. Do so in the following YAML format:

    <final-output-yaml>
    output_flag: {output_flag}
    file_location: {file_location}
    </final-output-yaml>

    Remember, `output_flag` must be one of the following:
    - LOCATION_NOT_MENTIONED
    - LOCATION_NOT_LOCAL
    - LOCATION_OKAY
    `file_location` should be a path string if `output_flag` is `LOCATION_OKAY`, and otherwise blank.
    """
    request = dedent_and_strip(request)
    messages = [
        SystemMessage(content=context),
        HumanMessage(content=request),
    ]
    output = query_model(
        model=FAST_MODEL,
        messages=messages,
        preamble=f"Extracting file location from conversation...\n{format_messages(messages)}",
        printout=True,
        color=AGENT_COLOR,
    )
    extracted_output = extract_and_unpack(
        output,
        start_block_type="<final-output-yaml>",
        end_block_type="</final-output-yaml>",
        prefix="",
    )
    extracted_output = YAML.load(extracted_output)
    output_flag = extracted_output["output_flag"]
    file_location = Path(extracted_output["file_location"])
    if output_flag == "LOCATION_OKAY":
        if file_location.exists():
            return file_location
        raise LocationNotFoundError(
            reason_code=LocationNotFoundError.ReasonCode.NO_FILE_AT_LOCATION
        )
    reason_code = LocationNotFoundError.ReasonCode(output_flag)
    raise LocationNotFoundError(reason_code=reason_code)


MESSAGE_MAPPING = {
    LocationNotFoundError.ReasonCode.LOCATION_NOT_MENTIONED: "Query failed: there is no mention of the location of the file to be queried.",
    LocationNotFoundError.ReasonCode.LOCATION_NOT_LOCAL: "Query failed: the location mentioned does not appear to be for a local file.",
    LocationNotFoundError.ReasonCode.NO_FILE_AT_LOCATION: "Query failed: the file does not exist at location.",
}


def answer_query(
    task_description: TaskDescription,
    message_history: Conversation,
    file_location: Path,
    document_text: str,
) -> str:
    """Answer a query about a document."""
    context = """
    # MISSION
    You are an ASSISTANT that is answering a query from a USER about a document.

    ## DOCUMENT INFO
    The document was loaded from the following location: {file_location}

    ## DOCUMENT TEXT
    The text of the document has been loaded and is ready for you to reference.
    <document-text>
    {document_text}
    </document-text>

    ## CONVERSATION
    The following is the conversation so far between you and the USER.
    <conversation>
    {conversation}
    </conversation>
    Remember, you are the ASSISTANT in the conversation.

    ## REASONING_PROCESS
    This is the reasoning process you must follow to answer the USER's query:
    <reasoning-process-yaml>
    - query_determination:
        action: determine the query the USER is asking about.
        notes: usually the first message contains the main overall question. Subsequent messages may contain follow-ups or refinements.
    - relevant_information_retrieval:
        action: retrieve parts of the document that are relevant to the query.
        notes: information doesn't need to directly answer the query—this is just brainstorming.
    - determine_information_completion:
        action: determine if the information retrieved is sufficient to answer the query.
        notes: if not, consider what additional information is needed, and whether it can reasonably be provided by the USER.
    - response_formulation:
        action: formulate a response to the query based on the relevant information.
        notes: |-
          this should be one of 3 options:
          - a direct answer (if there is sufficient information)
          - a request for more information or clarification from the USER (if more information is needed and can be provided by the USER)
          - admission that the query cannot be answered (if there is insufficient information and the USER couldn't reasonably be expected to provide it)
          - ending the conversation (if the user indicates they are done asking questions)
    </reasoning-process-yaml>
    """
    context = dedent_and_strip(context).format(
        file_location=file_location,
        document_text=document_text,
        conversation=format_conversation(task_description, message_history),
    )
    request = """
    # REQUEST
    Your task is to follow the REASONING_PROCESS above to answer the USER's query.

    ## REASONING_OUTPUT
    You must output the intermediate results of the REASONING_PROCESS in the following format:
    <reasoning-output-yaml>
    {reasoning_output}
    </reasoning-output-yaml>
    The `reasoning-output-yaml` should have a similar structure to the YAML inside the `reasoning-process-yaml` block above, except with the results of each part of the process.

    ## FINAL_OUTPUT
    After you have posted the REASONING_OUTPUT, output the response to the USER's query in the following format:
    <final-response-text>
    {response_text}
    </final-response-text>
    """
    request = dedent_and_strip(request)
    request = context + "\n\n" + request
    messages = [
        HumanMessage(content=request),
        # HumanMessage(content=request),
    ]
    output = query_model(
        model=BROAD_MODEL,
        messages=messages,
        preamble=f"Answering query from user...\n{format_messages(messages)}",
        printout=True,
        color=AGENT_COLOR,
    )
    return extract_and_unpack(
        output,
        start_block_type="<final-response-text>",
        end_block_type="</final-response-text>",
        prefix="",
    )


def load_bot(*_) -> BotCore:
    """Load the bot core."""

    document = DocumentProxy()

    def run(
        task_description: TaskDescription,
        message_history: Conversation,
        output_dir: Path,
    ) -> ExecutionReport:
        """Run the bot."""
        if not document.file_location:
            try:
                document.file_location = extract_file_location(
                    task_description, message_history
                )
            except LocationNotFoundError as e:
                return ExecutionReport(MESSAGE_MAPPING[e.reason_code])
        if not document.text:
            try:
                document.text = load_markdown(document.file_location)
            except MarkdownLoadError as e:
                return ExecutionReport(e.message)

        answer = answer_query(
            task_description, message_history, document.file_location, document.text
        )
        return ExecutionReport(answer)

    return BotCore(run)
