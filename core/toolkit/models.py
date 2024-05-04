"""Model utilities."""

from typing import Any, Callable, Sequence

from colorama import Fore
from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, ToolCall
from langchain_core.tools import tool
from langchain_community.chat_models.perplexity import ChatPerplexity
from langchain_openai.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic

from core.config import GPT_4_TURBO

load_dotenv(override=True)

# broad_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", verbose=False)
# anthropic models: https://docs.anthropic.com/claude/reference/selecting-a-model

PRECISE_MODEL = ChatOpenAI(
    temperature=0, model_name=GPT_4_TURBO, verbose=False
)
VARIANT_MODEL = ChatOpenAI(
    temperature=1.0, model_name=GPT_4_TURBO, verbose=False
)
FAST_MODEL = ChatGroq(temperature=0, model_name="llama3-70b-8192", verbose=False)
SEARCH_MODEL = ChatPerplexity(temperature=0, model="sonar-medium-online", verbose=False)
CREATIVE_MODEL = ChatAnthropic(temperature=1.0, model="claude-3-opus-20240229", verbose=False)


def query_model(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    color: str = Fore.RESET,
    preamble: str | None = None,
    printout: bool = True,
) -> str:
    """Query an LLM chat model. `preamble` is printed before the result."""
    if preamble is not None and printout:
        print(f"\033[1;34m{preamble}\033[0m")
    result = str(model.invoke(messages).content)
    if printout:
        print(f"{color}{result}{Fore.RESET}")
    return result


def query_model_tool_call(
    model: BaseChatModel,
    messages: Sequence[BaseMessage],
    tools: Sequence[Callable[..., Any]],
    color: str = Fore.RESET,
    preamble: str | None = None,
    printout: bool = True,
) -> list[ToolCall]:
    """Query an LLM chat model, returning the result of a tool call."""
    model = model.bind_tools([tool(t) for t in tools])
    if preamble is not None and printout:
        print(f"\033[1;34m{preamble}\033[0m")
    result: AIMessage = model.invoke(messages)
    call_results = result.tool_calls
    if printout:
        print(f"{color}{call_results}{Fore.RESET}")
    return call_results


def call_tools(
    tools: Sequence[Callable[..., Any]], call_results: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run the tools on the call args."""
    for call_result in call_results:
        next_tool = next(t for t in tools if t.__name__ == call_result["name"])
        call_result["output"] = next_tool(**call_result["args"])
    return call_results


def format_messages(messages: Sequence[BaseMessage]) -> str:
    """Format model messages into something printable."""
    return "\n\n---\n\n".join(
        [f"[{message.type.upper()}]:\n\n{message.content}" for message in messages]  # type: ignore
    )
