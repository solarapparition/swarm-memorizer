"""Model utilities."""

from typing import Sequence

from colorama import Fore
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain.chat_models.base import BaseChatModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.chat_models.perplexity import ChatPerplexity

load_dotenv(override=True)

PRECISE_MODEL = ChatOpenAI(
    temperature=0, model_name="gpt-4-turbo-2024-04-09", verbose=False
)
# creative_model = ChatOpenAI(temperature=0.7, model_name="gpt-4", verbose=False)  # type: ignore
# super_creative_model = ChatOpenAI(temperature=1.0, model_name="gpt-4", verbose=False)  # type: ignore
SUPER_CREATIVE_MODEL = ChatOpenAI(
    temperature=1.0, model_name="gpt-4-turbo-2024-04-09", verbose=False
)
fast_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", verbose=False)
broad_model = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", verbose=False)
# anthropic models: https://docs.anthropic.com/claude/reference/selecting-a-model
PERPLEXITY = ChatPerplexity(temperature=0, model="sonar-medium-online", verbose=False)


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
    # result = model(list(messages)).content
    result = str(model.invoke(messages).content)
    if printout:
        print(f"{color}{result}{Fore.RESET}")
    return result


def format_messages(messages: Sequence[BaseMessage]) -> str:
    """Format model messages into something printable."""
    return "\n\n---\n\n".join(
        [f"[{message.type.upper()}]:\n\n{message.content}" for message in messages]  # type: ignore
    )
