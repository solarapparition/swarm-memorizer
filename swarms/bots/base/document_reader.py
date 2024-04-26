import hashlib
from pathlib import Path
from typing import Any, Hashable
import uuid
from embedchain import App

from core.toolkit.yaml_tools import DEFAULT_YAML as YAML

CONFIG = """
app:
  config:
    id: document-reader-bot
llm:
  provider: google
  config:
    model: models/gemini-1.5-pro-latest
    max_tokens: 4000
    temperature: 0.0
    top_p: 1
    stream: false
embedder:
  provider: google
  config:
    model: 'models/embedding-001'
    task_type: "retrieval_document"
    title: "Embeddings for Embedchain"
vectordb:
  provider: chroma
  config:
    collection_name: '{collection_name}'
    dir: {db_dir}
    allow_reset: true
"""


def chat_with_resource(query: str, source: Any, db_dir: Path) -> str:
    """Chat with the bot overlay of a resource."""
    collection_name = hashlib.sha256(repr(source).encode()).hexdigest()[:62]
    config = CONFIG.format(collection_name=collection_name, db_dir=db_dir)
    app = App.from_config(config=YAML.load(config))
    app.add(source)
    return str(app.chat(query))


# figure out what blah should be named # needs to be same name as what's passed in as input artifact
def query_artifact_agent(query: str, blah: Any) -> str:
    """Query an agent about an artifact."""


# answer = chat_with_resource(
#     query="What is the net worth of Elon Musk?",
#     source="https://www.forbes.com/profile/elon-musk",
#     db_dir=Path("db"),
# )


# > need intermediary to parse user input
