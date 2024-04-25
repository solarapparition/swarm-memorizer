from embedchain import App

# > update db
# > parametrize collection_name
# > need intermediary to parse user input

CONFIG = """
llm:
  provider: google
  config:
    id: document-reader-bot
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
    collection_name: 'my-collection'
    dir: db
    allow_reset: true
"""


from core.toolkit.yaml_tools import DEFAULT_YAML as YAML

config_dict = YAML.load(CONFIG)
app = App.from_config(config=config_dict)
app.add("https://www.forbes.com/profile/elon-musk")
response = app.query("What is the net worth of Elon Musk?")
if app.llm.config.stream: # if stream is enabled, response is a generator
    for chunk in response:
        print(chunk)
else:
    print(response)
