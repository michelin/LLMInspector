# Models

The provider abstraction. `BaseLLM` / `BaseEmbeddingModel` are the ABCs; Azure OpenAI is the
first concrete provider. `Settings` is the code-driven replacement for the old `config.ini`.

## Settings

```python
from llminspector import Settings

# Explicit (recommended for apps):
settings = Settings(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_version="2024-02-01",
    api_key="...",                       # optional here; can pass to the model instead
    azure_deployment="gpt-5-mini-dzs",   # defaults promoted from the legacy hard-coded values
    model_name="gpt-5-mini",
    embedding_deployment="text-embedding-ada-002",
    embedding_name="text-embedding-ada-002",
)

# Or from environment variables (azure_endpoint / api_version / api_key / ...):
settings = Settings.from_env()           # optionally Settings.from_env(".env")
```

## AzureOpenAIModel & AzureOpenAIEmbedding

Both unify the two legacy auth styles. Provide **exactly one** of `api_key` or
`azure_ad_token_provider` (passing both, or neither, raises):

```python
from llminspector import AzureOpenAIModel, AzureOpenAIEmbedding

# API-key auth (taken from settings.api_key if omitted):
model = AzureOpenAIModel(settings, api_key="...")

# Azure AD token-provider auth:
model = AzureOpenAIModel(settings, azure_ad_token_provider=my_token_provider)

# Deployment / model names default from Settings but can be overridden:
model = AzureOpenAIModel(settings, model_name="gpt-4o-mini", azure_deployment="gpt-4o-mini")

embedding = AzureOpenAIEmbedding(settings, api_key="...")
```

### What you get

```python
model.generate("Hello")            # -> str  (sync)
await model.a_generate("Hello")    # -> str  (async)
model.get_model_name()             # -> "gpt-5-mini"
model.client                       # underlying langchain AzureChatOpenAI
model.ragas_llm()                  # ragas LangchainLLMWrapper (used by context metrics)

embedding.embed_text("hi")         # -> list[float]
embedding.embed_texts(["a", "b"])  # -> list[list[float]]
embedding.ragas_embeddings()       # ragas LangchainEmbeddingsWrapper
```

Metrics take the model in their constructor and use these handles internally — you rarely call
`generate()` directly. See [Metrics](03_metrics.md).
