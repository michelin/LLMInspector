# Models

The provider abstraction. `BaseLLM` / `BaseEmbeddingModel` are the ABCs; Azure OpenAI is the
first concrete provider. `AzureSettings` is the code-driven replacement for the old `config.ini`.

## AzureSettings

```python
from llminspector.config import AzureSettings

# Explicit (recommended for apps):
settings = AzureSettings(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_version="2024-02-01",
    api_key="...",                       # optional here; can pass to the model instead
    azure_deployment="gpt-5-mini-dzs",   # defaults promoted from the legacy hard-coded values
    model_name="gpt-5-mini",
    embedding_deployment="text-embedding-ada-002",
    embedding_name="text-embedding-ada-002",
)

# Or from environment variables:
settings = AzureSettings.from_env()      # optionally AzureSettings.from_env(".env")
```

The deployment/model defaults come from one specific Azure resource — override them for any other
tenant.

### Environment variables

Every field maps to `LLMINSPECTOR_<FIELD>`:

```
LLMINSPECTOR_AZURE_ENDPOINT      LLMINSPECTOR_AZURE_DEPLOYMENT
LLMINSPECTOR_API_VERSION         LLMINSPECTOR_MODEL_NAME
LLMINSPECTOR_API_KEY             LLMINSPECTOR_EMBEDDING_DEPLOYMENT
                                 LLMINSPECTOR_EMBEDDING_NAME
```

> **Deprecated:** the old unnamespaced lowercase names (`azure_endpoint`, `api_key`, …) still work
> for one release and emit a `DeprecationWarning`. They were case-sensitive on Linux, and `api_key`
> is generic enough to be claimed by an unrelated tool in the same environment.

> **Deprecated:** the class was previously called `Settings`. That name is still importable and
> warns on construction. It promised provider-neutrality it never had — every field on it is
> Azure-specific.

## AzureOpenAIModel & AzureOpenAIEmbedding

Both unify the two legacy auth styles. Provide **exactly one** of `api_key` or
`azure_ad_token_provider` (passing both, or neither, raises):

```python
from llminspector.models import AzureOpenAIModel, AzureOpenAIEmbedding

# API-key auth (taken from settings.api_key if omitted):
model = AzureOpenAIModel(settings, api_key="...")

# Azure AD token-provider auth:
model = AzureOpenAIModel(settings, azure_ad_token_provider=my_token_provider)

# Deployment / model names default from AzureSettings but can be overridden:
model = AzureOpenAIModel(settings, model_name="gpt-4o-mini", azure_deployment="gpt-4o-mini")

embedding = AzureOpenAIEmbedding(settings, api_key="...")
```

### What you get

```python
model.generate("Hello")            # -> str  (sync)
await model.a_generate("Hello")    # -> str  (async)
model.get_model_name()             # -> "gpt-5-mini"
model.client                       # underlying langchain AzureChatOpenAI (Azure-specific)
model.ragas_llm()                  # ragas LangchainLLMWrapper (optional capability)

embedding.embed_text("hi")           # -> list[float]
embedding.embed_texts(["a", "b"])    # -> list[list[float]]
await embedding.a_embed_texts([...]) # -> list[list[float]]  (async)
embedding.ragas_embeddings()         # ragas LangchainEmbeddingsWrapper (optional capability)
```

Both embedding calls go through the shared rate-limit backoff. Embedding a
document corpus — hundreds of chunks in one batch — is the most 429-prone
workload in the package, so `max_retries` is a constructor argument here too
(`0` disables it).

Metrics take the model in their constructor and use these handles internally — you rarely call
`generate()` directly. See [Metrics](03_metrics.md).

### Structured output

When you need a typed object rather than a string, hand the model a pydantic
schema:

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    score: float
    reason: str

verdict = model.generate_structured("Rate this answer...", Verdict)
verdict = await model.a_generate_structured("Rate this answer...", Verdict)
verdict.score   # -> float, already validated
```

The default implementation appends the schema and a "return only JSON" directive
to your prompt, parses the reply (tolerating code fences and surrounding prose),
and validates it. On a parse or validation failure it **reasks exactly once**
with the error text appended; a second failure raises `StructuredOutputError`.

That is a different retry axis from the rate-limit backoff, deliberately. A
malformed response is not transient — retrying it five times with exponential
delay just buys the same prose five times over.

`AzureOpenAIModel` overrides both methods to add `response_format={"type":
"json_object"}` and inherits everything else. An explicit `response_format` you
pass yourself wins.

## Knowing what a run cost

`generate()` returns `str`, so the provider's usage metadata is discarded.
`MeteredModel` wraps any provider and counts tokens with `tiktoken`:

```python
from llminspector.models import MeteredModel

model = MeteredModel(AzureOpenAIModel(settings))
model.generate("Hello")

model.usage        # {'calls': 1, 'prompt_tokens': 3, 'completion_tokens': 8, 'total_tokens': 11}
model.reset()      # meter one phase of a longer session
model.unwrap()     # give the plain provider back
```

It is a `BaseLLM` itself, so it drops in anywhere a provider goes — metrics,
generation stages, another decorator — and forwards everything it does not
define, including `max_workers` and the provider's own structured-output
implementation. Azure's native JSON mode survives metering.

Counts are an estimate, roughly ±10%, not a bill. Reasks are counted: a
structured request that needed a second attempt reports two calls, which is the
case you most want visibility on.

For a generation run, `GenerationConfig(track_usage=True)` wraps the models for
you and puts the totals on `GenerationResult.usage` — see
[Generation](05_synthesizers.md).

> **It mutates the model it wraps.** Counting from outside would miss every call
> the provider makes to itself, and `generate_structured` is built on
> `self.generate` — so an outside-only wrapper reports zero for a pipeline that
> uses structured output throughout. `unwrap()` reverses it.

## Writing another provider

The required contract is three methods. Metrics reach the model **only** through `a_generate`, so
this is enough to run every metric except the five ragas-backed context ones:

```python
from llminspector.models.base_model import BaseLLM

class MyProvider(BaseLLM):
    def get_model_name(self) -> str: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    async def a_generate(self, prompt: str, **kwargs) -> str: ...
```

Three methods is still the whole requirement. Everything the package added since
is **concrete with a working default**, so your provider gets it for free:

| You inherit | Default behaviour | Override when |
|---|---|---|
| `generate_structured` / `a_generate_structured` | Prompt-and-parse into the schema, one reask | Your API has a native JSON mode |
| `a_embed_text` / `a_embed_texts` | Sync call offloaded via `asyncio.to_thread` | Your client is genuinely async |

`ragas_llm()` / `ragas_embeddings()` are **optional capabilities** — concrete on the ABC, raising
`NotImplementedError` unless you override them. Only `RagasBackedMetric` subclasses
(`ContextPrecision`, `ContextRecall`, `ContextUtilisation`, `ContextRelevance`,
`ContextEntityRecall`) and the ragas testset backend ask for them; hand one a provider without
ragas support and you get a directed error naming the metric, not an `AttributeError`.

`.client` is **not** part of the contract — it is an Azure-specific escape hatch.
