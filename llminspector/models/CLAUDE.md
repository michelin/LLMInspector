# models/

Provider abstraction. `base_model.py` holds the ABCs, `azure_openai.py` the one
concrete provider, `retry.py` the shared backoff, `errors.py` the one exception
this layer raises on its own behalf.

## The contract is deliberately small

`BaseLLM`: `get_model_name`, `generate`, `a_generate`. That's it. Every metric
except the five ragas-backed ones, and every golden source except the ragas
testset backend, needs nothing more — **a new provider is a class with three
methods**. Keeping it that small is the point; resist adding required members.

`BaseEmbeddingModel`: `get_model_name`, `embed_text`, `embed_texts`.

### New capabilities are concrete-with-default, never abstract

This is the rule that keeps "three methods" true as the package grows. Adding an
`@abstractmethod` breaks every existing provider and turns the contract into a
moving target; adding a concrete method with a working default breaks nobody.

| Method | Default | Overridden by |
|---|---|---|
| `generate_structured` / `a_generate_structured` | Append the JSON Schema + a return-only-JSON directive, call `generate`, parse, validate, reask once | `AzureOpenAIModel`, to add `response_format` |
| `a_embed_text` / `a_embed_texts` | `asyncio.to_thread` around the sync call | a provider with a truly async client |
| `ragas_llm` / `ragas_embeddings` | raise `NotImplementedError` | providers that can supply the wrappers |

`tests/test_structured_output.py` asserts a three-method stub gets working
structured output. If that test needs changing, the contract grew — which is the
thing this file exists to prevent.

### Structured output has its own retry axis — keep it separate

`generate_structured` reasks **exactly once** on a parse or validation failure,
then raises `StructuredOutputError`. `retry.py` is rate-limit-only. Do not merge
them and do not teach `is_rate_limit_error` about parse failures: a malformed
response is not transient, so exponential backoff would buy the same prose five
times over, and a 429 is not fixed by appending an error message to the prompt.

The Azure override adds `response_format={"type": "json_object"}` via
`setdefault` and inherits all the parsing. There is deliberately no second copy
of that logic, no `.client` escape hatch, and no `with_structured_output`
dependency. Note `json_object` mode requires the word "JSON" in the prompt —
`STRUCTURED_OUTPUT_INSTRUCTION` is what satisfies that, so don't reword it
casually.

### Ragas support is an optional capability, not part of the contract

`BaseLLM.ragas_llm()` and `BaseEmbeddingModel.ragas_embeddings()` are *concrete*
and raise `NotImplementedError` with a directed message. Providers that can
supply the wrappers override them; providers that can't simply don't.
`RagasBackedMetric` and `engines/ragas_testset.py` are the only callers.

## Retry

`retry.py` is provider-agnostic on purpose: rate-limit detection works off status
codes (`status_code` / `status` / `http_status` / `code`), then class names
containing `RateLimit`, then message text — **never by importing a vendor's
exception classes**. A new provider gets backoff for free; don't reimplement it
per provider. Defaults: 5 retries, 1s initial, 60s cap, factor 2, jittered.

Without this, a 429 surfaced as a metric exception, got swallowed by the metric
error path, and became a `None` score indistinguishable from *skipped*.

Both `AzureOpenAIEmbedding.embed_text` and `.embed_texts` go through it. They
were raw, and embedding a document corpus is the most 429-prone workload here.

## max_workers

A provider may expose `max_workers`. `evaluate()` reads the strictest value
across the metric set and uses it as the default `batch_size`, so one number
governs both row concurrency and the ragas `RunConfig`. If you add a provider
with a concurrency ceiling, expose it as `max_workers` rather than throttling
internally.

## Adding a provider

Follow `.claude/skills/add-provider`. `tests/test_provider_contract.py` is the
spec — a new provider is done when it passes that file unmodified. Config goes
through `config/settings.py` (`LLMINSPECTOR_*` env vars); credentials never
appear in code, tests, notebooks, or docs.
