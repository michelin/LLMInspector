# models/

Provider abstraction. `base_model.py` holds the ABCs, `azure_openai.py` the one
concrete provider, `retry.py` the shared backoff.

## The contract is deliberately small

`BaseLLM`: `get_model_name`, `generate`, `a_generate`. That's it. Every metric
except the five ragas-backed ones, and every synthesizer engine except the ragas
testset backend, needs nothing more — **a new provider is a class with three
methods**. Keeping it that small is the point; resist adding required members.

`BaseEmbeddingModel`: `get_model_name`, `embed_text`, `embed_texts`.

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
