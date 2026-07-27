---
name: add-provider
description: Add a new LLM or embedding provider to llminspector/models — the three-method contract, config wiring, and the contract test that defines done. Use when asked to support OpenAI, Anthropic, Bedrock, a local model, or any new backend.
---

# Adding a provider

Read `llminspector/models/CLAUDE.md` first. `tests/test_provider_contract.py` is
the specification — **a provider is done when it passes that file unmodified.**

## 1. The required contract is three methods

```python
from llminspector.models.base_model import BaseLLM

class MyProviderModel(BaseLLM):
    def get_model_name(self) -> str: ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    async def a_generate(self, prompt: str, **kwargs) -> str: ...
```

Embeddings: `BaseEmbeddingModel` with `get_model_name`, `embed_text`,
`embed_texts`.

That is the whole requirement. Every metric except the ragas-backed ones, and
every synthesizer engine except the ragas testset backend, needs nothing more.
**Do not widen the ABC to accommodate one provider** — put provider-specific
behaviour on the subclass.

## 2. Optional capabilities

- `ragas_llm()` / `ragas_embeddings()` — override *only* if this provider can
  supply the wrappers. The base raises a directed `NotImplementedError`, which is
  the correct behaviour for a provider that can't. Without them, the five context
  metrics and the RAG testset engine are unavailable with this provider, and
  that is a supported configuration.
- `max_workers` — expose it if the provider has a concurrency ceiling.
  `evaluate()` reads the strictest value across the metric set and uses it as
  the default `batch_size`. Don't throttle internally instead.

## 3. Imports stay deferred

The SDK import goes **inside** the method or a lazily-built client property, not
at module top level. `import llminspector` must not pull in every vendor SDK, and
a user who installed only one provider's dependencies must still be able to
import the package.

## 4. Retry comes for free

Don't write per-provider backoff. `retry.py` detects rate limits from status
codes, then class names containing `RateLimit`, then message text — deliberately
without importing any vendor's exception types. Wire your calls through it.

## 5. Config

Settings go through `llminspector/config/settings.py` and read `LLMINSPECTOR_*`
environment variables. **No credentials in code, tests, notebooks, or docs** —
`tests/test_settings.py` scrubs the environment with an autouse fixture, and new
settings should follow that pattern.

## 6. Export and document

- Export from `llminspector/models/__init__.py` (not the package root).
- Add the provider to `docs/guides/02_models.md`, including which optional
  capabilities it supports.
- If it supports ragas, say so explicitly — that is the difference between a
  full and a partial metric set.

## 7. Verify

```bash
.venv/bin/python -m pytest -q tests/test_provider_contract.py tests/test_models.py
.venv/bin/python -m pytest -q
```

If the contract test needs editing to pass, stop: either the contract is
genuinely changing (a decision for the user) or the provider is wrong.
