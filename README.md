# LLMInspector

When we use LLMs/GenAI in enterprise applications, understanding, evaluating and navigating their
capabilities, limitations and risks is important. LLMInspector helps make sure an LLM application
is in alignment with functional and non-functional requirements and is safe & robust against
adversarial queries.

LLMInspector is a Python package for **generating** test data, **evaluating** LLM/RAG applications
against a broad metric suite, and **reporting** the results.

## Key features

- **Datasets** — a typed schema (`LLMTestCase`, `Golden`, `EvaluationDataset`) with pandas / Excel
  I/O and column mapping.
- **Models** — a provider abstraction (`BaseLLM` / `BaseEmbeddingModel`), Azure OpenAI first,
  unifying API-key and Azure-AD-token auth. A provider is still three methods; structured
  output, async embeddings and token metering are concrete defaults you inherit.
- **Metrics** — 20+ class-based metrics: quality (BERTScore), RAG (faithfulness, answer
  correctness/relevancy, conciseness, context precision/recall/utilisation/relevance/entity
  recall), safety (PII, content moderation, jailbreak, refusal, hallucination, code detection),
  NLP (sentiment, emotion, language, readability, tokens), and policy compliance.
- **Evaluate** — an async engine (`a_evaluate()` / `evaluate()`) with availability filtering,
  rate-limit backoff, visible per-metric failures, and stable, ordered output that the metrics
  themselves declare.
- **Generation** — an async pipeline that turns a `GoldenSource` into goldens through an ordered
  chain of `Stage`s, with per-golden lineage, seeded reproducibility, explained rejections, and
  optional token metering. Five sources ship: a curated adversarial bank (offline), contexts you
  supply, your own document corpus (chunking, embedding, critic-scored context selection), a
  description alone, and an existing golden set to grow.
- **Reporting** — export results to DataFrame / Excel / numeric summary.

## Getting started

### 1. Create a virtual environment

Python 3.12+ is required.

Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
```

Or the standard library:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install the package

```bash
pip install -e .
```

Three optional extras, each confined to one capability:

| Extra | Buys | Needed for |
|---|---|---|
| `ragas` | the five context metrics (`ContextPrecision`, `ContextRecall`, `ContextUtilisation`, `ContextRelevance`, `ContextEntityRecall`) | those five metrics only |
| `documents` | `pypdf`, `python-docx` | generating from PDF or DOCX corpora |
| `faiss` | `faiss-cpu` | an alternative vector index for large corpora |

```bash
pip install -e ".[ragas]"
pip install -e ".[documents]"
pip install -e ".[ragas,documents,faiss]"
```

Everything else runs on a bare `BaseLLM` provider and needs no extra: all other
metrics, the evaluate engine, the adversarial generator, and document generation
from `.txt` / `.md` / `.mdx`. Reaching for a capability whose extra is missing
raises an `ImportError` naming the extra rather than failing obscurely.

> **Note — build backend.** LLMInspector builds with the Michelin-internal
> `pydnx_packaging` backend, which is not on public PyPI. On a machine **with**
> Artifactory access the command above works as-is. **Without** it, disable build
> isolation after preinstalling the build tools:
>
> ```bash
> uv pip install --python .venv/bin/python setuptools wheel
> uv pip install --python .venv/bin/python -e . --no-build-isolation
> ```

### 3. Verify

```bash
python -c "import llminspector; print(llminspector.__version__)"
pytest -q
```

The offline notebooks are the quickest smoke test:

```bash
jupyter lab examples/     # then run 01_dataset_roundtrip.ipynb top to bottom
```

## Quickstart

```python
from llminspector import evaluate, reporting
from llminspector.config import AzureSettings
from llminspector.dataset import EvaluationDataset
from llminspector.metrics import AnswerCorrectnessMetric, FaithfulnessMetric
from llminspector.models import AzureOpenAIModel
from llminspector.test_case import LLMTestCase

model = AzureOpenAIModel(AzureSettings.from_env())

dataset = EvaluationDataset(test_cases=[
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        expected_output="Paris",
        retrieval_context=["The capital of France is Paris."],
    ),
])

result = evaluate(dataset, [FaithfulnessMetric(model), AnswerCorrectnessMetric(model)])
print(result.to_pandas())
```

## Where things live

Each layer owns its own names, so every class has exactly one import path. Only the evaluate
entry points sit at the package root:

| import from | what it holds |
|---|---|
| `llminspector` | `evaluate`, `a_evaluate`, `EvaluationResult`, `__version__` |
| `llminspector.test_case` | `LLMTestCase` |
| `llminspector.dataset` | `EvaluationDataset`, `Golden`, `ColumnMapping`, `goldens_to_dataframe` |
| `llminspector.config` | `AzureSettings` |
| `llminspector.models` | `BaseLLM`, `AzureOpenAIModel`, `AzureOpenAIEmbedding`, `MeteredModel` |
| `llminspector.metrics` | `BaseMetric` + all 24 metric classes |
| `llminspector.generation` | `Generator`, `GenerationResult`, `DocumentSource`, `ContextSource`, `ScratchSource`, `SeedGoldenSource`, `AdversarialGenerator`, `default_stages` |
| `llminspector.reporting` | `summary`, `errors` |

## Documentation

- **Usage guides:** [`docs/guides/`](docs/guides/) — datasets, models, metrics, evaluate,
  generation, reporting.
- **Runnable examples:** [`examples/`](examples/) — five notebooks, from the schema layer to a
  full generate → evaluate → export run.

### Building the API docs locally

The Sphinx build renders [`docs/api.rst`](docs/api.rst) from the package docstrings:

```bash
pip install -e ".[dev]"        # see the build-backend note above if this fails
cd docs && make html
python -m http.server -d _build/html 8000   # then open http://localhost:8000
```

It needs only `sphinx` and `sphinx-copybutton`. The version shown in the docs comes from the
installed package metadata, falling back to `dev` in a checkout that was never installed. The
Markdown guides under `docs/guides/` are read directly and are not part of the Sphinx build.

## Authors

Sourabh Potnis · Ankit Zade · Kiran Prasath · Arpit Kumar · Shraddha Pawar
