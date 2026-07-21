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
  unifying API-key and Azure-AD-token auth.
- **Metrics** — 20+ class-based metrics: quality (BERTScore), RAG (faithfulness, answer
  correctness/relevancy, conciseness, context precision/recall/utilisation/relevance/entity
  recall), safety (PII, content moderation, jailbreak, refusal, hallucination, code detection),
  NLP (sentiment, emotion, language, readability, tokens), and policy compliance.
- **Evaluate** — an async engine (`evaluate()`) with availability filtering, dependency handling,
  and stable, ordered output.
- **Synthesizers** — alignment (tag-augment → paraphrase → perturb), adversarial (curated bank /
  red-team seam), and RAG (testset generation + ground-truth refinement), each behind a swappable
  engine so custom generators drop in without a refactor.
- **Reporting** — export results to DataFrame / Excel / numeric summary.

## Getting started

### 1. Create a virtual environment

Python 3.9+ is required (developed and tested on 3.12).

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
python examples/01_dataset_roundtrip.py     # runs fully offline
pytest -q
```

## Quickstart

```python
from llminspector import (
    EvaluationDataset, LLMTestCase, Settings, AzureOpenAIModel,
    FaithfulnessMetric, AnswerCorrectnessMetric, evaluate, reporting,
)

model = AzureOpenAIModel(Settings.from_env())

dataset = EvaluationDataset(test_cases=[
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="Paris is the capital of France.",
        expected_output="Paris",
        retrieval_context=["The capital of France is Paris."],
    ),
])

result = evaluate(dataset, [FaithfulnessMetric(model), AnswerCorrectnessMetric(model)])
print(reporting.to_dataframe(result))
```

## Documentation

- **Usage guides:** [`docs/guides/`](docs/guides/) — datasets, models, metrics, evaluate,
  synthesizers, reporting.
- **Runnable examples:** [`examples/`](examples/) — scripts + a `getting_started.ipynb` notebook.
- **Architecture:** [`REFACTOR_TARGET.md`](REFACTOR_TARGET.md) (end-state) and
  [`REFACTOR_PHASES.md`](REFACTOR_PHASES.md) (build history).

## Authors

Sourabh Potnis · Ankit Zade · Kiran Prasath · Arpit Kumar · Shraddha Pawar
