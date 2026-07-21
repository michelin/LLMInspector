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

## Installation

```bash
pip install -e .
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
