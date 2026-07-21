# LLMInspector Usage Guides

Task-oriented guides that sit on top of the API reference. Read them in order for a first pass,
or jump to the module you need.

1. [Datasets & Test Cases](01_datasets.md) — `LLMTestCase`, `Golden`, `EvaluationDataset`
2. [Models](02_models.md) — `Settings`, `AzureOpenAIModel`, both auth styles
3. [Metrics](03_metrics.md) — the metric catalogue and how to construct them
4. [Evaluate](04_evaluate.md) — the `evaluate()` engine, availability, `overall_accuracy`
5. [Synthesizers](05_synthesizers.md) — alignment / adversarial / RAG, and the swappable engines
6. [Reporting](06_reporting.md) — exporting results

Runnable code lives in [`examples/`](../../examples/). Architecture and the refactor history are
in [`REFACTOR_TARGET.md`](../../REFACTOR_TARGET.md) and [`REFACTOR_PHASES.md`](../../REFACTOR_PHASES.md).

## Install

```bash
pip install -e .
```

## 30-second tour

```python
from llminspector import (
    EvaluationDataset, LLMTestCase, Settings, AzureOpenAIModel,
    FaithfulnessMetric, evaluate, reporting,
)

model = AzureOpenAIModel(Settings.from_env())
dataset = EvaluationDataset(test_cases=[
    LLMTestCase(input="...", actual_output="...", retrieval_context=["..."]),
])
result = evaluate(dataset, [FaithfulnessMetric(model)])
print(reporting.to_dataframe(result))
```
