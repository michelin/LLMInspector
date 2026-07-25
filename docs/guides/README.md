# LLMInspector Usage Guides

Task-oriented guides that sit on top of the API reference. Read them in order for a first pass,
or jump to the module you need.

1. [Datasets & Test Cases](01_datasets.md) — `LLMTestCase`, `Golden`, `EvaluationDataset`
2. [Models](02_models.md) — `AzureSettings`, `AzureOpenAIModel`, both auth styles
3. [Metrics](03_metrics.md) — the metric catalogue and how to construct them
4. [Evaluate](04_evaluate.md) — the `evaluate()` engine, availability, the unified correctness judge
5. [Synthesizers](05_synthesizers.md) — alignment / adversarial / RAG, and the swappable engines
6. [Reporting](06_reporting.md) — exporting results

Runnable code lives in [`examples/`](../../examples/). Architecture and the refactor history are
in [`notes/REFACTOR_TARGET.md`](../../notes/REFACTOR_TARGET.md) and
[`notes/REFACTOR_PHASES.md`](../../notes/REFACTOR_PHASES.md).

## Install

See [Getting started](../../README.md#getting-started) for venv setup and the
`pydnx` build-backend note. In short:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -e .
```

## 30-second tour

```python
from llminspector import evaluate, reporting
from llminspector.config import AzureSettings
from llminspector.dataset import EvaluationDataset
from llminspector.metrics import FaithfulnessMetric
from llminspector.models import AzureOpenAIModel
from llminspector.test_case import LLMTestCase

model = AzureOpenAIModel(AzureSettings.from_env())
dataset = EvaluationDataset(test_cases=[
    LLMTestCase(input="...", actual_output="...", retrieval_context=["..."]),
])
result = evaluate(dataset, [FaithfulnessMetric(model)])
print(result.to_pandas())
```
