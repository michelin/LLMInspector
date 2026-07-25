# Examples

Runnable notebooks for the `llminspector` package. Open them from this directory
(`jupyter lab examples/`) — the relative paths to `../tests/test_sample/` assume it.

| Notebook | What it shows | Needs |
|----------|---------------|-------|
| [`01_dataset_roundtrip.ipynb`](01_dataset_roundtrip.ipynb) | Build / load / serialize an `EvaluationDataset` | nothing |
| [`02_evaluate.ipynb`](02_evaluate.ipynb) | Model + metrics + `a_evaluate()` + reporting | live Azure model + deps |
| [`03_synthesize.ipynb`](03_synthesize.ipynb) | Alignment / adversarial / RAG synthesis | adversarial: nothing |
| [`04_end_to_end.ipynb`](04_end_to_end.ipynb) | synthesize → evaluate → export | evaluate step: live model |
| [`getting_started.ipynb`](getting_started.ipynb) | The full tour in one notebook | see per-cell notes |

The **dataset**, **adversarial synthesis**, and **export** paths run fully offline.
Anything that calls an LLM (most metrics) or ragas (RAG synthesis) needs a live Azure OpenAI
model configured via `AzureSettings` and the corresponding dependencies installed:

```
LLMINSPECTOR_AZURE_ENDPOINT
LLMINSPECTOR_API_VERSION
LLMINSPECTOR_API_KEY
```

## `await a_evaluate(...)`, not `evaluate(...)`

Jupyter already runs an event loop, and the synchronous `evaluate()` calls `asyncio.run()` — which
raises inside one. The notebooks use the async entry point:

```python
from llminspector import a_evaluate

result = await a_evaluate(dataset, metrics)
```

`evaluate()` is the form for scripts, and raises a directed error if you call it from a loop.

## Imports follow the package hierarchy

Each layer owns its own names; only the evaluate entry points live at the root:

```python
from llminspector import a_evaluate, reporting
from llminspector.test_case import LLMTestCase
from llminspector.dataset import EvaluationDataset
from llminspector.config import AzureSettings
from llminspector.models import AzureOpenAIModel
from llminspector.metrics import FaithfulnessMetric
from llminspector.synthesizer import AdversarialSynthesizer
```

See the [usage guides](../docs/guides/) for a deeper reference on each module.
