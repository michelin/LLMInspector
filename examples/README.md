# Examples

Runnable notebooks for the `llminspector` package. Open them from this directory
(`jupyter lab examples/`) — the relative paths to `../tests/test_sample/` assume it.

| Notebook | What it shows | Needs |
|----------|---------------|-------|
| [`01_dataset_roundtrip.ipynb`](01_dataset_roundtrip.ipynb) | Build / load / serialize an `EvaluationDataset` | nothing |
| [`02_evaluate.ipynb`](02_evaluate.ipynb) | Model + metrics + `a_evaluate()` + reporting | live Azure model + deps |
| [`03_synthesize.ipynb`](03_synthesize.ipynb) | Generation: sources, stages, adversarial | adversarial: nothing |
| [`04_end_to_end.ipynb`](04_end_to_end.ipynb) | generate → evaluate → export | evaluate step: live model |
| [`05_dataset_creation.ipynb`](05_dataset_creation.ipynb) | **Every way to build a dataset**, with Azure OpenAI | ways 1–2: nothing; 3–6: live model |
| [`getting_started.ipynb`](getting_started.ipynb) | The full tour in one notebook | see per-cell notes |

The **dataset**, **adversarial generation**, and **export** paths run fully offline.
Anything that calls an LLM — most metrics, and every generation source except the
adversarial bank — needs a live Azure OpenAI model configured via `AzureSettings`:

```
LLMINSPECTOR_AZURE_ENDPOINT
LLMINSPECTOR_API_VERSION
LLMINSPECTOR_API_KEY
LLMINSPECTOR_EMBEDDING_DEPLOYMENT     # document-backed generation only
```

Generating from a document corpus also wants the optional extra
`llminspector[documents]` for PDF and DOCX; `.txt` / `.md` / `.mdx` are core.

## `await` the async entry points

Jupyter already runs an event loop, and the synchronous `evaluate()` calls `asyncio.run()` — which
raises inside one. The notebooks use the async entry point:

```python
from llminspector import a_evaluate

result = await a_evaluate(dataset, metrics)
```

The same applies to generation, which is async-first for the same reason:

```python
result = await gen.a_generate()
```

`evaluate()` and `Generator.generate()` are the forms for scripts, and raise a directed error if
you call them from a running loop.

## Imports follow the package hierarchy

Each layer owns its own names; only the evaluate entry points live at the root:

```python
from llminspector import a_evaluate, reporting
from llminspector.test_case import LLMTestCase
from llminspector.dataset import EvaluationDataset
from llminspector.config import AzureSettings
from llminspector.models import AzureOpenAIModel
from llminspector.metrics import FaithfulnessMetric
from llminspector.generation import AdversarialGenerator, DocumentSource, Generator, default_stages
```

See the [usage guides](../docs/guides/) for a deeper reference on each module.
