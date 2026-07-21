# Examples

Runnable templates for the `llminspector` package. Run from the repo root.

| File | What it shows | Needs |
|------|---------------|-------|
| [`01_dataset_roundtrip.py`](01_dataset_roundtrip.py) | Build / load / serialize an `EvaluationDataset` | nothing |
| [`02_evaluate.py`](02_evaluate.py) | Model + metrics + `evaluate()` + reporting | live Azure model + deps |
| [`03_synthesize.py`](03_synthesize.py) | Alignment / adversarial / RAG synthesis | adversarial: nothing |
| [`04_end_to_end.py`](04_end_to_end.py) | synthesize → evaluate → export | evaluate step: live model |
| [`getting_started.ipynb`](getting_started.ipynb) | The full tour as a notebook | see per-cell notes |

The **adversarial synthesis**, **dataset**, and **reporting-shape** paths run fully offline.
Anything that calls an LLM (most metrics) or ragas (RAG synthesis) needs a live Azure OpenAI
model configured via `Settings` and the corresponding dependencies installed.

```bash
python examples/01_dataset_roundtrip.py
python examples/04_end_to_end.py          # runs the offline parts, skips evaluate without creds
```

See the [usage guides](../docs/guides/) for a deeper reference on each module.
