# Datasets & Test Cases

The schema layer. Three types:

- **`LLMTestCase`** — one evaluable row: a prompt plus (optionally) the model's answer, a
  reference answer, retrieval context, and a policy.
- **`Golden`** — a *seed* row for synthesis (no `actual_output`; the answer is what synthesis /
  your app produces). Carries a free-form `metadata` dict.
- **`EvaluationDataset`** — a container of test cases and/or goldens with pandas / Excel I/O.

## LLMTestCase

```python
from llminspector import LLMTestCase

tc = LLMTestCase(
    input="What is the capital of France?",   # required (non-empty)
    actual_output="Paris is the capital.",     # optional
    expected_output="Paris",                   # optional (ground truth)
    retrieval_context=["France's capital is Paris."],  # optional list[str]
    policy="No financial advice.",             # optional
)
```

Field names are DeepEval-shaped and map onto the legacy five inputs:

| LLMTestCase | Legacy |
|-------------|--------|
| `input` | question |
| `actual_output` | answer |
| `expected_output` | ground_truth |
| `retrieval_context` | contexts |
| `policy` | policy |

`retrieval_context` accepts a single string or an iterable of strings; blanks are dropped. Only
`input` is required — every metric that needs more filters itself out when its inputs are missing.

## EvaluationDataset

```python
from llminspector import EvaluationDataset

# From a spreadsheet (default column names: question/answer/ground_truth/contexts/policy)
dataset = EvaluationDataset.from_excel("data.xlsx")

# Custom headers via per-column overrides:
dataset = EvaluationDataset.from_excel(
    "data.xlsx", input_col="prompt", actual_output_col="response",
)

# Or from a DataFrame:
dataset = EvaluationDataset.from_pandas(df)

# Back out:
df = dataset.to_pandas()
dataset.to_excel("out.xlsx")
```

For synthesis seeds, use the golden variants: `goldens_from_excel` / `goldens_from_pandas` /
`goldens_to_pandas` / `goldens_to_excel`.

## Golden

```python
from llminspector import Golden

g = Golden(input="...", expected_output="...", context=["..."], metadata={"capability": "toxicity"})
```

`metadata` is where synthesizers stash their extra columns (`augmentation_type`, `Capability`,
`synthesizer_name`, …) so every synthesizer emits a uniform shape — see
[Synthesizers](05_synthesizers.md).
