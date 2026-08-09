# Datasets & Test Cases

The schema layer. Three types:

- **`LLMTestCase`** — one evaluable row: a prompt plus (optionally) the model's answer, a
  reference answer, retrieval context, and a policy.
- **`Golden`** — a *seed* row for synthesis (no `actual_output`; the answer is what synthesis /
  your app produces). Carries a free-form `metadata` dict.
- **`EvaluationDataset`** — a container of test cases and/or goldens with pandas / Excel I/O.

## LLMTestCase

```python
from llminspector.test_case import LLMTestCase

tc = LLMTestCase(
    input="What is the capital of France?",   # required (non-empty)
    actual_output="Paris is the capital.",     # optional
    expected_output="Paris",                   # optional (ground truth)
    retrieval_context=["France's capital is Paris."],  # optional list[str]
    policy="No financial advice.",             # optional
)
```

Field names map onto the five spreadsheet columns:

| LLMTestCase | Column |
|-------------|--------|
| `input` | question |
| `actual_output` | answer |
| `expected_output` | ground_truth |
| `retrieval_context` | contexts |
| `policy` | policy |

`retrieval_context` accepts a single string or an iterable of strings; blanks are dropped. Only
`input` is required — every metric that needs more filters itself out when its inputs are missing.

Two further fields exist for tracing a row back to where it came from:
`golden_id` (the `Golden.id` it was promoted from, else `None`) and `metadata`
(copied from that golden). Neither is exported — `to_pandas` and
`EvaluationResult.to_pandas` name their columns explicitly, so result tables stay
the width they have always been.

## EvaluationDataset

```python
from llminspector.dataset import EvaluationDataset

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

For generation seeds, use the golden variants: `goldens_from_excel` / `goldens_from_pandas` /
`goldens_to_pandas` / `goldens_to_excel`.

## Golden

```python
from llminspector.dataset import Golden

g = Golden(input="...", expected_output="...", context=["..."], metadata={"capability": "toxicity"})
g.id           # '3f9a…' — minted automatically, stable across a model_copy
```

`metadata` is where generators stash their extra columns (`augmentation_type`, `Capability`,
`synthesizer_name`, lineage, quality scores, …) so every generator emits a uniform shape — see
[Synthesizers](05_synthesizers.md).

### Metadata survives the round trip

`goldens_to_pandas` / `goldens_to_excel` write `id` and one column per metadata
key (the union across all goldens, in first-seen order). `goldens_from_pandas` /
`goldens_from_excel` read them back: any column that is not one of the four
mapped core fields (`id` / `question` / `ground_truth` / `contexts`) is collected
into `metadata`. Blank cells are dropped rather than stored as `None`, so a
golden never inherits another golden's keys.

Ids round-trip too. Without that, every reload would mint fresh ids and sever the
`golden_id` link on anything promoted from those goldens.

### Promoting goldens to test cases

A golden becomes evaluable once there is an answer to score:

```python
tc = g.to_test_case(actual_output="Paris is the capital.")
tc.golden_id == g.id        # True — results stay traceable to their seed

# Or for a whole dataset, positionally aligned with `dataset.goldens`:
cases = dataset.to_test_cases(answers=[...], policies=[...])
scored = EvaluationDataset(test_cases=cases)
```

`context` becomes `retrieval_context` — the same passages, named for what they
are on each side of the pipeline. `policy` is supplied at promotion time because
it belongs to the evaluation, not to the seed. A mismatched `answers` length
raises rather than zipping to the shorter sequence, which would silently attach
answers to the wrong questions.
