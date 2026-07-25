# Evaluate

`evaluate()` runs a list of metric objects over a dataset and returns an `EvaluationResult`.

```python
from llminspector import evaluate

result = evaluate(dataset, metrics, show_progress=True)
result.rows          # list[dict] — one ordered metric dict per test case
result.errors        # metric failures, if any
result.to_pandas()   # source columns + metric columns
result.to_excel("out.xlsx")
```

## In a notebook or a server: `a_evaluate`

The engine is async all the way down. `evaluate()` calls `asyncio.run()`, which **raises** inside
an already-running event loop — Jupyter, FastAPI, anything with a loop. Await the async form there:

```python
from llminspector import a_evaluate

result = await a_evaluate(dataset, metrics)
```

`evaluate()` is a thin wrapper around it and raises a directed `RuntimeError` pointing here if you
call it from inside a loop.

## When metrics fail

A metric that raises is isolated: it scores `None` and the run continues. `None` in the table is
ambiguous on its own — it also means *skipped because a required input was missing* — so failures
are recorded separately:

```python
result.errors          # [{"row": 0, "error": "faithfulness: AuthenticationError: 401 ..."}, ...]
result.error_summary() # {"faithfulness": 20}  -> every row failed, not "all scores were None"
```

Failures also go to the standard `logging` module (logger `llminspector.*` at `WARNING`), not to
stdout. A run where every call returned 401 is now obvious instead of looking like a clean table of
empty results.

## Rate limits

Provider calls retry HTTP 429 with exponential backoff and full jitter, honouring a `Retry-After`
header when the server sends one. Nothing else is retried. Tune with
`AzureOpenAIModel(..., max_retries=5)`; `max_retries=0` disables it.

## How it works

- **Per row**, the metrics whose `required_inputs` are all present are run concurrently
  (`asyncio.gather`); rows are processed in batches of `batch_size`.
- **Availability filtering** — a metric missing an input on a given row is skipped there (its
  column is left `None`), not errored.
- **Result order is preserved** — rows come back in input order even though they complete
  concurrently.
- **Column expansion** — a metric declares the columns it owns via `expand()`: code detection →
  `*_code_detected` / `*_code_language`; content moderation → per-category columns; policy →
  `is_policy_violated` / `policy_violation_reason`.
- **Aggregates** — `total_tokens` is summed.
- **No hidden metrics** — `evaluate()` runs exactly the metrics you hand it. Nothing is auto-added
  and no column is overwritten after the fact.
- Columns are ordered by a stable, legacy-compatible key order.

## answer_correctness is a single unified judge

`AnswerCorrectnessMetric` takes **question + answer + ground truth + retrieval context** and judges
ground-truth agreement, faithfulness to the context, and relevancy to the question *inside one
prompt*. The value in the `answer_correctness` column is that judge's score — nothing rescales it.

The reason it is one prompt rather than a blend of three separate metrics:

> Content in the answer that is **absent from the ground truth** is **not** penalised when it is
> (a) supported by the retrieval context and (b) relevant to the question.

A blend cannot express that rule, because agreement-with-ground-truth is computed before it can know
whether the "extra" content was vouched for by the context. A RAG system that correctly surfaces a
true, relevant fact the golden answer happens to omit used to score as if it had hallucinated.

`retrieval_context` is **optional** and never gates availability — rows without one are still
judged, with one factor fewer:

| | context present (RAG) | context absent (plain LLM) |
|---|---|---|
| **factors** | 3 — GT agreement, faithfulness, relevancy | 2 — GT coverage, relevancy |
| **emphasis** | GT agreement + faithfulness carry the score; relevancy modifies | GT coverage dominates; relevancy carries very low weight |
| **don't-penalise rule** | active | inactive — no context to vouch for extra content |

Alongside the headline score the judge emits its three sub-judgements, so a low score is
diagnosable without a second run:

```
answer_correctness  answer_correctness_gt_agreement  answer_correctness_faithfulness  answer_correctness_relevancy
```

`answer_correctness_faithfulness` is `None` on rows with no retrieval context — faithfulness is
undefined when there is nothing to be faithful to.

`FaithfulnessMetric` and `AnswerRelevancyMetric` remain available as standalone metrics; they are
simply no longer added behind your back.

## Metrics carry their own model

You pass **instantiated** metric objects, each already built with its model:

```python
metrics = [
    FaithfulnessMetric(model),
    AnswerCorrectnessMetric(model),
    BertScoreMetric(),                       # local-only, no model
    SentimentMetric(model, target="actual_output"),
]
result = evaluate(dataset, metrics)
```
