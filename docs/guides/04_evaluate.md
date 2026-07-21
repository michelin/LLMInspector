# Evaluate

`evaluate()` runs a list of metric objects over a dataset and returns an `EvaluationResult`.

```python
from llminspector import evaluate

result = evaluate(dataset, metrics, batch_size=5, show_progress=True)
result.rows          # list[dict] — one ordered metric dict per test case
result.to_pandas()   # source columns + metric columns
result.to_excel("out.xlsx")
```

## How it works

- **Per row**, the metrics whose `required_inputs` are all present are run concurrently
  (`asyncio.gather`); rows are processed in batches of `batch_size`.
- **Availability filtering** — a metric missing an input on a given row is skipped there (its
  column is left `None`), not errored.
- **Result order is preserved** — rows come back in input order even though they complete
  concurrently.
- **Column expansion** — dict-valued metrics fan out into columns: code detection →
  `*_code_detected` / `*_code_language`; content moderation → per-category columns; policy →
  `is_policy_violated` / `policy_violation_reason`.
- **Aggregates** — `total_tokens` is summed; the reported `answer_correctness` is the weighted
  `overall_accuracy` blend (see below).
- Columns are ordered by a stable, legacy-compatible key order.

## answer_correctness = overall_accuracy

When you include `AnswerCorrectnessMetric`, the reported `answer_correctness` value is a weighted
blend, and `evaluate()` **auto-adds** its dependencies to compute it:

- 3 signals present: `0.5*correctness + 0.3*faithfulness + 0.2*relevancy`
- correctness + relevancy only: `0.75*correctness + 0.25*relevancy`

`faithfulness` / `answer_relevancy` are pulled in automatically if you didn't request them, then
dropped from the output. (This blend is a documented stopgap slated to move into the
answer-correctness judge prompt — see REFACTOR_PHASES.md "Future work".)

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
