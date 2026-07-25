# Reporting

## Serializing a result

An `EvaluationResult` (from [`evaluate()`](04_evaluate.md)) knows how to serialize itself:

```python
df = result.to_pandas()          # source columns + metric columns
result.to_excel("eval.xlsx")
```

> Phase 8.6 removed `reporting.to_dataframe` / `reporting.to_excel`. They forwarded to these two
> methods and added nothing, so every export had two doors. Call the methods on the result.

### Naming the source columns

The first five columns default to `question` / `answer` / `ground_truth` / `contexts` / `policy`.
Override with `ResultColumns` — the **write** schema, deliberately separate from the dataset's
`ColumnMapping` **read** schema, so renaming an input column no longer renames an output one:

```python
from llminspector.evaluate import ResultColumns

result.to_pandas(columns=ResultColumns(input_col="prompt", actual_output_col="response"))

# or echo the names the dataset was read with:
result.to_pandas(columns=ResultColumns.from_column_mapping(mapping))
```

## Analysis

`reporting` holds what it adds on top of serialization:

```python
from llminspector import reporting

reporting.summary(result)   # per-numeric-metric {count, mean, min, max}
reporting.errors(result)    # DataFrame of metric failures: row | metric | error
```

`summary()` reports only numeric metric columns; label / list / dict metrics (sentiment, PII,
content moderation, …) are skipped.

`errors()` is empty for a clean run. Read the two together: a metric whose `summary()` count is
lower than the row count usually has rows in `errors()` — a `None` score means either *the metric
failed* or *a required input was missing*, and only this tells you which.

## Example

```python
result = evaluate(dataset, [FaithfulnessMetric(model), AnswerCorrectnessMetric(model)])

print(result.to_pandas())
# question | answer | ... | bert_score | faithfulness | faithfulness_reasoning | answer_correctness | ...

print(reporting.summary(result))
# {'faithfulness': {'count': 10, 'mean': 0.82, 'min': 0.4, 'max': 1.0}, 'answer_correctness': {...}}

print(reporting.errors(result))
# Empty DataFrame  ->  every None above is a skip, not a failure
```
