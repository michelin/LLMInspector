# Reporting

Turn an `EvaluationResult` (from [`evaluate()`](04_evaluate.md)) into a table, a spreadsheet, or a
numeric summary.

```python
from llminspector import reporting

df = reporting.to_dataframe(result)     # source columns + metric columns
reporting.to_excel(result, "eval.xlsx")
stats = reporting.summary(result)       # per-numeric-metric {count, mean, min, max}
```

`summary()` reports only numeric metric columns; label / list / dict metrics (sentiment, PII,
content moderation, …) are skipped.

The same methods exist directly on the result object (`result.to_pandas()`, `result.to_excel()`)
— the `reporting` functions are the stable public entry point that wraps them.

## Example

```python
result = evaluate(dataset, [FaithfulnessMetric(model), AnswerCorrectnessMetric(model)])

print(reporting.to_dataframe(result))
# question | answer | ... | bert_score | faithfulness | faithfulness_reasoning | answer_correctness | ...

print(reporting.summary(result))
# {'faithfulness': {'count': 10, 'mean': 0.82, 'min': 0.4, 'max': 1.0}, 'answer_correctness': {...}}
```
