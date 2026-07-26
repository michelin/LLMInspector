# Metrics

Metrics are **class-based**. Construct one with the model it needs, then either call it directly
or hand it to [`evaluate()`](04_evaluate.md).

```python
from llminspector.metrics import FaithfulnessMetric
from llminspector.test_case import LLMTestCase

metric = FaithfulnessMetric(model, threshold=0.7)
score = metric.measure(LLMTestCase(input="...", actual_output="...", retrieval_context=["..."]))
metric.score        # the value
metric.reason       # rationale (LLM-judge metrics)
metric.is_successful()   # score >= threshold, or None for non-numeric / no threshold
```

`measure()` is sync; `a_measure()` is the async form the engine drives. Each metric declares
`required_inputs` — the `LLMTestCase` attributes it needs — so `evaluate()` can skip it on rows
that lack them.

## Thresholds and the `_success` column

`threshold` defaults to `None` on every metric, so pass/fail is off until you ask for it. Set one
and `evaluate()` exports a `{metric_name}_success` column alongside the score:

```python
evaluate(dataset, [FaithfulnessMetric(model, threshold=0.7)])
# -> columns: faithfulness, faithfulness_reasoning, faithfulness_success
```

Metrics **without** a threshold contribute no `_success` column at all — otherwise a default run
would carry twenty-odd all-blank columns. The value is blank even with a threshold when the score
is not numeric (sentiment labels, moderation flag dicts), and on rows where the metric was skipped
or failed.

## ragas-backed metrics

The five `Context*` metrics subclass `RagasBackedMetric` and need both the `llminspector[ragas]`
extra and a provider exposing `ragas_llm()`. They are the only metrics that do; every other one
runs on a bare `BaseLLM`. Without the extra they score `None` and record an `ImportError` naming
it on `EvaluationResult.errors`, rather than aborting the run.

## Catalogue

| Metric | Needs | Output |
|--------|-------|--------|
| `BertScoreMetric` | actual_output, expected_output | float (local BERTScore) |
| `FaithfulnessMetric` | input, actual_output, retrieval_context | float + reason |
| `AnswerCorrectnessMetric` | input, actual_output, expected_output (context optional) | float + reason + 3 sub-scores |
| `AnswerRelevancyMetric` | input, actual_output | float + reason |
| `ConcisenessMetric` | input, actual_output | float + reason |
| `ContextPrecisionMetric` | input, expected_output, retrieval_context | float (ragas) |
| `ContextRecallMetric` | input, actual_output, expected_output, retrieval_context | float (ragas) |
| `ContextUtilisationMetric` | input, actual_output, retrieval_context | float (ragas) |
| `ContextRelevanceMetric` | input, retrieval_context | float (ragas) |
| `ContextEntityRecallMetric` | expected_output, retrieval_context | float (ragas) |
| `PIIDetectionMetric` | one text field | list of PII entity types (presidio) |
| `ContentModerationMetric` | one text field | per-category flag dict |
| `QuestionJailbreakMetric` | input | 0/1 |
| `AnswerJailbreakMetric` | actual_output | 0/1 |
| `RefusalMetric` | input, actual_output | 0/1 |
| `HallucinationMetric` | retrieval_context, actual_output | 0/1 |
| `CodeDetectMetric` | one text field | `{code_detected, code_language}` |
| `SentimentMetric` | one text field | label |
| `EmotionMetric` | one text field | label |
| `LanguageDetectionMetric` | one text field | language name |
| `ReadabilityMetric` | one text field | Flesch-Kincaid grade |
| `TokenCountMetric` | one text field | int (tiktoken) |
| `PolicyComplianceMetric` | input, actual_output, policy | `{is_policy_violated, policy_violation_reason}` |

### Dual (question/answer) metrics

`Sentiment`, `Emotion`, `LanguageDetection`, `Readability`, `TokenCount`, `PIIDetection`,
`ContentModeration`, and `CodeDetect` run on either the question or the answer. Pick with
`target`:

```python
SentimentMetric(model, target="input")           # -> name "question_sentiment"
SentimentMetric(model, target="actual_output")    # -> name "answer_sentiment"
```

Local-only metrics (`BertScore`, `PIIDetection`, `Readability`, `TokenCount`,
`LanguageDetection`) don't need a model — construct them with no arguments.

## A metric owns its output columns

`evaluate()` contains no metric names. Each metric declares what it contributes to the exported
table:

| attribute | meaning | default |
|---|---|---|
| `expand(score)` | score → `{column: value}` | `{name: score}` |
| `output_columns` | every column it owns, in order | derived from `expand`, with `{name}_reasoning` behind the headline |
| `sort_key` | where its block sits in the table (lower first) | `1000` |
| `sort_key_by_target` | per-target override for dual metrics | `{}` |

Metrics whose score is structured override `expand`: `ContentModerationMetric` returns its nine
category columns, `CodeDetectMetric` two, `PolicyComplianceMetric` two, `AnswerCorrectnessMetric`
its headline score plus three sub-judgements. `expand` must return the **same key set for every
score including `None`** — the engine calls `expand(None)` to reserve columns on rows where the
metric is skipped, which is what keeps the header stable across rows.

To add a metric, subclass `BaseMetric`, set `metric_name` / `required_inputs` / `sort_key`, and
implement `a_measure`. No other file needs to change.

## Aggregates

`calculate_total_tokens(results_dict)` operates on a per-row results dict (not a single test case).
`evaluate()` applies it for you; see [Evaluate](04_evaluate.md).

`calculate_overall_accuracy` has been removed. Its 0.5/0.3/0.2 blend now lives inside
`AnswerCorrectnessMetric`'s prompt as a single holistic judgement — see
[Evaluate](04_evaluate.md#answer_correctness-is-a-single-unified-judge).
