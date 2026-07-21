# Metrics

Metrics are **class-based**. Construct one with the model it needs, then either call it directly
or hand it to [`evaluate()`](04_evaluate.md).

```python
from llminspector import FaithfulnessMetric, LLMTestCase

metric = FaithfulnessMetric(model, threshold=0.7)
score = metric.measure(LLMTestCase(input="...", actual_output="...", retrieval_context=["..."]))
metric.score        # the value
metric.reason       # rationale (LLM-judge metrics)
metric.is_successful()   # score >= threshold, or None for non-numeric / no threshold
```

`measure()` is sync; `a_measure()` is the async form the engine drives. Each metric declares
`required_inputs` — the `LLMTestCase` attributes it needs — so `evaluate()` can skip it on rows
that lack them.

## Catalogue

| Metric | Needs | Output |
|--------|-------|--------|
| `BertScoreMetric` | actual_output, expected_output | float (local BERTScore) |
| `FaithfulnessMetric` | input, actual_output, retrieval_context | float + reason |
| `AnswerCorrectnessMetric` | input, actual_output, expected_output | float + reason |
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

## Aggregates

`calculate_overall_accuracy(results_dict)` and `calculate_total_tokens(results_dict)` operate on a
per-row results dict (not a single test case). `evaluate()` applies them for you; see
[Evaluate](04_evaluate.md).
