"""Example 02 — evaluate a dataset with metrics.

Shows the Phase 2 model layer + Phase 3 metrics + Phase 4 evaluate engine wired
together. The metrics that call an LLM / ragas need a live Azure OpenAI model,
so this script builds one from environment variables via ``Settings.from_env()``.

Environment variables (see Settings.from_env):
    azure_endpoint, api_version, api_key   (or use azure_ad_token_provider in code)

Run:  python examples/02_evaluate.py
Requires: live Azure creds + the model dependencies (langchain/ragas/...).
"""

from llminspector import (
    AnswerCorrectnessMetric,
    AzureOpenAIModel,
    BertScoreMetric,
    EvaluationDataset,
    FaithfulnessMetric,
    LLMTestCase,
    PIIDetectionMetric,
    Settings,
    SentimentMetric,
    evaluate,
    reporting,
)

# 1. Configure the provider. In code you can pass api_key OR an
#    azure_ad_token_provider explicitly instead of reading the environment.
settings = Settings.from_env()
model = AzureOpenAIModel(settings)  # picks up settings.api_key

# 2. Build the metric set. Each metric is an object constructed with the model.
#    Local-only metrics (BERTScore, PII) don't need the model.
metrics = [
    FaithfulnessMetric(model),
    AnswerCorrectnessMetric(model),  # reported as the overall_accuracy blend
    SentimentMetric(model, target="actual_output"),
    BertScoreMetric(),
    PIIDetectionMetric(target="actual_output"),
]

# 3. The dataset under test (here inline; normally EvaluationDataset.from_excel).
dataset = EvaluationDataset(
    test_cases=[
        LLMTestCase(
            input="What is the capital of France?",
            actual_output="Paris is the capital of France.",
            expected_output="Paris",
            retrieval_context=["The capital of France is Paris."],
        ),
    ]
)

# 4. Evaluate. Metrics whose inputs are missing on a row are skipped for that row.
result = evaluate(dataset, metrics, batch_size=5)

# 5. Report.
print(reporting.to_dataframe(result).T)
print("\nNumeric summary:", reporting.summary(result))
reporting.to_excel(result, "/tmp/llminspector_eval.xlsx")
