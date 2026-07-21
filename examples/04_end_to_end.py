"""Example 04 — end to end: synthesize -> evaluate -> export.

The synthesis + export steps run fully offline. The evaluate step needs a live
Azure model + the model dependencies, so it runs only when Azure credentials are
present in the environment; otherwise it is skipped with a message.

Run:  python examples/04_end_to_end.py
"""

import os

from llminspector import (
    AdversarialSynthesizer,
    EvaluationDataset,
    LLMTestCase,
    reporting,
)

# 1. SYNTHESIZE — build an adversarial testset from the curated bank (offline).
synth = AdversarialSynthesizer.from_excel(
    "tests/test_sample/test_adversarialdata.xlsx", capability="all", sample_size=5
)
seed_dataset = synth.generate()
print(f"Synthesized {len(seed_dataset.goldens)} adversarial goldens.")

# 2. (Your system) — produce answers for each seed prompt. Here we stub answers
#    so the flow is self-contained; in practice you'd call your LLM app.
answered = EvaluationDataset(
    test_cases=[
        LLMTestCase(input=g.input, actual_output="(stubbed answer)")
        for g in seed_dataset.goldens
    ]
)

# 3. EVALUATE — only when a live model is configured.
have_creds = all(os.getenv(k) for k in ("azure_endpoint", "api_version", "api_key"))
if have_creds:
    from llminspector import (
        AnswerJailbreakMetric,
        AzureOpenAIModel,
        Settings,
        SentimentMetric,
        evaluate,
    )

    model = AzureOpenAIModel(Settings.from_env())
    metrics = [
        SentimentMetric(model, target="actual_output"),
        AnswerJailbreakMetric(model),
    ]
    result = evaluate(answered, metrics)
    print(reporting.to_dataframe(result))
    reporting.to_excel(result, "/tmp/llminspector_e2e_eval.xlsx")
    print("Wrote /tmp/llminspector_e2e_eval.xlsx")
else:
    print("No Azure credentials found — skipping the evaluate step.")

# 4. EXPORT — the synthesized seed set (always).
synth.to_excel("/tmp/llminspector_e2e_seeds.xlsx")
print("Wrote /tmp/llminspector_e2e_seeds.xlsx")
