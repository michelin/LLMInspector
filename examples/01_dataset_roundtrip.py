"""Example 01 — build an EvaluationDataset and round-trip it.

Shows the Phase 1 schema layer: constructing test cases in code, loading them
from a spreadsheet with column mapping, and serializing back to pandas / Excel.

Run:  python examples/01_dataset_roundtrip.py
No credentials or heavy dependencies required.
"""

from llminspector import EvaluationDataset, LLMTestCase

# 1. Construct test cases directly in code.
test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
        expected_output="Paris",
        retrieval_context=["France is a country in Europe. Its capital is Paris."],
    ),
    LLMTestCase(
        input="Summarize the refund policy.",
        actual_output="Refunds are available within 30 days.",
        expected_output="Customers may request a refund within 30 days of purchase.",
    ),
]
dataset = EvaluationDataset(test_cases=test_cases)
print("Constructed:", dataset)

# 2. Serialize to a DataFrame (columns map back to the legacy names by default:
#    question / answer / ground_truth / contexts / policy).
df = dataset.to_pandas()
print("\nAs DataFrame:\n", df[["question", "answer", "ground_truth"]])

# 3. Round-trip through Excel.
dataset.to_excel("/tmp/llminspector_dataset.xlsx")
reloaded = EvaluationDataset.from_excel("/tmp/llminspector_dataset.xlsx")
print("\nReloaded:", reloaded)

# 4. Load with a custom column mapping (if your sheet uses different headers).
#    reloaded = EvaluationDataset.from_excel(
#        "my_sheet.xlsx", input_col="prompt", actual_output_col="response",
#    )
