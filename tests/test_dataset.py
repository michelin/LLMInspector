"""Phase 1 exit-criteria tests for ``EvaluationDataset``.

Covers the required round-trip of a seed ``.xlsx`` into an
``EvaluationDataset`` and back, plus pandas round-tripping of full test cases
(including list-valued ``retrieval_context``) and column mapping.

The seed workbook is built in-test rather than committed as a binary: it
reproduces the shape of the golden sample that used to live under
``example/Data/`` (a ``UserInput`` / ``Expected_Result`` pair, tag
placeholders intact, no context column).
"""

import pandas as pd
import pytest

from llminspector.dataset import EvaluationDataset, Golden
from llminspector.test_case import LLMTestCase

# 20 rows, mirroring the retired example/Data/Golden_data_sample.xlsx.
_GOLDEN_SEED_ROWS = [
    ("{greeting}, how are you?", "Hey, welcome! How can I help you today?"),
    ("What are your opening hours?", "We are open Monday to Friday, 9am to 6pm."),
    (
        "How do I reset my password?",
        "Use the 'Forgot password' link on the sign-in page.",
    ),
    (
        "Where can I find my invoice?",
        "Invoices are under Account > Billing > Invoices.",
    ),
    ("Can I change my delivery address?", "Yes, until the order has been dispatched."),
    (
        "{greeting}, I need help with an order.",
        "Of course — could you share the order number?",
    ),
    ("Do you ship internationally?", "We ship to most countries in the EU and the UK."),
    ("What is your return policy?", "Returns are accepted within 30 days of delivery."),
    (
        "How long does delivery take?",
        "Standard delivery arrives in 3 to 5 working days.",
    ),
    (
        "Is there a warranty on tyres?",
        "All tyres carry a two-year manufacturer warranty.",
    ),
    (
        "How do I cancel my subscription?",
        "Go to Account > Subscription and select Cancel.",
    ),
    ("Can I speak to a human agent?", "Sure — I can transfer you to an agent now."),
    (
        "What payment methods do you accept?",
        "We accept major cards, PayPal, and bank transfer.",
    ),
    (
        "My package has not arrived.",
        "I'm sorry — let's track it with your order number.",
    ),
    ("Do you offer a student discount?", "Yes, 10% off with a valid student ID."),
    (
        "How do I update my email address?",
        "Change it under Account > Profile > Contact details.",
    ),
    ("{greeting}, thanks for your help!", "Happy to help — have a great day!"),
    (
        "Are your tyres suitable for winter?",
        "Our winter range is certified for cold conditions.",
    ),
    ("Can I get a VAT receipt?", "Yes, VAT receipts are downloadable from Billing."),
    (
        "How do I contact support by phone?",
        "Call 0800 123 456, Monday to Friday, 9am to 6pm.",
    ),
]


@pytest.fixture
def sample_xlsx(tmp_path):
    """Write the seed golden workbook and return its path."""
    path = tmp_path / "Golden_data_sample.xlsx"
    pd.DataFrame(_GOLDEN_SEED_ROWS, columns=["UserInput", "Expected_Result"]).to_excel(
        path, index=False
    )
    return path


# -- exit criterion: sample .xlsx -> dataset -> back ------------------------


def test_golden_sample_xlsx_roundtrip(sample_xlsx, tmp_path):
    ds = EvaluationDataset.goldens_from_excel(
        str(sample_xlsx),
        input_col="UserInput",
        expected_output_col="Expected_Result",
        context_col="__none__",  # column absent -> context stays None
    )
    assert len(ds.goldens) == 20
    assert all(isinstance(g, Golden) for g in ds.goldens)
    assert ds.goldens[0].input == "{greeting}, how are you?"
    assert ds.goldens[0].expected_output == "Hey, welcome! How can I help you today?"
    assert ds.goldens[0].context is None

    # write back out, reload, and confirm the seed columns survive
    out = tmp_path / "roundtrip.xlsx"
    ds.goldens_to_excel(
        str(out),
        input_col="UserInput",
        expected_output_col="Expected_Result",
        context_col="context",
    )
    reloaded = EvaluationDataset.goldens_from_excel(
        str(out),
        input_col="UserInput",
        expected_output_col="Expected_Result",
        context_col="context",
    )
    assert len(reloaded.goldens) == len(ds.goldens)
    assert [g.input for g in reloaded.goldens] == [g.input for g in ds.goldens]
    assert [g.expected_output for g in reloaded.goldens] == [
        g.expected_output for g in ds.goldens
    ]


# -- test-case pandas round-trip incl. list contexts -----------------------


def _sample_frame():
    return pd.DataFrame(
        {
            "question": ["q1", "q2", "q3"],
            "answer": ["a1", "a2", None],
            "ground_truth": ["gt1", None, "gt3"],
            "contexts": [["c1", "c2"], "single ctx", None],
            "policy": ["p1", None, "p3"],
        }
    )


def test_from_pandas_default_column_mapping():
    ds = EvaluationDataset.from_pandas(_sample_frame())
    assert len(ds) == 3
    assert all(isinstance(tc, LLMTestCase) for tc in ds.test_cases)

    first = ds.test_cases[0]
    assert first.input == "q1"
    assert first.actual_output == "a1"
    assert first.expected_output == "gt1"
    assert first.retrieval_context == ["c1", "c2"]
    assert first.policy == "p1"

    # single-string context cell coerces to a one-element list
    assert ds.test_cases[1].retrieval_context == ["single ctx"]
    # missing values become None
    assert ds.test_cases[2].actual_output is None
    assert ds.test_cases[2].retrieval_context is None


def test_pandas_roundtrip_preserves_test_cases():
    ds = EvaluationDataset.from_pandas(_sample_frame())
    df = ds.to_pandas()
    assert list(df.columns) == [
        "question",
        "answer",
        "ground_truth",
        "contexts",
        "policy",
    ]
    ds2 = EvaluationDataset.from_pandas(df)
    assert [tc.model_dump() for tc in ds2.test_cases] == [
        tc.model_dump() for tc in ds.test_cases
    ]


def test_excel_roundtrip_preserves_list_contexts(tmp_path):
    ds = EvaluationDataset.from_pandas(_sample_frame())
    out = tmp_path / "tc.xlsx"
    ds.to_excel(str(out))
    ds2 = EvaluationDataset.from_excel(str(out))
    assert [tc.model_dump() for tc in ds2.test_cases] == [
        tc.model_dump() for tc in ds.test_cases
    ]


def test_custom_column_mapping():
    df = pd.DataFrame({"prompt": ["hi"], "reply": ["hello"]})
    ds = EvaluationDataset.from_pandas(
        df, input_col="prompt", actual_output_col="reply"
    )
    assert ds.test_cases[0].input == "hi"
    assert ds.test_cases[0].actual_output == "hello"


def test_unknown_column_argument_raises():
    with pytest.raises(TypeError):
        EvaluationDataset.from_pandas(_sample_frame(), bogus_col="x")


def test_rows_without_input_are_skipped():
    df = pd.DataFrame({"question": ["q1", None, "  "], "answer": ["a", "b", "c"]})
    ds = EvaluationDataset.from_pandas(df)
    assert len(ds) == 1
    assert ds.test_cases[0].input == "q1"
