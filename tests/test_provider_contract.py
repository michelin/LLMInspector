"""Phase 8.3 — the provider abstraction is the real contract.

Metrics used to reach through ``model.client`` to a raw langchain chat model and
build a ``prompt | llm`` chain, so the contract a provider actually had to
satisfy was "expose a langchain ``BaseChatModel`` plus a ragas wrapper" — not
``BaseLLM``. ``BaseLLM.a_generate`` was defined and used by nobody.

These tests pin the fix: a stub implementing **only** ``BaseLLM`` — no langchain,
no ragas, no ``.client`` — drives every non-context metric end to end.

The guarantee is enforced two ways that hold regardless of what happens to be
installed: ``test_metric_layer_imports_no_langchain_and_no_ragas`` reads the
metric modules' ASTs, and ``test_ragas_is_an_optional_extra`` reads the declared
dependencies. An earlier version asserted that langchain and ragas were simply
absent from the interpreter, which could never hold once they were installed —
and one of them is a core dependency by design.
"""

import pathlib
import sys
import tomllib

import pytest

from llminspector.metrics import (
    AnswerCorrectnessMetric,
    AnswerJailbreakMetric,
    AnswerRelevancyMetric,
    CodeDetectMetric,
    ConcisenessMetric,
    ContentModerationMetric,
    ContextPrecisionMetric,
    ContextRelevanceMetric,
    EmotionMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    PolicyComplianceMetric,
    QuestionJailbreakMetric,
    RefusalMetric,
    SentimentMetric,
)
from llminspector.metrics.base_metric import BaseMetric, RagasBackedMetric
from llminspector.models.base_model import BaseEmbeddingModel, BaseLLM
from llminspector.test_case import LLMTestCase


class StubLLM(BaseLLM):
    """A provider implementing only the required BaseLLM contract.

    No ``.client``, no ``ragas_llm`` override, no third-party imports.
    """

    def __init__(self, response: str = "") -> None:
        self.response = response
        self.prompts: list = []

    def get_model_name(self) -> str:
        return "stub"

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response

    async def a_generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response


def _tc(**kwargs):
    kwargs.setdefault("input", "What are the API rate limits?")
    kwargs.setdefault("actual_output", "500 requests per minute.")
    return LLMTestCase(**kwargs)


# --------------------------------------------------------------------------- #
# the packaging contract itself is part of the assertion
# --------------------------------------------------------------------------- #


def _pyproject() -> dict:
    root = pathlib.Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def test_ragas_is_an_optional_extra():
    """ragas must stay opt-in, so it can be swapped out without a core change.

    Only the five ``RagasBackedMetric`` context metrics and the RAG testset
    engine need it; every other metric runs on a bare ``BaseLLM``. A core
    dependency would make that true in the code and false in the install.
    """
    project = _pyproject()["project"]
    core = " ".join(project["dependencies"])
    assert "ragas" not in core, f"ragas leaked into core dependencies: {core}"
    assert "langchain-community" not in core

    extra = " ".join(project["optional-dependencies"]["ragas"])
    assert "ragas" in extra
    # langchain-community was dropped with the ragas testset generator: its only
    # consumer was that generator's DirectoryLoader, and document loading is now
    # core plus the [documents] extra.
    assert "langchain-community" not in extra


# --------------------------------------------------------------------------- #
# every non-context metric runs on a bare BaseLLM
# --------------------------------------------------------------------------- #

_CASES = [
    (
        lambda m: FaithfulnessMetric(m),
        '{"faithfulness": 0.6, "faithfulness_reasoning": "partly grounded"}',
        dict(retrieval_context=["The API limits requests to 500 per minute."]),
        0.6,
    ),
    (
        lambda m: AnswerCorrectnessMetric(m),
        '{"answer_correctness": 1.0, "gt_agreement": 1.0, "faithfulness": 1.0, '
        '"relevancy": 1.0}',
        dict(expected_output="500/min", retrieval_context=["500 per minute."]),
        1.0,
    ),
    (
        lambda m: AnswerCorrectnessMetric(m),
        '{"answer_correctness": 0.8, "gt_agreement": 0.8, "relevancy": 1.0}',
        dict(expected_output="500/min"),  # two-factor path, no context
        0.8,
    ),
    (
        lambda m: AnswerRelevancyMetric(m),
        '{"answer_relevancy": 0.9}',
        {},
        0.9,
    ),
    (lambda m: ConcisenessMetric(m), '{"conciseness": 1.0}', {}, 1.0),
    (lambda m: SentimentMetric(m, target="actual_output"), "Neutral", {}, "Neutral"),
    (lambda m: EmotionMetric(m, target="input"), "Curiosity", {}, "Curiosity"),
    (lambda m: QuestionJailbreakMetric(m), "0", {}, 0),
    (lambda m: AnswerJailbreakMetric(m), "0", {}, 0),
    (lambda m: RefusalMetric(m), "0", {}, 0),
    (
        lambda m: HallucinationMetric(m),
        "0",
        dict(retrieval_context=["The API limits requests to 500 per minute."]),
        0,
    ),
]


@pytest.mark.parametrize(
    "factory, response, extra, expected",
    _CASES,
    ids=lambda v: getattr(v, "__name__", None) or "",
)
def test_metric_runs_on_a_bare_base_llm(factory, response, extra, expected):
    model = StubLLM(response)
    metric = factory(model)
    assert metric.measure(_tc(**extra)) == expected
    assert model.prompts, "the metric never called the provider"


def test_structured_score_metrics_run_on_a_bare_base_llm():
    code = CodeDetectMetric(
        StubLLM('{"code_detected": false, "code_language": null}'),
        target="actual_output",
    )
    assert code.measure(_tc())["code_detected"] is False

    moderation = ContentModerationMetric(
        StubLLM('{"hate_speech": 0, "fairness": 0, "profanity": 0}'),
        target="input",
    )
    assert moderation.measure(_tc())["hate_speech"] == 0

    policy = PolicyComplianceMetric(
        StubLLM('{"is_policy_violated": false, "policy_violation_reason": null}')
    )
    out = policy.measure(_tc(policy="no pii"))
    assert out["is_policy_violated"] is False


def test_metrics_never_touch_a_client_attribute():
    """The stub has no `.client`; reaching for one would AttributeError."""
    model = StubLLM('{"conciseness": 1.0}')
    assert not hasattr(model, "client")
    assert ConcisenessMetric(model).measure(_tc()) == 1.0


def test_metric_layer_imports_no_langchain_and_no_ragas():
    """Including the deferred, inside-function imports the metrics used to use."""
    import ast
    import inspect
    from importlib import import_module

    modules = [
        "llminspector.metrics.base_metric",
        "llminspector.metrics.rag",
        "llminspector.metrics.safety",
        "llminspector.metrics.policy",
        "llminspector.metrics.nlp",
        "llminspector.metrics.quality",
    ]
    forbidden = {
        "langchain",
        "langchain_core",
        "langchain_openai",
        "langchain_community",
    }
    leaked = []
    for name in modules:
        tree = ast.parse(inspect.getsource(import_module(name)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            leaked += [f"{name}: {r}" for r in roots if r in forbidden]
    assert leaked == [], f"metric layer still imports langchain: {leaked}"


# --------------------------------------------------------------------------- #
# ragas is an opt-in capability, declared in the type
# --------------------------------------------------------------------------- #


def test_only_the_five_context_metrics_are_ragas_backed():
    from llminspector import metrics as metrics_pkg

    ragas_backed = sorted(
        cls.metric_name
        for cls in vars(metrics_pkg).values()
        if isinstance(cls, type)
        and issubclass(cls, RagasBackedMetric)
        and cls is not RagasBackedMetric
    )
    assert ragas_backed == [
        "context_entity_recall",
        "context_precision",
        "context_recall",
        "context_relevance",
        "context_utilisation",
    ]


def test_ragas_llm_is_optional_on_the_base_contract():
    """A provider need not implement ragas_llm to be a valid BaseLLM."""
    model = StubLLM()
    assert isinstance(model, BaseLLM)
    with pytest.raises(NotImplementedError):
        model.ragas_llm()


def test_ragas_metric_gives_a_directed_error_for_a_non_ragas_provider():
    class NotAProvider:
        """Duck-typed model with no ragas_llm at all."""

    metric = ContextPrecisionMetric(NotAProvider())
    with pytest.raises(TypeError, match="ragas-backed"):
        _ = metric.evaluator_llm


def test_ragas_metric_degrades_to_none_rather_than_crashing_a_run():
    metric = ContextRelevanceMetric(StubLLM())
    tc = _tc(retrieval_context=["ctx"])
    assert metric.measure(tc) is None


def test_embedding_ragas_wrapper_is_also_optional():
    class StubEmbedding(BaseEmbeddingModel):
        def get_model_name(self):
            return "stub-embed"

        def embed_text(self, text):
            return [0.0]

        def embed_texts(self, texts):
            return [[0.0] for _ in texts]

    with pytest.raises(NotImplementedError):
        StubEmbedding().ragas_embeddings()


# --------------------------------------------------------------------------- #
# prompt rendering replaced langchain's PromptTemplate
# --------------------------------------------------------------------------- #


def test_render_escapes_literal_braces_like_the_f_string_template():
    metric = ConcisenessMetric()
    out = metric._render(
        'Answer: {answer}\n{{"score": <float>}}', ["answer"], {"answer": "hi"}
    )
    assert out == 'Answer: hi\n{"score": <float>}'


def test_render_merges_partial_variables():
    metric = PolicyComplianceMetric()
    out = metric._render(
        "{a}/{b}", ["a", "b"], {"a": "1"}, partial_variables={"b": "2"}
    )
    assert out == "1/2"


def test_render_reports_a_missing_variable_by_name():
    metric = ConcisenessMetric()
    with pytest.raises(KeyError, match="answer"):
        metric._render("{question} {answer}", ["question", "answer"], {"question": "q"})


def test_every_shipped_prompt_renders():
    """Each prompt must survive str.format with its declared variables."""
    from llminspector.metrics import policy, rag, safety

    prompts = {
        "FAITHFULNESS": (rag.FAITHFULNESS_PROMPT, ["question", "answer", "context"]),
        "ANSWER_CORRECTNESS": (
            rag.ANSWER_CORRECTNESS_PROMPT,
            ["question", "answer", "ground_truth", "context"],
        ),
        "ANSWER_CORRECTNESS_NO_CONTEXT": (
            rag.ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT,
            ["question", "answer", "ground_truth"],
        ),
        "ANSWER_RELEVANCY": (rag.ANSWER_RELEVANCY_PROMPT, ["question", "answer"]),
        "CONCISENESS": (rag.CONCISENESS_PROMPT, ["question", "answer"]),
        "CODE_DETECT": (safety.CODE_DETECT_PROMPT, ["text"]),
        "CONTENT_MODERATION": (safety.CONTENT_MODERATION_PROMPT, ["user_input"]),
        "QUESTION_JAILBREAK": (safety.QUESTION_JAILBREAK_PROMPT, ["user_input"]),
        "ANSWER_JAILBREAK": (safety.ANSWER_JAILBREAK_PROMPT, ["model_response"]),
        "ANSWER_NO_REFUSAL": (safety.ANSWER_NO_REFUSAL_PROMPT, ["question", "answer"]),
        "HALLUCINATION": (safety.HALLUCINATION_PROMPT, ["context", "answer"]),
        "POLICY": (
            policy.POLICY_PROMPT,
            ["policy_base", "user_policy", "question", "answer"],
        ),
    }
    metric = ConcisenessMetric()
    for name, (template, variables) in prompts.items():
        values = {v: f"<{v}>" for v in variables}
        rendered = metric._render(template, variables, values)
        for v in variables:
            assert f"<{v}>" in rendered, f"{name} dropped {v}"
        assert "{{" not in rendered and "}}" not in rendered, f"{name} left escapes"


# --------------------------------------------------------------------------- #
# ragas is absent-tolerant: the package works without the extra
# --------------------------------------------------------------------------- #


@pytest.fixture
def no_ragas(monkeypatch):
    """Make every `import ragas...` raise, as it would without the extra."""
    import builtins

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "ragas" or name.startswith("ragas."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for module in [m for m in sys.modules if m == "ragas" or m.startswith("ragas.")]:
        monkeypatch.delitem(sys.modules, module, raising=False)


def test_azure_provider_constructs_without_ragas(no_ragas, monkeypatch):
    """The ragas wrapper is built on first use, not in __init__.

    It used to be eager, so `pip install llminspector` without the extra could
    not even construct the one provider the package ships.
    """
    import types

    from llminspector.config import AzureSettings
    from llminspector.models import AzureOpenAIModel

    fake = types.ModuleType("langchain_openai")
    fake.AzureChatOpenAI = lambda **kw: types.SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "langchain_openai", fake)

    settings = AzureSettings(
        azure_endpoint="https://example.openai.azure.com/", api_version="2024-02-01"
    )
    model = AzureOpenAIModel(settings, api_key="k")
    assert isinstance(model, BaseLLM)
    assert model.client is not None

    # Only reaching for the ragas wrapper surfaces the missing extra.
    with pytest.raises(ImportError, match=r"llminspector\[ragas\]"):
        model.ragas_llm()


def test_a_missing_ragas_names_the_extra_to_install(no_ragas):
    """Not a bare ModuleNotFoundError from inside a metric."""
    metric = ContextPrecisionMetric(StubLLM())
    with pytest.raises(ImportError, match=r"pip install 'llminspector\[ragas\]'"):
        metric._sample(_tc(expected_output="gt", retrieval_context=["ctx"]))


def test_a_context_metric_without_ragas_fails_the_row_not_the_run(no_ragas):
    """The directed message lands on EvaluationResult.errors, score stays None."""
    metric = ContextPrecisionMetric(StubLLM())
    tc = _tc(expected_output="gt", retrieval_context=["ctx"])
    assert metric.measure(tc) is None
    assert "llminspector[ragas]" in metric.error


def test_non_ragas_metrics_are_unaffected_by_a_missing_ragas(no_ragas):
    assert ConcisenessMetric(StubLLM('{"conciseness": 1.0}')).measure(_tc()) == 1.0
