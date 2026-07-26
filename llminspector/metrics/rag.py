"""RAG metrics.

LLM-judge metrics (Faithfulness, AnswerCorrectness, AnswerRelevancy,
Conciseness) port the exact prompts + JSON parsing from
``EvalMetrics.*_eval_async``. Context metrics (ContextPrecision, ContextRecall,
ContextUtilisation, ContextRelevance, ContextEntityRecall) port the ragas
``SingleTurnSample`` + scorer + ``round(score, 2)`` bodies verbatim.
"""

from __future__ import annotations

import gc
from typing import Any, Optional, cast

from ..utils.json_utils import parse_json_response
from ..utils.optional import optional_dependency
from .base_metric import BaseMetric, RagasBackedMetric

# --------------------------------------------------------------------------- #
# LLM-judge prompts (verbatim from eval_metrics.py)
# --------------------------------------------------------------------------- #

FAITHFULNESS_PROMPT = """
        You are an expert evaluator. Systematically assess the **Faithfulness** of the **Answer** relative to the **Context**.

        ### Input Data
        - **Question**: {question}
        - **Answer**: {answer}
        - **Context**: {context}

        ### Evaluation Rules (Strict Adherence)
        1.  **Closed World Assumption**: Evaluate **only** based on the provided Context. Do not use outside knowledge.
            - If a claim is factually true in the real world but *not* present in the Context, it is **Ungrounded** (Hallucination).
        2.  **Claim Verification**: Break the Answer into atomic claims. Check if each claim is supported by the Context.
        3.  **Scoring**:
            - **1.0**: All claims are fully supported by the Context.
            - **0.8**: Most claims (>80%) supported; minor exaggerations or harmless inferences.
            - **0.6**: Moderate grounding; contains specific details (dates, names, causes) not found in Context.
            - **0.4**: Significant hallucinations; introduces major new information not in Context.
            - **0.2**: Mostly ungrounded; barely relates to the Context.
            - **0.0**: Totally irrelevant or contradicts Context.

        ### One-Shot Example
        **Context**: "The new API limits requests to 500 per minute. It returns a 429 error when the limit is exceeded."
        **Answer**: "The API limits requests to 500 per minute to prevent DDoS attacks and returns a 429 error."
        **Evaluation**:
        - "Limits requests to 500/min": Supported.
        - "Returns 429 error": Supported.
        - "To prevent DDoS attacks": **Ungrounded** (Context mentions the limit, but not the *reason*).
        **Score**: 0.6 (Key logic added that wasn't in source).

        ### Output Format
        Output **only** a valid JSON object. Do not output reasoning text before the JSON.

        ```json
        {{
        "faithfulness": <float>,
        "faithfulness_reasoning": "<Concise explanation under 50 words>",
        }}
        ```
        """

# --------------------------------------------------------------------------- #
# answer_correctness — the unified judge (Phase 8.1)
#
# One prompt does the whole job. The retired design scored agreement-with-ground-
# truth in isolation and then blended it externally with faithfulness and
# relevancy (0.5/0.3/0.2). That penalised the single most important RAG success
# case: an answer that surfaces a true, context-supported, question-relevant fact
# the golden answer happens to omit. Judging all factors inside one prompt lets
# the judge see that the "extra" content is vouched for by the context.
#
# retrieval_context is optional, so there are two prompts:
#   * three-factor (context present) — GT agreement + faithfulness + relevancy
#   * two-factor  (context absent)   — GT coverage dominant, relevancy minor
# Faithfulness is undefined without a context to be faithful to, so it drops out
# of the two-factor mode rather than being scored against nothing.
# --------------------------------------------------------------------------- #

ANSWER_CORRECTNESS_PROMPT = """
        You are an expert evaluator. Judge the overall **Correctness** of the **Answer** using the
        **Ground Truth** and the **Context** together, in a single holistic assessment.

        ### Input Data
        - **Question**: {question}
        - **Answer**: {answer}
        - **Ground Truth**: {ground_truth}
        - **Context**: {context}

        ### The Three Factors
        Assess all three, then combine them into one score.

        1. **Ground-Truth Agreement** — decompose the Ground Truth into atomic facts (dates,
           entities, actions, specific values) and check the Answer against each. Credit semantic
           matches: synonyms and paraphrase count, style does not. A specific number or proper noun
           present in the Ground Truth but missing from the Answer is a **Miss**.
        2. **Faithfulness to Context** — decompose the Answer into atomic claims and check each
           against the Context. A claim that is factually true in the real world but absent from
           both the Ground Truth and the Context is **Ungrounded** (a hallucination).
        3. **Relevancy to Question** — what share of the Answer addresses the user's actual intent.
           Fluff, filler, and tangents lower this. On-topic-but-wrong still counts as relevant.

        ### THE OVERRIDING RULE — do not penalise supported extras
        Content in the Answer that is **absent from the Ground Truth** must **NOT** be penalised
        when it is BOTH:
          (a) supported by the Context, AND
          (b) relevant to the Question.
        The Ground Truth is a reference answer, not an exhaustive one. A retrieval system that
        correctly surfaces a true, relevant fact the Ground Truth omits is doing its job, and must
        score as such. Only penalise extra content when it is ungrounded in the Context, or
        irrelevant to the Question, or contradicts the Ground Truth.

        ### Combining
        Ground-Truth agreement and faithfulness carry the score. Relevancy modifies it: it pulls a
        score down when the Answer is padded or off-topic, but a fully relevant answer does not earn
        credit it has not earned on the other two factors. Contradicting the Ground Truth or the
        Context is the most severe fault. Score on this scale:
        - **1.0**: Covers the Ground Truth, every claim grounded, no padding.
        - **0.8**: Minor Ground-Truth details missing, or slight verbosity. Nothing ungrounded.
        - **0.6**: A core fact missing, or a specific ungrounded detail introduced.
        - **0.4**: Multiple core facts missing, or significant hallucination.
        - **0.2**: Vague topical link only.
        - **0.0**: Irrelevant, empty, or contradicts the Ground Truth or Context.

        ### One-Shot Example
        **Question**: "What are the API rate limits?"
        **Ground Truth**: "The API allows 500 requests per minute."
        **Context**: "The new API limits requests to 500 per minute. It returns a 429 error when the
        limit is exceeded."
        **Answer**: "The API allows 500 requests per minute, and returns a 429 error if you exceed
        that."
        **Evaluation**:
        - "500 requests per minute": matches the Ground Truth. ✓
        - "returns a 429 error": absent from the Ground Truth, but **supported by the Context** and
          **relevant to the Question** -> NOT penalised under the overriding rule. ✓
        **Score**: 1.0 (the extra fact is a retrieval success, not a hallucination).

        ### Output Format
        Output **only** a valid JSON object. Do not output reasoning text before the JSON.

        ```json
        {{
        "answer_correctness": <float>,
        "gt_agreement": <float>,
        "faithfulness": <float>,
        "relevancy": <float>,
        "answer_correctness_reasoning": "<Concise explanation under 50 words>"
        }}
        ```
        """

ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT = """
        You are an expert evaluator. Judge the overall **Correctness** of the **Answer** against the
        **Ground Truth**. No retrieval context was supplied for this row.

        ### Input Data
        - **Question**: {question}
        - **Answer**: {answer}
        - **Ground Truth**: {ground_truth}

        ### The Two Factors
        There is no Context, so faithfulness cannot be assessed — do not attempt it. Judge only:

        1. **Ground-Truth Coverage** — decompose the Ground Truth into atomic facts (dates,
           entities, actions, specific values) and check the Answer against each. Credit semantic
           matches: synonyms and paraphrase count, style does not. A specific number or proper noun
           present in the Ground Truth but missing from the Answer is a **Miss**.
        2. **Relevancy to Question** — whether the Answer addresses the user's actual intent.

        ### Combining
        **Ground-Truth coverage dominates the score.** Relevancy carries **very low weight**: use it
        only to break ties or to dock an Answer that is padded with off-topic material. Never let a
        highly relevant Answer compensate for missing Ground-Truth facts.

        With no Context to vouch for it, content in the Answer that is absent from the Ground Truth
        earns no credit — but do not treat it as a hallucination either, since there is nothing to
        check it against. Judge coverage of what the Ground Truth *does* state.

        Score on this scale:
        - **1.0**: Perfect coverage (all atomic facts present).
        - **0.8**: High coverage (only minor or trivial details missing).
        - **0.6**: Moderate coverage (a core concept or key entity missing).
        - **0.4**: Low coverage (multiple core concepts missing).
        - **0.2**: Minimal coverage (vague link to the topic).
        - **0.0**: Irrelevant, empty, or contradicts the Ground Truth.

        ### One-Shot Example
        **Ground Truth**: "The project requires Python 3.9, AWS Lambda, and a DynamoDB table."
        **Answer**: "You need Python and a database."
        **Evaluation**:
        - Python 3.9: Partial (mentioned Python, missed the version) -> ⚠
        - AWS Lambda: Absent -> ✗
        - DynamoDB: Partial (mentioned a database, missed the type) -> ⚠
        **Score**: 0.4 (significant gaps in specificity; relevancy is high but barely moves it).

        ### Output Format
        Output **only** a valid JSON object. Do not output reasoning text before the JSON.
        Omit "faithfulness" — it is undefined without a Context.

        ```json
        {{
        "answer_correctness": <float>,
        "gt_agreement": <float>,
        "relevancy": <float>,
        "answer_correctness_reasoning": "<Concise explanation under 50 words>"
        }}
        ```
        """

ANSWER_RELEVANCY_PROMPT = """
        You are an expert evaluator. Systematically assess the **Relevancy** of the **Answer** to the **Question**.

        ### Input Data
        - **Question**: {question}
        - **Answer**: {answer}

        ### Evaluation Rules (Strict Adherence)
        1.  **Signal-to-Noise Ratio**: Assess what percentage of the Answer directly addresses the user's specific intent.
        2.  **Ignore Accuracy**: Do not check if the facts are true. Only check if they are *on-topic*.
            - *Example:* If the user asks "What is 2+2?" and the answer is "2+2 equals 5", this is **Relevant** (on-topic) but incorrect. If the answer is "I like pizza", it is **Irrelevant**.
        3.  **Penalty Factors**:
            - **Fluff/Filler**: Long introductions or conversational filler lower the score.
            - **Tangents**: Information about related but unasked topics (e.g., answering about "Cars" when asked about "Trucks") lowers the score significantly.
        4.  **Scoring**:
            - **1.0**: 100% Relevant. Direct answer, no fluff.
            - **0.8**: Mostly relevant; contains minor conversational filler or unnecessary context.
            - **0.6**: Mixed; contains valid answer mixed with a distinct off-topic claim.
            - **0.4**: Low relevance; answers a slightly different question or is mostly fluff.
            - **0.2**: Minimal relevance; barely touches the topic.
            - **0.0**: Completely off-topic or non-responsive.

        ### One-Shot Example
        **Question**: "How do I reset my password?"
        **Answer**: "To reset your password, go to Settings > Security > Reset. By the way, we also launched a new Dark Mode feature yesterday that you might like."
        **Evaluation**:
        - "Go to Settings...": Relevant (Direct answer).
        - "Launched Dark Mode...": **Irrelevant** (Marketing noise unrelated to the user's problem).
        **Score**: 0.6 (Valid answer diluted by unrelated topic).

        ### Output Format
        Output **only** a valid JSON object. Do not output reasoning text before the JSON.

        ```json
        {{
        "answer_relevancy": <float>,
        "answer_relevancy_reasoning": "<Concise explanation under 50 words>",
        }}
        ```
        """

CONCISENESS_PROMPT = """
        You are an expert evaluator. Systematically assess the **Conciseness** of the **Answer**.

        ### Input Data
        - **Question**: {question}
        - **Answer**: {answer}

        ### Evaluation Rules (Strict Adherence)
        1.  **Information Density**: Measure the ratio of *unique value* to *total words*.
            - High Conciseness = High Value / Low Word Count.
        2.  **Redundancy Check**:
            - **Looping**: Penalize heavily if the answer states a fact, then restates it in different words immediately after.
            - **Filler**: Penalize "fluff" phrases (e.g., "It is important to note that," "As previously mentioned").
            - **Recursive Definition**: Penalize defining common terms unless asked (e.g., "Rain, which is water falling from the sky...").
        3.  **Elaboration vs. Repetition**: Do *not* penalize detailed explanations if every sentence adds *new* information. Only penalize if the *same* information is repeated.
        4.  **Scoring**:
            - **1.0**: Efficient. Every word serves a purpose.
            - **0.8**: Good. Minor verbosity or one small repetition.
            - **0.6**: Moderate. Noticeable looping or unnecessary summaries.
            - **0.4**: Bloated. Significant repetition; could be half the length.
            - **0.2**: Verbose. High noise-to-signal ratio.
            - **0.0**: Empty or nonsensical.

        ### One-Shot Example
        **Question**: "How do I save a file?"
        **Answer**: "To save a file, you must perform the save action. First, go to the File menu. Then, click Save. By clicking Save, you will save your document to the disk. This ensures the file is stored."
        **Evaluation**:
        - "Perform the save action": Redundant introduction.
        - "By clicking Save...": Restates the previous sentence.
        - "Ensures the file is stored": Restates the concept of saving.
        **Score**: 0.4 (4 sentences used to convey 2 steps).

        ### Output Format
        Output **only** a valid JSON object. Do not output reasoning text before the JSON.

        ```json
        {{
        "conciseness": <float>,
        "conciseness_reasoning": "<Concise explanation under 50 words>",
        }}
        ```
        """


class _JsonJudgeMetric(BaseMetric):
    """Shared body for the LLM-judge JSON metrics.

    Subclasses set ``metric_name``, ``required_inputs``, ``_prompt``,
    ``_input_variables``, and implement ``_values(test_case)``.
    """

    _prompt: str = ""
    _input_variables: list = []
    _error_label: str = "metric"
    produces_reasoning = True

    def _values(self, test_case: Any) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    # Prompt selection goes through these hooks so a subclass can branch on the
    # test case (see AnswerCorrectnessMetric, which switches on retrieval_context).
    # The base implementations ignore it; the parameter is the hook contract.
    # pylint: disable=unused-argument
    def _prompt_for(self, test_case: Any) -> str:
        return self._prompt

    def _variables_for(self, test_case: Any) -> list:
        return self._input_variables

    # pylint: enable=unused-argument

    def _read_sub_scores(self, parsed: dict) -> None:
        """Pull any extra judgements off the parsed JSON. Default: none."""

    def _clear_sub_scores(self) -> None:
        """Reset extra judgements after a failure. Default: nothing to reset."""

    async def a_measure(self, test_case: Any) -> Any:
        reasoning_key = f"{self.metric_name}_reasoning"
        try:
            result = await self._arun_prompt(
                self._prompt_for(test_case),
                self._variables_for(test_case),
                self._values(test_case),
            )
            parsed = parse_json_response(result)
            self.score = parsed.get(self.metric_name)
            self.reason = parsed.get(reasoning_key)
            self._read_sub_scores(parsed)
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            self.record_failure(e)
            self.score = None
            self.reason = None
            self._clear_sub_scores()
        self.is_successful()
        return self.score


class FaithfulnessMetric(_JsonJudgeMetric):
    metric_name = "faithfulness"
    sort_key = 310
    required_inputs = {"input", "actual_output", "retrieval_context"}
    _prompt = FAITHFULNESS_PROMPT
    _input_variables = ["question", "answer", "context"]
    _error_label = "faithfulness async"

    def _values(self, test_case: Any) -> dict:
        return {
            "question": test_case.input,
            "answer": test_case.actual_output,
            "context": test_case.retrieval_context,
        }


def _has_context(test_case: Any) -> bool:
    """True when the test case carries a usable retrieval context.

    Mirrors the ``retrieval_context`` rule in ``evaluate._availability``: present,
    non-empty, and not made up entirely of blank entries.
    """
    ctx = getattr(test_case, "retrieval_context", None)
    if not ctx:
        return False
    return any(str(c).strip() for c in ctx)


class AnswerCorrectnessMetric(_JsonJudgeMetric):
    """The unified correctness judge.

    Takes question + answer + ground truth + retrieval context and assesses
    ground-truth agreement, faithfulness to context, and relevancy to the
    question *inside a single prompt*, so that context-supported, question-
    relevant content missing from the ground truth is not penalised.

    ``retrieval_context`` is optional and never gates availability: rows without
    one degrade to a two-factor judgement (ground-truth coverage dominant,
    relevancy at very low weight) rather than being skipped. ``faithfulness`` is
    undefined in that mode and is reported as ``None``.

    Alongside ``score`` / ``reason`` the judge exposes its three sub-judgements
    as ``answer_correctness_gt_agreement`` / ``_faithfulness`` / ``_relevancy``
    columns, so a low score is diagnosable without a second run.
    """

    metric_name = "answer_correctness"
    sort_key = 320
    # Context is read when present but is deliberately NOT required: a plain-LLM
    # row must still be judged, just with one factor fewer.
    required_inputs = {"input", "actual_output", "expected_output"}
    _prompt = ANSWER_CORRECTNESS_PROMPT
    _input_variables = ["question", "answer", "ground_truth", "context"]
    _error_label = "answer correctness async"

    #: Sub-judgement column suffix -> key in the judge's JSON response.
    _SUB_SCORES = {
        "gt_agreement": "gt_agreement",
        "faithfulness": "faithfulness",
        "relevancy": "relevancy",
    }

    def __init__(self, model: Any = None, threshold: Optional[float] = None) -> None:
        super().__init__(model=model, threshold=threshold)
        self.sub_scores: dict = {k: None for k in self._SUB_SCORES}

    def _prompt_for(self, test_case: Any) -> str:
        return (
            ANSWER_CORRECTNESS_PROMPT
            if _has_context(test_case)
            else ANSWER_CORRECTNESS_NO_CONTEXT_PROMPT
        )

    def _variables_for(self, test_case: Any) -> list:
        if _has_context(test_case):
            return ["question", "answer", "ground_truth", "context"]
        return ["question", "answer", "ground_truth"]

    def _values(self, test_case: Any) -> dict:
        values = {
            "question": test_case.input,
            "answer": test_case.actual_output,
            "ground_truth": test_case.expected_output,
        }
        if _has_context(test_case):
            values["context"] = test_case.retrieval_context
        return values

    def _read_sub_scores(self, parsed: dict) -> None:
        # faithfulness is absent by design on the two-factor path; .get leaves None.
        self.sub_scores = {
            column: parsed.get(json_key)
            for column, json_key in self._SUB_SCORES.items()
        }

    def _clear_sub_scores(self) -> None:
        self.sub_scores = {k: None for k in self._SUB_SCORES}

    def expand(self, score: Any) -> dict:
        """Headline score plus the three sub-judgements as their own columns."""
        columns = {self.metric_name: score}
        for column in self._SUB_SCORES:
            columns[f"{self.metric_name}_{column}"] = self.sub_scores.get(column)
        return columns

    def clone(self) -> "AnswerCorrectnessMetric":
        new = cast(AnswerCorrectnessMetric, super().clone())
        new.sub_scores = {k: None for k in self._SUB_SCORES}
        return new


class AnswerRelevancyMetric(_JsonJudgeMetric):
    metric_name = "answer_relevancy"
    sort_key = 330
    required_inputs = {"input", "actual_output"}
    _prompt = ANSWER_RELEVANCY_PROMPT
    _input_variables = ["question", "answer"]
    _error_label = "answer relevance async"

    def _values(self, test_case: Any) -> dict:
        return {"question": test_case.input, "answer": test_case.actual_output}


class ConcisenessMetric(_JsonJudgeMetric):
    metric_name = "conciseness"
    sort_key = 340
    required_inputs = {"input", "actual_output"}
    _prompt = CONCISENESS_PROMPT
    _input_variables = ["question", "answer"]
    _error_label = "conciseness async"

    def _values(self, test_case: Any) -> dict:
        return {"question": test_case.input, "answer": test_case.actual_output}


# --------------------------------------------------------------------------- #
# ragas context metrics (verbatim from eval_metrics.py *_eval_async)
# --------------------------------------------------------------------------- #


class _RagasContextMetric(RagasBackedMetric):
    """Shared body for ragas ``SingleTurnSample`` context metrics.

    These five are the only metrics in the package that need ragas; every other
    one runs on a bare ``BaseLLM``.
    """

    _error_label: str = "context_metric"

    def _sample(self, test_case: Any):  # pragma: no cover - overridden
        raise NotImplementedError

    def _scorer(self):  # pragma: no cover - overridden
        raise NotImplementedError

    async def a_measure(self, test_case: Any) -> Any:
        try:
            sample = self._sample(test_case)
            scorer = self._scorer()
            score = await scorer.single_turn_ascore(sample)
            del scorer
            gc.collect()
            self.score = round(score, 2)
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            self.record_failure(e)
            self.score = None
        self.is_successful()
        return self.score


class ContextPrecisionMetric(_RagasContextMetric):
    metric_name = "context_precision"
    sort_key = 380
    required_inputs = {"input", "expected_output", "retrieval_context"}
    _error_label = "context_precision_eval"

    def _sample(self, test_case: Any):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas.metrics import LLMContextPrecisionWithReference

        return LLMContextPrecisionWithReference(
            name="context_precision", llm=self.evaluator_llm
        )


class ContextRecallMetric(_RagasContextMetric):
    metric_name = "context_recall"
    sort_key = 390
    required_inputs = {"input", "actual_output", "expected_output", "retrieval_context"}
    _error_label = "context_recall_eval"

    def _sample(self, test_case: Any):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas.metrics import LLMContextRecall

        return LLMContextRecall(llm=self.evaluator_llm)


class ContextUtilisationMetric(_RagasContextMetric):
    metric_name = "context_utilisation"
    sort_key = 360
    required_inputs = {"input", "actual_output", "retrieval_context"}
    _error_label = "context_utilisation_eval"

    def _sample(self, test_case: Any):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas.metrics import LLMContextPrecisionWithoutReference

        return LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)


class ContextRelevanceMetric(_RagasContextMetric):
    metric_name = "context_relevance"
    sort_key = 350
    required_inputs = {"input", "retrieval_context"}
    _error_label = "context_relevance_eval"

    def _sample(self, test_case: Any):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas.metrics import ContextRelevance

        return ContextRelevance(name="context_relevance", llm=self.evaluator_llm)


class ContextEntityRecallMetric(_RagasContextMetric):
    metric_name = "context_entity_recall"
    sort_key = 370
    required_inputs = {"expected_output", "retrieval_context"}
    _error_label = "context_entity_recall_eval"

    def _sample(self, test_case: Any):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas import SingleTurnSample

        return SingleTurnSample(
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        with optional_dependency("ragas", extra="ragas", feature="the context metrics"):
            from ragas.metrics import ContextEntityRecall

        return ContextEntityRecall(llm=self.evaluator_llm)
