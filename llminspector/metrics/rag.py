"""RAG metrics.

LLM-judge metrics (Faithfulness, AnswerCorrectness, AnswerRelevancy,
Conciseness) port the exact prompts + JSON parsing from
``EvalMetrics.*_eval_async``. Context metrics (ContextPrecision, ContextRecall,
ContextUtilisation, ContextRelevance, ContextEntityRecall) port the ragas
``SingleTurnSample`` + scorer + ``round(score, 2)`` bodies verbatim.
"""

from __future__ import annotations

import gc
from typing import Any

from ..utils.json_utils import parse_json_response
from .base_metric import BaseMetric

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

ANSWER_CORRECTNESS_PROMPT = """
        You are an expert evaluator. Systematically assess the **Answer** against the **Ground Truth** for completeness.

        ### Input Data
        - **Question**: {question}
        - **Answer**: {answer}
        - **Ground Truth**: {ground_truth}

        ### Evaluation Rules (Strict Adherence)
        1. **Decomposition**: Mentally break the Ground Truth into unique atomic facts (dates, entities, actions, specific values).
        2. **Semantic Matching**: Credit the Answer if it conveys the *same meaning* using synonyms or paraphrasing. Do not penalize for style.
        3. **Missing Information**: If a specific number, proper noun, or chemical formula is in the Ground Truth but missing from the Answer, it is a **Miss**.
        4. **Scoring**:
        - **1.0**: Perfect coverage (All atomic facts present).
        - **0.8**: High coverage (Only minor/trivial details missing).
        - **0.6**: Moderate coverage (A core concept or key entity is missing).
        - **0.4**: Low coverage (Multiple core concepts missing).
        - **0.2**: Minimal coverage (Vague link to topic).
        - **0.0**: Irrelevant/Empty.

        ### One-Shot Example
        **Ground Truth**: "The project requires Python 3.9, AWS Lambda, and a DynamoDB table." (3 Atomic Facts)
        **Answer**: "You need Python and a database."
        **Evaluation**:
        - Python 3.9: Partial (Mentioned Python, missed version) -> ⚠
        - AWS Lambda: Absent -> ✗
        - DynamoDB: Partial (Mentioned database, missed type) -> ⚠
        **Score**: 0.4 (Significant gaps in specificity).

        ### Output Format
        Output **only** a valid JSON object. Do not output reasoning text before the JSON.

        ```json
        {{
        "answer_correctness": <float>,
        "answer_correctness_reasoning": "<Concise explanation under 50 words>",
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

    async def a_measure(self, test_case: Any) -> Any:
        reasoning_key = f"{self.metric_name}_reasoning"
        try:
            result = await self._arun_prompt(
                self._prompt, self._input_variables, self._values(test_case)
            )
            parsed = parse_json_response(result)
            self.score = parsed.get(self.metric_name)
            self.reason = parsed.get(reasoning_key)
        except Exception as e:  # noqa: BLE001 - mirror legacy behavior
            print(f"{self._error_label}: error occurred:", str(e))
            self.score = None
            self.reason = None
        self.is_successful()
        return self.score


class FaithfulnessMetric(_JsonJudgeMetric):
    metric_name = "faithfulness"
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


class AnswerCorrectnessMetric(_JsonJudgeMetric):
    metric_name = "answer_correctness"
    required_inputs = {"input", "actual_output", "expected_output"}
    _prompt = ANSWER_CORRECTNESS_PROMPT
    _input_variables = ["question", "answer", "ground_truth"]
    _error_label = "answer correctness async"

    def _values(self, test_case: Any) -> dict:
        return {
            "question": test_case.input,
            "answer": test_case.actual_output,
            "ground_truth": test_case.expected_output,
        }


class AnswerRelevancyMetric(_JsonJudgeMetric):
    metric_name = "answer_relevancy"
    required_inputs = {"input", "actual_output"}
    _prompt = ANSWER_RELEVANCY_PROMPT
    _input_variables = ["question", "answer"]
    _error_label = "answer relevance async"

    def _values(self, test_case: Any) -> dict:
        return {"question": test_case.input, "answer": test_case.actual_output}


class ConcisenessMetric(_JsonJudgeMetric):
    metric_name = "conciseness"
    required_inputs = {"input", "actual_output"}
    _prompt = CONCISENESS_PROMPT
    _input_variables = ["question", "answer"]
    _error_label = "conciseness async"

    def _values(self, test_case: Any) -> dict:
        return {"question": test_case.input, "answer": test_case.actual_output}


# --------------------------------------------------------------------------- #
# ragas context metrics (verbatim from eval_metrics.py *_eval_async)
# --------------------------------------------------------------------------- #

class _RagasContextMetric(BaseMetric):
    """Shared body for ragas ``SingleTurnSample`` context metrics."""

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
            print(f"Error in {self._error_label}: {str(e)}")
            self.score = None
        self.is_successful()
        return self.score


class ContextPrecisionMetric(_RagasContextMetric):
    metric_name = "context_precision"
    required_inputs = {"input", "expected_output", "retrieval_context"}
    _error_label = "context_precision_eval"

    def _sample(self, test_case: Any):
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        from ragas.metrics import LLMContextPrecisionWithReference

        return LLMContextPrecisionWithReference(
            name="context_precision", llm=self.evaluator_llm
        )


class ContextRecallMetric(_RagasContextMetric):
    metric_name = "context_recall"
    required_inputs = {"input", "actual_output", "expected_output", "retrieval_context"}
    _error_label = "context_recall_eval"

    def _sample(self, test_case: Any):
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        from ragas.metrics import LLMContextRecall

        return LLMContextRecall(llm=self.evaluator_llm)


class ContextUtilisationMetric(_RagasContextMetric):
    metric_name = "context_utilisation"
    required_inputs = {"input", "actual_output", "retrieval_context"}
    _error_label = "context_utilisation_eval"

    def _sample(self, test_case: Any):
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        from ragas.metrics import LLMContextPrecisionWithoutReference

        return LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)


class ContextRelevanceMetric(_RagasContextMetric):
    metric_name = "context_relevance"
    required_inputs = {"input", "retrieval_context"}
    _error_label = "context_relevance_eval"

    def _sample(self, test_case: Any):
        from ragas import SingleTurnSample

        return SingleTurnSample(
            user_input=test_case.input,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        from ragas.metrics import ContextRelevance

        return ContextRelevance(name="context_relevance", llm=self.evaluator_llm)


class ContextEntityRecallMetric(_RagasContextMetric):
    metric_name = "context_entity_recall"
    required_inputs = {"expected_output", "retrieval_context"}
    _error_label = "context_entity_recall_eval"

    def _sample(self, test_case: Any):
        from ragas import SingleTurnSample

        return SingleTurnSample(
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieval_context,
        )

    def _scorer(self):
        from ragas.metrics import ContextEntityRecall

        return ContextEntityRecall(llm=self.evaluator_llm)
