"""Default RAG testset backend — ragas ``TestsetGenerator`` + GT refinement.

Ports ``RagEval.generate_testset`` / ``enhance_ground_truth`` / ``refine_answer``.
**All ragas imports are confined to this file** (lazily), so the planned
ragas-free custom backend is a drop-in :class:`TestsetBackend` replacement that
simply never imports this module.
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from ...dataset.golden import Golden
from .base import TestsetBackend

_DEFAULT_REFINE_PROMPT = (
    "Given the question: {question}\n"
    "the context: {context}\n"
    "and the draft answer: {answer}\n"
    "produce an improved, accurate ground-truth answer."
)


class RagasTestsetBackend(TestsetBackend):
    """Generate a RAG testset from documents, then refine each ground truth.

    Parameters
    ----------
    model:
        A :class:`~llminspector.models.base_model.BaseLLM` — ``generate``
        refines answers, and ``ragas_llm()`` drives testset generation.
    embedding:
        An :class:`~llminspector.models.azure_openai.AzureOpenAIEmbedding`
        (its ``.ragas_embeddings()`` drives generation).
    documents:
        Pre-loaded langchain documents. If ``None``, ``document_dir`` is loaded.
    document_dir:
        Directory of source docs (pdf/docx/txt) to load when ``documents`` is None.
    test_size:
        Number of test rows to generate.
    refine_prompt:
        Template with ``{question}`` / ``{context}`` / ``{answer}`` placeholders.
    """

    #: declared so to_pandas()'s column set is knowable without running it
    metadata_keys = ("synthesizer_name",)

    def __init__(
        self,
        model: Any,
        embedding: Any,
        documents: Optional[list] = None,
        document_dir: Optional[str] = None,
        test_size: int = 10,
        refine_prompt: str = _DEFAULT_REFINE_PROMPT,
    ) -> None:
        self.model = model
        self.embedding = embedding
        self.documents = documents
        self.document_dir = document_dir
        self.test_size = test_size
        self.refine_prompt = refine_prompt

    def _load_documents(self) -> list:
        from langchain_community.document_loaders import DirectoryLoader

        loader = DirectoryLoader(
            self.document_dir,
            glob=["**/*.pdf", "**/*.docx", "**/*.txt"],
            use_multithreading=True,
        )
        return loader.load()

    def refine_answer(self, question, context, answer) -> str:
        # Goes through BaseLLM.generate rather than reaching into a langchain
        # client, so the refinement step works with any provider (only testset
        # *generation* below actually needs ragas).
        return self.model.generate(
            self.refine_prompt.format(question=question, context=context, answer=answer)
        )

    def enhance_ground_truth(self, test_df: pd.DataFrame) -> pd.DataFrame:
        responses = []
        for _, row in test_df.iterrows():
            question = row["question"]
            answer = row["ground_truth"]
            context = row["reference_contexts"]
            responses.append(
                self.refine_answer(question, answer=answer, context=context)
            )

        test_df["responses"] = responses
        test_df.drop(columns=["ground_truth"], inplace=True)
        test_df.rename(columns={"responses": "ground_truth"}, inplace=True)
        column_order = [
            "question",
            "ground_truth",
            "reference_contexts",
            "synthesizer_name",
        ]
        return test_df[column_order]

    def _generate_testset_df(self) -> pd.DataFrame:
        from ragas.testset import TestsetGenerator

        documents = (
            self.documents if self.documents is not None else self._load_documents()
        )

        evaluator_llm = self.model.ragas_llm()
        embeddings = self.embedding.ragas_embeddings()
        run_config = self.model.run_config

        generator = TestsetGenerator(llm=evaluator_llm, embedding_model=embeddings)
        dataset = generator.generate_with_langchain_docs(
            documents,
            testset_size=self.test_size,
            transforms_llm=evaluator_llm,
            transforms_embedding_model=embeddings,
            run_config=run_config,
        )
        testset_df = dataset.to_pandas().rename(
            columns={"user_input": "question", "reference": "ground_truth"}
        )
        return self.enhance_ground_truth(testset_df)

    def generate(self) -> List[Golden]:
        df = self._generate_testset_df()
        goldens: List[Golden] = []
        for _, row in df.iterrows():
            context = row.get("reference_contexts")
            goldens.append(
                Golden(
                    input=str(row["question"]),
                    expected_output=(
                        None
                        if row.get("ground_truth") is None
                        else str(row["ground_truth"])
                    ),
                    context=list(context) if context is not None else None,
                    metadata={"synthesizer_name": row.get("synthesizer_name")},
                )
            )
        return goldens
