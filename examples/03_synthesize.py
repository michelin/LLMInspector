"""Example 03 — synthesize test data (alignment / adversarial / RAG).

Shows the Phase 5 synthesizers. Each returns an ``EvaluationDataset`` of
``Golden`` seeds; synthesizer-specific columns live in ``Golden.metadata`` and
flatten out via ``to_pandas()``.

The ADVERSARIAL example runs fully offline (pure pandas). The ALIGNMENT and RAG
examples need heavy deps (transformers / ragas) + a model, so they are shown but
guarded.

Run:  python examples/03_synthesize.py
"""

from llminspector import AdversarialSynthesizer

# --- Adversarial: sample/filter a curated bank (offline) ------------------- #
adv = AdversarialSynthesizer.from_excel(
    "tests/test_sample/test_adversarialdata.xlsx",
    capability="all",       # or a specific capability / sub-capability
    sample_size=5,
)
adv_dataset = adv.generate()
print("Adversarial goldens:", len(adv_dataset.goldens))
print(adv.to_pandas().head())

# --- Alignment: tag-augment -> HF-T5 paraphrase -> perturb ----------------- #
# from llminspector import AlignmentSynthesizer
# align = AlignmentSynthesizer.from_excel(
#     "tests/test_sample/test_alignmentdata.xlsx",
#     augmentations={"uppercase": ("case_change", 1.0), "typo": ("noise", 0.5)},
#     paraphrase_count=3,
# )
# align_dataset = align.generate()   # downloads the T5 paraphraser on first run
# print(align.to_pandas().head())

# --- RAG: ragas TestsetGenerator + ground-truth refinement ----------------- #
# from llminspector import AzureOpenAIModel, AzureOpenAIEmbedding, RagSynthesizer, Settings
# settings = Settings.from_env()
# rag = RagSynthesizer(
#     model=AzureOpenAIModel(settings),
#     embedding=AzureOpenAIEmbedding(settings),
#     document_dir="path/to/docs",
#     test_size=10,
# )
# rag_dataset = rag.generate()
#
# To swap the engine later (e.g. a ragas-free backend), pass backend=YourBackend()
# — no other code changes.
