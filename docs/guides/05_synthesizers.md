# Synthesizers

Generate test data. Three synthesizers, all with the same stable contract:

```python
dataset = synthesizer.generate()   # -> EvaluationDataset of Goldens
synthesizer.to_pandas()            # flattens Golden core fields + metadata into columns
synthesizer.to_excel("out.xlsx")
```

Synthesizer-specific columns live in `Golden.metadata`, so the output shape is uniform regardless
of which synthesizer (or engine) produced it.

## Two layers: shell + engine

Each synthesizer is a thin **shell** that delegates its core to a swappable **engine**. The shell
and its `generate()` contract are stable; the engine is where the algorithm lives and is the seam
future implementations replace — **without changing your calling code**.

| Synthesizer | Engine ABC | Default engine | Future drop-in |
|-------------|-----------|----------------|----------------|
| `AlignmentSynthesizer` | `AlignmentEngine` | `LegacyTagT5Engine` | custom generator |
| `RagSynthesizer` | `TestsetBackend` | `RagasTestsetBackend` | ragas-free custom |
| `AdversarialSynthesizer` | `AttackSource` | `CuratedBankSource` | red-teaming generator |

To swap: implement the ABC and inject it (`engine=` / `backend=` / `source=`). Nothing else changes.

## Adversarial (offline)

```python
from llminspector import AdversarialSynthesizer

synth = AdversarialSynthesizer.from_excel(
    "adversarial_bank.xlsx", capability="all", sample_size=100,
)
dataset = synth.generate()
```

Samples / filters a curated bank by capability / sub-capability. Pure pandas — no model needed.

## Alignment

```python
from llminspector import AlignmentSynthesizer

synth = AlignmentSynthesizer.from_excel(
    "alignment_seeds.xlsx",
    augmentations={"uppercase": ("case_change", 1.0), "typo": ("noise", 0.5)},
    paraphrase_count=3,
)
dataset = synth.generate()   # tag-augment -> HF-T5 paraphrase -> perturb
```

Needs `transformers` (downloads `humarin/chatgpt_paraphraser_on_T5_base` on first run). The
perturbation transforms live in `llminspector.synthesizer.perturbations` and reuse the lookup
tables in `llminspector.data`.

## RAG

```python
from llminspector import AzureOpenAIModel, AzureOpenAIEmbedding, RagSynthesizer, Settings

settings = Settings.from_env()
synth = RagSynthesizer(
    model=AzureOpenAIModel(settings),
    embedding=AzureOpenAIEmbedding(settings),
    document_dir="path/to/docs",
    test_size=10,
)
seeds = synth.generate()     # ragas TestsetGenerator + per-row ground-truth refinement
```

RAG **scoring** goes through the [`evaluate()`](04_evaluate.md) engine:

```python
# after your RAG app answers the seed questions into `answered` (an EvaluationDataset):
result = synth.rag_evaluation(answered, metrics=[FaithfulnessMetric(model)])
synth.export_eval(result, "rag_eval.xlsx")
```
