# Synthesizers

Generate test data. Three synthesizers, all with the same stable contract:

```python
dataset = synthesizer.generate()   # -> EvaluationDataset of Goldens
synthesizer.to_pandas()            # flattens Golden core fields + metadata into columns
synthesizer.to_excel("out.xlsx")
```

Synthesizer-specific columns live in `Golden.metadata`, so the output shape is uniform regardless
of which synthesizer (or engine) produced it. Each engine **declares** the metadata keys it
emits, so the column set is knowable without running it:

```python
synth.metadata_keys   # ('Capability', 'Sub Capability', 'Char Len')
# -> to_pandas() columns are ['input', 'expected_output', 'context', *metadata_keys]
```

## Two layers: shell + engine

Each synthesizer is a thin **shell** that delegates its core to a swappable **engine**. The shell
and its `generate()` contract are stable; the engine is where the algorithm lives and is the seam
future implementations replace — **without changing your calling code**.

| Synthesizer | Engine ABC | Default engine | Future drop-in |
|-------------|-----------|----------------|----------------|
| `AlignmentSynthesizer` | `AlignmentEngine` | `LegacyTagT5Engine` | custom generator |
| `RagSynthesizer` | `TestsetBackend` | `RagasTestsetBackend` | ragas-free custom |
| `AdversarialSynthesizer` | `AttackSource` | `CuratedBankSource` | red-teaming generator |

**Construction is split in two.** The constructor takes the engine and nothing else; the
`from_*` classmethods build the default engine from raw data:

```python
AlignmentSynthesizer(engine)                    # bring your own engine
AlignmentSynthesizer.from_dataframe(df, ...)    # build the default one
AlignmentSynthesizer.from_excel(path, ...)      # ...from a spreadsheet
```

One signature used to accept *either* a DataFrame plus engine-config kwargs *or* a pre-built
engine, raising `ValueError` when given neither — a call the type signature said was valid but
never was. Each path now has an honest signature.

## Adversarial (offline)

```python
from llminspector.synthesizer import AdversarialSynthesizer

synth = AdversarialSynthesizer.from_excel(
    "adversarial_bank.xlsx", capability="all", sample_size=100,
)
dataset = synth.generate()
```

Samples / filters a curated bank by capability / sub-capability. Pure pandas — no model needed.

## Alignment

```python
from llminspector.synthesizer import AlignmentSynthesizer

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
from llminspector.config import AzureSettings
from llminspector.models import AzureOpenAIModel, AzureOpenAIEmbedding
from llminspector.synthesizer import RagSynthesizer

settings = AzureSettings.from_env()
synth = RagSynthesizer.from_documents(
    model=AzureOpenAIModel(settings),
    embedding=AzureOpenAIEmbedding(settings),
    document_dir="path/to/docs",
    test_size=10,
)
seeds = synth.generate()     # ragas TestsetGenerator + per-row ground-truth refinement
```

RAG **scoring** goes through the [`evaluate()`](04_evaluate.md) engine directly:

```python
from llminspector import a_evaluate

# after your RAG app answers the seed questions into `answered` (an EvaluationDataset):
result = await a_evaluate(answered, [FaithfulnessMetric(model)])
result.to_excel("rag_eval.xlsx")
```

> `RagSynthesizer.rag_evaluation()` / `export_eval()` were removed in Phase 8.6. They forwarded
> to exactly the two calls above and added nothing; a second door into `evaluate()` is worse
> than no door.
