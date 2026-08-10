# Changelog

## 0.2.0

Testset generation rewritten. `synthesizer/` becomes `generation/`, built on a
`GoldenSource` + `Stage` pipeline instead of the previous pair of parallel
hierarchies.

This is a **breaking release**. Pre-1.0 semver plus `Private :: Do Not Upload`
permits it; every removal is listed below so an upgrade is a search-and-replace
rather than an investigation.

### Removed

| Removed | Replacement |
|---|---|
| `llminspector.synthesizer` (the whole package) | `llminspector.generation` |
| `BaseSynthesizer` | `Generator` |
| `AdversarialSynthesizer` | `AdversarialGenerator` |
| `RagSynthesizer`, `RagGenerator` | `DocumentSource` + `default_stages()` |
| `AlignmentSynthesizer` | — removed outright |
| `AlignmentEngine`, `LegacyTagT5Engine`, `alignment_tag` | — removed outright |
| `TestsetBackend`, `RagasTestsetBackend` | `DocumentSource` |
| `AttackSource` | `GoldenSource` / `SyncGoldenSource` |
| `llminspector.synthesizer.engines` (package) | `llminspector.generation.sources` |
| `goldens_to_dataframe` in `synthesizer` | same name in `llminspector.dataset` |
| `langchain-community` from the `ragas` extra | not needed |

Alignment generation is gone entirely: a HuggingFace T5 paraphrase loop that
produced low-quality variants and pulled a top-level numpy import into the
package import path. `transformers` and `torch` remain core dependencies for
BERTScore and the safety metrics.

### Added

- **`Generator(source, stages)`** returning `GenerationResult` with `goldens`,
  `errors`, `rejected`, `error_summary()`, `usage`.
- **Sources**: `ContextSource`, `DocumentSource`, `ScratchSource`,
  `SeedGoldenSource`, `CuratedBankSource`.
- **Stages**: `FiltrationStage`, `EvolutionStage`, `StylingStage`,
  `ExpectedOutputStage`, `PerturbationStage`, plus `default_stages()`.
- **Config**: `GenerationConfig`, `FiltrationConfig`, `EvolutionConfig`,
  `StylingConfig`, `ContextConfig`.
- **Document handling**: token-aware chunking over `tiktoken`, `NumpyIndex` /
  `FaissIndex`, critic-scored context selection, optional cross-file merging.
- **`Golden.id`** and `Golden.to_test_case()`; `LLMTestCase.golden_id` /
  `metadata`; `EvaluationDataset.to_test_cases()`.
- **`BaseLLM.generate_structured` / `a_generate_structured`** and
  `BaseEmbeddingModel.a_embed_text` / `a_embed_texts` — concrete with defaults,
  so the provider contract is still three methods.
- **`MeteredModel`** and `GenerationConfig(track_usage=True)` for token totals.
- **`StructuredOutputError`** in `llminspector.models`.
- **Extras**: `[documents]` (pypdf, python-docx) and `[faiss]` (faiss-cpu).

### Fixed

- **`policy_violation_reason` has never been populated.** The policy prompt asked
  the model for `policy_voilation_reason` while the parser read
  `policy_violation_reason`, so the column was always `"None"`. Fixing the typo
  means the column now carries the model's explanation — a **behaviour change**
  for anyone reading that column or asserting it is empty.
- `goldens_from_pandas` silently discarded every column that was not one of the
  three mapped core fields, so metadata written by an export vanished on reload.
  Unmapped columns are now collected into `Golden.metadata`, and `id`
  round-trips.
- `goldens_to_pandas` dropped `metadata` entirely on export.
- `AzureOpenAIEmbedding.embed_text` / `embed_texts` called the client without the
  shared rate-limit backoff.
- The curated attack bank's `melt` was a no-op on the real schema, emitted
  duplicate goldens under a mislabelled `Char Len` column on the schema it
  claimed to serve, and produced *zero* goldens for a bank with no extra columns.
- `CuratedBankSource` ignored the run seed, leaving unfiltered adversarial runs
  unreproducible.

### Changed

- **`to_pandas()` raises when nothing has been generated.** The old
  `BaseSynthesizer.to_pandas()` silently started a run — for an LLM-backed
  pipeline, an accidental paid one.
- Golden exports now lead with an `id` column.
- The `ragas` extra backs only the five `RagasBackedMetric` context metrics.
