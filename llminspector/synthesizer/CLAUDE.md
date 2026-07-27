# synthesizer/

Dataset generation: alignment (perturbed variants of seed prompts), adversarial
(attack prompts), RAG (question/ground-truth/context from documents).

## Two layers — this is the load-bearing design

**Stable shell** (`base.py`, `alignment.py`, `adversarial.py`, `rag.py`)
— `BaseSynthesizer.generate() -> EvaluationDataset` is the fixed contract.
Callers depend only on this.

**Swappable engines** (`engines/`) — the parts slated for replacement:

| ABC | Today's implementation | Replaces with |
|---|---|---|
| `AlignmentEngine` | `LegacyTagT5Engine` (tag-augment → HF T5 → perturb) | custom generation |
| `TestsetBackend` | `RagasTestsetBackend` | our own ragas-free testset generation |
| `AttackSource` | `CuratedBankSource` (static bank) | live red-teaming |

These seams exist **specifically so the replacement is a new subclass injected
into the synthesizer, with no change to the synthesizer classes, their callers,
or the `Golden` output contract.** Work that erodes that boundary — a
synthesizer reaching into engine internals, ragas leaking out of
`engines/ragas_testset.py`, an engine returning something other than
`List[Golden]` — collapses the two layers back together and forces a second
refactor when the engines are replaced. Don't.

## Uniform output

Every engine returns `List[Golden]`. Engine-specific columns hang off
`Golden.metadata`, and each engine declares `metadata_keys` (ordered) so
`to_pandas()`'s column set is knowable without running the engine. If you add an
engine, declare `metadata_keys` — an undiscoverable output shape was the exact
cost this bought out.

## Ragas containment

All ragas imports in this package live in `engines/ragas_testset.py` (plus
`metrics/`'s `RagasBackedMetric` subclasses), wrapped in
`optional_dependency("ragas", extra="ragas", feature=...)`. `langchain-community`
rides along for one thing only: the `DirectoryLoader` in that engine.

## perturbations.py

Engine-agnostic string transforms (typos, contractions, abbreviations, OCR
noise, context prefixes/suffixes) driven by the JSON tables in `data/`. Shared by
any engine; keep it free of engine or provider knowledge.

Two legacy bugs were fixed here — `add_contraction` and `add_abbreviation` were
silent no-ops. Tests in `tests/test_perturbations.py` pin the fixed behaviour.

## Known legacy behaviours (preserved deliberately)

- Legacy alignment drops `Expected_Result` at the perturbation stage. Preserved
  as-is; not a bug to fix without asking.
- RAG scoring runs through `evaluate()` via the `rag_evaluation()` /
  `export_eval()` wrappers rather than a separate scoring path.
