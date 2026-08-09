# generation/

Dataset generation. One pipeline shape, whatever is being generated:

```
Generator(source, stages) -> GenerationResult
```

A `GoldenSource` produces the starting goldens; an ordered list of `Stage`
objects transforms each one. `AdversarialGenerator` / `RagGenerator` are presets
that pick a source and add no machinery.

This replaced a pair of parallel hierarchies — `BaseSynthesizer` subclasses on
one side, `engines/` ABCs on the other, one engine per synthesizer. Three
synthesizers meant three ABCs, three wrappers and three `metadata_keys`
forwarding properties, all to express "produce a list of goldens". Adding a
fourth way to generate meant adding to both hierarchies. Don't reintroduce that
shape: a new way of seeding a run is **one `GoldenSource`**, and it inherits
every stage.

## The two abstractions

**`GoldenSource.a_produce(config) -> list[Golden]`** — the front of the
pipeline. Async because every non-trivial source is I/O bound. `SyncGoldenSource`
is for sources whose work is purely local (the curated attack bank is a pandas
filter); implement `produce` and the async wrapper is free.

**`Stage.a_apply(golden, ctx) -> Golden | None`** — one transformation.
Returning `None` **discards** the golden and must be paired with
`ctx.reject(reason)`. Raising means *the stage broke*, which is a different
outcome: it lands on `GenerationResult.errors`, not `.rejected`. Keep those two
distinct — "this golden didn't make the cut" and "this code failed" need
different responses from whoever reads the run.

Both are generic over `GoldenT`. That is the multi-turn seam: a
`ConversationalGolden` slots in without the generator, the stage chain, `a_map`,
or the export path changing. See `../dataset/CLAUDE.md` on `_BaseGolden`.

## Rules that are easy to break

**`to_pandas()` raises when nothing has been generated.** The old
`BaseSynthesizer.to_pandas()` silently called `generate()`. For an LLM-backed
pipeline that is an accidental paid run started by a method whose name promises
a read.

**Declare `metadata_keys`** on every source and stage, ordered. The union is the
run's column set, and it must be knowable *without running the pipeline* —
otherwise the only way to learn your output shape is to pay for it. A test
asserts the declaration stays true against what actually lands in metadata.

**One implementation, and it is the async one.** `generate()` is an
`asyncio.run` wrapper that raises inside a running loop, exactly like
`evaluate()`. Never write a parallel sync path — divergent sync/async behaviour
is the single worst defect in the design this was adapted from, where the async
branch truncated a result list and the sync branch did not.

**Fan out through `utils/concurrency.py::a_map`**, bounded by
`config.max_concurrent`. Don't hand-roll another semaphore loop.

**Each golden gets its own `StageContext`.** Stages are shared across concurrent
goldens, so per-golden state lives on the context, never on the stage — the same
reasoning as `BaseMetric.clone()` per row.

## config.py is not config/

`GenerationConfig` and friends are plain dataclasses living next to the pipeline
they steer. `llminspector/config/` is provider-connection settings read from the
environment. Mixing them would put an LLM temperature next to an API key.

`seed` threads into every `random` / `numpy` draw. Without it a run cannot be
reproduced and tests have to assert on ranges instead of values.

## Ragas containment

`sources/ragas_testset.py` is the only ragas importer here, lazily and through
`optional_dependency("ragas", extra="ragas", feature=...)`. It and `rag.py` are
**transitional** — the ragas-free document pipeline replaces them, at which point
both are deleted along with `langchain-community` from the extra.

## perturbations.py

Engine-agnostic string transforms (typos, contractions, abbreviations, OCR
noise, context prefixes/suffixes) driven by the JSON tables in `data/`. Keep it
free of provider or pipeline knowledge; it becomes an opt-in `Stage`.

Two legacy bugs were fixed here — `add_contraction` and `add_abbreviation` were
silent no-ops. `tests/test_perturbations.py` pins the fixed behaviour.

## Removed, deliberately

Alignment generation (`AlignmentSynthesizer`, `AlignmentEngine`,
`LegacyTagT5Engine`, `alignment_tag.py`) is gone: a HuggingFace T5 paraphrase
loop that produced low-quality variants and dragged a top-level numpy import
into the package import path. `transformers` / `torch` stay in `pyproject.toml`
for `metrics/quality.py` (BERTScore) and `metrics/safety.py`.

The curated bank's `melt` is gone too. It claimed to expand per-attack variant
columns; no shipped bank has them, so it was a no-op on the real schema, emitted
duplicate goldens under a mislabelled column on the schema it claimed to serve,
and produced *zero* goldens for a bank with no extra columns.
