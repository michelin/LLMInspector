# Generation

Generate test data — goldens you can answer with your application and then score with
[`evaluate()`](04_evaluate.md).

A generation run is one **source** followed by an ordered list of **stages**:

```python
from llminspector.generation import Generator

gen = Generator(source, stages)
result = gen.generate()            # scripts
result = await gen.a_generate()    # notebooks and async apps
result.to_pandas()
```

The source produces the run's starting goldens (a curated attack bank, a set of contexts, a
directory of documents). Each stage takes one golden and returns a golden — rewritten, evolved,
restyled — or `None` to discard it. That split is the whole design: adding a new way to seed a
run means writing one `GoldenSource`, and the stages that filter, evolve and style it are
inherited unchanged.

The pipeline is **async-first**. `generate()` is a thin `asyncio.run` wrapper over
`a_generate()`, so there is one code path and no sync/async behaviour drift. Inside Jupyter,
`await gen.a_generate()` — `generate()` raises in a running event loop, exactly like
`evaluate()`.

## Adversarial (offline, no model)

The one generator that needs nothing but a spreadsheet. It samples and filters a curated bank of
attack prompts by capability / sub-capability — pure pandas.

```python
from llminspector.generation import AdversarialGenerator

gen = AdversarialGenerator.from_excel(
    "adversarial_bank.xlsx", capability="toxicity", sample_size=100,
)
result = gen.generate()
result.to_pandas().head()
```

`capability` / `subcapability` accept `"all"` (or `None`) to mean "no filter", in which case
`sample_size` rows are sampled at random. `from_dataframe(bank_df, ...)` takes the bank
in memory instead.

Under the hood this is a `Generator` over `CuratedBankSource` with an empty stage list — the
classmethods exist so the common case is one line.

## Grounded generation from contexts

The full pipeline: hand it context chunks and it writes grounded inputs, filters
them, evolves them, re-checks them, and answers them.

```python
from llminspector.generation import (
    ContextSource, Generator, GenerationConfig, default_stages,
)

source = ContextSource(
    [
        ["The API allows 500 requests per minute.", "Bursts up to 750 are tolerated."],
        ["Invoices are issued on the first of the month."],
    ],
    max_goldens_per_context=2,
)

result = await Generator(
    source, default_stages(), config=GenerationConfig(model=model, seed=42),
).a_generate()
```

No optional dependency is involved — this runs on a core install.

### The chain, and why it is in this order

`default_stages()` returns **filter → evolve → filter → style → expected_output**.
Generation itself is not a stage: it turns one context into *many* inputs, and a
stage is one golden in, one golden out, so producing the initial goldens is the
source's job.

Two orderings are deliberate:

- **Evolution runs before a final filter pass, not after.** Evolving last and
  never re-checking lets a compounding chain of rewrites leave an input
  unanswerable from its context with nothing able to notice. The second pass is
  the cheap one — `max_rewrites=0`, so it scores once and applies the reject
  policy rather than re-running the repair loop.
- **Expected output runs last.** Everything before it can still change the input,
  and a reference answer written against a pre-evolution question is worse than
  none: it looks like ground truth and scores the wrong thing.

A clean golden costs five model calls: write, score, evolve, re-score, answer.
Styling adds one *only when configured* — an unconfigured `StylingStage` makes
zero calls.

### Filtration is a policy

```python
FiltrationConfig(quality_threshold=0.5, max_rewrites=3, on_reject="rewrite")
```

The input is scored on self-containment and clear objective, rewritten with the
critic's feedback while it fails, and **re-scored after every rewrite** — so the
`quality` column always describes the text actually stored, not a string that no
longer exists. When it still fails after `max_rewrites`:

| `on_reject` | Outcome |
|---|---|
| `"rewrite"` (default) | Kept, flagged `below_threshold` |
| `"discard"` | Dropped; the reason lands on `result.rejected` |
| `"keep"` | Kept, unflagged |

### Lineage

Every stage appends a compact record to `golden.metadata["lineage"]` and
promotes the columns an export needs — `quality`, `rewrites`, `evolutions`,
`below_threshold`, `styled`. Each stage declares them in `metadata_keys`, so the
output column set is knowable without paying for a run.

### Perturbation

`PerturbationStage` is opt-in and makes **no model call** — it roughens an input
with typos, OCR noise, or case changes for adversarial robustness testing. Put it
*after* everything else: filtration would score its own noise as a defect.

```python
from llminspector.generation import PerturbationStage

stages = [*default_stages(), PerturbationStage(["typo", "ocr_typo"])]
```

Seeded runs are reproducible: `GenerationConfig.seed` drives both the evolution
strategy draws and the perturbations. The seed is mixed with each golden's input
text, so a run replays exactly while goldens within it still vary.

## Generating from your own documents

`DocumentSource` loads a corpus, builds contexts from it, and then hands them to
**the same stage chain** as `ContextSource` — document handling is a way of
obtaining contexts, not a different kind of generation.

```python
from llminspector.generation import (
    ContextConfig, DocumentSource, Generator, GenerationConfig, default_stages,
)

source = DocumentSource(
    directory="./corpus",
    context_config=ContextConfig(chunk_size=1024, max_contexts=20),
)

result = await Generator(
    source,
    default_stages(),
    config=GenerationConfig(model=model, embedding=embedding, seed=42),
).a_generate()
```

`.txt`, `.md` and `.mdx` are read in **core**. PDF and DOCX need an extra:

```bash
pip install 'llminspector[documents]'    # pypdf, python-docx
```

Ask for a format you haven't installed and you get a directed error naming the
extra, not a bare `ModuleNotFoundError`.

### How a context is built

1. Each document is chunked on real tokens (`tiktoken`, already core — **not**
   `langchain-text-splitters`, which only arrives via the `ragas` extra and would
   break a core install).
2. All of one document's chunks are embedded in a **single** call.
3. A seeded sample of `candidate_pool` chunks is scored by the critic model on
   clarity, depth, structure and relevance; the best `max_contexts` become seeds.
4. Each context is the seed chunk plus its nearest neighbours **above
   `similarity_threshold`**.

`similarity_threshold` defaults to **0.5, not 0.0**. A threshold of zero accepts
every neighbour including orthogonal ones, which quietly defeats the point of
checking similarity at all.

### Validation happens before you pay

Chunk and context sizes are checked before the first embedding call, and the
error names your actual numbers and suggests concrete replacements:

```
The corpus is about 412 token(s), which splits into roughly 1 chunk(s) at
chunk_size=2048 — fewer than the 4 chunk(s) each context needs.
Try chunk_size=64 with chunk_overlap=12, or lower chunks_per_context.
```

### Cross-file contexts

`ContextConfig(cross_file=True)` merges contexts whose source files are disjoint,
so generated inputs require combining documents. Each context is consumed by at
most one group, so no chunk appears twice. Chunks are prefixed `[SOURCE: <file>]`
**only** when a context really spans two or more files.

### Choosing a vector index

`index_backend` is `"numpy"` (default), `"faiss"`, or `"auto"`. Both backends
normalise on insert and score with an inner product, so they return identical
cosine scores — a test asserts the same top-k for the same vectors. The choice is
performance, never behaviour.

`"auto"` picks faiss only when it is installed *and* the corpus is large, and
logs which it chose. It is never silent.

```bash
pip install 'llminspector[faiss]'
```

## Generating with no corpus at all

`ScratchSource` writes inputs from a description of the setting:

```python
from llminspector.generation import ScratchSource, StylingConfig

source = ScratchSource(
    StylingConfig(
        scenario="tyre retail support",
        task="answer billing and delivery questions",
        input_format="a short customer chat message",
    ),
    num_goldens=50,
)
```

All three styling fields are required — without source material they are the
only description the model has. Omit any and you get **one** error naming every
missing field, not one per rerun.

Nothing here is grounded, so there is no expected output by default. A reference
answer invented without source material is not ground truth; it is a second
opinion wearing ground truth's column name.

Large runs are batched (10 inputs per call) and de-duplicated, since asking for
hundreds in a single reply reliably degrades into near-duplicates.

## Growing an existing set

`SeedGoldenSource` produces more goldens in the vein of ones you already have:

```python
from llminspector.generation import SeedGoldenSource

source = SeedGoldenSource(existing_goldens, max_per_golden=2)
```

Omit `styling` and it is **reverse-engineered** from up to ten seed inputs with
one model call, so the augmented set sounds like the set it grew from rather than
like the model's default register.

Seeds are **partitioned**, not routed as a block: those carrying context go down
the grounded path, those without go down the scratch path, and both run. Routing
the whole batch on whether *any* seed has context — the obvious implementation —
silently drops every context-free seed, so a mixed set returns fewer goldens than
asked for with no indication why.

Every generated golden carries `seed_id`, so an augmented set stays traceable to
what it grew from.

## RAG

Question / ground-truth / context triples generated from your documents, backed by `ragas`:

```python
from llminspector.config import AzureSettings
from llminspector.generation import RagGenerator
from llminspector.models import AzureOpenAIEmbedding, AzureOpenAIModel

settings = AzureSettings.from_env()
gen = RagGenerator.from_documents(
    model=AzureOpenAIModel(settings),
    embedding=AzureOpenAIEmbedding(settings),
    document_dir="path/to/docs",
    test_size=10,
)
result = await gen.a_generate()
```

Needs the `ragas` extra (`pip install "llminspector[ragas]"`). Reaching for it without the extra
raises an `ImportError` naming the extra rather than failing obscurely.

Like the adversarial preset it runs **no stages**: ragas already filters and evolves internally,
and layering our own chain on top would double-process every golden.

## The result

`generate()` / `a_generate()` return a `GenerationResult`, shaped after
[`EvaluationResult`](04_evaluate.md) so there is one partial-failure idiom to learn — the run
always returns, and the reasons live on the result:

```python
len(result)             # goldens that made it through every stage
result.goldens
result.errors           # [{"index": 3, "stage": "filter", "error": "TimeoutError: …"}]
result.rejected         # [{"index": 7, "stage": "filter", "reason": "score 0.2 < 0.5"}]
result.error_summary()  # "1 error(s) [filter: 1]; 2 rejected [filter: 2]" — "" for a clean run
result.to_pandas()
result.to_excel("seeds.xlsx")
```

A stage that *raises* is an error — that stage broke, go and fix it. A stage that returns `None`
is a rejection — the golden did not make the cut, which is the filter working. Neither aborts
the run.

Unlike an evaluation, a generation genuinely ends up with fewer goldens than it started with, so
`goldens` holds only the survivors. `errors` and `rejected` carry the `index` the golden held in
the **source's** output, which is what lines a missing golden back up against what went in. A
`repr` always shows all three counts, so `produced=3` is never ambiguous about what was lost.

The generator exports the last run too:

```python
gen.to_pandas()
gen.to_excel("seeds.xlsx")
```

`gen.to_pandas()` **raises** if nothing has been generated yet. It does not silently call
`generate()` for you: for an LLM-backed pipeline that turns an export line into a paid run, and
another one on the next call. A run that produced nothing is a different state — that returns an
empty frame, because the caller needs to be able to see it.

## Knowing the columns before you pay for a run

Generator-specific columns live in `Golden.metadata`, so the output shape is uniform no matter
which source and stages produced it. The source and every stage **declare** the metadata keys
they emit, in export order:

```python
gen.metadata_keys   # ('Capability', 'Sub Capability', 'Char Len')
# -> to_pandas() columns are ['id', 'input', 'expected_output', 'context', *metadata_keys]
```

`metadata_keys` is the ordered, de-duplicated union across the source and every stage — knowable
without running the pipeline, which for an LLM-backed one means without paying for it.

The one column that rides along undeclared is `lineage`: every stage that changes a golden
appends a compact record to `golden.metadata["lineage"]` (`{"stage": …, "score": …}`), so a
generated row carries its own provenance rather than needing it reconstructed from logs.

## Configuration

Run knobs are plain dataclasses, exported from `llminspector.generation`. They are deliberately
*not* in `llminspector.config`, which holds provider connection settings — an LLM temperature
does not belong next to an API key.

```python
from llminspector.config import AzureSettings
from llminspector.generation import (
    EvolutionConfig,
    FiltrationConfig,
    GenerationConfig,
    StylingConfig,
)
from llminspector.models import AzureOpenAIModel

settings = AzureSettings.from_env()
config = GenerationConfig(
    model=AzureOpenAIModel(settings),
    critic_model=None,          # judging model; falls back to `model`
    max_concurrent=5,           # ceiling on in-flight model calls
    show_progress=True,
    seed=42,                    # reproducible evolution choices and sampling
    include_expected_output=True,
)
```

| Dataclass | Steers |
|---|---|
| `GenerationConfig` | models, concurrency, progress, `seed`, whether to generate expected outputs |
| `FiltrationConfig` | `quality_threshold`, `max_rewrites`, and `on_reject` |
| `EvolutionConfig` | `num_evolutions` and a name → weight map of strategies |
| `StylingConfig` | `scenario`, `task`, `input_format`, `expected_output_format` |

Two of these are worth reading closely:

- **`critic_model`** splits generating from judging. Generate with a large model and judge with a
  cheap one — the judging calls are where most of the cost sits. Unset, `config.critic` returns
  `config.model`.
- **`seed`** threads into every random draw in the pipeline. Without it a run cannot be
  reproduced, which makes both debugging and testing far more expensive than they need to be.

`FiltrationConfig.on_reject` is a policy, not a fixed behaviour. An input is re-scored after each
rewrite, so the stored score always describes the stored text:

| `on_reject` | An input still under the threshold after `max_rewrites` |
|---|---|
| `"rewrite"` | kept, flagged as below threshold (default) |
| `"discard"` | dropped, with the reason on `result.rejected` |
| `"keep"` | kept unflagged; the score is still recorded |

`StylingConfig` fields are free text handed to the model, not enumerations — the scenario of a
tyre-retail support bot is not drawn from a fixed list. Sources that generate without any context
need `scenario`, `task` and `input_format`, and `missing_fields()` reports all of the unset ones
at once.

## Writing your own source or stage

Both are ABCs, both async, and both generic over the golden type:

```python
from llminspector.dataset import Golden
from llminspector.generation import Generator, Stage, StageContext, SyncGoldenSource


class MySource(SyncGoldenSource):
    metadata_keys = ("origin",)

    def produce(self, config):
        return [Golden(input="…", metadata={"origin": "hand-written"})]


class DropEmpty(Stage):
    name = "drop_empty"

    async def a_apply(self, golden, ctx: StageContext):
        if not golden.input.strip():
            ctx.reject("empty input")
            return None
        self.record(golden, kept=True)
        return golden


gen = Generator(MySource(), [DropEmpty()])
```

Subclass `GoldenSource` and implement `a_produce` when the source does I/O; subclass
`SyncGoldenSource` and implement `produce` when it does not — the curated bank is a pandas filter
over an in-memory frame, and writing it as a coroutine would be theatre.

`StageContext` is passed rather than stored on the stage, so one stage instance is safe to use
across concurrent goldens. It carries the config, the context chunks the golden was grounded in,
the files they came from, and `ctx.reject(reason)`.

Declare `metadata_keys` on anything you write. An undiscoverable output shape is exactly the cost
the declaration buys out.

## Scoring what you generated

Generation stops at goldens. Answer them with your application, then score them through the
normal [`evaluate()`](04_evaluate.md) path — there is no second door:

```python
from llminspector import a_evaluate
from llminspector.dataset import EvaluationDataset
from llminspector.metrics import FaithfulnessMetric

answered = EvaluationDataset(
    test_cases=[g.to_test_case(actual_output=my_app(g.input)) for g in result.goldens]
)
scored = await a_evaluate(answered, [FaithfulnessMetric(model)])
scored.to_excel("rag_eval.xlsx")
```

`Golden.to_test_case()` carries the golden's `id` across as `golden_id`, so a scored row can be
traced back to the row that generated it.
