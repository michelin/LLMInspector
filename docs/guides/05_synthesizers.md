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
