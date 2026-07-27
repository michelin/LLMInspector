# metrics/

24 metric classes over `BaseMetric`. Grouped by concern, one module each:
`quality.py` (BERTScore), `rag.py` (faithfulness/correctness/context), `safety.py`
(PII, moderation, jailbreak, refusal, hallucination, code detection), `nlp.py`
(sentiment, emotion, language, readability, tokens), `policy.py`,
`aggregate.py` (token totals).

## The contract

A metric is self-contained: build it with a model, call `measure` / `a_measure`
with an `LLMTestCase`, read `score` / `reason` / `success` off the instance.
Subclasses implement **`a_measure` only** — `measure` is the `asyncio.run`
wrapper on the base.

Class attributes every new metric sets:

| Attribute | Why it matters |
|---|---|
| `metric_name` | The headline column name. |
| `required_inputs` | `LLMTestCase` attrs needed. The engine skips the metric on rows missing any of them — this is how a dataset with no `retrieval_context` still evaluates. |
| `produces_reasoning` | Adds a `{name}_reasoning` column right behind the score. |
| `sort_key` | Position in the exported table. Values are spaced so a new metric slots between two existing ones without renumbering. |

## Rules that are easy to break

**Never raise out of `a_measure`.** Catch, call `self.record_failure(exc)`, score
`None`. One broken metric must not abort a run over thousands of rows. The error
surfaces on `EvaluationResult.errors`, so nothing is actually swallowed — a
table of `None`s with no errors means *skipped*, with errors means *failed*, and
that distinction is the whole point.

**`clone()` must reset all mutable per-run state.** The engine clones each metric
per row so concurrent rows don't race. The base `clone` is a *shallow* copy that
resets only `score` / `reason` / `success` / `error`; the `model` is shared
deliberately. If your metric accumulates anything else, override `clone` — see
`AnswerCorrectnessMetric.clone` re-initialising `sub_scores`.

**Metrics own their output columns.** The evaluate engine contains zero metric
names. Override `expand(score)` when the score maps to more than one column
(moderation flag sets, code detection, policy verdicts). `expand` **must return
the same key set for every score including `None`** — the engine calls
`expand(None)` up front to reserve columns. `output_columns` is derived from
`expand`, so the two cannot drift.

**Talk to the model through `BaseLLM` only** — `generate` / `a_generate`, via the
`_run_prompt` / `_arun_prompt` helpers. No langchain import belongs in this
package. Reaching for `model.client` quietly makes "expose a langchain client"
the real provider contract, which is exactly what was removed.

## The three base classes

- `BaseMetric` — everything above.
- `DualTargetMetric` — one implementation scored against `input` *or*
  `actual_output`; `name` and `required_inputs` derive from `target`. Set
  `name_suffix` and `sort_key_by_target` (both positions, since the question and
  answer variants aren't always adjacent in the export).
- `RagasBackedMetric` — needs the provider's ragas wrapper, not just
  `a_generate`. Only the five context metrics (Precision, Recall, Utilisation,
  Relevance, EntityRecall) subclass it. **Ragas imports appear nowhere else in
  this directory**, and they go inside `optional_dependency(...)` from
  `utils/optional.py`.

## Local models

BERTScore, lingua, and presidio load through module-level `@lru_cache` loader
functions, imported inside the loader. A metric must never import transformers /
torch / presidio at module top level.

## Adding one

Follow `.claude/skills/add-metric`. The short version: class in the right module
→ export in `metrics/__init__.py` → entry in `tests/test_column_contract.py`'s
`GOLDEN_HEADER` → unit test in `tests/test_metrics.py` → update
`docs/guides/03_metrics.md`.
