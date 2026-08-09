# dataset/

`EvaluationDataset` — a container of `LLMTestCase`s and/or `Golden`s with
pandas / Excel (de)serialization.

## The two row types

`LLMTestCase` (in `../test_case/`) is the evaluable unit: `input` (required),
`actual_output`, `expected_output`, `retrieval_context`, `policy`, plus
`golden_id` / `metadata` for provenance.

`Golden` is the *seed* for generation: `id`, `input` (required),
`expected_output`, `context`, `metadata`. It has no `actual_output` — the answer
is what generation and evaluation produce. `Golden.to_test_case()` promotes one.

Both are pydantic with `extra="ignore"`, both require a non-empty `input`, and
both coerce a bare string context into a one-element list while dropping blanks.
Keep that symmetry when editing either.

`Golden.metadata` is the uniformity escape hatch: it lets every generator,
present or future, emit the same `Golden` shape while carrying its own extra
columns. See `../synthesizer/CLAUDE.md`.

## `_BaseGolden` is an extension seam — don't collapse it

`Golden` subclasses a private `_BaseGolden` holding the fields that describe a
golden's *identity and provenance* rather than its turn shape: `id`, `context`,
`metadata`. `Golden` itself declares only the single-turn pair `input` /
`expected_output`.

That split exists so a multi-turn `ConversationalGolden` can become a sibling
subclass substituting `scenario` / `expected_outcome`, with the export path and
the generation pipeline continuing to work against `_BaseGolden` untouched.
Moving a shared field down into `Golden` quietly breaks that.
`tests/test_golden.py` asserts which class declares which field.

## Identity and provenance

`Golden.id` is a uuid minted per object — **not** derived from the input, because
two goldens may legitimately share an input (different evolutions of one seed).
It survives the Excel round trip, and `to_test_case()` copies it onto
`LLMTestCase.golden_id`. That is the only link from a scored row back to the
golden and its lineage metadata, so anything that regenerates ids on load breaks
traceability.

`to_test_case()` **deep-copies** `metadata` rather than sharing the dict;
mutating the test case must not corrupt the golden. Deep, not shallow, because
metadata is not flat — the generation stages append to a `metadata["lineage"]`
list, which a shallow copy would leave shared.

`golden_id` and `metadata` are deliberately *not* exported by
`EvaluationDataset.to_pandas` or `EvaluationResult.to_pandas`. Both name their
columns explicitly, which is what keeps result tables from silently widening —
switching either to `model_dump()` would break that.

## Column mapping is a compatibility surface

`ColumnMapping` and `GoldenColumnMapping` default to the legacy `helper.py`
column names:

```
question → input          answer   → actual_output
ground_truth → expected_output      contexts → retrieval_context
policy → policy           id       → id   (goldens only)
```

**Existing user spreadsheets load unchanged because of these defaults.**
Changing a default breaks every workbook in the field — treat them as frozen and
add an override instead. `from_pandas` / `from_excel` /
`goldens_from_pandas` / `goldens_from_excel` all take a mapping plus per-column
overrides via `_resolve_mapping`.

## Parsing

`_cell` normalises a scalar to `str` or `None` (NaN-safe). `_parse_context` uses
`ast.literal_eval` to read list-shaped cells written by pandas, falling back to
the raw string. Spreadsheet round-tripping is covered by `tests/test_dataset.py`
— run it after any change here.

**Metadata columns do not go through `_cell`.** `goldens_from_pandas` collects
every unmapped column into `Golden.metadata` keeping the raw value, so an int
column comes back as an int. Missing/NaN cells are *dropped* rather than stored
as `None`: the export unions every golden's metadata keys, so retaining the
blanks would give each golden every other golden's keys after one round trip.

## `goldens_to_dataframe`

Lives here, not in `synthesizer/` — flattening goldens to a table is a dataset
concern and `goldens_to_pandas` is its main caller. It takes an optional mapping:

- `mapping=None` → the `Golden` **attribute** names (`id` / `input` /
  `expected_output` / `context`). What a generator's own export wants: a fresh
  artefact named after the model.
- a `GoldenColumnMapping` → the **spreadsheet** names, which is what
  `goldens_to_pandas` passes so its output stays readable by
  `goldens_from_pandas`.

Metadata columns follow, as the union of all keys in first-seen order. A metadata
key colliding with a core column name is skipped; the core field wins.
