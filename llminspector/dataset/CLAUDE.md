# dataset/

`EvaluationDataset` — a container of `LLMTestCase`s and/or `Golden`s with
pandas / Excel (de)serialization.

## The two row types

`LLMTestCase` (in `../test_case/`) is the evaluable unit: `input` (required),
`actual_output`, `expected_output`, `retrieval_context`, `policy`.

`Golden` is the *seed* for synthesis: `input` (required), `expected_output`,
`context`, `metadata`. It has no `actual_output` — the answer is what synthesis
and evaluation produce.

Both are pydantic with `extra="ignore"`, both require a non-empty `input`, and
both coerce a bare string context into a one-element list while dropping blanks.
Keep that symmetry when editing either.

`Golden.metadata` is the uniformity escape hatch: it lets every synthesizer,
present or future, emit the same `Golden` shape while carrying its own extra
columns. See `../synthesizer/CLAUDE.md`.

## Column mapping is a compatibility surface

`ColumnMapping` and `GoldenColumnMapping` default to the legacy `helper.py`
column names:

```
question → input          answer   → actual_output
ground_truth → expected_output      contexts → retrieval_context
policy → policy
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
