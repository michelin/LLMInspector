# evaluate/

The async batch engine (`evaluate.py`) and the result object (`result.py`).

```python
evaluate(dataset, metrics=[...], batch_size=None, show_progress=True) -> EvaluationResult
await a_evaluate(...)   # same, inside a running loop
```

`evaluate()` raises a directed `RuntimeError` when called from a running event
loop (Jupyter, FastAPI) telling the caller to await `a_evaluate` instead — the
engine is async all the way down.

## Invariants

**Rows come back in input order.** `as_completed` drives the progress bar, but
each task carries its index and results are placed by index. The legacy
`as_completed` + positional `pd.concat` could misalign rows against their
inputs. Any change here must preserve input ordering — `tests/test_evaluate_engine.py`
checks it.

**The engine knows no metric names.** Columns come from the metrics
(`expand` / `output_columns` / `sort_key`); the engine only sorts and merges.
The old hand-maintained 50-entry `_REORDER_KEYS` list plus three bespoke
`_expand_*` functions are gone. If you find yourself typing a metric name in
this directory, the change belongs in `metrics/`.

**Output columns reflect only the requested metrics** and their expansions — not
every metric the package can compute. An intentional divergence from the legacy
helper, noted inline.

**Availability filtering**: a metric whose `required_inputs` aren't present on a
row is skipped for that row (scored `None`), not an error.

**Failures are collected, never raised.** Metric errors land on
`EvaluationResult.errors` as `{"row": idx, "error": ...}`, sorted by row. A run
with failures logs a warning and still returns a result.

## batch_size

Defaults to the strictest `max_workers` across the metrics' providers, else 5.
This is what keeps row concurrency and the providers' own ceiling from being set
independently. An explicit `batch_size` wins; `< 1` raises.

## tqdm

Lazily imported with a null-bar fallback — it is a declared dependency but the
engine must not hard-fail if it's absent.

## Known stopgap

`aggregate.py::overall_accuracy` *overwrites* `answer_correctness` with a
weighted 0.5/0.3/0.2 blend. It is slated to be replaced by a unified,
context-optional answer-correctness judge — don't build new behaviour on top of
the blend.
