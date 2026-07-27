---
name: add-metric
description: Add a new metric class to llminspector — where the class goes, what it must declare, and every file that has to change alongside it. Use when asked to add, port, or split a metric.
---

# Adding a metric

Read `llminspector/metrics/CLAUDE.md` first — it holds the contract this
checklist operationalises.

## 1. Pick the module

| Module | Holds |
|---|---|
| `quality.py` | Similarity / overlap scoring |
| `rag.py` | Faithfulness, correctness, context metrics |
| `safety.py` | PII, moderation, jailbreak, refusal, hallucination, code detection |
| `nlp.py` | Sentiment, emotion, language, readability, tokens |
| `policy.py` | Policy compliance |

New concern with no home? Ask before creating a module — the grouping is part of
the public import surface.

## 2. Pick the base class

- `BaseMetric` — the normal case.
- `DualTargetMetric` — scores the question *or* the answer with one
  implementation. Set `name_suffix` and `sort_key_by_target` (both positions).
- `RagasBackedMetric` — only if it genuinely cannot run on a bare `BaseLLM`.
  Prefer not to: every ragas-backed metric is unavailable to users who skipped
  the optional extra.

## 3. Write the class

```python
class MyMetric(BaseMetric):
    metric_name = "my_metric"
    required_inputs = {"input", "actual_output"}
    produces_reasoning = True     # adds {name}_reasoning behind the score
    sort_key = 450                # position in the export; values are spaced

    async def a_measure(self, test_case):
        try:
            raw = await self._arun_prompt(_PROMPT, _VARS, {...})
            self.score = _parse(raw)
        except Exception as exc:          # never propagate
            self.record_failure(exc)
            self.score = None
        self.is_successful()
        return self.score
```

Checks before moving on:

- [ ] Implements `a_measure` only — `measure` comes from the base.
- [ ] Never raises: catches, calls `record_failure`, scores `None`.
- [ ] Reaches the model only through `_run_prompt` / `_arun_prompt`
      (i.e. `BaseLLM.generate` / `a_generate`). No langchain import.
- [ ] Any heavy import (transformers, torch, presidio, ragas) is inside the
      function or an `@lru_cache` module-level loader — never at module top.
- [ ] Multi-column score? Override `expand(score)`, returning **the same keys for
      every score including `None`**.
- [ ] Mutable per-run state? Override `clone()` and reset it — see
      `AnswerCorrectnessMetric.clone`.

## 4. Export it

Add to both the import block and `__all__` in `llminspector/metrics/__init__.py`,
in the right group. Do **not** add it to the package root.

## 5. Update the frozen column contract

`tests/test_column_contract.py` pins `GOLDEN_HEADER` — the exact exported column
order. Insert your columns at the position your `sort_key` implies. If the test
fails after that, your `sort_key` and the header disagree; fix the one that is
wrong rather than forcing the expectation.

## 6. Test it

In `tests/test_metrics.py`: a stub model whose `a_generate` returns a canned
response, one test per scoring branch plus the failure path (assert `score is
None` and `error` is set). No network.

## 7. Document it

`docs/guides/03_metrics.md` — a row in the metric table and, if the metric has
non-obvious inputs or output columns, a short runnable snippet.

## 8. Verify

```bash
.venv/bin/python -m pytest -q
```
