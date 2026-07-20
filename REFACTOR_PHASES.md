# LLMInspector Refactor — Phased Plan

Companion to [REFACTOR_TARGET.md](REFACTOR_TARGET.md) (the end-state). Each phase is
**independently implementable, checkable, and revertible**. **Do not start a phase until
the previous phase's exit criteria pass.** Every phase ends with a concrete verification step,
so we validate incrementally instead of bulk-implementing and debugging everything at once.

Legend: 🔨 build · ✅ exit criteria · ⚠️ decision to confirm during the phase.

---

## Phase 0 — Scaffolding & packaging rename
🔨
- Create the empty `llminspector/` package tree (all dirs from REFACTOR_TARGET.md with `__init__.py` stubs).
- `pyproject.toml`: `name = "llminspector"`; `include = ["llminspector*"]`; **remove** the
  streamlit `package-data` block and the `streamlit` / `streamlit-aggrid` deps; keep the
  `pydnx_packaging` build backend.
- `setup.py`: `PROJECT = "llminspector"`.
- Keep the old `llm_inspector/` in place for now (parallel) so nothing breaks mid-migration.

✅ `pip install -e .` succeeds · `import llminspector` works · `llminspector.__version__` resolves.

---

## Phase 1 — Schema layer (`test_case`, `dataset`, `config`)
🔨
- `LLMTestCase` (pydantic): `input, actual_output, expected_output, retrieval_context, policy`
  — maps the current 5 inputs (`question/answer/ground_truth/contexts/policy`).
- `Golden` + `EvaluationDataset` with `from_pandas/from_excel/to_pandas/to_excel` and
  column-name mapping (reproduces `helper.py`'s `*_col` defaults).
- `Settings` dataclass (+ optional `.from_env()`).

✅ Unit test round-trips a sample `.xlsx` (from `example/Data/`) into an `EvaluationDataset`
and back; `LLMTestCase` construction + validation covered.

---

## Phase 2 — Model layer (provider abstraction)
🔨
- `BaseLLM` / `BaseEmbeddingModel` ABCs.
- `AzureOpenAIModel` + `AzureOpenAIEmbedding` unifying **both** auth styles (`api_key` **or**
  `azure_ad_token_provider`); deployment / model / embedding names become constructor args
  (promote the hard-coded `gpt-5-mini` / `gpt-4o-mini` / `ada-002`).

✅ Instantiate a model from code; smoke `generate()` + `embed_text()` (live if creds
available, else mocked) return the expected types.

---

## Phase 3 — Metrics layer (class-based, logic ported)
🔨
- `BaseMetric` ABC. Port every **live** metric's computation verbatim into a subclass,
  grouped per REFACTOR_TARGET.md (`quality / rag / safety / nlp / policy`). Preserve prompts,
  ragas calls, presidio entities + threshold, tiktoken encoding, and arg-order quirks. Reuse
  the JSON-parsing utils.

⚠️ Confirm during this phase:
- Duplicate metric defs in `eval_metrics.py` (e.g. `context_precision` at `:892` and `:973`) — keep one.
- `accuracy_rouge` / `accuracy_bleu` are defined but never wired into the async map — include as metrics or drop.

✅ Per-metric unit tests (extending `tests/test_evaluate.py`) reproduce the legacy numbers on sample rows.

---

## Phase 4 — Evaluate engine
🔨
- Port `helper.py`'s async pipeline into `evaluate(dataset, metrics=[...])` → `EvaluationResult`:
  per-row `asyncio.gather`, batch `as_completed` + tqdm, availability filtering + dependency
  auto-add (answer_correctness → faithfulness / answer_relevancy), `overall_accuracy` /
  `total_tokens`, `reorder_results`. Drive the metric objects' `a_measure()`.
- Replaces the broken top-level `evaluate()` and the stale `MetricsCalculator`.

✅ `evaluate()` on a sample dataset returns enriched results with the same columns / ordering
as the legacy `reorder_results` contract.

---

## Phase 5 — Synthesizer
🔨
- `AlignmentSynthesizer` (3-stage: tag-augment → HF-T5 paraphrase → perturb; perturbations in `perturbations.py`).
- `AdversarialSynthesizer` (curated-bank sample/filter, unchanged algorithm).
- `RagSynthesizer` (ragas `TestsetGenerator` + per-row GT refinement); **fix** the
  `self.test_df` never-set gap; RAG *scoring* now goes through `evaluate()` (Phase 4), so the
  missing `rag_evaluation()` / `export_eval()` are rebuilt as thin wrappers over `evaluate()`.
- Split the 6 used tables out of `constants.py` into `llminspector/data/`; drop the rest.

⚠️ Confirm during this phase:
- The two silent perturbation bugs (`add_contraction` / `add_abbreviation` return the
  unmodified input) — **recommend fix + note** rather than preserve.

✅ Each synthesizer produces a testset from the sample inputs with the documented output columns.

---

## Phase 6 — Reporting, public API, cleanup
🔨
- `reporting/exporters.py` (`to_dataframe` / `to_excel`). Finalize the `llminspector/__init__.py` public surface.
- **Remove** `pages/`, `LLMInspector_main.py`, `.streamlit/`, and the old `llm_inspector/`
  package. Update `README.md` (drop the Streamlit playground section), Sphinx docs, and add
  `examples/` scripts / notebooks replacing the playground.

✅ An end-to-end example script (synthesize → evaluate → export) runs green; no import
references the old package or streamlit.

---

## Phase 7 — Packaging & Artifactory + CI
🔨
- Finalize `pyproject.toml` / `setup.py` for the `pydnx` wheel build; update `.gitlab-ci.yml`
  (and/or `.github/`) test / build / publish jobs; refresh the coverage `omit` paths.

✅ Wheel builds via pydnx; Artifactory upload dry-run succeeds; CI green (pytest + lint + docs).

---

## Overall verification
- After each phase: run that phase's exit-criteria unit tests (`pytest tests/...`).
- Regression anchor: keep a golden run of the legacy `evaluate` output columns to diff
  against the new `evaluate()` (Phase 4).
- Final: `pip install` the built wheel into a clean venv and run the `examples/` end-to-end script.
