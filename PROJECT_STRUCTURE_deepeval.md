# DeepEval — Project Structure

**DeepEval** ("The LLM Evaluation Framework", v4.1.1, Apache-2.0, by Confident AI) is a
Python (with a parallel TypeScript SDK) framework for unit-testing and
evaluating LLM applications and agents — metrics, datasets/synthetic data,
tracing/observability, red teaming, benchmarks, and integration with the
Confident AI platform. Ships as a `pip install deepeval` package **and** as a
set of installable **Agent Skills** for Claude Code / Claude.ai / Cursor.

## 1. Top-level layout

```
deepeval_scratch/
├── deepeval/              # The Python package (source of truth, see §2-3)
├── skills/                # Agent Skills for AI coding assistants (see §4)
├── typescript/             # Parallel TypeScript SDK (src/, examples/, test/)
├── docs/                  # Docusaurus/Fumadocs-style documentation site source
├── examples/               # Runnable example projects (RAG, MCP, tracing, DAG, notebooks…)
├── tests/                  # Pytest suite for the Python package (see §5)
├── scripts/, .scripts/     # Dev tooling: metric-template compiler, changelog generator
├── .github/workflows/      # CI: lint (black), per-area test suites, release, changelog
├── .claude-plugin/         # Claude Code plugin manifest (exposes skills/ as a plugin)
├── .cursor-plugin/         # Cursor plugin manifest (points at skills/)
├── demo_trace_scope/       # Small demo app for tracing
├── assets/                 # Images/logos used in README
├── pyproject.toml          # Python package definition (poetry), version 4.1.1
├── poetry.lock
├── README.md, CONTRIBUTING.md, LICENSE.md, MAINTAINERS.md, CITATION.cff
├── manual_after_evals_iterator.py, test_agentcore_agent.py, test_pydantic_agent.py
│                            # ad-hoc top-level scripts/examples
└── .env.example             # Env vars template (API keys, Confident AI, model providers)
```

## 2. `deepeval/` — the Python package

| Directory | What it does |
| --- | --- |
| `metrics/` (163 files, ~50 submodules) | The evaluation metrics themselves: `g_eval`, `answer_relevancy`, `faithfulness`, `hallucination`, `bias`, `toxicity`, `pii_leakage`, contextual precision/recall/relevancy, conversational & multi-turn variants (`turn_*`), agentic metrics (`tool_correctness`, `task_completion`, `plan_adherence`, `role_adherence`, `agent_loop_detection`, `mcp_use_metric`), `dag`/`conversational_dag` (custom decision-graph metrics), `arena_g_eval`. This is the largest and most central module. |
| `benchmarks/` (73 files) | Public LLM benchmark harnesses: MMLU, HellaSwag, ARC, GSM8K, TruthfulQA, BIG-Bench-Hard, HumanEval, BoolQ, DROP, IFEval, LogiQA, MathQA, SQuAD, WinoGrande, EquityMedQA, etc. |
| `integrations/` (32 files) | Hooks into agent/orchestration frameworks: `crewai`, `google_adk`, `langchain`, `llama_index`, `pydantic_ai`, `strands`, `agentcore`, `hugging_face`, `openinference`. |
| `tracing/` (22 files) | DeepEval's native tracing SDK — `@observe` decorator, span types, `otel/` submodule for OpenTelemetry interop, `offline_evals/`. |
| `optimizer/` (38 files) | Prompt/agent optimization: `algorithms/`, `rewriter/`, `scorer/` — iterative improvement loops driven by eval results. |
| `synthesizer/` (14 files) | Synthetic dataset / golden generation, with `chunking/` and `templates/` for document-grounded data generation. |
| `simulator/` (14 files) | Conversation/agent simulation for multi-turn testing, with `controller/` and `simulation_graph/`. |
| `evaluate/` (16 files) | Core `evaluate()` entrypoint and execution engine (`execute/`). |
| `models/` (32 files) | LLM & embedding model wrappers (`llms/`, `embedding_models/`) used by metrics/synthesizer. |
| `models_integrations/` | Thin adapters bridging third-party model clients into DeepEval's model interface. |
| `dataset/` | Dataset/golden data structures and loading. |
| `test_case/`, `test_run/` | Core test-case schema and pytest test-run lifecycle/reporting. |
| `cli/` (17 files) | `deepeval` CLI: `auth/`, `generate/`, `test/`, `diagnose/` subcommands. |
| `confident/` | Confident AI platform client (login, upload results/traces). |
| `red_teaming/` | Adversarial/safety testing utilities (empty of `.py` files at top — likely re-exports or in submodules). |
| `annotation/` | Human/LLM annotation utilities for datasets or traces. |
| `prompt/` | Prompt object/versioning abstractions. |
| `openai/`, `anthropic/`, `openai_agents/` | Provider-specific convenience wrappers/instrumentation. |
| `inspect/` (12 files) | Trace/eval inspection tooling, with `fixtures/` and `widgets/` (likely notebook/UI helpers). |
| `config/` | Settings, env var loading (`autoload_dotenv`, `get_settings`). |
| `templates/`, `scorer/`, `plugins/` | Prompt templates for metrics, scoring primitives, pytest plugin hooks. |
| Root files | `__init__.py` (public API surface: `evaluate`, `assert_test`, `login`, `instrument`, …), `constants.py`, `errors.py`, `key_handler.py`, `telemetry.py`, `utils.py`, `contextvars.py`, `progress_context.py`, `_version.py`. |

## 3. Base-class architecture (metrics, synthesizer, evaluate, models, dataset, test_case)

The package is not uniformly OOP — some subpackages are real ABC hierarchies
with enforced contracts, others are single concrete classes, and one has a
vestigial base class that's dead code. Findings below are grounded in the
actual source (file:line), `deepeval/tracing/` excluded as out of scope.

### 3.1 `deepeval/metrics/` — real hierarchy, most extended

`deepeval/metrics/base_metric.py` defines three sibling bases, each mixing in
`PromptMixin` (adds `_get_prompt()` for template resolution). They use
`@abstractmethod` for documentation but the classes are plain `class X(PromptMixin)`,
**not** `ABC` subclasses — the contract is advisory, not enforced at
instantiation. Every subclass is hooked into cost/token tracking via
`__init_subclass__` → `observe_methods(cls)`.

- `class BaseMetric(PromptMixin)` — single-turn (`LLMTestCase`). Abstract
  `measure()`/`a_measure() -> float`, `is_successful() -> bool`; concrete state
  `threshold`, `score`, `success`, `reason`, `strict_mode`, `async_mode`.
- `class BaseConversationalMetric(PromptMixin)` — same shape but over
  `ConversationalTestCase`. Used directly by conversational, DAG-conversational,
  MCP multi-turn, **and all `turn_*` metrics** — there is no separate
  intermediate "turn" base class.
- `class BaseArenaMetric(PromptMixin)` — `measure`/`a_measure` take
  `ArenaTestCase` and return a `str` verdict instead of a `float`.

Leaf examples:

| Metric | Extends | Overrides |
| --- | --- | --- |
| `AnswerRelevancyMetric` (`metrics/answer_relevancy/`) | `BaseMetric` | `measure`, `a_measure`, `is_successful` |
| `GEval` (`metrics/g_eval/`) | `BaseMetric` | same trio; also composed inside DAG `VerdictNode`s |
| `ToolCorrectnessMetric` (`metrics/tool_correctness/`) | `BaseMetric` | deterministic (non-LLM) `measure`/`a_measure` |
| `TurnRelevancyMetric` (`metrics/turn_relevancy/`) | `BaseConversationalMetric` | same pattern as `turn_faithfulness`, `turn_contextual_*` |
| `ConversationalGEval` | `BaseConversationalMetric` | multi-turn G-Eval |
| `ArenaGEval` | `BaseArenaMetric` | pairwise-comparison verdict |

**DAG metrics have their own, entirely separate node hierarchy** (not built on
`BaseMetric`/`BaseConversationalMetric` at the node level):

- `metrics/dag/nodes.py` `class BaseNode(PromptMixin)` → `VerdictNode`,
  `TaskNode`, `BinaryJudgementNode`, `NonBinaryJudgementNode`. Walked by
  `metrics/dag/dag.py` `class DAGMetric(BaseMetric)`.
- `metrics/conversational_dag/nodes.py` `class ConversationalBaseNode(PromptMixin)`
  → its own `ConversationalVerdictNode`/`ConversationalTaskNode`/etc., walked by
  `class ConversationalDAGMetric(BaseConversationalMetric)`. **No inheritance
  link between the two node hierarchies** — they're parallel, duplicated.

### 3.2 `deepeval/synthesizer/` — compositional, not a hierarchy

`synthesizer/base_synthesizer.py` defines `class BaseSynthesizer` — **confirmed
dead code**: it's never imported or subclassed anywhere in the package.
`synthesizer/synthesizer.py` `class Synthesizer` is a standalone concrete
class that does *not* extend it. `Synthesizer` composes a `DocumentChunker`
(`chunking/doc_chunker.py`) and a `ContextGenerator` (`chunking/context_generator.py`),
both standalone with no base class, plus static-method template "bags"
(`SynthesizerTemplate`, `FilterTemplate`, `EvolutionTemplate`,
`ExtractionTemplate`, `PromptSynthesizerTemplate`, …) that share no base
either. Extension here happens by composition/configuration, not subclassing.

### 3.3 `deepeval/evaluate/` — no classes at all

`grep -n "^class"` across every file in `evaluate/execute/` (`_common.py`,
`e2e.py`, `agentic.py`, `loop.py`, `trace_scope.py`) returns nothing. Single-turn
vs. conversational, sync vs. async, and traced vs. non-traced modes are all
handled by separate top-level functions (`execute_test_cases()`,
`execute_agentic_test_cases_from_loop()`, `a_execute_agentic_test_cases_from_loop()`)
and branching, not by polymorphism over a base "executor"/"strategy" class.
`evaluate/evaluate.py`'s public `evaluate()`/`assert_test()` are plain functions.

### 3.4 `deepeval/models/` — real ABCs, most disciplined hierarchy

`deepeval/models/base_model.py` defines three independent `ABC` subclasses
(these *are* enforced — instantiating a subclass that skips an abstract
method raises `TypeError`):

- `class DeepEvalBaseModel(ABC)` — generic scorer base: abstract `load_model()`,
  `_call()`; `__call__` delegates to `_call`.
- `class DeepEvalBaseLLM(ABC)` — abstract `load_model()`, `generate() -> str`,
  `async a_generate() -> str`, `get_model_name() -> str`; overridable
  `batch_generate`, `supports_log_probs/temperature/multimodal/structured_outputs/json_mode`,
  `generate_with_schema`. `__init_subclass__` wires every subclass into
  cost/token observability the same way `BaseMetric` does.
- `class DeepEvalBaseEmbeddingModel(ABC)` — abstract `load_model()`,
  `embed_text()`/`a_embed_text()`, `embed_texts()`/`a_embed_texts()`, `get_model_name()`.

Concrete `DeepEvalBaseLLM` wrappers (`models/llms/`), each implementing
`load_model`/`generate`/`a_generate`/`get_model_name`: `GPTModel`
(`openai_model.py`), `AnthropicModel`, `GeminiModel`, `OllamaModel`, `LocalModel`.
Concrete `DeepEvalBaseEmbeddingModel` wrappers (`models/embedding_models/`):
`OpenAIEmbeddingModel`, `LocalEmbeddingModel`. Legacy non-LLM scorer models
(`AnswerRelevancyModel`, `DetoxifyModel`, `UnBiasedModel`, `SummaCModels`)
extend `DeepEvalBaseModel` instead; `HallucinationModel` is a `Singleton`, not
a `DeepEvalBaseModel` subclass at all.

### 3.5 `deepeval/dataset/` — independent pydantic models, no shared base

- `dataset/golden.py` `class Golden(BaseModel)` — single-turn: `input`,
  `actual_output`, `expected_output`, `context`, `retrieval_context`,
  `tools_called`, `expected_tools`, `multimodal`.
- `dataset/golden.py` `class ConversationalGolden(BaseModel)` — **independent**
  pydantic model, *not* a subclass of `Golden`: `scenario`, `expected_outcome`,
  `user_description`, `turns: Optional[List[Turn]]`, `multimodal`. Duplicates
  `Golden`'s multimodal-validator logic rather than sharing it via a mixin.
- `dataset/dataset.py` `class EvaluationDataset` — a plain `@dataclass` (not
  pydantic) holding `_goldens`, `_conversational_goldens`, `_llm_test_cases`,
  `_conversational_test_cases`; single concrete class, no abstract base.

### 3.6 `deepeval/test_case/` — same pattern: parallel models, no shared base

- `test_case/llm_test_case.py` `class LLMTestCase(BaseModel)` — single-turn:
  `input`, `actual_output`, `expected_output`, `context`, `retrieval_context`,
  `tools_called`, `expected_tools`, `mcp_*`, `multimodal: bool`. **There is no
  separate `MLLMTestCase`** — multimodality is a flag + regex placeholder
  parsing (`[DEEPEVAL:IMAGE:...]`) inside `LLMTestCase` itself.
- `test_case/conversational_test_case.py` `class ConversationalTestCase(BaseModel)`
  — multi-turn: `turns: List[Turn]`, `scenario`, `expected_outcome`,
  `user_description`, `chatbot_role`. Same file's `class Turn(BaseModel)` is
  one conversation turn (`role`, `content`, `retrieval_context`, `tools_called`).
- `test_case/arena_test_case.py` `class ArenaTestCase` — a `@dataclass` (not
  `BaseModel`) holding `contestants: List[Contestant]`, where `Contestant`
  wraps a `name` + an `LLMTestCase` + `hyperparameters`.
- `test_case/llm_test_case.py` `class ToolCall(BaseModel)` — `name`, `type`,
  `input_parameters`, `output`, custom `__eq__`/`__hash__`.
- `test_case/mcp.py` — `MCPToolCall`, `MCPPromptCall`, `MCPResourceCall`, each
  an independent `BaseModel` (no shared base), plus plain-class `MCPServer`.

No inheritance links `LLMTestCase` ↔ `ConversationalTestCase` or `Golden` ↔
`ConversationalGolden` — the single-turn/multi-turn parallelism across
`dataset/` and `test_case/` is achieved purely by convention and duplicated
field/validator logic, not by a shared base class.

## 4. `skills/deepeval/` — main evaluation workflow skill

Exposed as a Claude Code plugin via [.claude-plugin/plugin.json](.claude-plugin/plugin.json)
(`"skills": "./skills/"`) and to Cursor via [.cursor-plugin/plugin.json](.cursor-plugin/plugin.json).
Installable standalone via `npx skills add confident-ai/deepeval --skill "deepeval"`
or manual copy into `.claude/skills/`.

Triggers when a user wants to **evaluate or improve an AI agent/RAG/chatbot**:
add evals, generate datasets/goldens, run `deepeval generate` / `deepeval test
run`, send results to Confident AI, or iterate on failures.

- [SKILL.md](skills/deepeval/SKILL.md) — workflow: inspect app → intake questions → reuse/generate
  dataset → instrument (delegates out to a separate tracing skill, not covered
  here) → `deepeval test run` → iterate (default 5 rounds).
- `references/`
  - `intake.md` — required clarifying questions to ask before starting.
  - `choose-use-case.md` — decision guide for which eval pattern fits.
  - `metrics.md` — which built-in metric to use for which failure mode.
  - `datasets.md` — reusing vs. generating goldens.
  - `synthetic-data.md` — `deepeval generate` synthetic golden generation.
  - `pytest-e2e-evals.md` — building a committed pytest eval suite.
  - `traced-evals.md` — running evals against traced spans.
  - `iteration-loop.md` — the failure → fix → re-run loop.
  - `confident-ai.md` — sending results/reports to Confident AI.
  - `artifact-contracts.md` — expected shape of generated files/artifacts.
- `templates/` — `metrics.py`, `test_single_turn_no_tracing.py`,
  `test_single_turn_tracing.py`, `test_multi_turn_e2e.py` (copy-paste pytest
  starters for each eval pattern).
- Carries its own `LICENSE` (Apache-2.0), and `skills/README.md` documents
  install paths for Claude.ai, Claude Code, Cursor, and the `npx skills` CLI.

## 5. Other top-level areas

- **`typescript/`** — parallel TS SDK: `src/{annotation,cli,confident,config,dataset,
  evaluate,governance,integrations,metrics,models,openai,prompt,simulate,
  templates,test-case,tracing}`, its own `examples/` and `test/`.
- **`docs/`** — the deepeval.com documentation site source (Next.js-style:
  `app/`, `content/{blog,changelog,docs,guides,integrations,tutorials}`,
  `components/`, `snippets/`, `public/`).
- **`examples/`** — `getting_started/`, `rag_evaluation/`, `mcp_evaluation/`,
  `tracing/`, `dag-examples/`, `notebooks/`, `community/` contributed examples.
- **`tests/`** — `test_core/`, `test_metrics/`, `test_integrations/`,
  `test_confident/`, `test_templates/`, `test_docs/` — mirrors CI workflows in
  `.github/workflows/` (`test_core.yml`, `test_metrics.yml`,
  `test_integrations.yml`, `test_confident.yml`, `test_metric_templates.yml`,
  plus `black.yml` lint, `release.yml`, `changelog.yml`).
- **`scripts/` / `.scripts/`** — `compile_metric_templates.py`,
  `check_openai_model_capabilities.py`, and changelog generation scripts.
