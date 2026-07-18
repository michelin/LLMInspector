# DeepEval — Complete Project Tree & Architecture

> **Repository**: `deepeval_scratch/` — a fork/scratch workspace of [confident-ai/deepeval](https://github.com/confident-ai/deepeval) (v4.1.1)
> **Purpose**: The open-source LLM Evaluation Framework, built to function like `pytest` but specialized for unit-testing LLM applications (agents, RAG pipelines, chatbots).

---

## Table of Contents

1. [Full Project Tree](#full-project-tree)
2. [Root-Level Files & Config](#root-level-files--config)
3. [Core Python Package — `deepeval/`](#core-python-package--deepeval)
4. [Agent Skills System — `skills/`](#agent-skills-system--skills)
5. [Coding Agent Plugin Manifests — `.claude-plugin/` & `.cursor-plugin/`](#coding-agent-plugin-manifests)
6. [Pre-Commit Hooks — `.pre-commit-config.yaml`](#pre-commit-hooks)
7. [Pytest Plugin & Hook System — `deepeval/plugins/` & `deepeval/test_run/hooks.py`](#pytest-plugin--hook-system)
8. [CI/CD Pipelines — `.github/workflows/`](#cicd-pipelines)
9. [Documentation Site — `docs/`](#documentation-site--docs)
10. [TypeScript SDK — `typescript/`](#typescript-sdk--typescript)
11. [Design Philosophy Summary](#design-philosophy-summary)

---

## Full Project Tree

```
deepeval_scratch/
│
├── .claude-plugin/                          # 🤖 Claude Code agent plugin manifest
│   ├── marketplace.json                     #    Marketplace listing metadata for plugin discovery
│   └── plugin.json                          #    Plugin manifest pointing to skills/
│
├── .cursor-plugin/                          # 🤖 Cursor IDE agent plugin manifest
│   └── plugin.json                          #    Plugin manifest pointing to skills/
│
├── .github/                                 # ⚙️ GitHub-specific config
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md                    #    Bug report issue template
│   │   └── feature_request.md               #    Feature request issue template
│   └── workflows/
│       ├── black.yml                        #    CI: Black code formatting check on PRs
│       ├── changelog.yml                    #    CI: Auto-generate changelog on release
│       ├── full_test_core_for_pr.yml        #    CI: Full core test suite for PR validation
│       ├── pr-title-check.yml               #    CI: Enforce PR title conventions
│       ├── release.yml                      #    CI: Publish to PyPI on release tag
│       ├── test_confident.yml               #    CI: Confident AI integration tests
│       ├── test_core.yml                    #    CI: Core library unit tests
│       ├── test_integrations.yml            #    CI: Third-party integration tests
│       ├── test_metrics.yml                 #    CI: Metric accuracy tests
│       ├── test_metric_templates.yml        #    CI: Metric template validation
│       ├── typescript_lint.yml              #    CI: ESLint for TypeScript SDK
│       └── typescript_test.yml              #    CI: Jest tests for TypeScript SDK
│
├── .scripts/                                # 🔧 Internal maintainer automation
│   └── changelog/
│       ├── generate.py                      #    Auto-generate changelog from git history
│       └── release_notes.py                 #    Format release notes for GitHub Releases
│
├── .vscode/                                 # 🖥️ VS Code workspace settings
│   └── settings.json                        #    Format-on-save with Prettier for JS/CSS
│
├── assets/                                  # 🖼️ README & docs media
│   ├── confident-mcp-architecture.png       #    MCP architecture diagram
│   ├── demo.gif                             #    Animated demo for README hero
│   └── hero/
│       ├── wordmark-dark.svg                #    Dark-mode logo
│       └── wordmark-light.svg               #    Light-mode logo
│
├── deepeval/                                # 📦 CORE PYTHON PACKAGE (see detailed section below)
│   ├── __init__.py                          #    Public API exports
│   ├── _version.py                          #    Single-source version string
│   ├── constants.py                         #    Global constants & env var names
│   ├── contextvars.py                       #    Python contextvars for async-safe state
│   ├── errors.py                            #    Custom exception hierarchy
│   ├── key_handler.py                       #    API key management & validation
│   ├── progress_context.py                  #    Rich progress bars for CLI output
│   ├── py.typed                             #    PEP 561 type-checking marker
│   ├── singleton.py                         #    Singleton metaclass for managers
│   ├── telemetry.py                         #    Anonymous usage analytics (PostHog)
│   ├── utils.py                             #    Shared utility functions
│   │
│   ├── annotation/                          #    Human annotation / labeling API
│   │   ├── __init__.py
│   │   ├── annotation.py                    #    Annotation data models & workflows
│   │   └── api.py                           #    Confident AI annotation REST calls
│   │
│   ├── anthropic/                           #    Anthropic Claude SDK patching
│   │   ├── __init__.py
│   │   ├── extractors.py                    #    Extract spans from Anthropic responses
│   │   ├── patch.py                         #    Monkey-patch Anthropic client for tracing
│   │   └── utils.py                         #    Anthropic-specific helpers
│   │
│   ├── benchmarks/                          #    Standard LLM benchmark runners
│   │   ├── __init__.py
│   │   ├── results.py                       #    Benchmark result aggregation
│   │   ├── schema.py                        #    Shared data schemas
│   │   ├── tasks/                           #    Task enum definitions
│   │   ├── utils.py                         #    Benchmark utilities
│   │   ├── big_bench_hard/                  #    BIG-Bench Hard (27 sub-tasks)
│   │   │   ├── cot_prompts/                 #    Chain-of-thought prompt files (.txt)
│   │   │   └── shot_prompts/                #    Few-shot prompt files (.txt)
│   │   ├── bool_q/                          #    BoolQ benchmark
│   │   ├── drop/                            #    DROP benchmark
│   │   ├── equity_med_qa/                   #    EquityMedQA benchmark
│   │   ├── gsm8k/                           #    GSM8K math benchmark
│   │   ├── hellaswag/                       #    HellaSwag benchmark
│   │   ├── human_eval/                      #    HumanEval code benchmark
│   │   ├── ifeval/                          #    IFEval benchmark
│   │   ├── lambada/                         #    LAMBADA benchmark
│   │   ├── logi_qa/                         #    LogiQA benchmark
│   │   ├── math_qa/                         #    MathQA benchmark
│   │   ├── mmlu/                            #    MMLU benchmark
│   │   ├── squad/                           #    SQuAD benchmark
│   │   ├── truthful_qa/                     #    TruthfulQA benchmark
│   │   └── winogrande/                      #    WinoGrande benchmark
│   │
│   ├── cli/                                 #    Command-line interface (Typer app)
│   │   ├── __init__.py
│   │   ├── main.py                          #    Typer app entry point (`deepeval` command)
│   │   ├── types.py                         #    CLI-specific enums & types
│   │   ├── utils.py                         #    CLI helper functions
│   │   ├── dotenv_handler.py                #    .env file auto-loading logic
│   │   ├── inspect.py                       #    `deepeval inspect` subcommand
│   │   ├── auth/                            #    `deepeval login` / auth flow
│   │   │   ├── api.py                       #    Auth REST API calls
│   │   │   ├── command.py                   #    Login CLI command
│   │   │   └── flow.py                      #    OAuth/browser auth flow
│   │   ├── generate/                        #    `deepeval generate` (synthetic data CLI)
│   │   │   ├── command.py                   #    Generate CLI command
│   │   │   └── utils.py                     #    Generation helpers
│   │   └── test/                            #    `deepeval test run` subcommand
│   │       └── command.py                   #    Test execution CLI command
│   │
│   ├── config/                              #    Runtime configuration management
│   │
│   ├── confident/                           #    Confident AI cloud platform API
│   │   ├── __init__.py
│   │   ├── api.py                           #    REST client for Confident AI
│   │   └── types.py                         #    API response/request models
│   │
│   ├── dataset/                             #    Dataset & Golden management
│   │   ├── __init__.py
│   │   ├── api.py                           #    Cloud dataset API calls
│   │   ├── dataset.py                       #    EvaluationDataset class
│   │   ├── golden.py                        #    Golden (ground-truth) data model
│   │   ├── test_run_tracer.py               #    Dataset-level trace orchestration
│   │   ├── types.py                         #    Dataset type enums
│   │   └── utils.py                         #    Dataset utility functions
│   │
│   ├── evaluate/                            #    Core evaluation execution engine
│   │   └── (evaluation runner, async orchestrator, result compilation)
│   │
│   ├── inspect/                             #    `deepeval inspect` TUI (Textual app)
│   │   ├── __init__.py
│   │   ├── __main__.py                      #    TUI entry point
│   │   ├── app.py                           #    Textual application class
│   │   ├── loader.py                        #    Load test run data for inspection
│   │   ├── styles.tcss                      #    Textual CSS stylesheet
│   │   ├── types.py                         #    TUI data types
│   │   ├── fixtures/
│   │   │   └── test_run_sample.json         #    Sample data for TUI development
│   │   └── widgets/                         #    Textual UI widgets
│   │       ├── details.py                   #    Detail pane widget
│   │       ├── header_bar.py                #    Header bar widget
│   │       ├── help_modal.py                #    Help overlay widget
│   │       ├── search_bar.py                #    Search/filter widget
│   │       ├── span_tree.py                 #    Span tree view widget
│   │       └── _styling.py                  #    Widget style helpers
│   │
│   ├── integrations/                        #    Third-party framework integrations
│   │   ├── __init__.py
│   │   ├── README.md                        #    Integration overview
│   │   ├── agentcore/                       #    AgentCore SDK integration
│   │   │   ├── instrumentator.py            #    Auto-instrumentation
│   │   │   └── otel.py                      #    OTel attribute mapping
│   │   ├── crewai/                          #    CrewAI multi-agent integration
│   │   │   ├── handler.py                   #    CrewAI callback handler
│   │   │   ├── subs.py                      #    Subscription wiring
│   │   │   ├── tool.py                      #    CrewAI tool wrapping
│   │   │   └── wrapper.py                   #    Agent wrapper
│   │   ├── google_adk/                      #    Google Agent Development Kit
│   │   │   └── otel.py                      #    OTel attribute mapping
│   │   ├── hugging_face/                    #    HuggingFace Transformers
│   │   │   ├── callback.py                  #    Trainer callback
│   │   │   ├── rich_manager.py              #    Rich progress integration
│   │   │   └── utils.py
│   │   ├── langchain/                       #    LangChain integration
│   │   │   ├── callback.py                  #    LangChain callback handler
│   │   │   ├── patch.py                     #    Monkey-patching for tracing
│   │   │   └── utils.py
│   │   ├── llama_index/                     #    LlamaIndex integration
│   │   │   ├── handler.py                   #    LlamaIndex callback handler
│   │   │   └── utils.py
│   │   ├── openinference/                   #    OpenInference standard
│   │   │   ├── instrumentator.py            #    Auto-instrumentation
│   │   │   └── otel.py                      #    OTel attribute mapping
│   │   ├── pydantic_ai/                     #    Pydantic AI integration
│   │   │   ├── README.md
│   │   │   ├── instrumentator.py            #    Auto-instrumentation
│   │   │   └── otel.py                      #    OTel attribute mapping
│   │   └── strands/                         #    Strands Agents integration
│   │       ├── instrumentator.py            #    Auto-instrumentation
│   │       └── otel.py                      #    OTel attribute mapping
│   │
│   ├── metrics/                             #    📐 ALL EVALUATION METRICS
│   │   ├── __init__.py
│   │   ├── README.md                        #    Metrics overview
│   │   ├── base_metric.py                   #    Abstract base metric class
│   │   ├── indicator.py                     #    Pass/fail indicator logic
│   │   ├── ragas.py                         #    RAGAS compatibility layer
│   │   ├── retrieval_context_display.py     #    Retrieval context formatting
│   │   ├── utils.py                         #    Metric utility functions
│   │   │
│   │   │  # — Each metric is a self-contained subdirectory with:
│   │   │  #   metric.py    — Metric class (inherits BaseMetric)
│   │   │  #   schema.py    — Pydantic models for LLM-judge responses
│   │   │  #   templates/   — Jinja2/text prompt templates for LLM-as-judge
│   │   │  #     class.txt  — System prompt defining the judge's persona
│   │   │  #     *.txt      — Step-specific prompt fragments
│   │   │
│   │   ├── answer_relevancy/                #    Answer Relevancy metric
│   │   ├── bias/                            #    Bias detection metric
│   │   ├── citation_faithfulness/           #    Citation Faithfulness metric
│   │   ├── contextual_precision/            #    Contextual Precision metric
│   │   ├── contextual_recall/               #    Contextual Recall metric
│   │   ├── contextual_relevancy/            #    Contextual Relevancy metric
│   │   ├── conversation_completeness/       #    Conversation Completeness metric
│   │   ├── dag/                             #    DAG-based deterministic metric builder
│   │   ├── exact_match/                     #    Exact Match metric
│   │   ├── faithfulness/                    #    Faithfulness (groundedness) metric
│   │   ├── g_eval/                          #    G-Eval (research-backed LLM judge)
│   │   ├── goal_accuracy/                   #    Agent Goal Accuracy metric
│   │   ├── hallucination/                   #    Hallucination detection metric
│   │   ├── image_coherence/                 #    Image Coherence metric
│   │   ├── image_editing/                   #    Image Editing quality metric
│   │   ├── image_helpfulness/               #    Image Helpfulness metric
│   │   ├── image_reference/                 #    Image Reference metric
│   │   ├── json_correctness/               #    JSON Correctness metric
│   │   ├── knowledge_retention/             #    Knowledge Retention metric
│   │   ├── mcp_task_completion/             #    MCP Task Completion metric
│   │   ├── mcp_use/                         #    MCP Tool Use metric
│   │   ├── non_advice/                      #    Non-Advice metric
│   │   ├── pattern_match/                   #    Pattern Match metric
│   │   ├── pii_leakage/                     #    PII Leakage detection metric
│   │   ├── plan_adherence/                  #    Agent Plan Adherence metric
│   │   ├── plan_quality/                    #    Agent Plan Quality metric
│   │   ├── prompt_alignment/                #    Prompt Alignment metric
│   │   ├── role_adherence/                  #    Role Adherence metric
│   │   ├── role_violation/                  #    Role Violation detection metric
│   │   ├── step_efficiency/                 #    Agent Step Efficiency metric
│   │   ├── summarization/                   #    Summarization quality metric
│   │   ├── task_completion/                 #    Agent Task Completion metric
│   │   ├── text_to_image/                   #    Text-to-Image metric
│   │   ├── tool_correctness/                #    Tool Correctness metric
│   │   ├── tool_permission/                 #    Tool Permission metric
│   │   ├── tool_use/                        #    Tool Use quality metric
│   │   ├── topic_adherence/                 #    Topic Adherence metric
│   │   ├── toxicity/                        #    Toxicity detection metric
│   │   ├── turn_contextual_precision/       #    Multi-turn Contextual Precision
│   │   ├── turn_contextual_recall/          #    Multi-turn Contextual Recall
│   │   ├── turn_contextual_relevancy/       #    Multi-turn Contextual Relevancy
│   │   ├── turn_faithfulness/               #    Multi-turn Faithfulness
│   │   └── turn_relevancy/                  #    Multi-turn Relevancy
│   │
│   ├── model_integrations/                  #    Model provider abstractions
│   │   ├── __init__.py
│   │   ├── types.py                         #    Provider type enums
│   │   └── utils.py                         #    Provider detection utilities
│   │
│   ├── models/                              #    LLM wrappers for "LLM-as-judge"
│   │   ├── __init__.py
│   │   ├── base_model.py                    #    Abstract DeepEvalBaseLLM
│   │   ├── retry_policy.py                  #    Retry/backoff for LLM calls
│   │   ├── utils.py                         #    Model utility functions
│   │   ├── answer_relevancy_model.py        #    Specialized relevancy NLP model
│   │   ├── detoxify_model.py                #    Detoxify (local toxicity) model
│   │   ├── hallucination_model.py           #    Hallucination NLP model
│   │   ├── summac_model.py                  #    SummaC faithfulness model
│   │   ├── _summac_model.py                 #    Internal SummaC implementation
│   │   ├── unbias_model.py                  #    Bias detection NLP model
│   │   ├── embedding_models/                #    Embedding model wrappers
│   │   │   ├── azure_embedding_model.py
│   │   │   ├── local_embedding_model.py
│   │   │   ├── ollama_embedding_model.py
│   │   │   └── openai_embedding_model.py
│   │   └── llms/                            #    LLM provider wrappers
│   │       ├── amazon_bedrock_model.py
│   │       ├── anthropic_model.py
│   │       ├── azure_model.py
│   │       ├── constants.py                 #    Default model names & limits
│   │       ├── deepseek_model.py
│   │       ├── gateway_model.py
│   │       ├── gemini_model.py
│   │       ├── grok_model.py
│   │       ├── kimi_model.py
│   │       ├── litellm_model.py
│   │       ├── local_model.py
│   │       ├── ollama_model.py
│   │       ├── openai_model.py
│   │       ├── openrouter_model.py
│   │       ├── portkey_model.py
│   │       └── utils.py
│   │
│   ├── openai/                              #    OpenAI SDK patching for tracing
│   │   ├── __init__.py
│   │   ├── extractors.py                    #    Extract spans from OpenAI responses
│   │   ├── patch.py                         #    Monkey-patch OpenAI client
│   │   └── utils.py
│   │
│   ├── openai_agents/                       #    OpenAI Agents SDK integration
│   │   ├── __init__.py
│   │   ├── agent.py                         #    Agent wrapper class
│   │   ├── callback_handler.py              #    Agent lifecycle callback handler
│   │   ├── extractors.py                    #    Extract spans from agent runs
│   │   ├── patch.py                         #    Monkey-patch Agents SDK
│   │   └── runner.py                        #    Wrapped agent runner
│   │
│   ├── optimizer/                           #    Prompt optimization algorithms
│   │   ├── __init__.py
│   │   ├── configs.py                       #    Optimizer configuration
│   │   ├── policies.py                      #    Optimization policies
│   │   ├── prompt_optimizer.py              #    Main optimizer class
│   │   ├── types.py                         #    Optimizer type enums
│   │   ├── utils.py
│   │   ├── algorithms/                      #    Optimization algorithm implementations
│   │   │   ├── base.py                      #    Abstract algorithm base
│   │   │   ├── configs.py                   #    Algorithm-level configs
│   │   │   ├── copro/                       #    COPRO algorithm
│   │   │   ├── gepa/                        #    GEPA algorithm
│   │   │   ├── miprov2/                     #    MIPROv2 algorithm
│   │   │   └── simba/                       #    SIMBA algorithm
│   │   ├── rewriter/                        #    Prompt rewriting engine
│   │   └── scorer/                          #    Optimization scoring
│   │
│   ├── plugins/                             #    🔌 PYTEST PLUGIN (see hooks section)
│   │   ├── __init__.py
│   │   └── plugin.py                        #    pytest11 entry point with hooks
│   │
│   ├── prompt/                              #    Prompt versioning & management
│   │   ├── __init__.py
│   │   ├── api.py                           #    Prompt API calls to Confident AI
│   │   ├── prompt.py                        #    Prompt class
│   │   └── utils.py
│   │
│   ├── red_teaming/                         #    Red teaming / adversarial testing
│   │   └── README.md
│   │
│   ├── scorer/                              #    Scoring utilities
│   │   ├── __init__.py
│   │   └── scorer.py                        #    Scorer implementation
│   │
│   ├── simulator/                           #    Conversation simulation engine
│   │   ├── __init__.py
│   │   ├── conversation_simulator.py        #    Main simulator class
│   │   ├── schema.py                        #    Simulator data models
│   │   ├── template.py                      #    Simulation prompt templates
│   │   ├── utils.py
│   │   ├── controller/                      #    Simulation flow controller
│   │   │   ├── controller.py
│   │   │   ├── template.py
│   │   │   └── types.py
│   │   └── simulation_graph/                #    Graph-based simulation engine
│   │       ├── default.py                   #    Default simulation graph
│   │       ├── node.py                      #    Graph node logic
│   │       ├── runner.py                    #    Graph runner
│   │       └── template.py
│   │
│   ├── synthesizer/                         #    Synthetic test data generation
│   │   ├── __init__.py
│   │   ├── base_synthesizer.py              #    Abstract synthesizer base
│   │   ├── config.py                        #    Synthesizer configuration
│   │   ├── schema.py                        #    Data schemas
│   │   ├── synthesizer.py                   #    Main synthesizer class
│   │   ├── types.py                         #    Type definitions
│   │   ├── utils.py
│   │   ├── chunking/                        #    Document chunking for generation
│   │   │   ├── context_generator.py         #    Context generation from chunks
│   │   │   └── doc_chunker.py               #    Document chunker
│   │   └── templates/                       #    Generation prompt templates
│   │       ├── template.py
│   │       ├── template_extraction.py
│   │       └── template_prompt.py
│   │
│   ├── templates/                           #    Compiled metric prompt templates
│   │   ├── __init__.py
│   │   ├── resolver.py                      #    Template loader & resolver
│   │   └── metrics/
│   │       ├── templates.json               #    Compiled template registry
│   │       └── fragments/                   #    Reusable prompt fragments
│   │           ├── faithfulness_verdicts_*.txt
│   │           └── multimodal_*.txt
│   │
│   ├── test_case/                           #    Test case data models
│   │   ├── __init__.py
│   │   ├── api.py                           #    Test case serialization for API
│   │   ├── arena_test_case.py               #    Arena (A/B comparison) test case
│   │   ├── conversational_test_case.py      #    Multi-turn conversation test case
│   │   ├── llm_test_case.py                 #    Core LLMTestCase (input/output/context)
│   │   ├── mcp.py                           #    MCP protocol test case
│   │   └── utils.py                         #    Test case utilities
│   │
│   ├── test_run/                            #    Test run lifecycle management
│   │   ├── __init__.py
│   │   ├── api.py                           #    Test run API serialization
│   │   ├── cache.py                         #    Result caching between runs
│   │   ├── hooks.py                         #    🪝 on_test_run_end hook (see section)
│   │   ├── hyperparameters.py               #    Hyperparameter tracking
│   │   └── test_run.py                      #    TestRun & TestRunManager classes
│   │
│   └── tracing/                             #    🔍 Tracing / observability engine
│       ├── __init__.py
│       ├── api.py                           #    Trace API serialization
│       ├── context.py                       #    Trace context management
│       ├── integrations.py                  #    Integration registry
│       ├── internal.py                      #    Internal tracing utilities
│       ├── patchers.py                      #    Framework monkey-patchers
│       ├── perf_epoch_bridge.py             #    Performance epoch bridging
│       ├── trace_context.py                 #    Async-safe trace context
│       ├── trace_test_manager.py            #    Trace ↔ test run bridging
│       ├── tracing.py                       #    @observe decorator & Observer class
│       ├── types.py                         #    Span types, EvalMode, EvalSession
│       ├── utils.py                         #    Tracing utilities
│       ├── offline_evals/                   #    Offline trace-based evaluation
│       │   ├── api.py                       #    Offline eval API calls
│       │   ├── span.py                      #    Span-level offline evals
│       │   ├── thread.py                    #    Thread-safe eval execution
│       │   └── trace.py                     #    Trace-level offline evals
│       └── otel/                            #    OpenTelemetry bridge
│           ├── context_aware_processor.py   #    Context-aware span processor
│           ├── exporter.py                  #    OTel span exporter to Confident AI
│           ├── test_exporter.py             #    Test-mode exporter
│           └── utils.py                     #    OTel utility functions
│
├── demo_trace_scope/                        # 🧪 Demo/experiment for tracing scopes
│   ├── __init__.py
│   └── test_observed_app.py                 #    Example traced application test
│
├── docs/                                    # 📖 Documentation site (Next.js / Fumadocs)
│   ├── package.json                         #    Node.js project for docs site
│   ├── postcss.config.mjs
│   ├── app/                                 #    Next.js app directory
│   │   ├── layout.tsx                       #    Root layout
│   │   ├── global.css                       #    Global styles
│   │   ├── sitemap.ts                       #    Sitemap generator
│   │   ├── robots.ts                        #    Robots.txt generator
│   │   ├── llms.txt                         #    LLM-readable docs (for agents)
│   │   ├── llms-full.txt                    #    Full LLM-readable docs
│   │   ├── llms.mdx                         #    LLM docs MDX source
│   │   ├── (home)/                          #    Homepage route
│   │   ├── api/                             #    API reference docs
│   │   ├── blog/                            #    Blog section
│   │   ├── changelog/                       #    Changelog section
│   │   ├── docs/                            #    Core documentation pages
│   │   ├── enterprise/                      #    Enterprise docs
│   │   ├── guides/                          #    How-to guides
│   │   ├── integrations/                    #    Integration docs
│   │   ├── og/                              #    OpenGraph image generation
│   │   └── tutorials/                       #    Tutorial pages
│   ├── content/                             #    MDX content source files
│   │   ├── blog/                            #    Blog post MDX files
│   │   └── integrations/                    #    Integration content
│   ├── enterprise/
│   │   └── read-me.mdx
│   └── scripts/                             #    Doc build scripts
│       ├── build-readme-hero.mjs
│       ├── generate-changelog-contributors.mjs
│       ├── generate-contributors.mjs
│       ├── generate-repo-contributors.mjs
│       ├── normalize-admonition-titles.mjs
│       ├── replace-img-with-image-displayer.mjs
│       ├── strip-redundant-mdx-imports.mjs
│       └── timeline-to-steps.mjs
│
├── examples/                                # 📚 Usage examples
│   ├── create_tests.py                      #    Example: creating test cases
│   ├── sample.txt                           #    Sample text for examples
│   ├── getting_started/
│   │   └── test_example.py                  #    Quickstart example
│   ├── community/
│   │   └── chatbot_evaluation/              #    Community chatbot eval example
│   ├── dag-examples/
│   │   └── conversational_dag.ipynb         #    DAG metric notebook
│   ├── mcp_evaluation/                      #    MCP eval examples
│   │   ├── mcp_eval_multi_turn.py
│   │   └── mcp_eval_single_turn.py
│   ├── notebooks/                           #    Jupyter notebooks
│   │   ├── crewai.ipynb
│   │   ├── langgraph.ipynb
│   │   ├── openai.ipynb
│   │   └── pydantic_ai.ipynb
│   ├── rag_evaluation/                      #    RAG evaluation example
│   │   └── rag_evaluation_with_qdrant.py
│   └── tracing/                             #    Tracing examples
│       ├── crewai_tracing.ipynb
│       └── test_chatbot.py
│
├── scripts/                                 # 🔧 Maintainer utility scripts
│   ├── check_openai_model_capabilities.py   #    Audit OpenAI model feature support
│   └── compile_metric_templates.py          #    Compile .txt templates → templates.json
│
├── skills/                                  # 🤖 AGENT SKILLS (see detailed section)
│   ├── README.md                            #    Skills overview & installation guide
│   ├── deepeval/                            #    Main evaluation workflow skill
│   │   ├── SKILL.md                         #    Skill instructions (YAML frontmatter + MD)
│   │   ├── LICENSE
│   │   ├── references/                      #    Detailed reference docs for the agent
│   │   │   ├── artifact-contracts.md        #    Expected file locations for eval artifacts
│   │   │   ├── choose-use-case.md           #    Decision tree: agent vs RAG vs chatbot
│   │   │   ├── confident-ai.md              #    Confident AI integration guide
│   │   │   ├── datasets.md                  #    Dataset loading & management
│   │   │   ├── intake.md                    #    Intake questions to ask the user
│   │   │   ├── iteration-loop.md            #    How to iterate on eval failures
│   │   │   ├── metrics.md                   #    Metric selection guide
│   │   │   ├── pytest-e2e-evals.md          #    Pytest eval suite patterns
│   │   │   ├── synthetic-data.md            #    Synthetic data generation guide
│   │   │   └── traced-evals.md              #    Traced eval shapes & span metrics
│   │   └── templates/                       #    Code templates the agent uses
│   │       ├── metrics.py                   #    Shared metric list template
│   │       ├── test_multi_turn_e2e.py       #    Multi-turn eval template
│   │       ├── test_single_turn_no_tracing.py  # Single-turn (no tracing) template
│   │       └── test_single_turn_tracing.py  #    Single-turn (tracing) template
│   ├── deepeval-otel/                       #    Raw OpenTelemetry export skill
│   │   ├── SKILL.md                         #    Skill instructions
│   │   ├── LICENSE
│   │   ├── references/                      #    OTel reference docs
│   │   │   ├── endpoint-and-exporter.md     #    OTLP endpoint config
│   │   │   ├── gen-ai-fallbacks.md          #    GenAI semantic convention fallbacks
│   │   │   ├── span-attributes.md           #    confident.span.* attribute spec
│   │   │   └── trace-attributes.md          #    confident.trace.* attribute spec
│   │   └── templates/
│   │       └── confident_otel_setup.py      #    Minimal Python OTel export setup
│   └── deepeval-tracing/                    #    DeepEval native tracing skill
│       ├── SKILL.md                         #    Skill instructions
│       ├── LICENSE
│       └── references/
│           ├── integrations.md              #    Integration selection & framework index
│           └── tracing.md                   #    @observe, span types, tags, metadata
│
├── tests/                                   # 🧪 Framework's own test suite
│   ├── __init__.py
│   ├── test_agent_loop_detection.py         #    Agent loop detection tests
│   ├── test_confident/                      #    Confident AI cloud API tests
│   ├── test_core/                           #    Core library tests
│   │   ├── test_tracing/                    #    Tracing subsystem tests
│   │   │   ├── schemas/                     #    Expected JSON schemas for assertions
│   │   │   │   ├── span_types/              #    Per-span-type expected outputs
│   │   │   │   ├── tags/                    #    Tag-related expected outputs
│   │   │   │   └── update_functions/        #    Update function expected outputs
│   │   │   ├── test_configuration/          #    Tracing config tests
│   │   │   ├── test_generators/             #    Async/sync generator tracing tests
│   │   │   ├── test_integration/            #    Integration-level tests
│   │   │   ├── test_masking/                #    Data masking tests
│   │   │   ├── test_metadata/               #    Metadata tests
│   │   │   ├── test_nested_spans/           #    Nested span tests
│   │   │   ├── test_span_types/             #    Span type tests
│   │   │   ├── test_tags/                   #    Tag tests
│   │   │   └── test_update_functions/       #    Update function tests
│   │   ├── test_trim_and_load_json.py       #    JSON trimming utility tests
│   │   └── test_utils.py                    #    Core utility tests
│   ├── test_docs/                           #    Docs code example smoke tests
│   │   ├── test_confident/                  #    Confident AI doc examples
│   │   │   ├── test_integrations/           #    Per-integration doc tests
│   │   │   └── test_tracing_features/       #    Per-feature doc tests
│   │   └── test_deepeval/                   #    DeepEval doc examples
│   │       ├── test_ai_agent_evals/         #    Agent eval tutorial tests
│   │       └── test_llm_evals/              #    LLM eval tutorial tests
│   ├── test_integrations/                   #    Third-party integration tests
│   │   ├── utils.py                         #    Shared test utilities
│   │   ├── test_crewai/                     #    CrewAI integration tests
│   │   ├── test_google_adk/                 #    Google ADK integration tests
│   │   ├── test_langchain/                  #    LangChain integration tests
│   │   ├── test_llama_index/                #    LlamaIndex integration tests
│   │   ├── test_openai/                     #    OpenAI integration tests
│   │   ├── test_openrouter/                 #    OpenRouter integration tests
│   │   ├── test_pydanticai/                 #    Pydantic AI integration tests
│   │   │   ├── apps/                        #    Test application fixtures
│   │   │   └── schemas/                     #    Expected output schemas (JSON)
│   │   └── test_strands/                    #    Strands Agents integration tests
│   │       ├── apps/                        #    Test application fixtures
│   │       └── schemas/                     #    Expected output schemas (JSON)
│   ├── test_metrics/                        #    Individual metric tests (40+ files)
│   │   ├── images/car.png                   #    Test image for multimodal metrics
│   │   ├── test_answer_relevancy_metric.py
│   │   ├── test_hallucination_metric.py
│   │   ├── test_task_completetion_metric.py
│   │   └── ...                              #    (one test file per metric)
│   └── test_templates/
│       └── test_metric_templates.py         #    Compiled template validation
│
├── typescript/                              # 📘 TypeScript SDK
│   ├── package.json                         #    npm package config
│   ├── package-lock.json
│   ├── tsconfig.json                        #    TypeScript compiler config
│   ├── eslint.config.mts                    #    ESLint config
│   ├── jest.config.js                       #    Jest test config
│   ├── .prettierrc                          #    Prettier formatting
│   ├── .prettierignore
│   ├── .gitignore
│   ├── README.md
│   ├── src/                                 #    TypeScript source code
│   ├── examples/                            #    TypeScript usage examples
│   │   ├── dataset/                         #    Dataset examples
│   │   ├── integrations/                    #    Integration examples
│   │   │   ├── langchain/
│   │   │   ├── langgraph/
│   │   │   └── openai/
│   │   ├── simulate/                        #    Simulation examples
│   │   └── tracing/                         #    Tracing examples
│   └── test/                                #    Jest tests
│       ├── test-core/                       #    Core functionality tests
│       └── test-integrations/               #    Integration tests
│           ├── test-ai-sdk/                 #    Vercel AI SDK tests
│           ├── test-langchain/              #    LangChain.js tests
│           ├── test-openai/                 #    OpenAI JS tests
│           └── test-openai-agents/          #    OpenAI Agents JS tests
│
│ ──── Root-Level Files ────────────────────────────────────────────────────────
│
├── .env.example                             #    📋 All supported env vars (12+ providers)
├── .gitignore                               #    Git ignore rules
├── .pre-commit-config.yaml                  #    🪝 Pre-commit hooks (Black + Ruff)
├── CITATION.cff                             #    Academic citation metadata
├── CONTRIBUTING.md                          #    Contributor guidelines
├── LICENSE.md                               #    Apache-2.0 license
├── MAINTAINERS.md                           #    Project maintainer list
├── MANIFEST.in                              #    sdist manifest include rules
├── README.md                                #    Main project README
├── pyproject.toml                           #    📦 Poetry project config & dependencies
├── poetry.lock                              #    Locked dependency versions
│
│ ──── Scratch / Workspace-Local Files ─────────────────────────────────────────
│
├── manual_after_evals_iterator.py           #    Scratch: manual evals iterator experiment
├── test_agentcore_agent.py                  #    Scratch: AgentCore agent testing
└── test_pydantic_agent.py                   #    Scratch: Pydantic AI agent testing
```

---

## Root-Level Files & Config

| File | Purpose |
|---|---|
| [pyproject.toml](file:///home/kiran/Project/LLMInspector/deepeval_scratch/pyproject.toml) | Poetry project definition. Defines the `deepeval` CLI entry point, the `pytest11` plugin registration, all dependencies (including optional integration groups), and tool configs for Black, Ruff, and pytest. |
| [poetry.lock](file:///home/kiran/Project/LLMInspector/deepeval_scratch/poetry.lock) | Locked dependency graph for reproducible installs. |
| [.pre-commit-config.yaml](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.pre-commit-config.yaml) | Pre-commit hook definitions (see [Pre-Commit Hooks](#pre-commit-hooks)). |
| [.env.example](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.env.example) | Comprehensive template listing every supported environment variable — 12+ model provider API keys, Azure configs, Confident AI key, DeepEval behavior flags, and test IDs. |
| [MANIFEST.in](file:///home/kiran/Project/LLMInspector/deepeval_scratch/MANIFEST.in) | Ensures `py.typed` marker is included in source distributions for PEP 561 compliance. |
| [CITATION.cff](file:///home/kiran/Project/LLMInspector/deepeval_scratch/CITATION.cff) | Machine-readable citation metadata for academic references. |

---

## Core Python Package — `deepeval/`

The package follows a **modular, domain-driven** architecture where each subdirectory is a self-contained domain:

### Design Decisions

1. **Every metric is a subdirectory** (e.g., `metrics/hallucination/`) containing its own `metric.py`, `schema.py`, and `templates/` folder. This isolates each metric's LLM-judge prompts and Pydantic response schemas, making it easy to add new metrics without touching shared code.

2. **Prompt templates are `.txt` files**, not hardcoded strings. Each metric's `templates/` directory has a `class.txt` (system prompt persona) plus task-specific fragments. The `scripts/compile_metric_templates.py` script compiles them into `templates/metrics/templates.json` for fast runtime loading via `templates/resolver.py`.

3. **`models/llms/` separates LLM providers** from metric logic. Metrics work with the abstract `DeepEvalBaseLLM` interface — users can swap OpenAI for Anthropic, Gemini, Ollama, or any other provider without changing metric code.

4. **`plugins/plugin.py` is registered as a `pytest11` entry point** in `pyproject.toml`, automatically activating when pytest loads. This is how `deepeval test run` seamlessly hooks into pytest.

5. **`tracing/` uses both a custom Observer pattern and an OpenTelemetry bridge** (`tracing/otel/`). The `@observe` decorator creates spans that work with both the native SDK and standard OTel exporters.

---

## Agent Skills System — `skills/`

> [!IMPORTANT]
> This is one of the most innovative parts of the repo. DeepEval ships **structured instructions for AI coding agents** (Claude Code, Cursor, and others) as first-class project artifacts.

### How Skills Work

A **skill** is a directory containing a `SKILL.md` file with YAML frontmatter metadata and Markdown instructions. When a coding agent (Claude Code, Cursor, etc.) is loaded with these skills via the plugin manifests, it gains domain-specific knowledge about how to use DeepEval correctly.

### The Three Skills

#### 1. `skills/deepeval/` — Main Evaluation Workflow

**Trigger**: When a user asks to "evaluate", "add evals", "generate datasets", "run deepeval test", or "iterate on an AI app".

**What the agent does** (from [SKILL.md](file:///home/kiran/Project/LLMInspector/deepeval_scratch/skills/deepeval/SKILL.md)):
1. Inspects the codebase for app type (agent vs RAG vs chatbot)
2. Asks structured intake questions (eval model, dataset source, tracing, Confident AI, iteration rounds)
3. Selects metrics based on use case
4. Prepares datasets (existing or synthetic via `deepeval generate`)
5. Instruments the app for tracing
6. Creates a committed pytest eval suite from templates
7. Runs `deepeval test run` and iterates for N rounds

**Reference docs** in `references/`:
| File | Purpose |
|---|---|
| `intake.md` | Exact questions the agent must ask before editing code |
| `choose-use-case.md` | Decision tree: chatbot > agent > RAG precedence |
| `metrics.md` | Which metrics to use for which use case |
| `datasets.md` | How to load local/cloud datasets |
| `synthetic-data.md` | How to generate synthetic goldens |
| `pytest-e2e-evals.md` | Pytest eval suite design patterns |
| `traced-evals.md` | How to wire traced evals & span-level metrics |
| `iteration-loop.md` | Failure analysis → fix → rerun loop |
| `artifact-contracts.md` | Expected file paths and naming conventions |
| `confident-ai.md` | Confident AI cloud integration |

**Code templates** in `templates/`:
| File | When Used |
|---|---|
| `test_single_turn_tracing.py` | Agent/RAG evals with tracing (preferred) |
| `test_single_turn_no_tracing.py` | When user declines tracing |
| `test_multi_turn_e2e.py` | Chatbot / multi-turn agent evals |
| `metrics.py` | Shared metric instances (kept separate from test files) |

#### 2. `skills/deepeval-otel/` — Raw OpenTelemetry Export

**Trigger**: When a user wants to send OTel/OTLP traces to Confident AI without the `deepeval` Python package.

**Key detail**: This skill is **language-agnostic** — the `confident.*` attribute keys are the contract, not Python code. It works with any OTLP-capable SDK.

**References**: endpoint config, `confident.span.*` attributes, `confident.trace.*` attributes, GenAI semantic convention fallbacks.

#### 3. `skills/deepeval-tracing/` — DeepEval Native Tracing

**Trigger**: When a user wants to add `@observe` decorators or framework integrations (LangChain, CrewAI, Pydantic AI, etc.) to their app.

**Key detail**: This skill stops at **producing traces**. It does not run evals — that's the `deepeval` skill's job. The three skills are designed as **complementary layers**.

### Skill Registration & Discovery

```
.claude-plugin/plugin.json  →  "skills": "./skills/"     # Points Claude Code to skills/
.cursor-plugin/plugin.json  →  "skills": "./skills/"     # Points Cursor to skills/
skills/README.md            →  Installation instructions  # Manual install for other agents
```

---

## Coding Agent Plugin Manifests

### `.claude-plugin/` — Claude Code Integration

| File | Purpose |
|---|---|
| [plugin.json](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.claude-plugin/plugin.json) | Plugin manifest for Claude Code. Declares the plugin name ("DeepEval"), version, description, author, and points to `./skills/` for skill discovery. |
| [marketplace.json](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.claude-plugin/marketplace.json) | Marketplace listing for Claude Code's plugin marketplace. Includes category ("developer-tools"), keywords, and references `plugin.json`. |

### `.cursor-plugin/` — Cursor IDE Integration

| File | Purpose |
|---|---|
| [plugin.json](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.cursor-plugin/plugin.json) | Nearly identical to Claude's `plugin.json` but includes a `"category": "developer-tools"` field inline. Points to `./skills/` for skill discovery. |

### How Agents Use These

1. When a user installs DeepEval as a plugin in Claude Code or Cursor, the IDE reads the `plugin.json`.
2. The `"skills": "./skills/"` directive tells the agent to scan `skills/` for `SKILL.md` files.
3. Each `SKILL.md`'s YAML frontmatter `name` and `description` are used for **trigger matching** — the agent decides which skill to activate based on the user's request.
4. Once triggered, the agent reads the full `SKILL.md` body, then follows the `references/` and `templates/` for precise, domain-specific code generation.

---

## Pre-Commit Hooks

Defined in [.pre-commit-config.yaml](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.pre-commit-config.yaml):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.8.0
    hooks:
      - id: black                    # Auto-format Python code (80-char line length)

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]               # Lint + auto-fix Python issues
```

### Design Rationale

- **Black** enforces consistent formatting with an 80-character line length (configured in `pyproject.toml` under `[tool.black]`).
- **Ruff** replaces flake8/isort/pyflakes as a single, fast linter with auto-fix capability.
- These run **before every commit**, ensuring all code entering the repo meets style standards.
- The CI pipeline (`workflows/black.yml`) **also runs Black as a check** on PRs, creating a double-gate: local pre-commit + CI verification.
- Dev dependency `pre-commit = "^4.3.0"` is declared in `pyproject.toml` so all contributors can `pre-commit install`.

---

## Pytest Plugin & Hook System

### `deepeval/plugins/plugin.py` — The Pytest Plugin

Registered as a `pytest11` entry point in `pyproject.toml`:

```toml
[tool.poetry.plugins."pytest11"]
deepeval = "deepeval.plugins.plugin"
```

This means **installing `deepeval` via pip automatically activates the pytest plugin** — no manual configuration needed.

#### Hook Implementations

From [plugin.py](file:///home/kiran/Project/LLMInspector/deepeval_scratch/deepeval/plugins/plugin.py):

| Hook | Purpose |
|---|---|
| `pytest_addoption` | Adds `--identifier` flag to `pytest` / `deepeval test run` for naming test runs |
| `pytest_sessionstart` | Creates a `TestRun` object when running via `deepeval test run` (checks `get_is_running_deepeval()`) |
| `pytest_runtest_protocol` (`tryfirst`) | Sets `PYTEST_RUN_TEST_NAME` env var to the current test's node ID before each test |
| `pytest_runtest_call` (`hookwrapper`) | **The key hook** — wraps each test in a DeepEval evaluation scope. Creates an `Observer` context manager so that `@observe` spans inside the test are automatically attached to the in-flight test run. Sets `EvalSession(mode=EvalMode.EVALUATE)`. |
| `pytest_sessionfinish` (`tryfirst`, `hookwrapper`) | Runs teardown after all tests complete |
| `pytest_terminal_summary` | Reports skipped tests with their reasons |

### `deepeval/test_run/hooks.py` — Custom Test Run Hooks

From [hooks.py](file:///home/kiran/Project/LLMInspector/deepeval_scratch/deepeval/test_run/hooks.py):

```python
on_test_run_end_hook = None

def on_test_run_end(func):
    """Decorator to register a callback invoked when a test run completes."""
    global on_test_run_end_hook
    on_test_run_end_hook = func
    return func

def invoke_test_run_end_hook():
    """Called internally after test run finishes. Fires once and clears."""
    global on_test_run_end_hook
    if on_test_run_end_hook:
        on_test_run_end_hook()
        on_test_run_end_hook = None
```

This allows users to register post-test-run callbacks (e.g., upload results, send Slack notifications) via the `@on_test_run_end` decorator.

---

## CI/CD Pipelines

Defined in [.github/workflows/](file:///home/kiran/Project/LLMInspector/deepeval_scratch/.github/workflows):

| Workflow | Trigger | Purpose |
|---|---|---|
| `black.yml` | PRs (excl. typescript/) | Runs `black --check` to enforce formatting |
| `pr-title-check.yml` | PRs | Validates PR title conventions |
| `test_core.yml` | PRs / pushes | Runs core library unit tests |
| `full_test_core_for_pr.yml` | PRs | Extended core tests for thorough PR validation |
| `test_metrics.yml` | PRs / pushes | Tests metric accuracy & correctness |
| `test_metric_templates.yml` | PRs / pushes | Validates compiled prompt templates |
| `test_integrations.yml` | PRs / pushes | Tests all third-party integrations |
| `test_confident.yml` | PRs / pushes | Tests Confident AI cloud API integration |
| `typescript_lint.yml` | PRs / pushes | ESLint check for TypeScript SDK |
| `typescript_test.yml` | PRs / pushes | Jest tests for TypeScript SDK |
| `changelog.yml` | Releases | Auto-generates changelog via `.scripts/changelog/generate.py` |
| `release.yml` | Release tags | Builds and publishes to PyPI |

### Test Organization Strategy

Tests are split into **5 separate CI workflows** to enable parallelism and targeted re-runs:
- `test_core` — fast, no external API calls
- `test_metrics` — requires LLM API keys (slower, costly)
- `test_integrations` — requires third-party SDK installs
- `test_confident` — requires Confident AI API key
- `test_metric_templates` — validates template compilation

---

## Documentation Site — `docs/`

Built with **Next.js** (using Fumadocs or similar MDX-based framework):

- `app/` — Next.js app directory with route groups for docs, blog, guides, tutorials, API reference, and integrations
- `content/` — MDX source files for blog posts and integration docs
- `scripts/` — Build-time scripts for processing MDX (normalizing admonitions, replacing images, generating contributor pages)
- **`llms.txt` & `llms-full.txt`** — Machine-readable versions of the docs designed to be consumed by AI agents (a standard pioneered by the community for making docs agent-friendly)

---

## TypeScript SDK — `typescript/`

A parallel implementation of DeepEval in TypeScript for the JavaScript/Node.js ecosystem:

- **`src/`** — TypeScript source matching the Python SDK's API surface
- **`examples/`** — Usage examples for datasets, integrations (LangChain, LangGraph, OpenAI), simulation, and tracing
- **`test/`** — Jest test suite mirroring the Python test organization (core tests + integration tests with fixture-based assertions using JSON schemas)

---

## Design Philosophy Summary

### 1. "Testing as Code" Paradigm
By hooking into `pytest` via a `pytest11` plugin, DeepEval lets developers write LLM evaluations using the same patterns they use for unit tests. `deepeval test run` is just pytest with extra superpowers.

### 2. Model Agnosticism
The `models/llms/` directory provides **15 different provider wrappers** (OpenAI, Anthropic, Gemini, Ollama, DeepSeek, etc.). Metrics use the abstract `DeepEvalBaseLLM` interface, so the judge model is completely decoupled from the metric logic.

### 3. Agent-First Design
The `skills/` system makes DeepEval **the first LLM eval framework to ship structured instructions for AI coding agents**. Instead of asking developers to read docs and write eval code manually, they can tell their coding agent "add evals to my app" and the agent follows a precise, battle-tested workflow.

### 4. Complementary Skill Layering
The three skills (`deepeval`, `deepeval-tracing`, `deepeval-otel`) form a clean separation of concerns:
- **Tracing**: Instrument the app → produce spans
- **OTel**: Export spans → get them to Confident AI
- **Evals**: Run metrics against traces → iterate on failures

### 5. Template-Driven Metric Prompts
Metric prompts are externalized as `.txt` files, compiled into `templates.json`, and validated by CI (`test_metric_templates.yml`). This prevents prompt drift and makes prompt engineering a reviewable, diffable process.

### 6. Comprehensive CI/CD
12 separate GitHub Actions workflows ensure that every dimension of the project (formatting, core, metrics, integrations, templates, TypeScript, releases) is independently tested and validated.
