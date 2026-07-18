# LLMInspector - Updated Package Structure Plan

## 1. Overview
The goal of this restructuring is to modernize the `LLMInspector` codebase by adopting a modular, class and function-driven architecture heavily inspired by `deepeval`. This architecture will ensure high maintainability, elegant async/sync function definitions, and a seamless developer experience.

Key architectural changes include:
- **Removing Streamlit UI**: The package will be focused purely on being a robust core library. All UI components (`pages/`, `.streamlit/`, `LLMInspector_main.py`) will be completely removed.
- **Dropping `config.ini`**: Configuration will shift from a static `config.ini` file to a Pythonic, variable-driven approach. 
- **Adopting DeepEval's Taxonomy**: Introducing `dataset`, `evaluate`, `metrics`, `models`, `synthesizer`, `scorer`, and `test_case` to intelligently decouple concerns.
- **Improving SDLC**: Implementing `.github` workflows and a `skills/` architecture to improve continuous integration and AI-agent compatibility.

## 2. Root-Level Cleanup
The following files and directories will be deleted or ignored to clean up the package:
- `LLMInspector_main.py` (Deleted)
- `.streamlit/` (Deleted)
- `pages/` (Deleted)
- `config.ini` (Deleted)

## 3. Proposed Project Tree

### A. The Core Package (`llminspector/`)
The main package will be split into explicit functional domains to mimic `deepeval`.

```text
LLMInspector/
├── README.md
├── pyproject.toml / setup.py
├── requirements.txt
├── tests/                             
│
├── .github/                           # GitHub Actions for CI/CD
│   └── workflows/
│       ├── test_core.yml              # Runs pytest on PRs
│       ├── lint.yml                   # Black/Ruff formatting checks
│       └── release.yml                # Publishes to PyPI on tags
│
├── skills/                            # Agent Skills for AI Assistants (Cursor/Claude)
│   ├── README.md
│   └── llminspector/                  # Skill instructions on how AI should use our package
│       ├── SKILL.md
│       └── templates/                 # Code templates for writing eval tests
│
└── llminspector/                      # Core Package
    ├── __init__.py                    
    ├── constants.py                   
    ├── errors.py                      
    ├── utils.py                       
    │
    ├── dataset/                       # Dataset & Ground-Truth Management
    │   ├── __init__.py
    │   └── dataset.py                 # EvaluationDataset class for managing multiple test cases
    │
    ├── evaluate/                      # Core Execution Engine
    │   ├── __init__.py
    │   └── evaluate.py                # Async/Sync evaluation loops (evaluate() function)
    │
    ├── metrics/                       # Individual Metric Definitions
    │   ├── __init__.py
    │   ├── base_metric.py             # Abstract base class
    │   ├── rouge.py                   
    │   ├── bert_score.py              
    │   ├── faithfulness.py            
    │   ├── safety.py                  # Toxicity, Maliciousness, Harmfulness
    │   └── linguistics.py             # Readability, Language, Sentiment
    │
    ├── models/                        # LLM Wrappers for Evaluation
    │   ├── __init__.py
    │   ├── base_model.py              # Base LLM Wrapper
    │   └── openai_model.py            # Used if metrics require LLM-as-a-judge
    │
    ├── scorer/                        # Scoring Logic
    │   ├── __init__.py
    │   └── scorer.py                  # Normalization, threshold evaluation, pass/fail indicators
    │
    ├── synthesizer/                   # Synthetic Data Generation (Formerly alignment, adversarial, rag)
    │   ├── __init__.py
    │   ├── alignment.py               # Alignment data generation logic
    │   ├── adversarial.py             # Adversarial data generation logic
    │   └── rag.py                     # RAG test set generation logic
    │
    └── test_case/                     # Test Case Data Models
        ├── __init__.py
        └── test_case.py               # LLMTestCase (input, actual_output, expected_output, context)
```

## 4. Architectural Details & Changes

### A. Synthetic Data Generation (`synthesizer/`)
The logic currently housed in `alignment.py`, `adversarial.py`, and `rag_eval.py` fits perfectly into a `synthesizer` module. They generate the data we ultimately evaluate.

```python
from llminspector.synthesizer.alignment import AlignmentSynthesizer

synthesizer = AlignmentSynthesizer(paraphrase_count=2, augmentations={'typo': 1})
dataset = synthesizer.generate(input_file="input.xlsx")
```

### B. Standardized Inputs (`test_case/` & `dataset/`)
Instead of pandas dataframes and column name mapping from `config.ini`, evaluations will use typed test cases grouped into datasets.

```python
from llminspector.test_case import LLMTestCase
from llminspector.dataset import EvaluationDataset

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital.",
    expected_output="Paris",
    context=["France is a country in Europe."]
)
dataset = EvaluationDataset(test_cases=[test_case])
```

### C. Execution and Scoring (`evaluate/`, `metrics/`, `scorer/`)
Metrics will be separate classes extending `BaseMetric`. They will contain their own scoring logic (which might lean on the `scorer/` module) and can be executed via the central `evaluate()` engine synchronously or asynchronously.

```python
from llminspector.evaluate import evaluate
from llminspector.metrics import Faithfulness, BertScore

metrics = [Faithfulness(threshold=0.8), BertScore(threshold=0.9)]

# evaluate handles the async execution loop and compiles results
results = evaluate(dataset=dataset, metrics=metrics)
```

### D. LLM-as-a-judge (`models/`)
For metrics that require an LLM to evaluate text (e.g., faithfulness, maliciousness), the `models/` directory will provide wrappers to abstract away the LLM provider, allowing users to pass their own instantiated model wrappers if needed.

## 5. Software Development Life Cycle (SDLC) Improvements

To mimic the high quality of `deepeval`'s repository, we will introduce:

1. **GitHub Actions (`.github/workflows/`)**:
   - `lint.yml`: Enforces code quality using Black or Ruff on every PR.
   - `test_core.yml`: Runs the `tests/` directory on every PR to prevent regressions.

2. **Agent Skills (`skills/`)**:
   - We will provide a `skills/llminspector/SKILL.md` file. This acts as a manifest for AI Coding Agents (like Cursor or Claude). It tells the AI exactly how to use the `llminspector` package, standardizes the metrics, and provides code templates. This significantly lowers the barrier to entry for users trying to write tests with AI assistance.

## 6. Summary of Next Steps
1. Delete the Streamlit files and `config.ini`.
2. Scaffold the new directory structure inside `llminspector/` (including `synthesizer`, `evaluate`, `metrics`, etc.).
3. Implement `.github/workflows` for linting and testing.
4. Draft the initial `skills/` manifest.
5. Begin migrating `eval_metrics.py` into the `metrics/` and `scorer/` modules.
6. Migrate `alignment.py`, `adversarial.py`, and `rag_eval.py` into the `synthesizer/` module.
7. Implement `LLMTestCase` and the `evaluate` orchestration engine.
