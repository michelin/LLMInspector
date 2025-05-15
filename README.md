# LLM Inspector - Agent Evaluation

A tool designed to aid in test set generation and evaluation of Agentic Workflow.

## Setup

### Prerequisites
- Python 3.11
- Conda

### Installation

1. Clone the repository
```bash
git clone https://github.com/michelin/LLMInspector.git
git branch agent_eval
git checkout agent_eval
git pull origin agent_eval
```

2. Create and activate conda environment
```bash
conda create -n llminspector python=3.11
conda activate llminspector
```

3. Install requirements
```bash
pip install -r req.txt
```

## Project Structure

```
LLMInspector/
│
├── agent_eval/                           # Database directory
|  ├── data_claused.py               # Script for clause dataset generation
|  ├── data_paraphrased.py           # Script for paraphrased dataset generation
|  ├── evaluation.py                 # Evaluation script
|  ├── guardrails.py                 # Guardrails implementation
|  ├── trace_reader.py               # Trace reader implementation
|
├── .gitignore                    # Git ignore file
├── config.py                     # Configuration parameters
├── README.md                     # Project documentation
├── req.txt                       # Requirements file
├── run_evaluation.py             # Script to run evaluation
├── run_traces.py                 # Script to run traces
├── run_dataset_generation.py     # Script to run testset generation
│
├── db/                           # Database directory
│   ├── Chinook.db                # SQLite database
|
├── input/                        # Input directory
│   ├── input_data.csv            # Input data
|
├── output/                       # Output directory
│   ├── claused_dataset.csv       # Generated claused dataset
│   ├── paraphrased_dataset.csv   # Generated paraphrased dataset
│   ├── traces-344b4b439b2673c047ab227d929af37e.pkl  # Trace data file
```


