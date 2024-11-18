## LLMInspector: Comprehensive Evaluation and Testing for LLM Applications

When deploying Large Language Models (LLMs) in enterprise applications, it's crucial to understand, evaluate, and navigate their capabilities, limitations, and risks. Ensuring alignment with functional and non-functional requirements while maintaining robustness against adversarial queries is paramount.

**LLMInspector** is a sophisticated Python package developed to address these challenges. It offers a comprehensive solution for evaluating and testing the alignment and security of LLM-based applications. Tailored for enterprise needs, LLMInspector ensures ethical and effective deployment of powerful language models.

![flow chart](https://github.com/michelin/LLMInspector/blob/main/docs/assets/images/llminspector_flow_v1.0.png)

## Key features:
- Generation of prompts from Goldendataset by exploding the prompts with tag augmentation and paraphrasing.
- Generation of prompts with various perturbations applied to test the robustness of the LLM application.
- Generation of question and ground truth from documents, that can be used for testing of RAG based application.
- Evaluation of RAG based LLM application using LLM based evaluation metrics.
- Evaluation of the LLM application through various accuracy based metrics, sentiment analysis, emotion analysis, PII detection, Readability scores.
- Adversarial red team testing using curated datasets to probe for risks and vulnerabilities in LLM applications

## Installation:
The source code is currently hosted on GitHub at: [llminspector](https://github.com/michelin/LLMInspector)

```python
pip install git+https://github.com/michelin/LLMInspector.git
```

The list of changes to LLMInspector between each release can be found [here](https://github.com/michelin/LLMInspector/releases). 


## Documentation:
Detailed package and API Documentation is available [here](https://github.com/michelin/LLMInspector/wiki)


## Getting Started

Set up the config.ini and .env by referring [here](https://github.com/michelin/LLMInspector/wiki/Getting-Started)

```python
from llm_inspector.llminspector import llminspector
import pandas as pd

obj = llminspector(config_path=<path-to-cofig.ini>, env_path=<path-to-.env>)
obj.alignment()
obj.adversarial()
obj.rag_evaluate()
obj.converse()
df = pd.read_excel(<path-to-input-csv>)
obj.evaluate(df)
```


## Licence

This project is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/michelin/LLMInspector/blob/main/LICENSE) file for more details.


## Authors

- Sourabh Potnis
- Ankit Zade
- Kiran Prasath
- Arpit Kumar 
