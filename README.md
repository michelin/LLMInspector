# LLMInspector


When we use LLMs/GenAI in enterprise applications, understanding, evaluating and navigating their capabilities, limitations as well as risks is very important.
So, we need to make sure LLM application is in alignment with functional and non-functional requirements and is also safe & robust against adversarial queries.

LLMInspector is internally developed comprehensive Python package designed for the evaluation and testing of alignment as well as adversaries in Large Language Models (LLMs) based applications. The package is tailored to address the unique challenges associated with ensuring the ethical and effective deployment of powerful language models in enterprises. 


LLMInspector helps to automatically generate the data and tests, run the tests suite and generate an insights report in which LLM capabilities and risks are quantified.

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


This Python package features a Streamlit application that serves as a playground for users to discover its capabilities. The Streamlit application can be executed using the following command from the package location: streamlit run LLMInspector_main.py

![alt text](https://gitlab.michelin.com/DAI_QA/llm_inspector/-/raw/main/images/llminspector_flow_v1.0.jpg)


## Key features:
- Contextualization of data and tests in accordance with enterprise application
- LLM Alignment testing 
- LLM Adversarial testing
- Automated conversational framework
- Comprehensive reporting
- Streamlit application as playground

## Where to get it:
The source code is currently hosted on GitHub at: [llminspector](https://gitlab.michelin.com/DAI_QA/llminspector)

```
pip install git+https://gitlab.michelin.com/DAI_QA/llm_inspector
```


The list of changes to LLMInspector between each release can be found [here](https://gitlab.michelin.com/DAI_QA/llminspector/-/releases). 

## Documentation:
Detailed package and API Documentation is available [here](https://gitlab.michelin.com/DAI_QA/llminspector/-/wikis/home)



- Sourabh Potnis
- Ankit Zade
- Kiran Prasath
- Arpit Kumar
- Shraddha Pawar
