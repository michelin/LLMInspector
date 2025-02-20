import streamlit as st
from PIL import Image

st.header("LLM Inspector Documents")
st.write("LLM Inspector provides a comprehensive array of metrics to evaluate the quality and effectiveness of LLM-based applications. Details of these metrices is provided below.")


markdown_text = """
###### Alignment Metrics
    1. Accuracy %
    2. Robustness %
    3. Text quality
        - Language
        - Sentiment
        - Emotion
        - Readability index
        - Perplexity
        - Coherence
        - Conciseness

                            
###### Adversarial Metrics
    1. ASR% (Attack Success Rate) for each adversary type
        - Bias
        - Content moderation
        - Prompt Injection
        - Training data extraction
    2. PII detected.
    3. Toxicity detected.


###### RAG Metrics
    1. Retrieval
        - Context Precision
        - Context Recall
        - Context Relevance
    2. Generation
        - Faithfulness
        - Answer Relevancy
    3. End to End
        - Answer Semantic Similarity
        - Answer Correctness 
                    """

st.markdown(markdown_text)

st.info("For detailed information on Tests and Metrics, please check our WIKI page on [Github](https://github.com/michelin/LLMInspector/wiki/Tests-And-Metrics)")




st.write("The below image represents the important aspects and metrics for evaluating an LLM based application.")

image = Image.open("./images/LLM Evaluation.png")
st.image(image, caption="LLM Evaluation - Why, What and How")

st.markdown("---")