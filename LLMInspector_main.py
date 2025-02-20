import ast

from llm_inspector.eval_metrics import EvalMetrics
from llm_inspector.rag_eval import RagEval
import streamlit as st
import pandas as pd
from configparser import ConfigParser
import io
import datetime



mystate = st.session_state
image_logo = "images/white_logo-removebg-preview 1.png" 
st.logo(image_logo, size="large", icon_image=image_logo)

info = st.Page("./pages/welcome_home.py", title="Welcome",  icon="🏠")
test_gen_alignment = st.Page("./pages/alignment_data_generation.py", title="Alignment", icon="🎯")
test_gen_adverserial = st.Page("./pages/adverserial_data_generation.py", title="Adverserial", icon="🚫")
test_gen_rag = st.Page("./pages/RAG_data_generation.py", title="RAG", icon="📋")
documents = st.Page("./pages/Documents_links.py", title="LLMInspector Documents", icon="📜")

metric_eval = st.Page("./pages/Metric_Evaluation.py", title = "Metric Evaluation", icon = "📊")
page_dict ={}


pg = st.navigation(
    {
        "Home": [info],
        "Test Data Generation": [test_gen_alignment, test_gen_adverserial, test_gen_rag],
        "Evaluations" : [metric_eval],
        "Resources" : [documents]
    }
)


st.set_page_config(page_title = "LLMInspector UI", page_icon=":material/browse_activity:", layout = "wide", initial_sidebar_state = "expanded")

pg.run()

