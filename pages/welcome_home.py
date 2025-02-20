import streamlit as st
from PIL import Image
#from st_pages import Page, Section, show_pages, add_page_title

custom_bkg_css = """
<style>
[data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://brandscenter.michelingroup.com/app/uploads/2023/03/sans-titre-4.png");
    # background-size: 180%;
    # background-position: top right;
    background-repeat: no-repeat;
    padding-top: 120px;
    background-position: 20px 20px;
}}

[data-testid="stSidebarContent"] {{
    color: white;

}}

</style>
"""
# st.markdown('<style>[data-testid="stSidebar"] > div:first-child {color: black;}</style>', unsafe_allow_html=True,) 

#st.markdown(custom_bkg_css, unsafe_allow_html=True)


st.title("Welcome to LLMInspector!!")


st.write("LLMInspector is a comprehensive framework designed for testing of alignment as well as adversaries in Large Language Models (LLMs) \
         It is specifically tailored to address the unique challenges of ensuring the responsible and effective deployment of powerful language models in enterprise production environments.")

st.markdown("This playground for LLMInspector allows you to not only explore the framework but also perform various tasks such as generating test data, evaluating results, and deriving insights.")

#st.markdown("Please check Sidebar for Test Data Generation and Playground for Insight Generations.")

#st.markdown("LLMInspector is a sophisticated Python package developed to address these challenges. It offers a comprehensive solution for evaluating and testing the alignment and security of LLM-based applications. Tailored for enterprise needs, LLMInspector ensures ethical and effective deployment of powerful language models.")

image = Image.open("./images/llminspector_flow_v1.0.jpg")
st.image(image, caption="llmInspector Flowchart")

st.info("For further details, please check the LLMInspector repository on [Github](https://github.com/michelin/LLMInspector)")

st.markdown("---")

#pg.run()