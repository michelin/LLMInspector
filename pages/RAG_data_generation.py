import ast
from llm_inspector.rag_eval import RagEval
import os
import streamlit as st
import pandas as pd
from configparser import ConfigParser
import io
import datetime
import os
import tempfile
from typing import List
from llama_index.core import SimpleDirectoryReader, Document
import logging

logger = logging.getLogger(__name__)

buffer = io.BytesIO()
config_path = "./config.ini"
env_path = "./.env"


config = ConfigParser()
config.read(config_path)
dt_time = datetime.datetime.now()
res = ''
result = ''
rag_file = config["RAG_File"]
test_size = int(rag_file["testset_size"])

def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def load_documents(uploaded_files):
    temp_dir = tempfile.TemporaryDirectory()
    for filename in os.listdir(temp_dir.name):
        file_path = os.path.join(temp_dir.name, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
    reader = SimpleDirectoryReader(input_dir=temp_dir.name)
    return reader.load_data()


st.header("Generate RAG Test Data")
st.markdown("For test data generation, please select required file(s)")

st.pressed_first_button = False
uploaded_file_rag = st.file_uploader("Choose Files for RAG Test Data", accept_multiple_files=True, type=["pdf","docx"])


st.sidebar.slider("Test Data Set Size", min_value=5, max_value=15, value=test_size, step=1)

# dir = "tempDir_data"
# path = os.path.join(dir)

# print("type:" + str(type(uploaded_file_rag)))
documents = []

if len(uploaded_file_rag) >= 1:
    st.write("Total no. of files uploaded: " + str(len(uploaded_file_rag)))
    if uploaded_file_rag:
        documents = load_documents(uploaded_file_rag)
        print("List of documents uploaded for RAG Test data generation: " +str(documents))
        logger.info("List of documents uploaded for RAG Test data generation: " +str(documents))

Gen_button = st.button("Generate Test Data", type="primary")
dataset = pd.DataFrame()
if Gen_button:


    with st.spinner("Generating RAG Data..."):

        try:

            rag_testset = RagEval(config, env_path, test_size=test_size, documentList=documents)
            res = rag_testset.generate_testset()

            if len(res) != 0:
                with st.expander("RAG Test Data Preview", icon=":material/dataset:"):
                    st.dataframe(res)


                df_xlsx = to_excel(res)
                st.download_button(label='Download Current Result ⬇️',
                                data=df_xlsx ,
                                file_name= "RAG_TestDataset_output" 
                            + "_"
                            + str(dt_time.year)
                            + str(dt_time.month)
                            + str(dt_time.day)
                            + "_"
                            + str(dt_time.hour)
                            + str(dt_time.minute)
                            + ".xlsx")

        except Exception as e:
            logger.info("Please make sure files are uploaded correctly from your end. Error: {e}")
            st.write(f"RAG Test data generation failed Please look into below error: \n Error: {e}")


