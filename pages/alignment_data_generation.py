import ast

from llm_inspector.alignment import Alignment
# import openai
import os
import streamlit as st
import pandas as pd
from configparser import ConfigParser
import io
import datetime
import logging

logger = logging.getLogger(__name__)

buffer = io.BytesIO()
config_path = "config.ini"
env_path = ".env"

config = ConfigParser()
config.read(config_path)
dt_time = datetime.datetime.now()
res = ''
result = ''

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

st.header("Generate Alignment Test Data")
st.markdown("For test data Generation, please insert Golden test data based on your requirement.")


golden_dataset = pd.read_excel("./example/Data/Golden_data_sample.xlsx")


df_out=to_excel(golden_dataset)
st.write("Sample template file used for test data generation is available here for download")
st.download_button(label ="Download existing Golden dataset template ⬇️", data=df_out, file_name="Golden_data_sample.xlsx",type="secondary")

st.write( "Please refer sidebar for Configuration details")

st.subheader("Upload file for test data generation")
st.caption(" For Alignment test data generation, upload excel file similar to Golden dataset template provided above. Please make sure to add TAGs as per configuration.")

st.sidebar.header("Alignmnet Test Dataset Configurations")

#Getting Data from config File
tag_file_path = config["Alignment_File"]
tag_augmentation = config["Tag_Augmentation_Key"]
augmentation_type = config["Augmentation_Type"]
augmentations = ast.literal_eval(tag_file_path["augmentations"])
num_return_sequences = int(tag_file_path["paraphrase_count"])

keyword_dict = ast.literal_eval(tag_augmentation["tag_keyword_dict"])
tag_key_list = keyword_dict.keys()

augmentation_dict = ast.literal_eval(augmentation_type["augmentation_dict"])
aug_value = keyword_dict.values()


expand = st.sidebar.expander("TAG List", icon=":material/edit_note:")
tags_list = []
checkboxes_1={}

with expand:
    for k in tag_key_list:
        checkboxes_1[k] = st.checkbox(k, key= k, value = True)
    
dfs = []


expand = st.sidebar.expander("Augmentation Values", icon=":material/store:")
Augmentation_List = []
checkboxes_2={}

with expand:
    for index, (k,v) in enumerate(augmentations.items()):
        checkboxes_2[k] = st.checkbox(k, key=k, value = True)
    


paraphrase_value = st.sidebar.slider("Paraphrase Threshold", min_value=1, max_value=5, value=num_return_sequences, step=1)

uploaded_file = st.file_uploader("Please upload Golden dataset excel file for Alignment test data generation", type=["xlsx", "xls"])


@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)
    return df


if uploaded_file is not None:
    # Read the Excel file
    golden_dataset = load_data(uploaded_file)
    inpFile = uploaded_file
        
    with st.expander("Data Preview of the uploaded Excel"):
        st.dataframe(golden_dataset)



Gen_button = st.button("Generate Alignment Data", type="primary")
st.caption("Disclaimer: Test data will not be generated if no excel file is uploaded")
if Gen_button:
    with st.spinner("Generating Alignment Data..."):

        try:
            alignment_obj = Alignment(config, inputfile=uploaded_file, paraphrase_count=paraphrase_value)
            res = alignment_obj.transform_df()

        except Exception as e:
            logger.info("Alignment test Data generation failed, please rerun and check logs for errors. Error: {e}")
            st.write(f"Alignment test Data generation failed, please rerun and check raise the error. Error: {e}")


if len(res) != 0:
    dir = "output"
    path = os.path.join(dir)
    # if os.path.isdir('output'):
    #     output_path=path
    # else:
    #     try:
    #         os.mkdir(path)
    #         logger.info("% s Folder created successfully" % path)"
    #         #print("% s Folder created successfully" % path)
    #     except OSError as error:
    #         print(error)
    #         print("Folder path cannot be created")

    #df_out = res.to_excel("./" + path + "/Alignment_TestDataset.xlsx", index=False)
    
    with st.expander("Alignment Test Data Preview", icon=":material/dataset:"):
        st.dataframe(res)

    df_xlsx = to_excel(res)
    st.download_button(label='Download Current Result ⬇️',
                data=df_xlsx,
                file_name= "Alignment_Test_Data" 
            + "_"
            + str(dt_time.year)
            + str(dt_time.month)
            + str(dt_time.day)
            + "_"
            + str(dt_time.hour)
            + str(dt_time.minute)
            + ".xlsx")

# else:
#     st.write("Alignment test Data generation failed, please rerun and check logs for errors")




