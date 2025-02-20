import streamlit as st
from st_aggrid import AgGrid
import pandas as pd
from configparser import ConfigParser
import io
import datetime
from xlsxwriter import Workbook
import duckdb


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

dt_time = datetime.datetime.now()
st.header("Generate Adverserial Test Data")
#st.markdown("For Test Data Generations, please insert Golden TestData based on your requirement.")
st.write("Adverserial test data suite contains more than 3000 records of various capabilities like Jailbraiks, OutofScope, Content Moderation, etc. \
         By Using this screen, you can choose capabilities based on your requirement")


@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)
    return df

dataset = load_data("./example/Data/AdversarialAttack_Data.xlsx")


#cap = ["Jailbreak", "Out of scope"]

unique_capability = duckdb.sql (
f"""
SELECT distinct Capability from dataset
"""
).df()

#st.write(unique_capability)

capability = list(unique_capability['Capability'])

st.markdown("##### Please select required capabilities below:")
expand = st.expander("Select Capabilities", icon=":material/edit_note:")
capability_list = []
#Augmentation_List = []
checkboxes={}

with expand:
    for k in capability:
        checkboxes[k] = st.checkbox(k, key= k, value = True)

    for key, value in checkboxes.items():
        if value:
            capability_list.append(key)

st.markdown("###### You selected:" + str(capability_list))
Adv_data = dataset[dataset["Capability"].isin(capability_list)]

Adv_data.sort_values(by=['Capability'], inplace=True)

st.write("Adverserial test data generated and available for preview.")
with st.expander("Dataset Preview", icon=":material/dataset:"):
    st.dataframe(Adv_data)

df_out=to_excel(Adv_data)
st.write("Adverserial test data available for download below")
st.download_button(label ="Download ⬇️", data=df_out, file_name="Adversarial_Test_Data"
            + "_"
            + str(dt_time.year)
            + str(dt_time.month)
            + str(dt_time.day)
            + "_"
            + str(dt_time.hour)
            + str(dt_time.minute)
            + ".xlsx",
            type="secondary")
