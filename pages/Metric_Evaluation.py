import ast

from llm_inspector.eval_metrics import EvalMetrics
import os
import streamlit as st
import pandas as pd
from configparser import ConfigParser
import io
import datetime
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)

#Variable Definations
buffer = io.BytesIO()
config_path = "config.ini"
env_path = ".env"

config = ConfigParser()
config.read(config_path)
dt_time = datetime.datetime.now()
res = ''
result = ''


eval_file = config["Insights_File"]
eval_thresholds = ast.literal_eval(eval_file["thresholds"])
eval_metrics_list = ast.literal_eval(eval_file['Metrics'])


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

@st.cache_data
def load_data(path: str):
    df = pd.read_excel(path)
    return df

def create_bar_chart(category_names_list, category_values_list, chart_title, tab_value):
    tab_value = tab_value
    categories = category_names_list
    values = category_values_list
    
    colors = '#27509b'
    
    fig = go.Figure()

    for i in range(len(categories)):
        fig.add_trace(go.Bar(
                            y=[categories[i]],  # Change to y-axis
                            x=[values[i]],  # Change to x-axis
                            width = 0.1,
                            marker=dict(color=colors),
                            orientation='h',  # Set orientation to horizontal
                            name=categories[i],
                            showlegend= True,
                            text=[f'{values[i]:.0f}'],  # Set text to display on the bar
                            textposition='auto',
                            #text_auto=".2s"
                        ))

    # Update layout
    fig.update_layout(paper_bgcolor="#aec5e5",
                    title= chart_title,
                    xaxis=dict(title='Scores'),  # Update x-axis title
                    yaxis=dict(title='Capability Name'),  # Update y-axis title
                    barmode='stack',                    
                )
    fig.update_traces(
        textfont_size=20, textangle=0, textposition="outside", cliponaxis=False
    )
    # Display chart
    tab_value.plotly_chart(fig, use_container_width=True)

def gauge_chart(value, chart_title, tab_value):
    tab_value = tab_value
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 0.5], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#27509b",
                'thickness': 0.2},
            'steps' : [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}],
    #         'threshold' : {'line': {'color': "red", 'width': }, 'thickness': 0.75}
                },
            title = {'text': chart_title}
            ))

    # Update layout
    fig.update_layout(title = {'text': "Average of Metrics" }, paper_bgcolor="#aec5e5")
    tab_value.plotly_chart(fig, use_container_width=True)

def scatter_plot_chart(category_names_list, category_values_list, chart_title, range_end, tab_value):
    tab_value = tab_value
    fig = go.Figure(data=go.Scatterpolar(
        r=category_values_list,
        theta=category_names_list,
        fill='toself',
    ))

    # Update layout
    fig.update_layout(
        title={
                "text": chart_title
            },
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, range_end]
            )
        ),
        paper_bgcolor="#aec5e5"
    )
    tab_value.plotly_chart(fig, use_container_width=True)


def graph_call(df, tab_value):
    tab_value = tab_value
    out_df = df
    df_read = pd.DataFrame()
    df_al = pd.DataFrame()
    df_ad = pd.DataFrame()
    output_column_list = out_df.columns

    column_list_reading = ['flesch_kincaid_grades', 'flesch_reading_eases', 'automated_readability_indexes']
    column_list_Alignmentcategory = ['rouge-l','bert_score', 'answer_relevancy','context_recall','answer_similarity','answer_correctness',
                                     'faithfulness','coherence','conciseness',"answer_relevance","answer_factualConsistency",
                                     "context_entity_recall", "noise_sensitivity", "context_utilization"]
    column_list_AdverserialCategory = ["harmfulness",
                                       "maliciousness",
                                       "answer_toxicity",
                                    "question_toxicity",
                                    "question_banCode",
                                    "question_code",
                                    "question_gibberish",
                                    "question_promptInjection",
                                    "question_secrets",
                                    "question_regex",
                                    "answer_banSubstrings",
                                    "answer_banTopics",
                                    "answer_regex",
                                    "answer_gibberish",
                                    "answer_code",
                                    "answer_bias",
                                    "answer_language_same",
                                    "answer_maliciousURLs",
                                    "answer_noRefusal"
                                    ]

    for i in out_df.columns:
        if i in column_list_reading:
            df_read[i] = [out_df.loc[:, i].mean(skipna=True)]

    for j in out_df.columns:
        if j in column_list_Alignmentcategory:
            df_al[j] = [out_df.loc[:, j].mean(skipna=True)]

    for j in out_df.columns:
        if j in column_list_AdverserialCategory:
            df_ad[j] = [out_df.loc[:, j].mean(skipna=True)]

  

    if not df_al.empty:

        value_list_al = df_al.loc[0, :].values.flatten().tolist()

        val = []
        for i in value_list_al:
            val.append(i*100)
        df2 = df_al.transpose()
        df2.index.name = 'metrics'
        df2.rename(columns={0: 'values'}, inplace=True)

        val_metrics_avg_al = (df2['values'].mean(skipna=True))*100
        metric_graph = gauge_chart(val_metrics_avg_al, "Overall Alignment Metric Score", tab_value)
        scatter_pl = scatter_plot_chart(list(df_al.columns), val, "Avg. Alignment Scores", 100, tab_value)

    
   
    if not df_read.empty:
        value_list_reading = df_read.loc[0, :].values.flatten().tolist()

        df1 = df_read.transpose()
        df1.index.name = 'metrics'
        df1.rename(columns={0: 'values'}, inplace=True)
        y = create_bar_chart(list(df_read.columns), value_list_reading, 'Text Quality', tab_value)
        

    if not df_ad.empty:
        value_list_ad = df_ad.loc[0, :].values.flatten().tolist()
        val = []
        for i in value_list_ad:
            val.append(i)
        df3 = df_ad.transpose()
        df3.index.name = 'metrics'
        df3.rename(columns={0: 'values'}, inplace=True)
        val_metrics_avg_ad = (df3['values'].mean(skipna=True))
        metric_graph = gauge_chart(val_metrics_avg_ad*100, "Overall Adverserial Metrics", tab_value)
        scatter_plot_ad = scatter_plot_chart(list(df_ad.columns), val, "Avg. Adverserial Scores", 1, tab_value)
        y = create_bar_chart(list(df_ad.columns), val, 'Adverserial Metrics Score', tab_value)




#UI

st.title("LLMInspector - Metric Evaluation")
# Side Bar

st.sidebar.header("Metric Evaluation Settings: ")
metrics_opions = ["Eval_Metrics"]
metrics_selection = st.sidebar.selectbox("Select Validation metrics", metrics_opions)
st.sidebar.write("List of Evaluation Metrics:")

if metrics_selection == "Eval_Metrics":

    scanner_eval_data = eval_metrics_list.sort()

    expand = st.sidebar.expander("Select Eval metrics Scanners", icon=":material/display_settings:")
    scanner_eval = []

    checkboxes = {}

    with expand:
        for i in eval_metrics_list:
            checkboxes[i] = st.checkbox(i, key= i, value = True)

    dfs = []

    for key, value in checkboxes.items():
        if value:
            scanner_eval.append(key)

    eval_with_threshold = {}
    st.sidebar.write(":material/tune: Temperature Control:")
    for index, (k, v)in enumerate(eval_thresholds.items()):
        if k in scanner_eval:
            eval_threshold_value = st.sidebar.slider(str(k) + " threshold:", min_value=0.0, max_value=1.0, value=float(v), step=0.1)
            eval_with_threshold[k] = eval_threshold_value


#Single Prompt evaluation

tab1, tab2 = st.tabs(["LLMInspector Playground", "Metric Evaluation using Excel"])

tab1.subheader("LLMInspector Metric Evaluation Playground")
tab1.caption("Please insert below fields to generate Insights for Single input")

user_question_help = '''For below metrics evaluation, this is a mandatory input field - \n
answer_banSubstrings,
answer_banTopics,
\nanswer_bias,
answer_code,
\nanswer_correctness,
answer_factualConsistency,
\nanswer_gibberish,
answer_language_same,
\nanswer_maliciousURLs,
answer_noRefusal,
\nanswer_regex,
answer_relevance,
answer_relevancy,
\nanswer_senstivity,
answer_similarity,
\ncoherence,
conciseness,
\ncontext_precision,
context_recall,
\nfaithfulness,
harmfulness,
\nmaliciousness,
question_code,
\nquestion_emotion,
question_gibberish,
\nquestion_language,
question_promptInjection,
\nquestion_regex,
question_secrets,
\nquestion_sentiment,
question_toxicity
'''

user_answer_help = '''For below metrics evaluation, this is a mandatory input field -
\nanswer_banSubstrings,
\nanswer_banTopics,
\nanswer_bias,
\nanswer_code,
\nanswer_correctness,
\nanswer_emotion,
\nanswer_factualConsistency,
\nanswer_gibberish,
\nanswer_language,
\nanswer_language_same,
\nanswer_maliciousURLs,
\nanswer_noRefusal,
\nanswer_regex,
\nanswer_relevance,
\nanswer_relevancy,
\nanswer_senstivity,
\nanswer_sentiment,
\nanswer_similarity,
\nanswer_toxicity,
\nbert_score,
\ncoherence,
\nconciseness,
\ncontext_precision,
\ncontext_recall,
\nfaithfulness,
\nharmfulness,
\nmaliciousness,
\npii_detection,
\nreadability,
\nrouge_score,
\ntranslate
'''
user_groundtruth_help = '''For below metrics evaluation, this is a mandatory input field -
\nanswer_correctness,
\nanswer_similarity,
\nbert_score,
\ncontext_precision,
\ncontext_recall,
\nrouge_score
'''
user_context_help = '''For below metrics evaluation, this is a mandatory input field -
\nanswer_relevancy,
\ncoherence,
\nconciseness,
\ncontext_precision,
\ncontext_recall,
\nfaithfulness,
\nharmfulness,
\nmaliciousness
'''


user_question = tab1.text_area("Question:", height=40, help=user_question_help)
user_answer = tab1.text_area("Answer:", height=40, help = user_answer_help)
user_groundtruth = tab1.text_area("Ground truth:", height=40, help = user_groundtruth_help)
user_context = tab1.text_area("Context:", height=40, help = user_context_help)

#Select Scanners
#if metrics_selection == 'Eval_Metrics' :


def callback():
    st.session_state.button_clicked = True


# Disable the submit button after it is clicked
def disable():
    st.session_state.disabled = True
    #st.session_state.button_clicked = False

# Initialize disabled for form_submit_button to False
if "disabled" not in st.session_state:
    st.session_state.disabled = False

#Gen_button = st.button("Generate Metrics", type="primary", on_click=disable, disabled=st.session_state.disabled)

# if "button_clicked" not in st.session_state:    
#     st.session_state.button_clicked = False

# if st.session_state.get("Gen_button", False):
#     st.session_state.disabled = False
# elif st.session_state.get("download_button", True):
#     if st.session_state.button_clicked == True:
#         st.session_state.disabled = True
#         st.session_state.button_clicked = False
#     else:
#         st.session_state.disabled = False
#         st.session_state.button_clicked = True

Gen_button = tab1.button("Generate Metrics", type="primary")




#if metrics_selection =='Eval_Metrics':
# Check if a file has been uploaded


if metrics_selection == 'Eval_Metrics':  
    if len(scanner_eval) != 0:
        if (user_question or user_answer or user_groundtruth or user_context):
            #data = [[user_question], [user_answer], [ user_groundtruth]]

            eval_input_df = pd.DataFrame()
            if len(user_question) > 0 : 
                eval_input_df['question'] = [user_question]
            if len(user_answer) > 0 :    
                eval_input_df['answer'] = [user_answer]
            if len(user_groundtruth) > 0 :  
                eval_input_df['ground_truth'] = [user_groundtruth]
            if len(user_context) > 0 :  
                eval_input_df['contexts'] = [user_context]
            #eval_input_df = pd.DataFrame(data, columns=['question', 'answer','ground_truth_col'])
            #metric_list = scanner_eval_options
            metric_list = scanner_eval
            threshold = eval_with_threshold
            logger.info(eval_with_threshold)
            print(eval_with_threshold)
            #env_path = "llmInspectorEnv.env"

            #Gen_button = st.button("Generate Metrics", type="primary")
            if (Gen_button):
                with st.spinner("Generating metrics..."):
                    try:
                        evalu = EvalMetrics(eval_input_df, config, env_path, metric_list, threshold)   
                        result = evalu.calculate_metrics()
                    #res = evalu.export_insights()
                    #res = '/home/azureuser/cloudfiles/code/Users/shraddha.pawar/LLMInspector_Streamlit/output/insights_output_file_name_.xlsx'
                    except Exception as e:
                        logger.info("Metric Evaluation failed with below error -   Error: {e}")
                        st.write(f"Please make sure inputs are provided correctly. Metric Evaluation failed with below error -  Error: {e}")

                
                if len(result) != 0:
                    # dir = "output"
                    # path = os.path.join(dir)
                    # if os.path.isdir('output'):
                    #     output_path=path
                    # else:
                    #     try:
                    #         os.mkdir(path)
                    #         print("% s Folder created successfully" % path)
                    #     except OSError as error:
                    #         print(error)
                    #         print("Folder path cannot be created")
                    # # df_out = res.to_excel("Alignment_TestDataset.xlsx", index=False)
                    # df_out = result.to_excel("./" + path + "/Metric_Evaluation_result.xlsx")


                    # out_df = pd.read_excel("./" + path + "/Metric_Evaluation_result.xlsx")
                    tab1.markdown("### Generated Metrics:")
                    expand = tab1.expander("Evaluation Result Data Preview", icon=":material/table:")
                    with expand:
                        tab1.dataframe(result)

                    df_xlsx = to_excel(result)
                    tab1.download_button(label='Download Current Result ⬇️',
                                    data=df_xlsx,
                                    file_name= "Metric_Evaluation_result_" 
                                + str(dt_time.year)
                                + str(dt_time.month)
                                + str(dt_time.day)
                                + "_"
                                + str(dt_time.hour)
                                + str(dt_time.minute)
                                + ".xlsx", on_click=disable)
                    graph_call(result , tab1)

    else:
        tab1.warning("Please enter above inputs.")



tab2.subheader("Excel File Upload for evaluation")

uploaded_file = tab2.file_uploader("Upload an Excel File for Metric Evaluation", type=["xlsx", "xls"])
tab2.caption(" For Eval Metrics Generation, uploaded excel should have column names: 'question', 'answer','ground_truth' or 'context")

tab2.markdown("\n\n")


def callback():
    st.session_state.button_clicked = True


# Disable the submit button after it is clicked
def disable():
    st.session_state.disabled = True
    st.session_state.button_clicked = False

# Initialize disabled for form_submit_button to False
if "disabled" not in st.session_state:
    st.session_state.disabled = False


if "button_clicked" not in st.session_state:    
    st.session_state.button_clicked = False

if st.session_state.get("Gen_button", False):
    st.session_state.disabled = False
elif st.session_state.get("download_button", True):
    if st.session_state.button_clicked == True:
        st.session_state.disabled = True
        st.session_state.button_clicked = False
    else:
        st.session_state.disabled = False
        st.session_state.button_clicked = True

#Generate button
Gen_button2 = tab2.button("Generate Evaluation Metrics", type="primary")

if metrics_selection =='Eval_Metrics':
    # Check if a file has been uploaded
    if len(scanner_eval) != 0:
        if uploaded_file is not None:
            # Read the Excel file
            dataset = load_data(uploaded_file)
            
            metric_list = scanner_eval
            threshold = eval_with_threshold

            if (Gen_button2):
                with st.spinner("Generating metrics..."):
                    try:
                        evalu = EvalMetrics(dataset, config, env_path, metric_list, threshold)
                        result = evalu.calculate_metrics()
                    except Exception as e:
                        logger.info("Please make sure inputs are provided correctly. Error: {e}")
                        st.write(f"Please make sure inputs are provided correctly. Metric Evaluation failed with below error -  Error: {e}")

                if len(result) != 0:
                    dir = "output"
                    path = os.path.join(dir)
                    if os.path.isdir('output'):
                        output_path=path
                    else:
                        try:
                            os.mkdir(path)
                            print("% s Folder created successfully" % path)
                        except OSError as error:
                            print(error)
                            print("Folder path cannot be created")

                    df_out = result.to_excel("./" + path + "/MetricEvaluation_result.xlsx", index=False)
                    
                    out_df = pd.read_excel("./" + path + "/MetricEvaluation_result.xlsx")
                    tab2.markdown("### Generated Metrics:")

                    expand = st.expander("Evaluation Result Data Preview", icon=":material/table:")
                    with expand:
                        tab2.dataframe(out_df)
                    df_xlsx = to_excel(out_df)
                    tab2.download_button(label='Download Current Result ⬇️',
                                    data=df_xlsx ,
                                    file_name= "Metrics_output_" + metrics_selection
                                + "_"
                                + str(dt_time.year)
                                + str(dt_time.month)
                                + str(dt_time.day)
                                + "_"
                                + str(dt_time.hour)
                                + str(dt_time.minute)
                                + ".xlsx")
                        
                    graph_call(result, tab2)
