import ast
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu
import textstat
import numpy as np
from deep_translator import GoogleTranslator
from transformers import pipeline
from evaluate import load
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline,
)
from presidio_analyzer import AnalyzerEngine
from langdetect import detect
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
    context_entity_recall,
    context_utilization,
    noise_sensitivity_irrelevant,
    noise_sensitivity_relevant,
)
from ragas.metrics.critique import (
    harmfulness,
    coherence,
    conciseness,    
    maliciousness,
)
from ragas import evaluate
import datetime
from transformers import logging
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel

from bert_score import BERTScorer

#llm_guard
from llm_guard.input_scanners import BanCode, Code, Gibberish, PromptInjection, Regex
from llm_guard.input_scanners import Secrets
from llm_guard.input_scanners.gibberish import MatchType  as gibberish_MatchType
from llm_guard.input_scanners.prompt_injection import MatchType as promptInj_MatchType
from llm_guard.input_scanners.regex import MatchType as regex_MatchType
#from llm_guard.input_scanners.toxicity import MatchType

from llm_guard.output_scanners import Gibberish as out_gibberish
from llm_guard.output_scanners.gibberish import MatchType as gibberish_out_matchType

from llm_guard.input_scanners.ban_substrings import MatchType as ban_substrings_matchType

from llm_guard.output_scanners import Regex as out_regex

from llm_guard.output_scanners import BanTopics, BanSubstrings, Bias, LanguageSame, MaliciousURLs, NoRefusal, Relevance
from llm_guard.output_scanners.bias import MatchType as bias_matchType

from llm_guard.output_scanners import Code as out_code
from llm_guard.output_scanners.no_refusal import MatchType as noRefusal_matchType
#from llm_guard.output_scanners import Relevance
from llm_guard.output_scanners import FactualConsistency
from llm_guard.output_scanners import Sensitive
from llm_guard.output_scanners import Toxicity
from llm_guard.output_scanners.toxicity import MatchType as output_toxicity_MatchType
from llm_guard.input_scanners import Toxicity as input_toxicity
from llm_guard.input_scanners.toxicity import MatchType as input_toxicity_MatchType
from lingua import LanguageDetectorBuilder
import logging


logger = logging.getLogger(__name__)

#logging.set_verbosity(logging.ERROR)


class EvalMetrics:
    def __init__(self, df, config, env_path, metrics=None, threshold=None, out_dir=None):
        dotenv_path = Path(env_path)
        load_dotenv(dotenv_path=dotenv_path)
        nltk.download('punkt_tab')
        self.api_version = os.getenv("api_version")
        self.azure_endpoint = os.getenv("azure_endpoint")
        self.api_key = os.getenv("api_key")
        self.initialize_models()
        self.df = df
        insight_file = config["Insights_File"]
        dt_time = datetime.datetime.now()
        self.question_col = insight_file["question_col"]
        self.answer_col = insight_file["answer_col"]
        self.ground_truth_col = insight_file["ground_truth_col"]
        self.context_col = insight_file["context_col"]
        #self.threshold = threshold
        
        metric_config = insight_file["Metrics"]
        self.metrics_list = metrics if metrics is not None else metric_config

        
        self.output_lang = insight_file["output_lang"]
        self.output_path = (
            insight_file["Insights_output_path"]
            + insight_file["Insights_Output_fileName"]
            + "_"
            + str(dt_time.year)
            + str(dt_time.month)
            + str(dt_time.day)
            + "_"
            + str(dt_time.hour)
            + str(dt_time.minute)
            + ".xlsx"
        )

        self.eval_threshold_config = ast.literal_eval(insight_file["thresholds"])
        self.threshold = threshold if threshold is not None else self.eval_threshold_config
        self.banStrings_values = insight_file["ban_substring_values"]
        
        self.threshold_values = {}
        for index, (k, v)in enumerate(self.threshold.items()):
            if k in self.threshold.keys():
                key_variable = str(k) + "_threshold"
                #key_variable = v
                exec(f"{key_variable} = v")
                self.threshold_values[key_variable] = v


        self.ins_output = out_dir if out_dir is not None else self.output_path

        if len(self.metrics_list) == 0:
            self.metrics_list = [
                "rouge_score",
                "bert_score",
                "answer_similarity",
                "answer_correctness",
                "question_toxicity",
                "answer_toxicity",
                "pii_detection",
                "readability",
                "translate",
                "question_language",
                "answer_language",
                "question_sentiment",
                "answer_sentiment",
                "question_emotion",
                "answer_emotion",
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
                "harmfulness",
                "coherence",
                "conciseness",
                "maliciousness",
            ]
        

        def string_to_list(s):
    # Use ast.literal_eval to safely parse string to list
            return ast.literal_eval(s)

        self.context = []
        if "contexts" in self.df.columns:
            self.context = df['contexts'].apply(string_to_list)
            logger.info("contexts available in input: " +str(self.df.columns))
            print("contexts -- " +str(self.df.columns))
        
        self.result_df = df
        logger.info("eval_metrics initiated")
        print("eval_metrics initiated")
        logger.info("List of columns in Input: " +str(self.df.columns))
        print("contexts check >  " +str(self.df.columns))

    def initialize_models(self):
        """
        Initializes the required models.
        """
        self.azure_model = AzureChatOpenAI(
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment="gpt-35-turbo-16k",
            model="gpt-35-turbo-16k",
            api_key=self.api_key,
            validate_base_url=False,
            timeout=180,
            temperature=1.0,
        )

        self.azure_embeddings = AzureOpenAIEmbeddings(
            openai_api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment="text-embedding-ada-002",
            model="text-embedding-ada-002",
            api_key=self.api_key,
            timeout=180,
        )

    def calculate_metrics(self):
        
        if (self.question_col in self.df.columns and  self.answer_col in self.df.columns  and self.ground_truth_col in self.df.columns and 'contexts' in self.df.columns) :
            if "context_precision" in self.metrics_list:
                try:
                    context_precision = self.context_precision_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.df[self.ground_truth_col],
                        self.context,
                    )
                    self.result_df["context_precision"] = context_precision
                    logger.info("context_precision done")
                    print("context_precision done")
                except Exception as e:
                    logger.info("Failed to execute - context_precision. Please refer Error: {e}")
                    print("Failed to execute - context_precision. Please refer Error: {e}")
                    self.result_df["context_precision"] = None
                    l_new=[]
                    for x in self.result_df["context_precision"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["context_precision"] = l_new
                    #continue
                    
                #self.result_df["context_precision"] = context_precision

            if "context_recall" in self.metrics_list:
                try:
                    context_recall = self.context_recall_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.df[self.ground_truth_col],
                        self.context,
                    )
                    self.result_df["context_recall"] = context_recall
                    print("context_recall done")
                    logger.info("context_recall done")
                except Exception as e:
                    logger.info("Failed to execute - context_recall. Please refer Error: {e}")
                    print("Failed to execute - context_recall. Please refer Error: {e}")
                    self.result_df["context_recall"] = None
                    l_new=[]
                    for x in self.result_df["context_recall"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["context_recall"] = l_new
            
            if "noise_sensitivity" in self.metrics_list:
                try:
                    noise_sensitivity_relevant, noise_sensitivity_irrelevant  = self.noise_sensitivity_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.df[self.ground_truth_col],
                        self.context,
                    )
                    self.result_df["noise_sensitivity_relevant"] = noise_sensitivity_relevant
                    self.result_df["noise_sensitivity_irrelevant"] = noise_sensitivity_irrelevant
                    print("noise_sensitivity done")
                    logger.info("noise_sensitivity done")
                except Exception as e:
                    logger.info("Failed to execute - noise_sensitivity. Please refer Error: {e}")
                    print("Failed to execute - noise_sensitivity. Please refer Error: {e}")
                    self.result_df["noise_sensitivity"] = None #self.result_df.apply(lambda x: np.NaN, axis=1)
                    print(type(self.result_df["noise_sensitivity"]))
                    l_new=[]
                    for x in self.result_df["noise_sensitivity"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["noise_sensitivity"] = l_new
                    #continue
                    
        if (self.question_col in self.df.columns and  self.answer_col in self.df.columns and 'contexts' in self.df.columns) :
            if "faithfulness" in self.metrics_list:
                try:
                    faithfulness = self.faithfulness_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context,
                    )
                    self.result_df["faithfulness"] = faithfulness
                    print("faithfulness done")
                    logger.info("faithfulness done")
                except Exception as e:
                    logger.info("Failed to execute - faithfulness. Please refer Error: {e}")
                    print("Failed to execute - faithfulness. Please refer Error: {e}")
                    self.result_df["faithfulness"] = None
                    l_new=[]
                    for x in self.result_df["faithfulness"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["faithfulness"] = l_new

            if "answer_relevancy" in self.metrics_list:
                try:
                    answer_relevancy = self.answer_relevancy_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context
                    )
                    self.result_df["answer_relevancy"] = answer_relevancy
                    print("answer_relevancy done")
                    logger.info("answer_relevancy done")
                except Exception as e:
                    logger.info("Failed to execute - answer_relevancy. Please refer Error: {e}")
                    print("Failed to execute - answer_relevancy. Please refer Error: {e}")
                    self.result_df["answer_relevancy"] = None
                    l_new=[]
                    for x in self.result_df["answer_relevancy"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["answer_relevancy"] = l_new

            if "harmfulness" in self.metrics_list:
                try:
                    harmfulness = self.harmfulness_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context,
                    )
                    self.result_df["harmfulness"] = harmfulness
                    print("harmfulness done")
                    logger.info("harmfulness done")
                except Exception as e:
                    logger.info("Failed to execute - harmfulness. Please refer Error: {e}")
                    print("Failed to execute - harmfulness. Please refer Error: {e}")
                    self.result_df["harmfulness"] = None
                    l_new=[]
                    for x in self.result_df["harmfulness"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["harmfulness"] = l_new

            if "coherence" in self.metrics_list:
                try:
                    coherence = self.coherence_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context,
                    )
                    self.result_df["coherence"] = coherence
                    print("coherence done")
                    logger.info("coherence done")
                except Exception as e:
                    logger.info("Failed to execute - coherence. Please refer Error: {e}")
                    print("Failed to execute - coherence. Please refer Error: {e}")
                    self.result_df["coherence"] = None
                    l_new=[]
                    for x in self.result_df["coherence"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["coherence"] = l_new

            if "conciseness" in self.metrics_list:
                try:
                    conciseness = self.conciseness_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context,
                    )
                    self.result_df["conciseness"] = conciseness
                    print("conciseness done")
                    logger.info("coherence done")
                except Exception as e:
                    logger.info("Failed to execute - conciseness. Please refer Error: {e}")
                    print("Failed to execute - conciseness. Please refer Error: {e}")
                    self.result_df["conciseness"] = None
                    l_new=[]
                    for x in self.result_df["conciseness"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                            
                    self.result_df["conciseness"] = l_new


            if "maliciousness" in self.metrics_list:
                try:
                    maliciousness = self.maliciousness_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context,
                    )
                    self.result_df["maliciousness"] = maliciousness
                    print("maliciousness done")
                    logger.info("maliciousness done")
                except Exception as e:
                    logger.info("Failed to execute - maliciousness. Please refer Error: {e}")
                    print("Failed to execute - maliciousness. Please refer Error: {e}")
                    self.result_df["maliciousness"] = None
                    l_new=[]
                    for x in self.result_df["maliciousness"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["maliciousness"] = l_new

            if "context_utilization" in self.metrics_list:
                try:
                    context_utilization = self.context_utilization_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.context,
                    )
                    self.result_df["context_utilization"] = context_utilization
                    print("context_utilization done")     
                    logger.info("context_utilization done")
                except Exception as e:
                    logger.info("Failed to execute - context_utilization. Please refer Error: {e}")
                    print("Failed to execute - context_utilization. Please refer Error: {e}")
                    self.result_df["context_utilization"] = None
                    l_new=[]
                    for x in self.result_df["context_utilization"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["context_utilization"] = l_new            
        
        if (self.question_col in self.df.columns and self.answer_col in self.df.columns and self.ground_truth_col in self.df.columns) :
            if "answer_similarity" in self.metrics_list:
                try:
                    answer_similarity = self.answer_similarity_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.df[self.ground_truth_col],
                    )
                    self.result_df["answer_similarity"] = answer_similarity
                    print("answer similarity done")
                    logger.info("answer_similarity done")
                except Exception as e:
                    logger.info("Failed to execute - answer_similarity. Please refer Error: {e}")
                    print("Failed to execute - answer_similarity. Please refer Error: {e}")
                    self.result_df["answer_similarity"] = None
                    l_new=[]
                    for x in self.result_df["answer_similarity"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_similarity"] = l_new

            if "answer_correctness" in self.metrics_list:
                try:
                    answer_correctness = self.answer_correctness_eval(
                        self.df[self.question_col],
                        self.df[self.answer_col],
                        self.df[self.ground_truth_col]
                    )
                    self.result_df["answer_correctness"] = answer_correctness
                    print("answer_correctness done")
                    logger.info("answer_correctness done")
                except Exception as e:
                    logger.info("Failed to execute - answer_correctness. Please refer Error: {e}")
                    print("Failed to execute - answer_correctness. Please refer Error: {e}")
                    self.result_df["answer_correctness"] = None
                    l_new=[]
                    for x in self.result_df["answer_correctness"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_correctness"] = l_new                

        
        if (self.answer_col in self.df.columns  and self.ground_truth_col in self.df.columns) :

            if "rouge_score" in self.metrics_list:
                rouge_score = self.accuracy_rouge(
                    self.df[self.ground_truth_col], self.df[self.answer_col]
                )
                rouge_score = self.accuracy_rouge(
                    self.df[self.ground_truth_col], self.df[self.answer_col]
                )

                self.result_df["rouge-l"] = rouge_score
                print("rouge done")

        if (self.ground_truth_col in self.df.columns and 'contexts' in self.df.columns):

            if "context_entity_recall" in self.metrics_list:
                try:
                    context_entity_recall = self.context_entity_recall_eval(
                        self.df[self.ground_truth_col],
                        self.context,
                        )
                    self.result_df["context_entity_recall"] = context_entity_recall
                    print("context_entity_recall done")
                    logger.info("context_entity_recall done")
                except Exception as e:
                    logger.info("Failed to execute - context_entity_recall. Please refer Error: {e}")
                    print("Failed to execute - context_entity_recall. Please refer Error: {e}")
                    self.result_df["context_entity_recall"] = None
                    l_new=[]
                    for x in self.result_df["context_entity_recall"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["context_entity_recall"] = l_new  

        if (self.df[self.question_col] is not None and self.df[self.answer_col] is not None) :
            if "answer_regex" in self.metrics_list:
                try:
                    answer_regex = self.answer_regex_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list()
                    )
                    self.result_df["answer_regex"] = answer_regex
                    print("answer_regex done")  
                    logger.info("answer_regex done")
                except Exception as e:
                    logger.info("Failed to execute - answer_regex. Please refer Error: {e}")
                    print("Failed to execute - answer_regex. Please refer Error: {e}")
                    self.result_df["answer_regex"] = None
                    l_new=[]
                    for x in self.result_df["answer_regex"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_regex"] = l_new

            if "answer_gibberish" in self.metrics_list:
                try:
                    answer_gibberish = self.answer_gibberish_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list()
                    )
                    self.result_df["answer_gibberish"] = answer_gibberish
                    print("answer_gibberish done")
                    logger.info("answer_gibberish done")
                except Exception as e:
                    logger.info("Failed to execute - answer_gibberish. Please refer Error: {e}")
                    print("Failed to execute - answer_gibberish. Please refer Error: {e}")
                    self.result_df["answer_gibberish"] = None
                    l_new=[]
                    for x in self.result_df["answer_gibberish"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_gibberish"] = l_new

            if "answer_banSubstrings" in self.metrics_list:
                try:
                    answer_banSubstrings = self.answer_banSubstrings_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list()
                    )
                    self.result_df["answer_banSubstrings"] = answer_banSubstrings
                    print("answer_banSubstrings done")  
                    logger.info("answer_banSubstrings done")
                except Exception as e:
                    logger.info("Failed to execute - answer_banSubstrings. Please refer Error: {e}")
                    print("Failed to execute - answer_banSubstrings. Please refer Error: {e}")
                    self.result_df["answer_banSubstrings"] = None
                    l_new=[]
                    for x in self.result_df["answer_banSubstrings"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_banSubstrings"] = l_new

            if "answer_banTopics" in self.metrics_list:
                try:
                    answer_banTopics_Threshold = 0.5
                    answer_banTopics = self.answer_banTopics_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(), answer_banTopics_Threshold
                    )
                    self.result_df["answer_banTopics"] = answer_banTopics
                    print("answer_banTopics done")
                    logger.info("answer_banTopics done")
                except Exception as e:
                    logger.info("Failed to execute - answer_banTopics. Please refer Error: {e}")
                    print("Failed to execute - answer_banTopics. Please refer Error: {e}")
                    self.result_df["answer_banTopics"] = None
                    l_new=[]
                    for x in self.result_df["answer_banTopics"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_banTopics"] = l_new

            if "answer_code" in self.metrics_list:
                try:
                    answer_code = self.answer_code_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list()
                    )
                    self.result_df["answer_code"] = answer_code
                    print("answer_code done") 
                    logger.info("answer_code done")
                except Exception as e:
                    logger.info("Failed to execute - answer_code. Please refer Error: {e}")
                    print("Failed to execute - answer_code. Please refer Error: {e}")
                    self.result_df["answer_code"] = None
                    l_new=[]
                    for x in self.result_df["answer_code"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_code"] = l_new
                
            if "answer_bias" in self.metrics_list:
                try:
                    answer_bias = self.answer_bias_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(), self.threshold_values['answer_bias_threshold']
                    )
                    self.result_df["answer_bias"] = answer_bias
                    print("answer_bias done") 
                    logger.info("answer_bias done")
                except Exception as e:
                    logger.info("Failed to execute - answer_bias. Please refer Error: {e}")
                    print("Failed to execute - answer_bias. Please refer Error: {e}")
                    self.result_df["answer_bias"] = None
                    l_new=[]
                    for x in self.result_df["answer_bias"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_bias"] = l_new
            if "answer_language_same" in self.metrics_list:
                try:
                    answer_language_same = self.answer_language_same_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list()
                                            )
                    self.result_df["answer_language_same"] = answer_language_same
                    print("answer_language_same done") 
                    logger.info("answer_language_same done")
                except Exception as e:
                    logger.info("Failed to execute - answer_language_same. Please refer Error: {e}")
                    print("Failed to execute - answer_language_same. Please refer Error: {e}")
                    self.result_df["answer_language_same"] = None
                    l_new=[]
                    for x in self.result_df["answer_language_same"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_language_same"] = l_new
                
            if "answer_maliciousURLs" in self.metrics_list:
                try:
                    answer_maliciousURLs = self.answer_maliciousURLs_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(), self.threshold_values['answer_maliciousURLs_threshold']
                                            )
                    self.result_df["answer_maliciousURLs"] = answer_maliciousURLs
                    print("answer_maliciousURLs done") 
                    logger.info("answer_maliciousURLs done")
                except Exception as e:
                    logger.info("Failed to execute - answer_maliciousURLs. Please refer Error: {e}")
                    print("Failed to execute - answer_maliciousURLs. Please refer Error: {e}")
                    self.result_df["answer_maliciousURLs"] = None
                    l_new=[]
                    for x in self.result_df["answer_maliciousURLs"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_maliciousURLs"] = l_new

            if "answer_noRefusal" in self.metrics_list:
                try:
                    answer_noRefusal = self.answer_noRefusal_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(), self.threshold_values['answer_noRefusal_threshold']
                                            )
                    self.result_df["answer_noRefusal"] = answer_noRefusal
                    print("answer_noRefusal done") 
                    logger.info("answer_noRefusal done")
                except Exception as e:
                    logger.info("Failed to execute - answer_noRefusal. Please refer Error: {e}")
                    print("Failed to execute - answer_noRefusal. Please refer Error: {e}")
                    self.result_df["answer_noRefusal"] = None
                    l_new=[]
                    for x in self.result_df["answer_noRefusal"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_noRefusal"] = l_new
                
            if "answer_relevance" in self.metrics_list:
                try:
                    answer_relevance = self.answer_relevance_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(), self.threshold_values['answer_relevance_threshold']
                                            )
                    self.result_df["answer_relevance llm_guard"] = answer_relevance
                    print("answer_relevance done") 
                    logger.info("answer_relevance done")
                except Exception as e:
                    logger.info("Failed to execute - answer_relevance. Please refer Error: {e}")
                    print("Failed to execute - answer_relevance. Please refer Error: {e}")
                    self.result_df["answer_relevance"] = None
                    l_new=[]
                    for x in self.result_df["answer_relevance"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_relevance"] = l_new
                
            if "answer_factualConsistency" in self.metrics_list:
                try:
                    answer_factualConsistency = self.answer_factualConsistency_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(), self.threshold_values['answer_factualConsistency_threshold']
                                            )
                    self.result_df["answer_factualConsistency"] = answer_factualConsistency
                    print("answer_factualConsistency done") 
                    logger.info("answer_factualConsistency done")
                except Exception as e:
                    logger.info("Failed to execute - answer_factualConsistency. Please refer Error: {e}")
                    print("Failed to execute - answer_banTopics. Please refer Error: {e}")
                    self.result_df["answer_factualConsistency"] = None
                    l_new=[]
                    for x in self.result_df["answer_factualConsistency"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_factualConsistency"] = l_new
                
            if "answer_senstivity" in self.metrics_list:
                try:
                    answer_senstivity = self.answer_senstivity_detect(
                        self.df[self.question_col].to_list(), self.df[self.answer_col].to_list()
                                            )
                    self.result_df["answer_senstivity"] = answer_senstivity
                    print("answer_senstivity done")
                    logger.info("answer_senstivity done")
                except Exception as e:
                    logger.info("Failed to execute - answer_senstivity. Please refer Error: {e}")
                    print("Failed to execute - answer_senstivity. Please refer Error: {e}")
                    self.result_df["answer_senstivity"] = None
                    l_new=[]
                    for x in self.result_df["answer_senstivity"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_senstivity"] = l_new
                
            if "answer_toxicity" in self.metrics_list:
                try:
                    answer_toxicity_llm_guard = self.answer_toxicity_llm_guard_detect(self.df[self.question_col].to_list(), self.df[self.answer_col].to_list(),
                                                                                    self.threshold_values['answer_toxicity_threshold'])
                    self.result_df["answer_toxicity"] = answer_toxicity_llm_guard
                    print("answer_toxicity done") 
                    logger.info("answer_toxicity done")
                except Exception as e:
                    logger.info("Failed to execute - answer_toxicity. Please refer Error: {e}")
                    print("Failed to execute - answer_toxicity. Please refer Error: {e}")
                    self.result_df["answer_toxicity"] = None
                    l_new=[]
                    for x in self.result_df["answer_toxicity"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_toxicity"] = l_new
                
        if (self.df[self.answer_col] is not None):
            if "pii_detection" in self.metrics_list:
                try:
                    pii_detection = self.pii_detection(self.df[self.answer_col].to_list())
                    self.result_df["PII"] = pii_detection
                    print("pii_detection done")
                    logger.info("pii_detection done")
                except Exception as e:
                    logger.info("Failed to execute - pii_detection. Please refer Error: {e}")
                    print("Failed to execute - pii_detection. Please refer Error: {e}")
                    self.result_df["pii_detection"] = None
                    l_new=[]
                    for x in self.result_df["pii_detection"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["pii_detection"] = l_new
                
            if "readability" in self.metrics_list:
                try:
                    (
                        flesch_kincaid_grades,
                        flesch_reading_eases,
                        automated_readability_indexes,
                    ) = self.text_quality(self.df[self.answer_col].to_list())
                    self.result_df["flesch_kincaid_grades"] = flesch_kincaid_grades
                    self.result_df["flesch_reading_eases"] = flesch_reading_eases
                    self.result_df["automated_readability_indexes"] = (
                        automated_readability_indexes
                    )
                    print("readability_index done")
                    logger.info("readability_index done")
                except Exception as e:
                    logger.info("Failed to execute - readability_index. Please refer Error: {e}")
                    print("Failed to execute - readability_index. Please refer Error: {e}")
                    self.result_df["readability_index"] = None
                    l_new=[]
                    for x in self.result_df["readability_index"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["readability_index"] = l_new
        


            if "translate" in self.metrics_list:
                try:
                    traslated_text = self.translate(
                        self.df[self.answer_col].to_list(), output_lang=self.output_lang
                    )
                    self.result_df["answer_traslated_text"] = traslated_text
                    print("Translation done")
                    logger.info("answer_translate done")
                except Exception as e:
                    logger.info("Failed to execute - answer_translate. Please refer Error: {e}")
                    print("Failed to execute - answer_translate. Please refer Error: {e}")
                    self.result_df["answer_traslated_text"] = None
                    l_new=[]
                    for x in self.result_df["answer_translate"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_traslated_text"] = l_new
        

            if "answer_language" in self.metrics_list:
                try:
                    answer_language = self.detect_language(self.df[self.answer_col].to_list())
                    self.result_df["answer_language"] = answer_language
                    print("answer lang detect done")
                    logger.info("answer_language done")
                except Exception as e:
                    logger.info("Failed to execute - answer_language. Please refer Error: {e}")
                    print("Failed to execute - answer_language. Please refer Error: {e}")
                    self.result_df["answer_language"] = None
                    l_new=[]
                    for x in self.result_df["answer_language"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["answer_language"] = l_new
                                
        if (self.df[self.question_col] is not None ):
            if "question_language" in self.metrics_list:
                try:
                    question_language = self.detect_language(self.df[self.question_col].to_list())
                    self.result_df["question_language"] = question_language
                    print("question lang detect done")
                    logger.info("question_language done")
                except Exception as e:
                    logger.info("Failed to execute - question_language. Please refer Error: {e}")
                    print("Failed to execute - question_language. Please refer Error: {e}")
                    self.result_df["question_language"] = None
                    l_new=[]
                    for x in self.result_df["question_language"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_language"] = l_new
                
            if "question_sentiment" in self.metrics_list:
                try:
                    question_sentiment = self.sentiment_analysis(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_sentiment"] = question_sentiment
                    print("question_sentiment done")
                    logger.info("question_sentiment done")
                except Exception as e:
                    logger.info("Failed to execute - question_sentiment. Please refer Error: {e}")
                    print("Failed to execute - question_sentiment. Please refer Error: {e}")
                    self.result_df["question_sentiment"] = None
                    l_new=[]
                    for x in self.result_df["question_sentiment"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_sentiment"] = l_new
                
            if "question_emotion" in self.metrics_list:
                try:
                    question_emotion = self.emotion_analysis(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_emotion"] = question_emotion
                    print("question_emotion done")
                    logger.info("question_emotion done")
                except Exception as e:
                    logger.info("Failed to execute - question_emotion. Please refer Error: {e}")
                    print("Failed to execute - question_emotion. Please refer Error: {e}")
                    self.result_df["question_emotion"] = None
                    l_new=[]
                    for x in self.result_df["question_emotion"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_emotion"] = l_new
                
            if "question_banCode" in self.metrics_list:
                try:
                    question_banCode = self.question_ban_code_detect(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_ban_code"] = question_banCode
                    print("question_ban_code done")
                    logger.info("question_ban_code done")
                except Exception as e:
                    logger.info("Failed to execute - question_ban_code. Please refer Error: {e}")
                    print("Failed to execute - question_ban_code. Please refer Error: {e}")
                    self.result_df["question_ban_code"] = None
                    l_new=[]
                    for x in self.result_df["question_ban_code"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_banCode"] = l_new
                
            if "question_code" in self.metrics_list:
                try:
                    question_code = self.question_code_detect(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_code"] = question_code
                    print("question_code done")
                    logger.info("question_code done")
                except Exception as e:
                    logger.info("Failed to execute - question_code. Please refer Error: {e}")
                    print("Failed to execute - answer_banTopics. Please refer Error: {e}")
                    self.result_df["question_code"] = None
                    l_new=[]
                    for x in self.result_df["question_code"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_code"] = l_new
                
            if "question_gibberish" in self.metrics_list:
                try:
                    question_gibberish = self.question_gibberish_detect(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_gibberish"] = question_gibberish
                    print("question_gibberish done")           
                    logger.info("question_gibberish done")
                except Exception as e:
                    logger.info("Failed to execute - question_gibberish. Please refer Error: {e}")
                    print("Failed to execute - question_gibberish. Please refer Error: {e}")
                    self.result_df["question_gibberish"] = None
                    l_new=[]
                    for x in self.result_df["question_gibberish"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_gibberish"] = l_new
                
            if "question_promptInjection" in self.metrics_list:
                try:
                    question_promptInjection = self.question_promptInjection_detect(
                        self.df[self.question_col].to_list(), self.threshold_values['question_promptInjection_threshold']
                    )
                    self.result_df["question_promptInjection"] = question_promptInjection
                    print("question_promptInjection done") 
                    logger.info("question_promptInjection done")
                except Exception as e:
                    logger.info("Failed to execute - question_promptInjection. Please refer Error: {e}")
                    print("Failed to execute - question_promptInjection. Please refer Error: {e}")
                    self.result_df["question_promptInjection"] = None
                    l_new=[]
                    for x in self.result_df["question_promptInjection"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_promptInjection"] = l_new
                
            if "question_secrets" in self.metrics_list:
                try:
                    question_secrets = self.question_secret_detect(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_secrets"] = question_secrets
                    print("question_secrets done")  
                    logger.info("question_secrets done")
                except Exception as e:
                    logger.info("Failed to execute - question_secrets. Please refer Error: {e}")
                    print("Failed to execute - question_secrets. Please refer Error: {e}")
                    self.result_df["question_secrets"] = None
                    l_new=[]
                    for x in self.result_df["question_secrets"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_secrets"] = l_new
                
            if "question_regex" in self.metrics_list:
                try:
                    question_regex = self.question_regex_detect(
                        self.df[self.question_col].to_list()
                    )
                    self.result_df["question_regex"] = question_regex
                    print("question_regex done")
                    logger.info("question_regex done")
                except Exception as e:
                    logger.info("Failed to execute - question_regex. Please refer Error: {e}")
                    print("Failed to execute - question_regex. Please refer Error: {e}")
                    self.result_df["question_regex"] = None
                    l_new=[]
                    for x in self.result_df["question_regex"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_regex"] = l_new
                
            if "question_toxicity" in self.metrics_list:
                try:
                    question_toxicity_llm_guard = self.question_toxicity_llm_guard_detect(self.df[self.question_col].to_list(), 
                                                                                        self.threshold_values['question_toxicity_threshold'])
                    self.result_df["question_toxicity"] = question_toxicity_llm_guard
                    print("question_toxicity done")
                    logger.info("question_toxicity done")
                except Exception as e:
                    logger.info("Failed to execute - question_toxicity. Please refer Error: {e}")
                    print("Failed to execute - question_toxicity. Please refer Error: {e}")
                    self.result_df["question_toxicity"] = None
                    l_new=[]
                    for x in self.result_df["question_toxicity"].tolist():
                        if x is None:
                            print(x)
                            l_new.append('Not calculated')
                    self.result_df["question_toxicity"] = l_new

        return self.result_df
    
    def add_result_column(self, df):
        def check_thresholds(row):
            for metric, threshold in self.threshold.items():
                if metric in ['harmfulness', 'coherence', 'conciseness', 'maliciousness']:
                    if row[metric] != threshold:
                        return 'Fail'
                else:
                    if row[metric] < threshold:
                        return 'Fail'
            return 'Pass'
        
        df['result'] = df.apply(check_thresholds, axis=0)
        
        return df
    


    def accuracy_rouge(self, expected_responses, actual_responses):
        rouge = Rouge()
        rouge_scores = []
        for i in range(len(actual_responses)):
            score = rouge.get_scores(actual_responses[i], expected_responses[i])
            rouge_scores.append(score[0]["rouge-l"]["f"])
        return rouge_scores

    def accuracy_bleu(self, expected_responses, actual_responses):
        bleu_scores = []
        for i in range(len(expected_responses)):
            reference = [expected_responses[i].split()]
            candidate = actual_responses[i].split()
            bleu_scores.append(sentence_bleu(reference, candidate))
        return bleu_scores

    def bertscore(self, references, predictions):
        results = []
        scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')
        reference = references
        candidate = predictions
        #scorer = BERTScorer(model_type='microsoft/deberta-xlarge-mnli')

        for i in range(len(candidate)):
            P, R, F1 = scorer.score([candidate[i]], [reference[i]])
            results.append(float(F1.mean()))

        return results
    
    def detect_language(self, texts):
        results = []
        self.lang_detector = LanguageDetectorBuilder.from_all_languages().build()
        
        for cnt, ele in enumerate(texts):
            detected_language = self.lang_detector.detect_language_of(ele)
            if detected_language:
                language = str(detected_language).split('.')[1]
                results.append(language)
        
        return results
        

    def toxicity_detection(self, texts):
        model_path = "martin-ha/toxic-comment-model"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        results = pipeline(texts)
        results = [i["label"] for i in results]
        return results

    def pii_detection(self, texts):
        analyzer = AnalyzerEngine()
        entities = [
            "CREDIT_CARD",
            "CRYPTO",
            "EMAIL_ADDRESS",
            "IBAN_CODE",
            "IP_ADDRESS",
            "NRP",
            "LOCATION",
            "PERSON",
            "PHONE_NUMBER",
            "MEDICAL_LICENSE",
            "URL",
            "IN_PAN",
            "IN_AADHAAR",
            "IN_VEHICLE_REGISTRATION",
        ]
        results = [
            analyzer.analyze(text=text, entities=entities, language="en")
            for text in texts
        ]
        return results

    def text_quality(self, actual_responses):
        flesch_kincaid_grades = []
        flesch_reading_eases = []
        automated_readability_indexes = []
        for i in actual_responses:
            flesch_kincaid_grades.append(textstat.flesch_kincaid_grade(i))
            flesch_reading_eases.append(textstat.flesch_reading_ease(i))
            automated_readability_indexes.append(
                textstat.automated_readability_index(i)
            )
        return (
            flesch_kincaid_grades,
            flesch_reading_eases,
            automated_readability_indexes,
        )

    def translate(self, texts, output_lang):
        translator = GoogleTranslator(source="auto", target=output_lang)
        batch_output = translator.translate_batch(texts)
        return batch_output

    def lang_detect(self, texts):
        det_langs = [detect(i) for i in texts]
        return det_langs

    def sentiment_analysis(self, texts):
        model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_task = pipeline(
            "sentiment-analysis", model=model_path, tokenizer=model_path
        )
        sentiments = sentiment_task(texts)
        sentiments = [i["label"] for i in sentiments]
        return sentiments

    def emotion_analysis(self, texts):
        classifier = pipeline(
            "text-classification", model="SamLowe/roberta-base-go_emotions"
        )
        emotions = classifier(texts)
        emotions = [i["label"] for i in emotions]
        return emotions
    
    def question_ban_code_detect(self,texts):
        ban_score = []
        for i in texts:
            scanner = BanCode()
            sanitized_prompt, is_valid, risk_score = scanner.scan(i)
            ban_score.append(risk_score)
        return ban_score
    
    def question_code_detect(self,texts):
        code_score = []
        for i in texts:
            scanner = Code(languages=["Python","Java"], is_blocked=True)  #sql
            sanitized_prompt, is_valid, risk_score = scanner.scan(i)
            code_score.append(risk_score)
        return code_score
    
    def question_gibberish_detect(self,texts):
        gibberish_score = []
        for i in texts:
            scanner = Gibberish(match_type=gibberish_MatchType.FULL)
            sanitized_prompt, is_valid, risk_score = scanner.scan(i)
            gibberish_score.append(risk_score)
        return gibberish_score

    def question_promptInjection_detect(self,texts, threshold_value):
        promptInjection_score = []
        for i in texts:
            scanner = PromptInjection(threshold=threshold_value, match_type=promptInj_MatchType.FULL)
            sanitized_prompt, is_valid, risk_score = scanner.scan(i)
            promptInjection_score.append(risk_score)
        return promptInjection_score

    def question_secret_detect(self,texts):
        secret_score = []
        for i in texts:
            scanner = Secrets(redact_mode=Secrets.REDACT_PARTIAL)
            sanitized_prompt, is_valid, risk_score = scanner.scan(i)
            secret_score.append(risk_score)
        return secret_score

    def question_regex_detect(self,texts):
        regex_score = []

        scanner = Regex(
                    patterns=[r"Bearer [A-Za-z0-9-._~+/]+"],  # List of regex patterns
                    is_blocked=True,  # If True, patterns are treated as 'bad'; if False, as 'good'
                    match_type=regex_MatchType.SEARCH,  # Can be SEARCH or FULL_MATCH
                    redact=True,  # Enable or disable redaction
                )

        for i in texts:
            sanitized_prompt, is_valid, risk_score = scanner.scan(i)
            regex_score.append(risk_score)
        return regex_score


    def answer_regex_detect(self,question, answer):
        answer_regex_score = []

        scanner = out_regex(
            patterns=[r"Bearer [A-Za-z0-9-._~+/]+"],  # List of regex patterns
            is_blocked=True,  # If True, patterns are treated as 'bad'; if False, as 'good'
            match_type=regex_MatchType.SEARCH,  # Can be SEARCH or FULL_MATCH
            redact=True,  # Enable or disable redaction
        )
        for i in range(len(question)):
            sanitized_prompt, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_regex_score.append(risk_score)
        return answer_regex_score

    def answer_gibberish_detect(self,question, answer):
        answer_gibberish_score = []

        for i in range(len(question)):
            scanner = out_gibberish(match_type=gibberish_out_matchType.FULL)
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_gibberish_score.append(risk_score)

        return answer_gibberish_score

    def answer_banSubstrings_detect(self, question, answer):
        answer_banSubstrings_score = []

        scanner = BanSubstrings(
        substrings=self.banStrings_values,
        match_type=ban_substrings_matchType.WORD,
        case_sensitive=False,
        redact=False,
        contains_all=False,
        )
        for i in range(len(question)):
            scanner = out_gibberish(match_type=gibberish_out_matchType.FULL)
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_banSubstrings_score.append(risk_score)

        return answer_banSubstrings_score

    def answer_banTopics_detect(self, question, answer, threshold):
        answer_banTopics_score = []

        for i in range(len(question)):
            scanner = BanTopics(topics=["violence"], threshold=threshold)
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_banTopics_score.append(risk_score)

        return answer_banTopics_score
    
    def answer_code_detect(self,question, answer):
        answer_code_score = []

        for i in range(len(question)):
            scanner = out_code(languages=["Python","Java"], is_blocked=True)
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_code_score.append(risk_score)

        return answer_code_score

    def answer_bias_detect(self,question, answer, threshold):
        answer_bias_score = []

        for i in range(len(question)):
            scanner = Bias(threshold=threshold, match_type=bias_matchType.FULL)
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_bias_score.append(risk_score)

        return answer_bias_score
    
    def answer_language_same_detect(self,question, answer):
        answer_language_same_score = []
        scanner = LanguageSame()
        
        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_language_same_score.append(risk_score)

        return answer_language_same_score


    def answer_maliciousURLs_detect(self,question, answer, threshold):
        answer_maliciousURLs_score = []
        scanner = MaliciousURLs(threshold=threshold)
        
        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_maliciousURLs_score.append(risk_score)

        return answer_maliciousURLs_score
    
    def answer_noRefusal_detect(self,question, answer, threshold):
        answer_noRefusal_score = []

        scanner = NoRefusal(threshold=threshold, match_type=noRefusal_matchType.FULL)

        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_noRefusal_score.append(risk_score)

        return answer_noRefusal_score

    def answer_relevance_detect(self,question, answer, threshold):
        answer_relevance_score = []

        scanner = Relevance(threshold=threshold)

        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_relevance_score.append(risk_score)

        return answer_relevance_score

    def answer_factualConsistency_detect(self,question, answer, threshold):
        answer_factualConsistency_score = []
        scanner = FactualConsistency(minimum_score=threshold)

        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_factualConsistency_score.append(risk_score)

        return answer_factualConsistency_score
    
    def answer_senstivity_detect(self,question, answer):
        answer_senstivity_score = []
        scanner = Sensitive(entity_types=["PERSON", "EMAIL"], redact=True)

        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_senstivity_score.append(risk_score)

        return answer_senstivity_score

    def answer_toxicity_llm_guard_detect(self, question, answer, threshold):
        answer_toxicity_llm_guard_score = []
        scanner = Toxicity(threshold=threshold, match_type=output_toxicity_MatchType.SENTENCE)

        for i in range(len(answer)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i], answer[i])
            answer_toxicity_llm_guard_score.append(risk_score)

        return answer_toxicity_llm_guard_score

    def question_toxicity_llm_guard_detect(self,question, threshold):
        question_toxicity_llm_guard_score = []
        scanner = input_toxicity(threshold=threshold, match_type=input_toxicity_MatchType.SENTENCE)

        for i in range(len(question)):
            sanitized_output, is_valid, risk_score = scanner.scan(question[i])
            question_toxicity_llm_guard_score.append(risk_score)

        return question_toxicity_llm_guard_score
    
    def answer_similarity_eval(self, questions, answer, ground_truth):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_similarity],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["answer_similarity"].to_list()

    def answer_correctness_eval(self, questions, answer, ground_truth):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_correctness],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["answer_correctness"].to_list()
    
    def faithfulness_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["faithfulness"].to_list()
    
    def context_precision_eval(self, questions, answer, ground_truth, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[context_precision],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["context_precision"].to_list()

    def context_recall_eval(self, questions, answer, ground_truth, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[context_recall],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["context_recall"].to_list()

    def answer_relevancy_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["answer_relevancy"].to_list()


    def harmfulness_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[harmfulness],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["harmfulness"].to_list()
    
    def coherence_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[coherence],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["coherence"].to_list()
    
    def conciseness_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[conciseness],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["conciseness"].to_list()
    
    def maliciousness_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[maliciousness],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["maliciousness"].to_list()
        
    def context_entity_recall_eval(self, ground_truth, contexts):
        data_samples = {
            "ground_truth": ground_truth,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[context_entity_recall],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["context_entity_recall"].to_list()

    def context_utilization_eval(self, questions, answer, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[context_utilization],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        return df["context_utilization"].to_list()

    def noise_sensitivity_eval(self, questions, answer, ground_truth, contexts):
        data_samples = {
            "question": questions,
            "answer": answer,
            "ground_truth": ground_truth,
            "contexts": contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[noise_sensitivity_relevant, noise_sensitivity_irrelevant],
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        df = score.to_pandas()
        #print(df)
        return df["noise_sensitivity_relevant"].to_list(), df["noise_sensitivity_irrelevant"].to_list()


    def export_insights(self):
        print("Insights can be obtained on the provided path:", self.ins_output)
        self.result_df.to_excel(self.ins_output)
        return self.ins_output
