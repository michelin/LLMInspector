from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain import PromptTemplate

from ragas.run_config import RunConfig
from ragas.testset.generator import TestsetGenerator

from ragas.testset.evolutions import simple, reasoning, multi_context

from llama_index.core import SimpleDirectoryReader
import ast
from datasets import Dataset

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)
from ragas.metrics.critique import (
    harmfulness,
    coherence,
    conciseness,
    maliciousness,
)
from ragas import evaluate

import pandas as pd
import datetime
import os
from tqdm import tqdm

from pathlib import Path
from dotenv import load_dotenv

dt_time = datetime.datetime.now()


class RagEval:
    """
    Class for performing RAG evaluation and exporting evaluation results.

    Attributes:
        config_path (str): Path to the configuration file.

    Methods:
        __init__(config_path): Initializes the RagEval object.
        initialize_models(): Initializes the required models.
        load_documents(): Loads documents for evaluation.
        generate_testset(): Generates the test set for RAG evaluation.
        rag_init(): Initializes the RAG model.
        rag_evaluation(): Performs RAG evaluation and exports the evaluation results.
        generate_responses(query_engine, test_questions, test_answers=None): Generates responses for given questions.
        getting_ai_response(query_str, context_str): Gets AI response for given query and context.
        getting_answer(query): Gets the answer for given query.
        export_testset(): Exports the generated test set.
        export_eval(): Exports the evaluation results.
    """

    def __init__(self, config, env_path, inpDir=None, df=None, threshold=None, test_size=None, documentList=None):
        """
        Initializes the RagEval object.

        Args:
            config (str): Path to the configuration file.
        """
        dotenv_path = Path(env_path)
        load_dotenv(dotenv_path=dotenv_path)
        self.rag_file = config["RAG_File"]
        self.api_version = os.getenv("api_version")
        self.azure_endpoint = os.getenv("azure_endpoint")
        self.api_key = os.getenv("api_key")
        #self.thresholds = ast.literal_eval(self.rag_file["thresholds"])
        self.thresholds = threshold if threshold is not None else ast.literal_eval(self.rag_file["thresholds"])
        self.input_dir = self.rag_file["RAG_testset_input_directory"]
        self.testset_filename = self.rag_file["RAG_testset_input_filename"]
        #self.file_dir = self.rag_file["RAG_testset_document_directory"]

        self.file_dir = inpDir if inpDir is not None else self.rag_file["RAG_testset_document_directory"]
        self.testsize = test_size if test_size is not None else int(self.rag_file['testset_size'])

        self.output_dir = self.rag_file["RAG_output_directory"]
        self.azure_model = None
        self.azure_embeddings = None
        self.loader = None
        self.documents = None
        self.run_config = None
        self.generator = None
        self.testset = None
        self.test_df = None
        self.df = df
        self.documents = documentList if documentList is not None else None

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

    def load_documents(self):
        """
        Loads documents for evaluation.
        """
        self.documents = SimpleDirectoryReader(
            self.file_dir,
            recursive=True,
            num_files_limit=100,
            required_exts=[".pdf", ".docx"],
            filename_as_id=True,
        ).load_data()

    def refine_answer(self, question, context, answer):
        prompt: str = self.rag_file['prompt']
        #prompt: str = 
        prompt = PromptTemplate.from_template(template=prompt)
        prompt_formatted_str = prompt.format(question=question, context=context, answer=answer)
        prediction = self.azure_model.predict(prompt_formatted_str)
        return prediction

    def enhance_ground_truth(self, test_df):
        responses = []
        for index, row in tqdm(test_df.iterrows(), desc="generating better GT:"):
            question = row["question"]
            answer = row["ground_truth"]
            context = row["contexts"]
            response = self.refine_answer(question, answer=answer, context=context)
            responses.append(response)

        test_df["responses"] = responses 
        test_df.drop(columns=['ground_truth'], inplace=True)
        test_df.rename(columns={'responses': 'ground_truth'}, inplace=True)
        column_order = ['question', 'ground_truth', 'contexts', 'metadata', 'evolution_type', 'episode_done']
        test_df = test_df[column_order]
        return test_df
    
    def add_result_column(self, df):
        def check_thresholds(row):
            for metric, threshold in self.thresholds.items():
                if metric in ['harmfulness', 'coherence', 'conciseness', 'maliciousness']:
                    if row[metric] != threshold:
                        return 'Fail'
                else:
                    if row[metric] < threshold:
                        return 'Fail'
            return 'Pass'
        
        df['result'] = df.apply(check_thresholds, axis=1)
        
        return df

    def generate_testset(self):
        """
        Generates the test set for RAG evaluation.

        Returns:
            DataFrame: Generated test set.
        """
        #test_size = int(self.rag_file["testset_size"])
        test_size = self.testsize
        print("initialising models")
        self.initialize_models()
        print("loading documents")
        if self.documents is None:
            self.load_documents()
            print("________________________________________")
            print(self.documents)
        self.run_config = RunConfig(timeout=180, max_retries=60, max_wait=180)
        self.generator = TestsetGenerator.from_langchain(
            generator_llm=self.azure_model, 
            critic_llm=self.azure_model,
            embeddings=self.azure_embeddings,
            run_config=self.run_config,
            chunk_size=256,
        )
        self.testset = self.generator.generate_with_llamaindex_docs(
            self.documents,
            test_size=test_size,  # langchain_docs
            with_debugging_logs=False,
            distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25},
            run_config=self.run_config,
        )

        self.test_df = self.testset.to_pandas()
        self.test_df = self.enhance_ground_truth(self.test_df)
        return self.test_df

    def rag_evaluation(self):
        """
        Performs RAG evaluation and exports the evaluation results.
        """
        def string_to_list(s):
            # Use ast.literal_eval to safely parse string to list
            return ast.literal_eval(s)
        
        self.initialize_models()
        self.test_df_ragEval = self.df if self.df is not None else pd.read_excel(self.input_dir + self.testset_filename, index_col=None)
        test_df = self.test_df_ragEval

        
        test_df['contexts'] = test_df['contexts'].apply(string_to_list)
        test_df1 = test_df.drop_duplicates(subset="question", keep="first")
        test_df1 = test_df1.dropna(subset=['question', 'ground_truth', 'answer', 'contexts']) # Updated to frop na based on selected list of columns

        # Typecasting below list of columns to keep data types consistent
        col_to_change = ['question', 'ground_truth', 'answer']
        test_df1[col_to_change] = test_df1[col_to_change].astype(str)

        result_ds = Dataset.from_pandas(test_df1)
        print("Dataset created3")

        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness,
            harmfulness,
            coherence,
            conciseness,
            maliciousness,
        ]

        #metrics = list(self.thresholds.keys())
        #y = list(rag_thresholds.keys())
        print("evaluating the questions")
        
        self.result = evaluate(
            result_ds,
            metrics=metrics,
            llm=self.azure_model,
            embeddings=self.azure_embeddings,
        )
        self.result_df = self.result.to_pandas()
        self.result_df = self.add_result_column(self.result_df)
        return self.result_df

    def export_testset(self):
        try:
            dt_time = datetime.datetime.now()
            filename = (
                self.rag_file["RAG_testset_Output_fileName"]
                + f"{dt_time.month}{dt_time.day}_{dt_time.hour}{dt_time.minute}.xlsx"
            )
            file_path = self.output_dir + filename
            self.test_df.to_excel(file_path, index=False)
            print("file saved in the path: ", file_path)
            return file_path
        except Exception as e:
            print(f"Error occurred while saving the testset: {e}")
            return None

    def export_eval(self):
        try:
            dt_time = datetime.datetime.now()
            filename = (
                self.rag_file["RAG_eval_Output_fileName"]
                + f"{dt_time.month}{dt_time.day}_{dt_time.hour}{dt_time.minute}.xlsx"
            )
            file_path = self.output_dir + filename
            if 'Unnamed: 0' in self.result_df.columns:
                self.result_df = self.result_df.drop('Unnamed: 0', axis=1)
            self.result_df.to_excel(file_path, index=False)
            print("file saved in the path: ", file_path)
            return file_path
        except Exception as e:
            print(f"Error occurred while saving the testset: {e}")
            return None
