from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import DirectoryLoader
import ast
from tqdm import tqdm
import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
logger = logging.getLogger(__name__)
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

    def __init__(self, config, env_path, inpDir=None, df=None, threshold=None, test_size=None, document_list=None, prompt_value=None):
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
        self.thresholds = threshold if threshold is not None else ast.literal_eval(self.rag_file["thresholds"])
        self.input_dir = self.rag_file["RAG_testset_input_directory"]
        self.testset_filename = self.rag_file["RAG_testset_input_filename"]
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
        self.documents = document_list if document_list is not None else None
        self.prompt_val: str = prompt_value if prompt_value is not None else self.rag_file['prompt']

    def initialize_models(self):
        """
        Initializes the required models.
        """
        azure_openai_configs = {
            "base_url": self.azure_endpoint,
            "api_version": self.api_version,
            "api_key": self.api_key,
            "model_deployment": "gpt-4o-mini",
            "model_name": "gpt-4o-mini",
            "embedding_deployment": "text-embedding-ada-002",
            "embedding_name": "text-embedding-ada-002",  
        }

        self.azure_llm = AzureChatOpenAI(
            api_key=azure_openai_configs["api_key"],
            openai_api_version=azure_openai_configs["api_version"],
            azure_endpoint=azure_openai_configs["base_url"],
            azure_deployment=azure_openai_configs["model_deployment"],
            model=azure_openai_configs["model_name"],
            validate_base_url=False,
            timeout=120
        )

        # init the embeddings for answer_relevancy, answer_correctness and answer_similarity
        azure_embeddings = AzureOpenAIEmbeddings(
            api_key=azure_openai_configs["api_key"],
            openai_api_version=azure_openai_configs["api_version"],
            azure_endpoint=azure_openai_configs["base_url"],
            azure_deployment=azure_openai_configs["embedding_deployment"],
            model=azure_openai_configs["embedding_name"],
        )

        self.evaluator_llm = LangchainLLMWrapper(self.azure_llm)
        self.azure_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)
        self.my_run_config = RunConfig(max_workers=6, timeout=120)

    def load_documents(self):
        """
        Loads documents for evaluation.
        """
        loader = DirectoryLoader(self.file_dir, glob=["**/*.pdf", "**/*.docx", "**/*.txt"], use_multithreading=True)
        self.documents = loader.load()

    def refine_answer(self, question, context, answer):
        prompt: str = self.prompt_val
        logger.info("Prompt for refining answer: " +str(prompt))
        prompt = PromptTemplate.from_template(template=prompt)
        prompt_formatted_str = prompt.format(question=question, context=context, answer=answer)
        prediction = self.azure_llm.invoke(prompt_formatted_str).content
        return prediction

    def enhance_ground_truth(self, test_df):
        responses = []
        for index, row in tqdm(test_df.iterrows(), desc="generating better GT:"):
            question = row["question"]
            answer = row["ground_truth"]
            context = row["reference_contexts"]
            response = self.refine_answer(question, answer=answer, context=context)
            responses.append(response)

        test_df["responses"] = responses 
        test_df.drop(columns=['ground_truth'], inplace=True)
        test_df.rename(columns={'responses': 'ground_truth'}, inplace=True)
        column_order = ['question', 'ground_truth', 'reference_contexts', 'synthesizer_name']
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
        logger.info("initialising models")
        self.initialize_models()

        if self.documents is None:
            logger.info("loading documents")
            self.load_documents()
            logger.info("List of documents uploaded for RAG Test data generation: " +str(self.documents))

        generator = TestsetGenerator(llm=self.evaluator_llm, embedding_model=self.azure_embeddings)
        dataset = generator.generate_with_langchain_docs(self.documents, 
                                                         testset_size=self.testsize, 
                                                         transforms_llm=self.evaluator_llm, 
                                                         transforms_embedding_model=self.azure_embeddings,
                                                         run_config=self.my_run_config)
        


        testset_df = dataset.to_pandas()
        testset_df = testset_df.rename(columns={
            'user_input': 'question',
            'reference': 'ground_truth',
        })
        print(testset_df.columns)
        testset_df = self.enhance_ground_truth(testset_df)
        return testset_df

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
            logger.info("file saved in the path: ", file_path)
            return file_path
        except Exception as e:
            print(f"Error occurred while saving the testset: {e}")
            logger.info(f"Error occurred while saving the testset: {e}")
            return None