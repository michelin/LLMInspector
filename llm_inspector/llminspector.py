from llm_inspector.adversarial import Adversarial
from llm_inspector.alignment import Alignment
from llm_inspector.eval_metrics import EvalMetrics
from llm_inspector.rag_eval import RagEval
from configparser import ConfigParser
import os
import subprocess


class llminspector:
    """
    Class for managing various tasks related to inspecting and evaluating language models.

    Attributes:
        config_path (str): Path to the configuration file (default is 'config.ini').
        env_path (str): Path to the .env file (default is '.env').

    Methods:
        __init__(config_path='config.ini', env_path='.env'): Initializes the llminspector object.
        alignment(): Export alignment data.
        adversarial(): Export adversarial data.
        evaluate(df): Calculate evaluation metrics and export insights.
        rag_testset_gen(): Generate test set for RAG evaluation.
        rag_evaluate(): Perform RAG evaluation and export evaluation results.
    """

    def __init__(self, config_path="config.ini", env_path=".env"):
        """
        Initializes the llminspector object.

        Args:
            config_path (str, optional): Path to the configuration file (default is 'config.ini').
        """
        self.config_path = config_path
        self.env_path = env_path
        self.config = ConfigParser()
        self.config.read(config_path)

    def alignment(self):
        """
        Export alignment data.
        """
        alignment_obj = Alignment(self.config)
        alignment_obj.export_alignment_data()

    def adversarial(self):
        """
        Export adversarial data.
        """
        adversarial_obj = Adversarial(self.config)
        adversarial_obj.export_adversarial_data()

    def evaluate(self, df):
        """
        Calculate evaluation metrics and export insights.

        Args:
            df (DataFrame): Input DataFrame for evaluation.
        """
        evalu = EvalMetrics(df, self.config, self.env_path)
        evalu.calculate_metrics()
        evalu.export_insights()

    def rag_testset_gen(self):
        """
        Generate test set for RAG evaluation.
        """
        rag_testset = RagEval(self.config, self.env_path)
        rag_testset.generate_testset()
        rag_testset.export_testset()

    def rag_evaluate(self):
        """
        Perform RAG evaluation and export evaluation results.
        """
        rag_eval = RagEval(self.config, self.env_path)
        rag_eval.rag_evaluation()
        rag_eval.export_eval()
    
    def run_streamlit_app(self):
        """
        Launch the Streamlit application for LLM Inspector.
        """
        import importlib.resources as pkg_resources
        import llm_inspector
        
        # Get the package's installed location
        try:
            # For Python 3.9+
            streamlit_file = pkg_resources.files(llm_inspector).joinpath("LLMInspector_main.py")
            streamlit_path = str(streamlit_file)
        except AttributeError:
            # Fallback for older Python versions
            streamlit_path = pkg_resources.resource_filename("llm_inspector", "LLMInspector_main.py")
        
        subprocess.run(["streamlit", "run", streamlit_path])