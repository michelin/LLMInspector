import ast
import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from presidio_analyzer.recognizer_result import RecognizerResult
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from llm_inspector.eval_metrics import EvalMetrics
import logging

logger = logging.getLogger(__name__)

@dataclass
class RAGMetrics:
    """
    A dataclass to store RAG (Retrieval-Augmented Generation) evaluation metrics.
    
    Provides a structured way to compute and store various RAG-related metrics.
    """
    bert_score: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    answer_groundedness: Optional[float] = None
    maliciousness: Optional[float] = None
    noise_sensitivity: Optional[float] = None
    context_relevance: Optional[float] = None
    context_utilization: Optional[float] = None
    context_entity_recall: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

@dataclass
class TextMetrics:
    """
    A dataclass to store text evaluation metrics.
    
    Provides a structured way to compute and store various text-related metrics.
    """
    emotion: Optional[str] = None
    sentiment: Optional[float] = None
    language: Optional[str] = None
    pii_detected: Optional[List[RecognizerResult]] = None
    flesch_kincaid_grade: Optional[float] = None
    tokens: Optional[int] = None

@dataclass
class QuestionMetrics:
    """
    Dataclass to represent metrics for evaluating a question's risk and quality.
    """
    question_ban_code_risk: Optional[float] = None
    question_code_risk: Optional[float] = None
    question_gibberish_risk: Optional[float] = None
    question_regex_risk: Optional[float] = None
    question_toxicity: Optional[float] = None

@dataclass
class AnswerMetrics:
    """
    Dataclass to represent metrics for evaluating an answer's risk and quality.
    """
    answer_regex_risk: Optional[float] = None
    answer_gibberish_risk: Optional[float] = None
    answer_ban_substrings_risk: Optional[float] = None
    answer_ban_topics_risk: Optional[float] = None
    answer_code_risk: Optional[float] = None
    answer_no_refusal_risk: Optional[float] = None
    answer_toxicity_risk: Optional[float] = None


@dataclass
class PolicyMetrics:
    """
    Dataclass to represent metrics for evaluating an answer is compliant with the policy.
    """
    is_policy_violated: Optional[str] = None
    policy_violation_reason: Optional[str] = None

def safe_metric_call(metric_func, *args):
    """
    Safely call a metric evaluation function with enhanced debugging.
    
    Args:
        metric_func (callable): The metric evaluation function
        *args: Arguments to pass to the metric function
    
    Returns:
        Optional[Any]: The metric value or None if calculation fails
    """
    try:       
        result = metric_func(*args)
        return result
        
    except Exception as e:
        logger.error(f"Error in {metric_func.__name__}:")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parallel_metric_calc(func_args):
    """Helper function to unpack arguments for parallel processing"""
    func, args = func_args
    return (func.__name__, safe_metric_call(func, *args))

class MetricsCalculator:
    """
    Class to handle metrics calculation with thread-based parallel processing
    and selective metric evaluation
    """
    def __init__(self, df, config, env_path, metrics_list=None, threshold=None, max_workers: Optional[int] = None):
        
        self.max_workers = max_workers or (cpu_count() * 2)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Define all available metrics by category
        self.available_metrics = {
            "rag": {
                "bert_score": self._calc_bert_score,
                "faithfulness": self._calc_faithfulness,
                "answer_correctness": self._calc_answer_correctness,
                "answer_relevancy": self._calc_answer_relevancy,
                "answer_groundedness": self._calc_response_groundedness,
                "maliciousness": self._calc_maliciousness,
                "context_relevance": self._calc_context_relevance,
                "context_utilization": self._calc_context_utilization,
                "context_entity_recall": self._calc_context_entity_recall,
                "context_precision": self._calc_context_precision,
                "context_recall": self._calc_context_recall,
                "noise_sensitivity": self._calc_noise_sensitivity
            },
            "text": {
                "emotion": self._calc_emotion,
                "sentiment": self._calc_sentiment,
                "language": self._calc_language,
                "pii_detected": self._calc_pii_detection,
                "flesch_kincaid_grade": self._calc_text_quality,
                "tokens": self._calc_tokens
            },
            "question": {
                "question_ban_code_risk": self._calc_question_ban_code_risk,
                "question_code_risk": self._calc_question_code_risk,
                "question_gibberish_risk": self._calc_question_gibberish_risk,
                "question_regex_risk": self._calc_question_regex_risk,
                "question_toxicity": self._calc_question_toxicity
            },
            "answer": {
                "answer_regex_risk": self._calc_answer_regex_risk,
                "answer_gibberish_risk": self._calc_answer_gibberish_risk,
                "answer_ban_substrings_risk": self._calc_answer_ban_substrings_risk,
                "answer_ban_topics_risk": self._calc_answer_ban_topics_risk,
                "answer_code_risk": self._calc_answer_code_risk,
                "answer_no_refusal_risk": self._calc_answer_no_refusal_risk,
                "answer_toxicity_risk": self._calc_answer_toxicity_risk
            }
        }

        self.env_path = env_path

        if metrics_list is not None:
            self.metrics_dict = {}
            for category, metrics in self.available_metrics.items():
                for key, function in metrics.items():
                    if key in metrics_list:
                        # print(key)
                        if category not in self.metrics_dict:
                            self.metrics_dict[category] = {}
                            self.metrics_dict[category][key] = metrics[key]
                        else:
                            if key not in self.metrics_dict[category]:
                                self.metrics_dict[category][key] = metrics[key]
            # print(f"##List of metrics to be calculated: {self.metrics_dict}")
        else:
            self.metrics_dict = self.available_metrics

        logger.info(f"Metrics Evaluation processed for: {self.metrics_dict}")

        # Store dataframe
        self.df = df
        # self.result_df = df
        
        # Extract config settings
        insight_file = config["Insights_File"]
        
        # Set column names
        self.question_col = insight_file["question_col"]
        self.answer_col = insight_file["answer_col"]
        self.ground_truth_col = insight_file["ground_truth_col"]
        self.context_col = insight_file["context_col"]
        
        # Set output language
        self.output_lang = insight_file["output_lang"]
        
        # Generate output path with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.output_path = (
            f"{insight_file['Insights_output_path']}"
            f"{insight_file['Insights_Output_fileName']}_{timestamp}.xlsx"
        )
        
        # Set evaluation thresholds
        self.eval_threshold_config = eval(insight_file["thresholds"])
        self.threshold = threshold if threshold is not None else self.eval_threshold_config
        
        # Process threshold values
        self.threshold_values = {
            f"{k}_threshold": v for k, v in self.threshold.items()
        }

        # self.ins_output = out_dir if out_dir is not None else self.output_path
        
       
        # Process contexts if available
        self.context = []
        if "context" in self.df.columns:
            self.context = self.df['context'].apply(ast.literal_eval)
            logger.info("Contexts available in input")

        self.eval_metrics = EvalMetrics(df=self.df, config=config, env_path=self.env_path)
        
        logger.info(f"Evaluation metrics initialized. Input columns: {', '.join(self.df.columns)}")


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
        
    # Helper methods for RAG metrics
    def _calc_bert_score(self, ground_truth, answer):
        return safe_metric_call(self.eval_metrics.bertscore, ground_truth, answer)
    
    def _calc_answer_correctness(self, answer, ground_truth):
        return safe_metric_call(self.eval_metrics.answer_correctness_eval, answer, ground_truth)
    
    def _calc_faithfulness(self, question, answer, context):
        return safe_metric_call(self.eval_metrics.faithfulness_eval, question, answer, context)
    
    def _calc_answer_relevancy(self, question, answer, context):
        return safe_metric_call(self.eval_metrics.answer_relevancy_eval, question, answer, context)
        
    def _calc_response_groundedness(self, answer, context):
        return safe_metric_call(self.eval_metrics.response_groundness_eval, answer, context)
    
    def _calc_maliciousness(self, question, answer):
        return safe_metric_call(self.eval_metrics.maliciousness_eval, question, answer)
    
    def _calc_context_utilization(self, question, answer, context):
        return safe_metric_call(self.eval_metrics.context_utilization_eval, question, answer, context)
    
    def _calc_context_entity_recall(self, ground_truth, context):
        return safe_metric_call(self.eval_metrics.context_entity_recall_eval, ground_truth, context)
    
    def _calc_context_precision(self, question, ground_truth, context):
        return safe_metric_call(self.eval_metrics.context_precision_eval, question, ground_truth, context)
    
    def _calc_context_relevance(self, question, context):
        return safe_metric_call(self.eval_metrics.context_relevance_eval, question, context)
    
    def _calc_context_recall(self, question, answer, ground_truth, context):
        return safe_metric_call(self.eval_metrics.context_recall_eval, question, answer, ground_truth, context)

    def _calc_noise_sensitivity(self, question, answer, ground_truth, context):
        return safe_metric_call(self.eval_metrics.noise_sensitivity_eval, question, answer, ground_truth, context)

    # Helper methods for Text metrics
    def _calc_emotion(self, text):
        return safe_metric_call(self.eval_metrics.emotion_analysis, text)
    
    def _calc_sentiment(self, text):
        return safe_metric_call(self.eval_metrics.sentiment_analysis, text)
    
    def _calc_language(self, text):
        return safe_metric_call(self.eval_metrics.detect_language, text)
    
    def _calc_pii_detection(self, text):
        return safe_metric_call(self.eval_metrics.pii_detection, text)
    
    def _calc_text_quality(self, text):
        return safe_metric_call(self.eval_metrics.text_quality, text)
    
    def _calc_tokens(self, text):
        return safe_metric_call(self.eval_metrics.num_tokens_from_string, text)
    
    # Helper methods for Question metrics
    def _calc_question_ban_code_risk(self, text):
        return safe_metric_call(self.eval_metrics.question_ban_code_detect, text)
    
    def _calc_question_code_risk(self, text):
        return safe_metric_call(self.eval_metrics.question_code_detect, text)
    
    def _calc_question_gibberish_risk(self, text):
        return safe_metric_call(self.eval_metrics.question_gibberish_detect, text)
    
    def _calc_question_regex_risk(self, text):
        return safe_metric_call(self.eval_metrics.question_regex_detect, text)
    
    def _calc_question_toxicity(self, text):
        return safe_metric_call(self.eval_metrics.question_toxicity_detect, text)
    
    # Helper methods for Answer metrics
    def _calc_answer_regex_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_regex_detect, question, answer)
    
    def _calc_answer_gibberish_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_gibberish_detect, question, answer)
    
    def _calc_answer_ban_substrings_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_ban_substrings_detect, question, answer)
    
    def _calc_answer_ban_topics_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_ban_topics_detect, question, answer)
    
    def _calc_answer_code_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_code_detect, question, answer)
    
    def _calc_answer_no_refusal_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_no_refusal_detect, question, answer)
    
    def _calc_answer_toxicity_risk(self, question, answer):
        return safe_metric_call(self.eval_metrics.answer_toxicity_detect, question, answer)

    def get_all_available_metrics(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary of all available metrics by category
        
        Returns:
            Dict[str, List[str]]: Dictionary with metric categories and their available metrics
        """
        return {
            category: list(metrics.keys()) 
            for category, metrics in self.available_metrics.items()
        }
        
    def get_question_metrics(
        self, 
        text: str, 
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate selected question metrics using thread pool
        
        Args:
            text (str): The question text to analyze
            metrics_to_calculate (Optional[List[str]]): List of specific question metrics to calculate.
                                                       If None, calculates all metrics.
        
        Returns:
            Dict[str, Optional[float]]: Dictionary of calculated metrics
        """
        metrics = QuestionMetrics()
        
        if not text or not text.strip():
            return asdict(metrics)
            
        try:
            available_metrics = self.metrics_dict["question"]
            
            # If no specific metrics requested, calculate all
            if metrics_to_calculate is None:
                metrics_to_calculate = list(available_metrics.keys())
            
            # Filter to only valid metrics
            valid_metrics = [m for m in metrics_to_calculate if m in available_metrics]
            
            # Create futures for each requested metric
            futures = []
            metric_names = []
            
            for metric_name in valid_metrics:
                metric_func = available_metrics[metric_name]
                futures.append(self.executor.submit(metric_func, text))
                metric_names.append(metric_name)
            
            # Get results as they complete
            results = [(f.result(), name) for f, name in zip(futures, metric_names)]
            
            # Set metrics attributes
            for value, name in results:
                setattr(metrics, name, value)
                    
        except Exception as e:
            logger.error(f"Error calculating question metrics: {e}")
            print(f"Error calculating question metrics: {e}")
        
        return asdict(metrics)

    def get_answer_metrics(
        self, 
        question: str, 
        answer: str,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate selected answer metrics using thread pool
        
        Args:
            question (str): The question text
            answer (str): The generated answer text
            metrics_to_calculate (Optional[List[str]]): List of specific answer metrics to calculate.
                                                       If None, calculates all metrics.
        
        Returns:
            Dict[str, Optional[float]]: Dictionary of calculated metrics
        """
        metrics = AnswerMetrics()
        
        if not question or not answer:
            return asdict(metrics)
            
        try:
            available_metrics = self.metrics_dict["answer"]
            
            # If no specific metrics requested, calculate all
            if metrics_to_calculate is None:
                metrics_to_calculate = list(available_metrics.keys())
            
            
            # Filter to only valid metrics
            valid_metrics = [m for m in metrics_to_calculate if m in available_metrics]

            # Create futures for each requested metric
            futures = []
            metric_names = []
            
            for metric_name in valid_metrics:
                metric_func = available_metrics[metric_name]
                futures.append(self.executor.submit(metric_func, question, answer))
                metric_names.append(metric_name)
            
            # Get results as they complete
            results = [(f.result(), name) for f, name in zip(futures, metric_names)]
            # Set metrics attributes
            for value, name in results:
                setattr(metrics, name, value)
                    
        except Exception as e:
            logger.error(f"Error calculating answer metrics: {e}")
            print(f"Error calculating answer metrics: {e}")
        
        return asdict(metrics)

    def get_metrics(
        self, 
        text: str,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[str, Optional[Any]]:
        """
        Calculate selected text metrics using thread pool
        
        Args:
            text (str): The text to analyze
            metrics_to_calculate (Optional[List[str]]): List of specific text metrics to calculate.
                                                       If None, calculates all metrics.
        
        Returns:
            Dict[str, Optional[Any]]: Dictionary of calculated metrics
        """
        metrics = TextMetrics()
        
        if not text or not text.strip():
            return asdict(metrics)
            
        try:
            available_metrics = self.metrics_dict["text"]
            
            # If no specific metrics requested, calculate all
            if metrics_to_calculate is None:
                metrics_to_calculate = list(available_metrics.keys())
            
            # Filter to only valid metrics
            valid_metrics = [m for m in metrics_to_calculate if m in available_metrics]
            
            # Create futures for each requested metric
            futures = []
            metric_names = []
            
            for metric_name in valid_metrics:
                metric_func = available_metrics[metric_name]
                futures.append(self.executor.submit(metric_func, text))
                metric_names.append(metric_name)
            
            # Get results as they complete
            results = [(f.result(), name) for f, name in zip(futures, metric_names)]
            
            # Set metrics attributes
            for value, name in results:
                setattr(metrics, name, value)
                    
        except Exception as e:
            logger.error(f"Error calculating text metrics: {e}")
            print(f"Error calculating text metrics: {e}")
        
        return asdict(metrics)

    def calculate_rag_metrics(
        self,
        question, 
        answer, 
        ground_truth, 
        context,
        metrics_to_calculate: Optional[List[str]] = None
    ) -> Dict[str, Optional[float]]:
        """
        Calculate selected RAG metrics using thread pool
        
        Args:
            question (str): The original question/question
            answer (str): The generated answer
            ground_truth (Optional[str]): The ground truth answer
            context (Optional[List[str]]): List of context passages
            metrics_to_calculate (Optional[List[str]]): List of specific RAG metrics to calculate.
                                                       If None, calculates all applicable metrics.
        
        Returns:
            Dict[str, Optional[float]]: Dictionary of calculated metrics
        """
        metrics = RAGMetrics()
        
        # Check basic requirements
        if not any([question, answer, ground_truth, context]):
            return asdict(metrics)
            
        try:
            # Define all available RAG metrics and their requirements
            available_metrics = self.metrics_dict["rag"]
            
            # If no specific metrics requested, calculate all applicable ones
            if metrics_to_calculate is None:
                metrics_to_calculate = list(available_metrics.keys())
            
            # Filter to only valid metrics
            valid_metrics = [m for m in metrics_to_calculate if m in available_metrics]

            futures = []
            metric_names = []
            
            # For each requested metric, check requirements and add to calculation queue if met
            for metric_name in valid_metrics:
                # Skip metrics that require inputs we don't have
                if metric_name in ["bert_score", "answer_correctness"] and (not answer or not ground_truth):
                    continue
                    
                if metric_name in ["faithfulness", "answer_relevancy", "context_utilisation"] and (not question or not answer or not context):
                    continue
                
                if metric_name == "maliciousness" and (not question or not answer):
                    continue
                
                if metric_name == "context_relevance" and (not question or not context):
                    continue

                if metric_name == "context_entity_recall" and (not ground_truth or not context):
                    continue

                if metric_name == "answer_groundedness" and (not answer or not context):
                    continue
                
                if metric_name == "context_precision" and (not question or not ground_truth or not context):
                    continue

                if metric_name in ["noise_sensitivity", "context_recall"] and (not question or not answer or 
                                                                            not ground_truth or not context):
                    continue
                
                # Add metric to calculation queue
                metric_func = available_metrics[metric_name]
                
                if metric_name == "bert_score":
                    futures.append(self.executor.submit(metric_func, ground_truth, answer))
                elif metric_name == "answer_correctness":
                    futures.append(self.executor.submit(metric_func, answer, ground_truth))
                elif metric_name in ["faithfulness", "answer_relevancy", "context_utilisation"]:
                    futures.append(self.executor.submit(metric_func, question, answer, context))
                elif metric_name == "maliciousness":
                    futures.append(self.executor.submit(metric_func, question, answer))
                elif metric_name == "context_relevance":
                    futures.append(self.executor.submit(metric_func, question, context))
                elif metric_name == "answer_groundedness":
                    futures.append(self.executor.submit(metric_func, answer, context))
                elif metric_name == "context_entity_recall":
                    futures.append(self.executor.submit(metric_func, ground_truth, context))
                elif metric_name == "context_precision":
                    futures.append(self.executor.submit(metric_func, question, ground_truth, context))
                elif metric_name in ["noise_sensitivity", "context_recall"]:
                    futures.append(self.executor.submit(metric_func, question, answer, ground_truth, context))
                
                metric_names.append(metric_name)
            
            # Get results as they complete
            results = [(f.result(), name) for f, name in zip(futures, metric_names)]
            # Set metrics attributes
            for value, name in results:
                setattr(metrics, name, value)
                    
        except Exception as e:
            logger.error(f"Error calculating RAG metrics: {e}")
            print(f"Error calculating RAG metrics: {e}")
        
        return asdict(metrics)
    
    def process_dataframe(self):
        """
        Process the input dataframe by calculating all specified metrics for each row
        and return a new dataframe with the original data and added metrics.
        
        Returns:
            pd.DataFrame: Enhanced dataframe with all calculated metrics
        """
        result_df = self.df.copy()

        metrics_data = {}

        print(self.metrics_dict)
        for idx in result_df.index:
            metrics_for_row = {}
            
            try:
                # Get the row
                row = result_df.loc[idx]
                
                # Extract required fields
                question = row[self.question_col] if self.question_col in result_df.columns else None
                answer = row[self.answer_col] if self.answer_col in result_df.columns else None
                ground_truth = row[self.ground_truth_col] if self.ground_truth_col in result_df.columns else None
                
                # Get context if available
                context = None
                if self.context_col in result_df.columns:
                    context = row[self.context_col]
                    # Convert string representation to list if needed
                    if isinstance(context, str):
                        try:
                            context = ast.literal_eval(context)
                        except (ValueError, SyntaxError):
                            # If not a proper list representation, keep as is
                            context = [context]
                elif hasattr(self, 'context') and len(self.context) > idx:
                    context = self.context[idx]
            

                # Calculate RAG metrics if we have all required components
                if 'rag' in self.metrics_dict:

                    if question and answer and (ground_truth or context):
                        rag_metrics = self.calculate_rag_metrics(question, answer, ground_truth, context)
                        for key, value in rag_metrics.items():
                            if value is not None:
                                metrics_for_row[key] = value
                            else:
                                metrics_for_row[key] = 'Not Applicable'
                else:
                    print(f"RAG metrics list not available in given metrics to calculate: {self.metrics_dict}")

                if 'question' in self.metrics_dict:
                    if question:
                        question_metrics = self.get_question_metrics(question)
                        for key, value in question_metrics.items():
                            # metric_name = key if key.startswith('question_') else f"question_{key}"
                            metrics_for_row[key] = value
                else:
                    print("Question metrics not available in given metrics to calculate")
            
                # Calculate answer metrics
                if 'answer' in self.metrics_dict:
                    if question and answer:
                        available_metrics = self.metrics_dict["answer"]
                        if available_metrics:
                            answer_metrics = self.get_answer_metrics(question, answer)
                            for key, value in answer_metrics.items():
                                # metric_name = key if key.startswith('answer_') else f"answer_{key}"
                                metrics_for_row[key] = value
                else:
                    print("Answer metrics not available in given metrics to calculate")
                    
                if 'text' in self.metrics_dict:
                    if question:
                        question_text_metrics = self.get_metrics(question)
                        for key, value in question_text_metrics.items():
                            if value != []:
                                metrics_for_row[f"question_{key}"] = value
                            else:
                                metrics_for_row[f"question_{key}"] = 'No pii detected'

                    if answer:
                        answer_text_metrics = self.get_metrics(answer)                 
                        for key, value in answer_text_metrics.items():
                            if value != []:
                                metrics_for_row[f"answer_{key}"] = value
                            else:
                                metrics_for_row[f"answer_{key}"] = 'No pii detected'
                
                else:
                    print(f"Answer metrics not available in given metrics to calculate")
                
                # Store metrics for this row
                metrics_data[idx] = metrics_for_row
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {e}")
                print(f"Error processing row {idx}: {e}")
                # Continue with next row instead of failing the entire process
                continue
        
        # Add all metrics to the result dataframe at once

        for idx, row_metrics in metrics_data.items():
            for metric_name, metric_value in row_metrics.items():
                if isinstance(metric_value, list):
                    for i, val in enumerate(metric_value):
                        result_df.at[idx, metric_name] = str(metric_value)
                else:
                    result_df.at[idx, metric_name] = metric_value
        
        return result_df
