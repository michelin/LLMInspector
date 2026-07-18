from typing import Dict, Any, List, Optional, Set, Callable, Awaitable, Tuple
import logging
import asyncio
import time
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("simple_scorer")
# logger.setLevel(logging.INFO)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)



def get_metric_definitions():
    """Define all possible metrics and their required inputs."""
    # RAG metrics and their required inputs
    rag_metrics = {
        "bert_score": {"answer", "ground_truth"},
        "faithfulness": {"question", "answer", "contexts"},
        "answer_correctness": {"question", "answer", "ground_truth"},
        "answer_relevancy": {"question", "answer"},
        "conciseness": {"question", "answer"},
        "context_relevance": {"question", "contexts"},
        "context_utilisation": {"question", "answer", "contexts"},
        "context_entity_recall": {"ground_truth", "contexts"},
        "context_precision": {"question", "ground_truth", "contexts"},
        "context_recall": {"question", "answer", "ground_truth", "contexts"},
    }
    
    # General LLM metrics - Question Only
    question_metrics = {
        "question_emotion": {"question"},
        "question_sentiment": {"question"},
        "question_language": {"question"},
        "question_pii_detected": {"question"},
        "question_flesch_kincaid_grade": {"question"},
        "question_tokens": {"question"},
        "question_code_detected": {"question"},
        "question_jailbreak_risk": {"question"},
        "question_content_moderation": {"question"},
    }
    
    # General LLM metrics - Answer Only
    answer_metrics = {
        "answer_emotion": {"answer"},
        "answer_sentiment": {"answer"},
        "answer_language": {"answer"},
        "answer_pii_detected": {"answer"},
        "answer_flesch_kincaid_grade": {"answer"},
        "answer_tokens": {"answer"},
        "answer_code_detected": {"answer"},
        "answer_no_refusal": {"question", "answer"},
        "answer_jailbreak_risk": {"answer"},
        "answer_content_moderation": {"answer"},
    }
    
    # Metrics requiring other combinations
    other_metrics = {
        "answer_hallucination_risk": {"contexts", "answer"},
        "policy_check": {"question", "answer", "policy"},
    }

    # Combine all metrics
    all_metrics = {}
    all_metrics.update(rag_metrics)
    all_metrics.update(question_metrics)
    all_metrics.update(answer_metrics)
    all_metrics.update(other_metrics)
    
    return all_metrics

def create_metric_functions(eval_metrics, question, answer, ground_truth, contexts, policy):
    """Create mapping of metric names to their async functions."""
    return {
        # RAG Metrics - these return JSON with score, reasoning, and key_findings
        "bert_score": lambda: eval_metrics.bertscore_async(answer, ground_truth),
        "faithfulness": lambda: eval_metrics.faithfulness_eval_async(question, answer, contexts),
        "answer_correctness": lambda: eval_metrics.answer_correctness_eval_async(question, answer, ground_truth),
        "answer_relevancy": lambda: eval_metrics.answer_relevancy_eval_async(question, answer),
        "conciseness": lambda: eval_metrics.conciseness_eval_async(question, answer),
        
        # Context metrics - these return simple numeric scores
        "context_relevance": lambda: eval_metrics.context_relevance_eval_async(question, contexts),
        "context_utilisation": lambda: eval_metrics.context_utilisation_eval_async(question, answer, contexts),
        "context_entity_recall": lambda: eval_metrics.context_entity_recall_eval_async(ground_truth, contexts),
        "context_precision": lambda: eval_metrics.context_precision_eval_async(question, ground_truth, contexts),
        "context_recall": lambda: eval_metrics.context_recall_eval_async(question, answer, ground_truth, contexts),

        # General LLM Metrics - simple string/list responses
        "question_emotion": lambda: eval_metrics.emotion_analysis_async(question),
        "answer_emotion": lambda: eval_metrics.emotion_analysis_async(answer),
        "question_sentiment": lambda: eval_metrics.sentiment_analysis_async(question),
        "answer_sentiment": lambda: eval_metrics.sentiment_analysis_async(answer),
        "question_language": lambda: eval_metrics.detect_language_async(question),
        "answer_language": lambda: eval_metrics.detect_language_async(answer),
        "question_pii_detected": lambda: eval_metrics.pii_detection_async(question),
        "answer_pii_detected": lambda: eval_metrics.pii_detection_async(answer),
        "question_flesch_kincaid_grade": lambda: eval_metrics.text_quality_async(question),
        "answer_flesch_kincaid_grade": lambda: eval_metrics.text_quality_async(answer),
        "question_tokens": lambda: eval_metrics.num_tokens_from_string_async(question),
        "answer_tokens": lambda: eval_metrics.num_tokens_from_string_async(answer),
        "answer_no_refusal": lambda: eval_metrics.answer_no_refusal_async(question, answer),
        
        # Code detect - returns JSON with code_detected and code_language
        "question_code_detected": lambda: eval_metrics.code_detect_async(question),
        "answer_code_detected": lambda: eval_metrics.code_detect_async(answer),

        # Guardrails - return simple numeric/boolean values
        "question_jailbreak_risk": lambda: eval_metrics.question_jailbreak_detect_async(question),
        "answer_jailbreak_risk": lambda: eval_metrics.answer_jailbreak_detect_async(answer),
        "answer_hallucination_risk": lambda: eval_metrics.hallucination_detect_async(contexts, answer),
        
        # Content moderation - returns JSON with multiple flags
        "question_content_moderation": lambda: eval_metrics.content_moderation_detect_async(question),
        "answer_content_moderation": lambda: eval_metrics.content_moderation_detect_async(answer),

        # Policy Check - returns JSON with is_policy_violated and policy_violation_reason
        "policy_check": lambda: eval_metrics.policy_eval_async(question, answer, policy)
    }

def check_input_availability(
    question: Optional[str], 
    answer: Optional[str], 
    ground_truth: Optional[str], 
    contexts: Optional[List[str]], 
    policy: Optional[str]
) -> Dict[str, bool]:
    """Check which inputs are available for metric calculation."""
    return {
        "question": question is not None and question.strip() != "",
        "answer": answer is not None and answer.strip() != "",
        "ground_truth": ground_truth is not None and ground_truth.strip() != "",
        "contexts": contexts is not None and len(contexts) > 0 and all(c.strip() != "" for c in contexts),
        "policy": policy is not None and policy.strip() != ""
    }

def filter_metrics_by_availability(
    metric_requirements: Dict[str, Set[str]],
    inputs_available: Dict[str, bool]
) -> List[str]:
    """Filter metrics based on input availability."""
    return [
        metric for metric, required_inputs in metric_requirements.items()
        if all(inputs_available.get(req, False) for req in required_inputs)
    ]

async def run_async_metric(
    metric_name: str, 
    metric_func: Callable[[], Awaitable[Any]]
) -> Tuple[str, Any]:
    """Run a single metric calculation asynchronously."""
    start_time = time.time()
    try:
        result = await metric_func()
        elapsed_time = time.time() - start_time
        # logger.warning(f"Metric '{metric_name}' calculated in {elapsed_time:.2f} seconds.")
        return metric_name, result
    except Exception as e:
        elapsed_time = time.time() - start_time
        # logger.error(f"Error calculating {metric_name} (took {elapsed_time:.2f}s): {str(e)}")
        return metric_name, f"Error: {str(e)}"


def process_json_metric_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process metrics that return JSON with detailed information.
    Extracts score, reasoning, and keeps key_findings as nested structure.
    """
    # Metrics that return JSON with score, reasoning, and key_findings
    json_metrics = [
        "faithfulness",
        "answer_correctness", 
        "answer_relevancy",
        "conciseness"
    ]
    
    for metric in json_metrics:
        if metric in results and results[metric] is not None:
            metric_data = results[metric]
            
            # Check if it's actually a dict (JSON response)
            if isinstance(metric_data, dict):
                # Extract main score
                score_key = metric
                if score_key in metric_data:
                    results[metric] = metric_data[score_key]
                
                # Extract reasoning
                reasoning_key = f"{metric}_reasoning"
                if reasoning_key in metric_data:
                    results[reasoning_key] = metric_data[reasoning_key]
                else:
                    results[reasoning_key] = None
    
    return results


def process_code_detect_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Process code detection results into separate columns."""
    for prefix in ["question", "answer"]:
        key = f"{prefix}_code_detected"
        try:
            if key in results and results[key] is not None:
                if isinstance(results[key], dict):
                    # Snapshot the dict first — key and f"{prefix}_code_detected" are the
                    # same string, so writing code_detected would overwrite the dict before
                    # the code_language read, causing "'bool' has no attribute 'get'".
                    code_info = results[key]
                    results[f"{prefix}_code_detected"] = code_info.get("code_detected", None)
                    results[f"{prefix}_code_language"] = code_info.get("code_language", None)
                else:
                    # If it's not a dict (e.g., boolean or other type), keep as-is and add language as None
                    results[f"{prefix}_code_detected"] = results[key]
                    results[f"{prefix}_code_language"] = None
            else:
                results[f"{prefix}_code_detected"] = None
                results[f"{prefix}_code_language"] = None
        except Exception as e:
            logger.error(f"Error processing {key}: {str(e)}")
            results[f"{prefix}_code_detected"] = None
            results[f"{prefix}_code_language"] = None
    
    return results


def process_content_moderation(results: Dict[str, Any]) -> Dict[str, Any]:
    """Process content moderation results into individual metrics."""
    for prefix in ["question", "answer"]:
        key = f"{prefix}_content_moderation"
        if key not in results or results[key] is None:
            # Set all moderation fields to None
            for subkey in ["hate_speech", "fairness", "sexually_explicit_information", 
                          "violence", "self_harm", "dangerous_content", 
                          "harassment", "profanity", "toxicity_risk"]:
                results[f"{prefix}_{subkey}"] = None
        else:
            # Extract moderation data if it's a dict
            if isinstance(results[key], dict):
                for subkey, value in results[key].items():
                    results[f"{prefix}_{subkey}"] = value
            else:
                # If not a dict, set all to None
                for subkey in ["hate_speech", "fairness", "sexually_explicit_information", 
                              "violence", "self_harm", "dangerous_content", 
                              "harassment", "profanity", "toxicity_risk"]:
                    results[f"{prefix}_{subkey}"] = None
        
        # Remove the original content moderation key
        if key in results:
            del results[key]
    
    return results


def process_policy_check(results: Dict[str, Any]) -> Dict[str, Any]:
    """Process policy check results into individual metrics."""
    if 'policy_check' in results:
        if results['policy_check'] is not None and isinstance(results['policy_check'], dict):
            results['is_policy_violated'] = results['policy_check'].get('is_policy_violated', None)
            results['policy_violation_reason'] = results['policy_check'].get('policy_violation_reason', None)
        else:
            results['is_policy_violated'] = None
            results['policy_violation_reason'] = None
        
        del results['policy_check']
    
    return results


def calculate_total_tokens(results: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate total tokens if possible."""
    if results.get("answer_tokens") is not None and results.get("question_tokens") is not None:
        results["total_tokens"] = results["answer_tokens"] + results["question_tokens"]
    else:
        results["total_tokens"] = None
    
    return results

def calculate_overall_accuracy(results: Dict[str, Any]) -> Optional[float]:
    """Calculate overall accuracy based on key metrics."""
    try:
        
        if results.get("answer_correctness") is not None and results.get("faithfulness") is not None and results.get("answer_relevancy") is not None:
            overall_accuracy = (0.5 * results["answer_correctness"] +
                                0.3 * results["faithfulness"] +
                                0.2 * results["answer_relevancy"])
            logger.info(f"  Calculated overall_accuracy (3 metrics): {overall_accuracy}")
        elif results.get("answer_correctness") is not None and results.get("answer_relevancy") is not None:
            overall_accuracy = (0.75 * results["answer_correctness"] +
                                0.25 * results["answer_relevancy"])
            logger.info(f"  Calculated overall_accuracy (2 metrics): {overall_accuracy}")
        else:
            logger.info("  Insufficient metrics for overall_accuracy calculation")
            return None

        return round(overall_accuracy, 2)
    except Exception as e:
        logger.error(f"Error calculating overall accuracy: {str(e)}")
        logger.error(f"  Metric values - answer_correctness: {results.get('answer_correctness')}, faithfulness: {results.get('faithfulness')}, answer_relevancy: {results.get('answer_relevancy')}")
        return None

def reorder_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder results dict according to a predefined key order."""
    ordered_keys = [
        # Question-related simple metrics
        "question_emotion", "question_sentiment", "question_language", 
        "question_pii_detected", "question_flesch_kincaid_grade", "question_tokens",
        
        # Question code detection
        "question_code_detected", "question_code_language",
        
        # Answer-related simple metrics
        "answer_emotion", "answer_sentiment", "answer_language", 
        "answer_pii_detected", "answer_flesch_kincaid_grade", "answer_tokens",
        "answer_no_refusal",
        
        # Answer code detection
        "answer_code_detected", "answer_code_language",
        
        # RAG Metrics - BERTScore (simple)
        "bert_score",
        
        # Faithfulness with details (nested key_findings)
        "faithfulness", "faithfulness_reasoning",
        
        # Answer Correctness with details (nested key_findings)
        "answer_correctness", "answer_correctness_reasoning",
        
        # Answer Relevancy with details (nested key_findings)
        "answer_relevancy", "answer_relevancy_reasoning",
        
        # Conciseness with details (nested key_findings)
        "conciseness", "conciseness_reasoning",
        
        # Context metrics (simple scores)
        "context_relevance", "context_utilisation", "context_entity_recall", 
        "context_precision", "context_recall",
        
        # Guardrails
        "question_jailbreak_risk", "answer_jailbreak_risk", "answer_hallucination_risk",
        
        # Content Moderation
        "question_hate_speech", "question_fairness", "question_sexually_explicit_information", 
        "question_violence", "question_self_harm", "question_dangerous_content",
        "question_harassment", "question_profanity", "question_toxicity_risk",
        "answer_hate_speech", "answer_fairness", "answer_sexually_explicit_information",
        "answer_violence", "answer_self_harm", "answer_dangerous_content", 
        "answer_harassment", "answer_profanity", "answer_toxicity_risk",
        
        # Policy check
        "is_policy_violated", "policy_violation_reason",
        
        # Token totals
        "total_tokens"
    ]

    # Create a new ordered dictionary with the specified key order
    ordered_results = {}
    for key in ordered_keys:
        if key in results:
            ordered_results[key] = results[key]
    
    # Add any remaining keys that weren't in the ordered_keys list
    for key in results:
        if key not in ordered_results:
            ordered_results[key] = results[key]

    return ordered_results


async def process_all_metrics(
    eval_metrics,
    question: Optional[str] = None,
    answer: Optional[str] = None,
    ground_truth: Optional[str] = None,
    contexts: Optional[List[str]] = None,
    policy: Optional[str] = None,
    kpi_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Process the input text and calculate requested RAG and General LLM metrics concurrently.

    Args:
        question: The question/prompt text.
        answer: The answer/completion text.
        ground_truth: The ground truth or reference answer (for RAG).
        contexts: List of contexts passages used for RAG evaluation.
        policy: Optional policy text for policy evaluation.
        kpi_list: Optional list of specific metrics (KPIs) to calculate.
                  If None, calculates all metrics for which inputs are available.

    Returns:
        Dict containing the calculated metrics with expanded columns for detailed results.
    """
    logger.info("Calculating all requested metrics asynchronously...")
    start_time = time.time()
    
    # Get metric definitions
    metric_requirements = get_metric_definitions()
    
    # Initialize results dict with None values for all possible base metrics
    all_possible_metrics = set(metric_requirements.keys())
    results = {metric: None for metric in all_possible_metrics}
    
    # Check which inputs are available
    inputs_available = check_input_availability(question, answer, ground_truth, contexts, policy)
    if not any(inputs_available.values()):
        logger.warning("No valid inputs provided. Skipping all metrics.")
        return results if not kpi_list else {k: v for k, v in results.items() if k in kpi_list}
    
    # Create metric functions
    metric_functions = create_metric_functions(eval_metrics, question, answer, ground_truth, contexts, policy)
    
    # Determine which metrics to calculate
    if kpi_list:
        # Calculate only metrics in kpi_list
        metrics_to_calculate = [m for m in kpi_list if m in all_possible_metrics]
        unknown_kpis = [m for m in kpi_list if m not in all_possible_metrics]
        if unknown_kpis:
            logger.warning(f"Ignoring unknown KPIs: {unknown_kpis}")
        
        # Ensure dependencies for answer_correctness (weighted average) are calculated
        if "answer_correctness" in metrics_to_calculate:
            for dep in ["faithfulness", "answer_relevancy"]:
                if dep not in metrics_to_calculate:
                    metrics_to_calculate.append(dep)
    else:
        # Calculate all metrics for which we have inputs
        metrics_to_calculate = list(all_possible_metrics)
    
    # Filter metrics based on available inputs
    available_metrics = filter_metrics_by_availability(
        {m: metric_requirements[m] for m in metrics_to_calculate},
        inputs_available
    )
    
    # Create tasks for metrics with available inputs
    tasks = []
    for metric in available_metrics:
        if metric in metric_functions:
            tasks.append(run_async_metric(metric, metric_functions[metric]))
        else:
            logger.warning(f"Metric '{metric}' defined but no function mapping found. Skipping.")
            # results[metric] = "Error: Function not found"
    
    # Execute tasks concurrently
    if tasks:
        logger.info(f"Running {len(tasks)} metrics concurrently...")
        gathered_results = await asyncio.gather(*tasks)
        for metric_name, metric_value in gathered_results:
            results[metric_name] = metric_value
    else:
        logger.info("No metric tasks to run.")
    
    # Post-process results
    results = process_json_metric_results(results)  # Process JSON metrics first
    results = process_code_detect_results(results)  # Process code detection
    results = process_content_moderation(results)   # Process content moderation
    results = process_policy_check(results)         # Process policy check
    results = calculate_total_tokens(results)       # Calculate total tokens
    
    # Calculate overall accuracy and assign to answer_correctness
    overall_acc = calculate_overall_accuracy(results)
    if overall_acc is not None:
        results["answer_correctness"] = overall_acc

    # Filter out metrics that were calculated only as dependencies
    if kpi_list:
        potential_deps = ["faithfulness", "answer_relevancy"]
        for metric in potential_deps:
            if metric not in kpi_list:
                # Remove the metric score
                if metric in results:
                    results[metric] = None
                # Remove related details
                for suffix in ["_reasoning", "_key_findings"]:
                    key = f"{metric}{suffix}"
                    if key in results:
                        results[key] = None

    total_time = time.time() - start_time
    logger.warning(f"Overall metrics calculation finished in {total_time:.2f} seconds.")

    # Return ordered results
    return reorder_results(results)


async def process_batch_metrics(
    eval_metrics,
    df: pd.DataFrame,
    batch_size: int = 5,
    kpi_list: Optional[List[str]] = None,
    question_col: str = "question",
    answer_col: str = "answer",
    ground_truth_col: str = "ground_truth",
    contexts_col: str = "contexts",
    policy_col: str = "policy",
) -> pd.DataFrame:
    """
    Process metrics for a dataframe in parallel batches with progress tracking.
    
    Args:
        eval_metrics: The metrics evaluation object
        df: DataFrame containing questions, answers, and related data
        batch_size: Number of items to process in parallel
        kpi_list: Optional list of specific metrics to calculate
        question_col: Column name for questions
        answer_col: Column name for answers
        ground_truth_col: Column name for ground truth references
        contexts_col: Column name for contexts (expected to contain lists of strings)
        policy_col: Column name for policy text
    
    Returns:
        DataFrame with original data and added metrics columns (including expanded detail columns)
    """
    logger.info(f"Starting batch processing of {len(df)} rows with batch size {batch_size}")
    start_time = time.time()
    
    # Create a copy of the dataframe to store results
    result_df = df.copy()
    
    # Initialize a list to store all results
    all_results = []
    
    # Calculate total number of batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    # Create main progress bar for overall progress
    main_pbar = tqdm(total=len(df), desc="Overall Progress", position=0)
    
    # Process the dataframe in batches
    for i in range(0, len(df), batch_size):
        batch_start_time = time.time()
        batch = df.iloc[i:i+batch_size]
        batch_size_actual = len(batch)
        batch_num = i//batch_size + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} (rows {i}-{min(i+batch_size-1, len(df)-1)})")
        
        # Create tasks for each row in the batch
        tasks = []
        for _, row in batch.iterrows():
            # Extract data from the row, handling missing columns gracefully
            question = row.get(question_col, None) if question_col in df.columns else None
            answer = row.get(answer_col, None) if answer_col in df.columns else None
            ground_truth = row.get(ground_truth_col, None) if ground_truth_col in df.columns else None
            contexts = row.get(contexts_col, None) if contexts_col in df.columns else None
            policy = row.get(policy_col, None) if policy_col in df.columns else None
            
            # Create a task for this row
            task = process_all_metrics(
                eval_metrics=eval_metrics,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                contexts=contexts,
                policy=policy,
                kpi_list=kpi_list
            )
            tasks.append(task)
        
        # Create a secondary progress bar for this batch with description showing batch number
        batch_desc = f"Batch {batch_num}/{total_batches}"
        with tqdm(total=batch_size_actual, desc=batch_desc, position=1, leave=False) as batch_pbar:
            # Custom callback to update progress for each completed task
            completed_tasks = 0
            
            async def gather_with_progress():
                nonlocal completed_tasks
                results = []
                for future in asyncio.as_completed(tasks):
                    result = await future
                    results.append(result)
                    completed_tasks += 1
                    batch_pbar.update(1)
                    main_pbar.update(1)
                return results
            
            # Execute all tasks in this batch concurrently with progress updates
            batch_results = await gather_with_progress()
            all_results.extend(batch_results)
        
        batch_end_time = time.time()
        logger.info(f"Batch {batch_num} completed in {batch_end_time - batch_start_time:.2f} seconds")
    
    # Close the main progress bar
    main_pbar.close()
    
    # Convert results to a DataFrame
    metrics_df = pd.DataFrame(all_results)
    
    # Join the metrics back to the original DataFrame
    result_df = pd.concat([result_df.reset_index(drop=True), metrics_df], axis=1)
    
    total_time = time.time() - start_time
    logger.info(f"Batch processing completed in {total_time:.2f} seconds. Processed {len(df)} rows.")
    
    return result_df


# Wrapper function for non-async code to call the async function
def process_dataframe_metrics(
    eval_metrics,
    df: pd.DataFrame,
    batch_size: int = 5,
    kpi_list: Optional[List[str]] = None,
    question_col: str = "question",
    answer_col: str = "answer",
    ground_truth_col: str = "ground_truth",
    contexts_col: str = "contexts", 
    policy_col: str = "policy",
) -> pd.DataFrame:
    """
    Non-async wrapper for batch processing metrics with progress tracking.
    
    Args: Same as process_batch_metrics
    
    Returns:
        DataFrame with original data and added metrics columns (including expanded detail columns)
    """
    return asyncio.run(process_batch_metrics(
        eval_metrics=eval_metrics,
        df=df,
        batch_size=batch_size,
        kpi_list=kpi_list,
        question_col=question_col,
        answer_col=answer_col,
        ground_truth_col=ground_truth_col,
        contexts_col=contexts_col,
        policy_col=policy_col
    ))