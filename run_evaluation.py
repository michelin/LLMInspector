import asyncio
import os
from typing import Dict, Any

from agent_eval.evaluation import TraceEvaluator
from ragas.messages import ToolCall as RToolCall
from deepeval.test_case import ToolCall as DToolCall
import config as config


async def main():
    """
    Main function to run evaluations on a trace.
    """

    # Set up the evaluator with the path to the trace file
    output_path = os.path.join(
        config.OUTPUT_DIR, 
        config.OUTPUT_FILENAME_TEMPLATE.format(trace_id=config.TRACE_ID)
    )
    evaluator = TraceEvaluator(output_path)
    
    # Load the trace
    trace_data = evaluator.load_trace()
    print(f"Loaded trace with {len(trace_data)} messages")
    
    # Set up Azure LLM
    evaluator.setup_azure_llm()
    print("Azure OpenAI LLM initialized successfully")
    
    
    # Example 1: Tool Call Accuracy
    # This example evaluates the accuracy of tool calls made by the agent, in this case this is
    # the ground truth that was considered for SQL agent based on open source chinook data.

    print("\nEvaluating Tool Call Accuracy (Example 1)...")
    reference_tool_calls1 = [
        RToolCall(name="transfer_to_sql_agent", args={}),
        RToolCall(name="check_relevance", args={"question": "Which country's customer spent the most?"}),
        RToolCall(name="convert_nl_to_sql", args={"question": "Which country's customer spent the most?"}),
        RToolCall(name="query_checker", args={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
        RToolCall(name="execute_sql", args={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
        RToolCall(name="transfer_back_to_supervisor", args={}),
    ]
    try:
        score = await evaluator.evaluate_tool_call_accuracy(reference_tool_calls1)
        print(f"Tool Call Accuracy Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate tool call accuracy: {e}")
    


    # Example 2: Tool Call Accuracy with alternative SQL
    print("\nEvaluating Tool Call Accuracy (Example 2)...")
    reference_tool_calls2 = [
        RToolCall(name="transfer_to_sql_agent", args={}),
        RToolCall(name="check_relevance", args={"question": "Which country's customer spent the most?"}),
        RToolCall(name="convert_nl_to_sql", args={"question": "Which country's customer spent the most?"}),
        RToolCall(name="query_checker", args={"sql_query": "SELECT c.Country AS country, SUM(i.Total) AS total_spent\nFROM Customer c\nJOIN Invoice i ON c.CustomerId = i.CustomerId\nGROUP BY c.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
        RToolCall(name="execute_sql", args={"sql_query": "SELECT c.Country AS country, SUM(i.Total) AS total_spent\nFROM Customer c\nJOIN Invoice i ON c.CustomerId = i.CustomerId\nGROUP BY c.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
        RToolCall(name="transfer_back_to_supervisor", args={}),
    ]
    try:
        score = await evaluator.evaluate_tool_call_accuracy(reference_tool_calls2)
        print(f"Tool Call Accuracy Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate tool call accuracy: {e}")
    



    # Topic Adherence
    print("\nEvaluating Topic Adherence (Single Topic)...")
    try:
        score = await evaluator.evaluate_topic_adherence(["Customer"])
        print(f"Topic Adherence Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate topic adherence: {e}")
    

    print("\nEvaluating Topic Adherence (Multiple Topics)...")
    try:
        score = await evaluator.evaluate_topic_adherence(["Customer", "Invoice"])
        print(f"Topic Adherence Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate topic adherence: {e}")
    


    # Goal Accuracy
    print("\nEvaluating Goal Accuracy (Reference 1)...")
    try:
        score = await evaluator.evaluate_goal_accuracy_with_reference("Customers from country spent the most")
        print(f"Goal Accuracy Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate goal accuracy: {e}")
    

    print("\nEvaluating Goal Accuracy (Reference 2)...")
    try:
        score = await evaluator.evaluate_goal_accuracy_with_reference("Customer spent the most in country")
        print(f"Goal Accuracy Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate goal accuracy: {e}")
    

    print("\nEvaluating Goal Accuracy (Without Reference)...")
    try:
        score = await evaluator.evaluate_goal_accuracy_without_reference()
        print(f"Goal Accuracy Score: {score}")
    except Exception as e:
        print(f"Failed to evaluate goal accuracy: {e}")
    
    
        
    # Tool Correctness
    # print("\nEvaluating Tool Correctness (Example 1)...")
    # expected_tools1 = [
    #     DToolCall(name="transfer_to_sql_agent", input_parameters={"input":""}),
    #     DToolCall(name="check_relevance", input_parameters={"question": "Which country's customer spent the most?"}),
    #     DToolCall(name="convert_nl_to_sql", input_parameters={"question": "Which country's customer spent the most?"}),
    #     DToolCall(name="query_checker", input_parameters={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
    #     DToolCall(name="execute_sql", input_parameters={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
    #     DToolCall(name="transfer_back_to_supervisor", input_parameters={"input":""}),
    # ]
    # try:
    #     result = evaluator.evaluate_tool_correctness(expected_tools1)
    #     print(f"Tool Correctness Score: {result['score']}")
    #     print(f"Reason: {result['reason']}")
    # except Exception as e:
    #     print(f"Failed to evaluate tool correctness: {e}")
    


    # # Example with fewer tools
    # print("\nEvaluating Tool Correctness (Example 2)...")
    # expected_tools2 = [
    #     DToolCall(name="transfer_to_sql_agent", input_parameters={"input":""}),
    #     DToolCall(name="check_relevance", input_parameters={"question": "Which country's customer spent the most?"}),
    #     DToolCall(name="convert_nl_to_sql", input_parameters={"question": "Which country's customer spent the most?"}),
    #     DToolCall(name="query_checker", input_parameters={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
    # ]
    # try:
    #     result = evaluator.evaluate_tool_correctness(expected_tools2)
    #     print(f"Tool Correctness Score: {result['score']}")
    #     print(f"Reason: {result['reason']}")
    # except Exception as e:
    #     print(f"Failed to evaluate tool correctness: {e}")
    


    # # Example with different order
    # print("\nEvaluating Tool Correctness (Example 3)...")
    # expected_tools3 = [
    #     DToolCall(name="transfer_to_sql_agent", input_parameters={"input":""}),
    #     DToolCall(name="check_relevance", input_parameters={"question": "Which country's customer spent the most?"}),
    #     DToolCall(name="convert_nl_to_sql", input_parameters={"question": "Which country's customer spent the most?"}),
    #     DToolCall(name="execute_sql", input_parameters={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
    #     DToolCall(name="query_checker", input_parameters={"sql_query": "SELECT Customer.Country AS country, SUM(Invoice.Total) AS total_spent\nFROM Customer\nJOIN Invoice ON Customer.CustomerId = Invoice.CustomerId\nGROUP BY Customer.Country\nORDER BY total_spent DESC\nLIMIT 1;"}),
    #     DToolCall(name="transfer_back_to_supervisor", input_parameters={"input":""}),
    # ]
    # try:
    #     result = evaluator.evaluate_tool_correctness(expected_tools3)
    #     print(f"Tool Correctness Score: {result['score']}")
    #     print(f"Reason: {result['reason']}")
    # except Exception as e:
    #     print(f"Failed to evaluate tool correctness: {e}")
    

    
    # Task Completion
    print("\nEvaluating Task Completion...")
    try:
        result = evaluator.evaluate_task_completion()
        print(f"Task Completion Score: {result['score']}")
        print(f"Reason: {result['reason']}")
    except Exception as e:
        print(f"Failed to evaluate task completion: {e}")

if __name__ == "__main__":
    asyncio.run(main())