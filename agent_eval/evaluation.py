import os
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai.chat_models import AzureChatOpenAI

# RAGAS imports
from ragas.integrations.langgraph import convert_to_ragas_messages
from ragas.metrics import (
    ToolCallAccuracy,
    TopicAdherenceScore,
    AgentGoalAccuracyWithReference,
    AgentGoalAccuracyWithoutReference
)
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import ToolCall as RagasToolCall
from ragas.llms import LangchainLLMWrapper

# DeepEval imports
from deepeval.test_case import LLMTestCase, ToolCallParams, ToolCall as DeepEvalToolCall
from deepeval.metrics import ToolCorrectnessMetric, TaskCompletionMetric
from deepeval.models import DeepEvalBaseLLM

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

class CustomAzureChatOpenAI(DeepEvalBaseLLM):
    """
    Custom Azure OpenAI adapter for DeepEval.
    """
    def __init__(self, model):
        self.model = model
 
    def load_model(self):
        return self.model
 
    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content
 
    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content
 
    def get_model_name(self):
        return "Custom Azure OpenAI Model"


class TraceEvaluator:
    """
    A class to evaluate traces from Phoenix using RAGAS and DeepEval metrics.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize the TraceEvaluator with trace output path.
        
        Args:
            output_path: Path to the pickled trace file
        """
        self.output_path = output_path
        self.trace_data = None
        self.ragas_trace = None
        self.deepeval_trace = None
        self.llm = None
        
    def load_trace(self) -> List:
        """
        Load the trace data from the pickle file.
        
        Returns:
            List of trace messages
        """
        try:
            with open(self.output_path, 'rb') as f:
                self.trace_data = pickle.load(f)
            print(f"File successfully loaded from {self.output_path}")
            return self.trace_data
        except FileNotFoundError as fnf_error:
            print(f"FileNotFoundError: {fnf_error}")
            raise
        except PermissionError as perm_error:
            print(f"PermissionError: {perm_error}")
            raise
        except pickle.UnpicklingError as unpickling_error:
            print(f"UnpicklingError: {unpickling_error}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise
    
    def setup_azure_llm(self) -> AzureChatOpenAI:
        """
        Set up the Azure OpenAI model for evaluations.
        
        Returns:
            Configured Azure OpenAI model
        """
        # Load environment variables
        load_dotenv()
        
        # Get API credentials from environment
        api_key = os.environ.get('api_key')
        azure_endpoint = os.environ.get('azure_endpoint')
        api_version = os.environ.get('api_version')
        
        if not all([api_key, azure_endpoint, api_version]):
            raise ValueError("Missing required environment variables for Azure OpenAI")
        
        # Initialize the Azure OpenAI model
        self.llm = AzureChatOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment="gpt-4o",
            model_name="gpt-4o",
            api_key=api_key,
            temperature=1.0,
        )
        
        return self.llm
    
    def convert_to_ragas_format(self) -> Any:
        """
        Convert trace data to RAGAS format.
        
        Returns:
            RAGAS-formatted trace data
        """
        if self.trace_data is None:
            self.load_trace()
            
        self.ragas_trace = convert_to_ragas_messages(self.trace_data)
        return self.ragas_trace
    
    def convert_to_deepeval_format(self) -> Dict:
        """
        Convert trace data to DeepEval format.
        
        Returns:
            DeepEval-formatted trace data
        """
        if self.trace_data is None:
            self.load_trace()
            
        trace = {}
        
        # Extract the first message (HumanMessage) as the input
        input_message = next(step for step in self.trace_data if isinstance(step, HumanMessage))
        input_text = input_message.content
        trace['input'] = input_text
        
        # Extract the last message (AIMessage) as the actual output
        output_message = next(step for step in reversed(self.trace_data) if isinstance(step, AIMessage))
        actual_output = output_message.content
        trace['actual_output'] = actual_output
        
        # Loop through output and extract tools called
        tool_call_map = {}
        for step in self.trace_data:
            if isinstance(step, AIMessage) and 'tool_calls' in step.additional_kwargs:
                for tool_call in step.additional_kwargs['tool_calls']:
                    tool_call_map[tool_call['id']] = {
                        'name': tool_call['function']['name'],
                        'args': tool_call['function']['arguments']
                    }

        tools_called = []
        for step in self.trace_data:
            if isinstance(step, ToolMessage):
                tool_call_id = step.tool_call_id
                tool_data = tool_call_map.get(tool_call_id)
                if tool_data:
                    args_dict = eval(tool_data['args'])

                    if args_dict == {}:
                        tools_called.append(DeepEvalToolCall(
                            name=tool_data['name'], 
                            input_parameters={"input": ""}, 
                            output=step.content
                        ))
                    else:
                        tools_called.append(DeepEvalToolCall(
                            name=tool_data['name'], 
                            input_parameters=args_dict, 
                            output=step.content
                        ))
                    
        trace['tools_called'] = tools_called
        
        self.deepeval_trace = trace
        return self.deepeval_trace
    
    async def evaluate_tool_call_accuracy(self, reference_tool_calls: List[RagasToolCall]) -> float:
        """
        Evaluate tool call accuracy using RAGAS.
        
        Args:
            reference_tool_calls: List of reference tool calls for comparison
            
        Returns:
            Tool call accuracy score
        """
        if self.ragas_trace is None:
            self.convert_to_ragas_format()
        
        sample = MultiTurnSample(
            user_input=self.ragas_trace,
            reference_tool_calls=reference_tool_calls
        )
        
        scorer = ToolCallAccuracy()
        score = await scorer.multi_turn_ascore(sample)
        return score
    
    async def evaluate_topic_adherence(self, reference_topics: List[str]) -> float:
        """
        Evaluate topic adherence using RAGAS.
        
        Args:
            reference_topics: List of reference topics for comparison
            
        Returns:
            Topic adherence score
        """
        if self.ragas_trace is None:
            self.convert_to_ragas_format()
            
        if self.llm is None:
            self.setup_azure_llm()
            
        sample = MultiTurnSample(
            user_input=self.ragas_trace,
            reference_topics=reference_topics
        )
        
        scorer = TopicAdherenceScore(llm=LangchainLLMWrapper(self.llm), mode="precision")
        score = await scorer.multi_turn_ascore(sample)
        return score
    
    async def evaluate_goal_accuracy_with_reference(self, reference: str) -> float:
        """
        Evaluate goal accuracy with reference using RAGAS.
        
        Args:
            reference: Reference goal for comparison
            
        Returns:
            Goal accuracy score
        """
        if self.ragas_trace is None:
            self.convert_to_ragas_format()
            
        if self.llm is None:
            self.setup_azure_llm()
            
        sample = MultiTurnSample(
            user_input=self.ragas_trace,
            reference=reference
        )
        
        scorer = AgentGoalAccuracyWithReference(llm=LangchainLLMWrapper(self.llm))
        score = await scorer.multi_turn_ascore(sample)
        return score
    
    async def evaluate_goal_accuracy_without_reference(self) -> float:
        """
        Evaluate goal accuracy without reference using RAGAS.
        
        Returns:
            Goal accuracy score
        """
        if self.ragas_trace is None:
            self.convert_to_ragas_format()
            
        if self.llm is None:
            self.setup_azure_llm()
            
        sample = MultiTurnSample(
            user_input=self.ragas_trace
        )
        
        scorer = AgentGoalAccuracyWithoutReference(llm=LangchainLLMWrapper(self.llm))
        score = await scorer.multi_turn_ascore(sample)
        return score
    
    def evaluate_tool_correctness(self, expected_tools: List[DeepEvalToolCall]) -> Dict:
        """
        Evaluate tool correctness using DeepEval.
        
        Args:
            expected_tools: List of expected tool calls for comparison
            
        Returns:
            Dictionary with score and reason
        """
        if self.deepeval_trace is None:
            self.convert_to_deepeval_format()
            
        test_case = LLMTestCase(
            input=self.deepeval_trace['input'],
            actual_output=self.deepeval_trace['actual_output'],
            tools_called=self.deepeval_trace['tools_called'],
            expected_tools=expected_tools
        )
        
        metric = ToolCorrectnessMetric(
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
            should_consider_ordering=True
        )
        
        metric.measure(test_case)
        
        return {
            "score": metric.score,
            "reason": metric.reason
        }
    
    def evaluate_task_completion(self) -> Dict:
        """
        Evaluate task completion using DeepEval.
        
        Returns:
            Dictionary with score and reason
        """
        if self.deepeval_trace is None:
            self.convert_to_deepeval_format()
            
        if self.llm is None:
            self.setup_azure_llm()
            
        custom_model = CustomAzureChatOpenAI(self.llm)
        
        test_case = LLMTestCase(
            input=self.deepeval_trace['input'],
            actual_output=self.deepeval_trace['actual_output'],
            tools_called=self.deepeval_trace['tools_called'],
        )
        
        metric = TaskCompletionMetric(
            threshold=0.7,
            model=custom_model,
            include_reason=True
        )
        
        metric.measure(test_case)
        
        return {
            "score": metric.score,
            "reason": metric.reason
        }