import os
import re
import ast
import json
import pickle
from typing import List, Dict, Any, Optional
import phoenix as px
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class TraceReader:
    """
    A class to read and process traces from Phoenix.
    """
    
    def __init__(self, project_name: str, span_name: str, trace_id: str, output_dir: str, output_filename_template: str):
        """
        Initialize the TraceReader with configuration parameters.
        
        Args:
            project_name: Name of the Phoenix project
            span_name: Name of the span to retrieve
            trace_id: ID of the trace to process
            output_dir: Directory to save the output
            output_filename_template: Template for the output filename
        """
        self.project_name = project_name
        self.span_name = span_name
        self.trace_id = trace_id
        self.output_dir = output_dir
        self.output_filename_template = output_filename_template
        self.client = px.Client()
        
    def fetch_spans_dataframe(self) -> Any:
        """
        Retrieve spans dataframe from Phoenix.
        
        Returns:
            Pandas DataFrame containing spans
        """
        return self.client.get_spans_dataframe(
            project_name=self.project_name, 
            filter_condition=f"name == '{self.span_name}'"
        )
    
    def extract_span_output(self, span: Any) -> str:
        """
        Extract output from the span for the specified trace ID.
        
        Args:
            span: The spans dataframe
            
        Returns:
            Span output as a string
        """
        return span.loc[span['context.trace_id'] == self.trace_id, 'attributes.output.value'].iloc[0]
    
    @staticmethod
    def extract_messages_from_output(span_output: str) -> List[str]:
        """
        Extract messages from the span output.
        
        Args:
            span_output: The raw span output
            
        Returns:
            List of message strings
        """
        data = json.loads(span_output)
        return data["messages"]
    
    @staticmethod
    def replace_space_after_symbols(text: str) -> str:
        """
        Replace spaces after certain symbols with commas for better parsing.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        return re.sub(r"([\'\}\"\]]) ", r"\1, ", text)
    
    @staticmethod
    def split_key_value_pairs(s: str) -> List[str]:
        """
        Split a string into key-value pairs.
        
        Args:
            s: String to split
            
        Returns:
            List of key-value pair strings
        """
        pairs = []
        current = ''
        depth = 0
        in_string = False
        string_char = ''

        for i, char in enumerate(s):
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif string_char == char and (i == 0 or s[i - 1] != '\\'):
                    in_string = False
            elif not in_string:
                if char in '([{':
                    depth += 1
                elif char in ')]}':
                    depth -= 1
                elif char == ',' and depth == 0:
                    pairs.append(current.strip())
                    current = ''
                    continue
            current += char
        if current:
            pairs.append(current.strip())
        return pairs
    
    @staticmethod
    def safe_eval(value: str) -> Any:
        """
        Safely evaluate a string as a Python expression.
        
        Args:
            value: String to evaluate
            
        Returns:
            Evaluated value or original string if evaluation fails
        """
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
    
    @classmethod
    def string_to_dict(cls, input_str: str) -> Dict:
        """
        Convert a string representation of a dictionary to a Python dictionary.
        
        Args:
            input_str: String to convert
            
        Returns:
            Converted dictionary
        """
        result = {}
        if input_str:
            pairs = cls.split_key_value_pairs(input_str)
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    parsed_value = cls.safe_eval(value)
                    if isinstance(parsed_value, str) and parsed_value.startswith('{') and parsed_value.endswith('}'):
                        parsed_value = cls.string_to_dict(parsed_value[1:-1])
                    result[key] = parsed_value
        return result
    
    def process_messages(self, messages: List[str]) -> List[Dict]:
        """
        Process a list of message strings into a list of dictionaries.
        
        Args:
            messages: List of message strings
            
        Returns:
            List of message dictionaries
        """
        # Replace spaces after special characters
        formatted_messages = [self.replace_space_after_symbols(message) for message in messages]
        
        # Convert strings to dictionaries
        return [self.string_to_dict(message) for message in formatted_messages]
    
    def convert_to_langchain_messages(self, message_dicts: List[Dict]) -> List:
        """
        Convert message dictionaries to LangChain message objects.
        
        Args:
            message_dicts: List of message dictionaries
            
        Returns:
            List of LangChain message objects
        """
        output = []
        
        # First message is always a HumanMessage
        output.append(HumanMessage(**message_dicts[0]))
        
        # Process the rest of the messages
        for message in message_dicts[1:]:
            if "tool_call_id" in message:
                output.append(ToolMessage(**message))
            else:
                if not message["additional_kwargs"]:
                    message["additional_kwargs"] = {'tool_calls': [{'id': message["tool_calls"][0]["id"], 'function': {'arguments': f'{message["tool_calls"][0]["args"]}', 'name': message["tool_calls"][0]["name"]}, 'type': 'function'}], 'refusal': None}
                output.append(AIMessage(**message))
                
        return output
    
    def save_to_pickle(self, langchain_messages: List) -> str:
        """
        Save LangChain messages to a pickle file.
        
        Args:
            langchain_messages: List of LangChain message objects
            
        Returns:
            Path to the saved file
        """
        output_path = os.path.join(
            self.output_dir, 
            self.output_filename_template.format(trace_id=self.trace_id)
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Save the file
        with open(output_path, 'wb') as f:
            pickle.dump(langchain_messages, f)
            
        return output_path
    
    def process_trace(self) -> str:
        """
        Process the entire trace from Phoenix to LangChain format.
        
        Returns:
            Path to the saved output file
        """
        try:
            # Get spans dataframe
            spans = self.fetch_spans_dataframe()
            
            # Extract span output
            span_output = self.extract_span_output(spans)
            
            # Extract messages from span output
            raw_messages = self.extract_messages_from_output(span_output)
            
            # Process messages
            message_dicts = self.process_messages(raw_messages)
            
            # Convert to LangChain messages
            langchain_messages = self.convert_to_langchain_messages(message_dicts)
            
            # Save to file
            output_path = self.save_to_pickle(langchain_messages)
            
            print(f"File successfully saved to {output_path}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            raise