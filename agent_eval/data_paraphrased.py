import os
import csv
import json
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI

# Load API keys from environment variables
load_dotenv()


class ParaphrasedDatasetGenerator:
    def __init__(self):
        """Initialize the dataset generator with API credentials and settings."""
        self.api_key = os.environ['api_key']
        self.azure_endpoint = os.environ['azure_endpoint']
        self.api_version = os.environ['api_version']

        self.llm = AzureChatOpenAI(
            api_version = self.api_version,
            azure_endpoint = self.azure_endpoint,
            azure_deployment = "gpt-4o",
            model_name = "gpt-4o",
            api_key = self.api_key,
            temperature = 1.0,
        )
        
 
    def generate_paraphrases(self, original_question: str, sql_query: str, 
                         num_paraphrases: int = 5) -> List[Dict[str, str]]:
        """Generate paraphrased versions of the original question using LangChain and an LLM."""
        
        # Create a prompt for the LLM
        combined_prompt = f"""You are a helpful assistant that generates paraphrased questions for text-to-SQL evaluation.
        Your task is to create variations of natural language questions that maintain the same semantic meaning but vary in structure, vocabulary, and phrasing.

        You need to create paraphrased variations of natural language questions for a text-to-SQL system evaluation dataset. 
        The paraphrased questions should maintain the same semantic meaning and intent but vary in structure, vocabulary, and phrasing.

        Original question: "{original_question}"

        Corresponding SQL query:
        ```sql
        {sql_query}
        ```

        Please generate {num_paraphrases} paraphrased versions of the original question that would map to the exact same SQL query. The paraphrases should:
        1. Vary in sentence structure (questions, commands, statements)
        2. Use different vocabulary and synonyms
        3. Range in complexity and formality levels
        4. Include both verbose and concise phrasings
        5. Maintain the same semantic meaning and intent

        For each paraphrase, add a brief note about what linguistic variation technique was applied.

        Format your response as a JSON array of objects with 'paraphrase' and 'technique' fields.
        """.format(original_question=original_question, sql_query=sql_query, num_paraphrases=num_paraphrases)
        
        # Create the chat prompt template - now with a single message
        prompt = ChatPromptTemplate.from_messages([
            ("system", combined_prompt),
        ])
        
        # Create the LLM chain
        paraphrase_chain = prompt | self.llm
        
        try:
            # Execute the chain
            result = paraphrase_chain.invoke({})
            content = result.content
            
            # Find JSON content in the response - handles cases where the model adds explanation text
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_content = content[json_start:json_end]
                paraphrases = json.loads(json_content)
            else:
                # Fallback: try to parse the whole response as JSON
                paraphrases = json.loads(content)
            
            return paraphrases
        
        except Exception as e:
            print(f"Error generating paraphrases: {e}")
            print(f"Response content: {content if 'content' in locals() else 'Not available'}")
            return []
    
    def process_query_batch(self, queries: List[Dict[str, str]]):
        """Process a batch of queries and collect their paraphrases in a DataFrame."""
        # Initialize a list to store all rows for the DataFrame
        all_rows = []
        
        for query in queries:
            original_question = query['Question']
            sql_query = query['SQL_Query']
            
            print(f"Generating paraphrases for: {original_question}")
            
            # Get paraphrased versions
            paraphrases = self.generate_paraphrases(
                original_question, 
                sql_query
            )
            
            # Add the original question
            all_rows.append({
                'original_question': original_question,
                'paraphrased_question': original_question,
                'sql_query': sql_query,
                'paraphrase_type': "original"
            })
            
            # Add all paraphrased versions
            for p in paraphrases:
                all_rows.append({
                    'original_question': original_question,
                    'paraphrased_question': p.get("paraphrase", ""),
                    'sql_query': sql_query,
                    'paraphrase_type': p.get("technique", "")
                })
                      
        # Create a DataFrame from all the rows
        return pd.DataFrame(all_rows)


    def process_from_input_file(self, df, output_file=None):
        """Process queries from a dataframe and return a DataFrame with all paraphrases."""

        # Load queries from input file (assuming CSV with Question,SQL_Query columns)
        queries = []
        
        for _, row in df.iterrows():
            queries.append({
                'Question': row['Question'],
                'SQL_Query': row['SQL_Query']
            })
        
        # Process all queries and get the DataFrame
        results_df = self.process_query_batch(queries)
        
        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Paraphrased dataset generated and saved to {output_file}")
        
        return results_df
