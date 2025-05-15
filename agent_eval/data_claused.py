import pandas as pd
import re
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI

# Load API keys from environment variables
load_dotenv()


class ClauseDatasetGenerator:
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
        
        self.agg_functions = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        
    
    # Function to identify and replace aggregation functions
    def replace_agg_functions(self, sql_query):
        found_funcs = set(func for func in self.agg_functions if re.search(rf'\b{func}\b', sql_query, re.IGNORECASE))
        variants = []

        for original_func in found_funcs:
            for new_func in self.agg_functions:
                if new_func != original_func:
                    new_query = re.sub(rf'\b{original_func}\b', new_func, sql_query, flags=re.IGNORECASE)
                    variants.append((original_func, new_func, new_query))

        return variants

    # Function to get natural language from LLM
    def generate_question_from_sql(self, sql_query):
        prompt = f"Translate the following SQL query into a natural language question:\nSQL: {sql_query}\nQuestion:"
        prompt = ChatPromptTemplate.from_messages([
                ("system", prompt),
            ])
        paraphrase_chain = prompt | self.llm
        response = paraphrase_chain.invoke({})
        return response.content.strip()

    def process_data(self, df, output_file=None):
        """Process queries from a dataframe and return a DataFrame with all paraphrases."""
        
        output_rows = []

        for idx, row in df.iterrows():
            original_query = row['SQL_Query']
            variants = self. replace_agg_functions(original_query)

            for original_func, new_func, new_query in variants:
                question = self.generate_question_from_sql(new_query)
                output_rows.append({
                    "original_query": original_query,
                    "replaced_func": f"{original_func}→{new_func}",
                    "new_query": new_query,
                    "generated_question": question
                })
        
        results_df = pd.DataFrame(output_rows)

        if output_file:
            results_df.to_csv(output_file, index=False)
            print(f"Paraphrased dataset generated and saved to {output_file}")
            
        return results_df

