from agent_eval.data_paraphrased import ParaphrasedDatasetGenerator
from agent_eval.data_claused import ClauseDatasetGenerator
import pandas as pd


input_file = "./input/input_data.csv"

# Paraphrased Dataset Generation

output_file = "./output/paraphrased_dataset.csv"
df = pd.read_csv(input_file) 

paraphrase_generator = ParaphrasedDatasetGenerator()
output_df = paraphrase_generator.process_from_input_file(df)

output_df.to_csv("./output/paraphrased_dataset.csv", index=False)



# Clause Manipulation

output_file = "./output/claused_dataset.csv"
df = pd.read_csv(input_file) 

clause_generator = ClauseDatasetGenerator()
output_df = clause_generator.process_data(df)

output_df.to_csv("./output/claused_dataset.csv", index=False)