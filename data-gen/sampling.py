import pandas as pd

# Load the CSV
df = pd.read_csv("~/workspace/catboost-code-rag/data/raw/catboost_code_dataset_templates_with_query.csv")

# Randomly select 500 rows
df_sample = df.sample(n=500, random_state=42)  # random_state ensures reproducibility

# Optionally, save to a new CSV
df_sample.to_csv("random_500.csv", index=False)

