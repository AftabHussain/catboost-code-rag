import pandas as pd

# Load your CSV
df = pd.read_csv("random_500.csv")

# Pick one random row
random_row = df.sample(n=1).to_dict(orient="records")[0]

print("=== Random Row ===")
print("Code:\n", random_row["code"], "\n")
print("Description:\n", random_row["description"], "\n")
print("Query:\n", random_row["query"])

