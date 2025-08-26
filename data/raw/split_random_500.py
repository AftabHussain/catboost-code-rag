import pandas as pd

# Load your CSV
df = pd.read_csv("random_500.csv")

# Split into 3 parts
num_parts = 3
chunk_size = len(df) // num_parts

for i in range(num_parts):
    start = i * chunk_size
    # Make sure last part gets any remainder rows
    end = (i + 1) * chunk_size if i < num_parts - 1 else len(df)
    
    part_df = df.iloc[start:end]
    part_df.to_csv(f"random_500_part_{i+1}.csv", index=False)

print("Split completed!")

