import json
import random

# Path to your JSONL file
file_path = "pairwise_prefs.jsonl"

# Read all lines
with open(file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# Pick one random item
random_item = random.choice(data)

print("=== Random Item ===")
print("Prompt:\n", random_item["prompt"], "\n")
print("Chosen:\n", random_item["chosen"], "\n")
print("Rejected:\n", random_item["rejected"])

