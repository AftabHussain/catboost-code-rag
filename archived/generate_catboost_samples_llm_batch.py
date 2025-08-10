import csv
import random
from transformers import pipeline

# Load open-source instruction-tuned model
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-alpha",  # or mistralai/Mistral-7B-Instruct-v0.1
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.95,
    do_sample=True
)

# Variations
variations = [
    "preprocessing missing values",
    "encoding categorical variables",
    "cross-validation using CatBoost",
    "using CatBoostPool for training",
    "hyperparameter tuning with CatBoost",
    "plotting CatBoost feature importance",
    "saving and loading a CatBoost model",
    "using early stopping with eval set",
    "predicting housing prices with custom loss",
    "using SHAP with CatBoost",
    "handling imbalanced housing data"
]

# Prompt maker
def make_prompt():
    variation = random.choice(variations)
    return f"""
Generate a Python code snippet that uses the CatBoost library applied to a Zillow-like housing dataset, focusing on {variation}. Also include a 1-2 sentence description of what the code does. Return the output in the following format:
### Code:
<insert code here>

### Description:
<insert description here>
""".strip()

# Output CSV
csv_path = "catboost_code_dataset.csv"
fieldnames = ["id", "code_snippet", "description"]

batch_size = 10
total_samples = 1000

with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    prompt_id = 1
    for start_idx in range(0, total_samples, batch_size):
        batch_prompts = [make_prompt() for _ in range(batch_size)]

        try:
            outputs = generator(batch_prompts)

            for prompt, output_list in zip(batch_prompts, outputs):
                output=output_list[0]
                text = output["generated_text"]
                code_start = text.find("### Code:")
                desc_start = text.find("### Description:")

                if code_start != -1 and desc_start != -1:
                    code_snippet = text[code_start + 9:desc_start].strip()
                    description = text[desc_start + 17:].strip()

                    if code_snippet and description:
                        writer.writerow({
                            "id": prompt_id,
                            "code_snippet": code_snippet,
                            "description": description
                        })
                        print(f"[✓] Sample {prompt_id} written.")
                    else:
                        print(f"[!] Sample {prompt_id} missing content, skipping.")
                else:
                    print(f"[!] Sample {prompt_id} did not parse correctly.")
                prompt_id += 1

        except Exception as e:
            print(f"[!] Error at batch starting {start_idx}: {e}")

print("✅ Finished generating 1000 diverse CatBoost samples.")

