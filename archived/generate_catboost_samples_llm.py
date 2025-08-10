import csv
import random
import time
from transformers import pipeline

# Load open-source instruction-tuned model (adjust to what you have installed)
generator = pipeline(
    "text-generation",
    #model="HuggingFaceH4/zephyr-7b-alpha",  # or mistralai/Mistral-7B-Instruct-v0.1
    model="mistralai/Mistral-7B-Instruct-v0.1",
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.95,
    do_sample=True
)

def make_prompt():
    variation = random.choice([
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
    ])

    example = """
### Code:
<code>

### Description:
<description>
""".strip()

    return f"""
You are a Python coding assistant. Generate a unique Python code snippet that uses the CatBoost library on a Zillow-like housing dataset, focusing on {variation}.
Follow the exact style and formatting shown in the example below, but create a new and different snippet:

{example}
""".strip()


# Output CSV path
csv_path = "catboost_code_dataset.csv"

# CSV fieldnames
fieldnames = ["id", "code_snippet", "description"]

# Generation loop
with open(csv_path, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1, 3):  # Generate 1000 samples
        prompt = make_prompt()
        try:
            output = generator(prompt)[0]["generated_text"]

            # Extract code and description using simple parsing
            code_start = output.find("### Code:")
            desc_start = output.find("### Description:")

            if code_start != -1 and desc_start != -1:
                code_snippet = output[code_start + 9:desc_start].strip()
                description = output[desc_start + 17:].strip()

                if code_snippet and description:
                    writer.writerow({
                        "id": i,
                        "code_snippet": code_snippet,
                        "description": description
                    })
                    print(f"[✓] Sample {i} written.")
                else:
                    print(f"[!] Sample {i} missing content, skipping.")
            else:
                print(f"[!] Sample {i} did not parse correctly.")
        except Exception as e:
            print(f"[!] Error at sample {i}: {e}")
        
        time.sleep(0.5)  # Optional: to avoid overload

print("✅ Finished generating 1000 diverse CatBoost samples.")

