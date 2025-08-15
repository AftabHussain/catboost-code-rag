import csv
import random
import os
import string

csv_path = "catboost_code_dataset_templates.csv"
templates_dir = "templates"

fieldnames = ["code", "description", "query"]

datasets = [
    "zillow_data.csv", "housing_prices.csv", "real_estate_train.csv",
    "property_data.csv", "house_sales.csv"
]
targets = ["price", "SalePrice", "MedianValue", "ListingPrice"]
loss_functions = ["RMSE", "MAE", "Quantile:alpha=0.9", "Logloss"]
depths = [4, 6, 8, 10]
learning_rates = [0.01, 0.05, 0.1, 0.2]
iterations = [100, 200, 500, 800]
categorical_features = [
    '["neighborhood"]',
    '["city", "zipcode"]',
    '["state"]',
    '["property_type", "zipcode"]'
]
folds = [3, 5, 10]
techniques = [
    "handling missing values",
    "encoding categorical variables",
    "hyperparameter tuning",
    "feature importance analysis",
    "early stopping",
    "cross-validation",
    "model saving and loading",
    "using CatBoostPool"
]

def load_templates(folder):
    templates = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if all(marker in content for marker in ["### CODE TEMPLATE ###", "### DESCRIPTION TEMPLATE ###", "### QUERY TEMPLATE ###"]):
                    parts = content.split("### CODE TEMPLATE ###")[1]
                    code_part, rest = parts.split("### DESCRIPTION TEMPLATE ###")
                    desc_part, query_part = rest.split("### QUERY TEMPLATE ###")
                    templates.append((code_part.strip(), desc_part.strip(), query_part.strip()))
                else:
                    print(f"[!] Warning: file {fname} missing required markers, skipping.")
    return templates

def extract_placeholders(template):
    formatter = string.Formatter()
    return {fname for _, fname, _, _ in formatter.parse(template) if fname}

def make_format_dict(keys):
    defaults = {
        "dataset": random.choice(datasets),
        "target": random.choice(targets),
        "loss_function": random.choice(loss_functions),
        "depth": random.choice(depths),
        "learning_rate": random.choice(learning_rates),
        "iterations": random.choice(iterations),
        "cat_features": random.choice(categorical_features),
        "folds": random.choice(folds),
        "technique": random.choice(techniques)
    }
    # For any key missing in defaults, assign empty string
    return {k: defaults.get(k, "") for k in keys}

template_pairs = load_templates(templates_dir)
print(f"Loaded {len(template_pairs)} template-description pairs.")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i in range(1000):  # generate 1000 samples
        idx = random.randint(0, len(template_pairs) - 1)

        code_template, description_template, query_template = template_pairs[idx]
        print(f"Using Template #{idx} for sample {i+1}")

        # Extract placeholders from both code and description templates
        keys = extract_placeholders(code_template).union(extract_placeholders(description_template)).union(extract_placeholders(query_template))


        format_dict = make_format_dict(keys)

        try:
            code_snippet = code_template.format(**format_dict).strip()
            description = description_template.format(**format_dict).strip()
            query = query_template.format(**format_dict).strip()


            writer.writerow({
                "code": code_snippet,
                "description": description,
                "query": query
            })

            if (i + 1) % 100 == 0:
                print(f"[✓] Generated {i + 1} samples.")

        except Exception as e:
            print(f"[!] Formatting error at sample {i+1}: {e}")

print(f"✅ Generated 1000 diverse CatBoost samples in {csv_path}")

