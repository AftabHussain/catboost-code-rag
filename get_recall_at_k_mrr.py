import numpy as np
import json
import pandas as pd
import os

# ==== Metrics functions ====
def recall_at_k(results, k=5):
    scores = []
    for r in results:
        gold_text = r["gold_code"] + " " + r["gold_description"]
        retrieved = r["retrieved_texts"][:k]
        found = any(gold_text.strip()[:50] in doc for doc in retrieved)
        scores.append(int(found))
    return np.mean(scores)

def mrr(results):
    ranks = []
    for r in results:
        gold_text = r["gold_code"] + " " + r["gold_description"]
        found_rank = None
        for i, doc in enumerate(r["retrieved_texts"], start=1):
            if gold_text.strip()[:50] in doc:
                found_rank = i
                break
        ranks.append(1.0 / found_rank if found_rank else 0.0)
    return np.mean(ranks)

def per_query_metrics(results):
    """Returns a list of dicts with per-query recall@k and rank info."""
    metrics = []
    for r in results:
        gold_text = r["gold_code"] + " " + r["gold_description"]
        retrieved = r["retrieved_texts"]
        found_rank = None
        for i, doc in enumerate(retrieved, start=1):
            if gold_text.strip()[:50] in doc:
                found_rank = i
                break

        metrics.append({
            "query": r["query"],
            "retrieved_count": len(retrieved),
            "found_rank": found_rank if found_rank else 0,
            "recall@1": int(found_rank is not None and found_rank <= 1),
            "recall@3": int(found_rank is not None and found_rank <= 3),
            "recall@5": int(found_rank is not None and found_rank <= 5),
        })
    return metrics

# ==== Load results ====
input_path = "output/random500_eval_results.json"
with open(input_path, "r", encoding="utf-8") as f:
    results = json.load(f)

# ==== Compute overall metrics ====
print("Recall@1:", recall_at_k(results, k=1))
print("Recall@3:", recall_at_k(results, k=3))
print("Recall@5:", recall_at_k(results, k=5))
print("MRR:", mrr(results))

# ==== Generate per-query CSV summary ====
metrics_list = per_query_metrics(results)
df_metrics = pd.DataFrame(metrics_list)

# Save CSV
os.makedirs("output", exist_ok=True)
csv_path = "output/random500_eval_retrieval_summary.csv"
df_metrics.to_csv(csv_path, index=False)
print(f"✅ Per-query retrieval metrics saved to {csv_path}")

