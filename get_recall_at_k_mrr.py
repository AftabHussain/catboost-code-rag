import numpy as np
import json

def recall_at_k(results, k=5):
    scores = []
    for r in results:
        gold_text = r["gold_code"] + " " + r["gold_description"]
        retrieved = r["retrieved_texts"][:k]
        found = any(gold_text.strip()[:50] in doc for doc in retrieved)  # loose match
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

# Load saved results
with open("output/random500_eval_results.json", "r") as f:
    results = json.load(f)

print("Recall@5:", recall_at_k(results, k=5))
print("MRR:", mrr(results))

