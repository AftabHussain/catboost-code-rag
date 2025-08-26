'''
1) Build pairs with self-comparison + heuristic ranking

Saves output/pairwise_prefs.jsonl (prompt, chosen, rejected).
'''

# file: build_pairwise_prefs.py
import os, json, random, re
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer, util

# --------------------------------
# Config - Base RAG Pipeline Model
# --------------------------------
DATA_CSV = "data/raw/random_500_part_3.csv"     # columns: query, code, description
VDB_PATH = "./data/vectordb/rag_vectorstore_db_v5"

BASE_LLM = "mistralai/Mistral-7B-Instruct-v0.1"
USE_4BIT = True
MAX_NEW_TOKENS = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [13, 17, 19, 23, 29, 31]
os.makedirs("output", exist_ok=True)

# ------------------------
# Load retriever
# ------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(VDB_PATH, embedding_model, allow_dangerous_deserialization=True)
#retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) #returns top five documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) #since our docs are dense (i.e., self-contained), top-1 is good enough

# ------------------------
# Build prompts
# ------------------------
def make_prompt(context, question):
    return f"""<s>[INST] Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question:
{question}
[/INST]"""

# ------------------------
# Load gen model (quantized if large)
# ------------------------
if USE_4BIT:
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    gen_model = AutoModelForCausalLM.from_pretrained(
        BASE_LLM, quantization_config=bnb_cfg, device_map="auto"
    )
else:
    gen_model = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.bfloat16).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(BASE_LLM, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

gen_model.eval()


# ------------------------------------------------------------------------------
# Config - For Accepted/Rejected Pair Generation (For Reward Model Construction)
# ------------------------------------------------------------------------------
NUM_CANDIDATES = 2          # N candidates per prompt
OUT_JSONL = "output/pairwise_prefs_part_3.jsonl"
# Heuristic weights
W_GROUND = 0.7   # similarity to retrieved context
W_KEYWD  = 0.3   # coverage of task-relevant keywords
W_LEN    = 0.05  # mild length penalty

# ------------------------
# Embedding model for grounding
# ------------------------
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

def extract_answer(full_text: str) -> str:
    # take text after the last [/INST]
    return full_text.split("[/INST]")[-1].strip()

def generate_candidates(prompt: str, n: int) -> list[str]:
    inputs = tokenizer([prompt], return_tensors="pt").to(gen_model.device)
    outs = []
    '''
	# Different seeds for diversity, makes a separate pass for each input
    # prompt. The whole generation process from 500 queries (from each query we
    # generate NUM_CANDIDATE prompts) takes ~8hrs.
    seeds = random.sample(SEEDS, k=min(n, len(SEEDS)))
    for sd in seeds:
        torch.manual_seed(sd)
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        txt = tokenizer.decode(out[0], skip_special_tokens=True)
        outs.append(extract_answer(txt))
    return outs
    '''

    # Different seeds for diversity.
    # PERF OPT NO.1: makes a single pass for a batch of seeds, # returning N
    # sequences. With this method, the whole generation from 500 input # queries
    # takes ~3hr.37mins, a huge reduction from about ~8hrs.
    seeds = random.sample(SEEDS, k=min(n, len(SEEDS)))
 
    with torch.no_grad():
      # REJECTED PERF OPT: Adding either of the following to enable FP16/BF16 for mixed-precision
      # generation increases the total generation time from ~3hr.37mins to ~5hrs.
      '''
      with torch.autocast("cuda", dtype=torch.float16):
      with torch.autocast("cuda", dtype=torch.float16):
      '''
      # Potential Reasons for the slowdown after autocast, suggested by chatgpt:
      # 4-bit quantization + autocast mismatch:
      # - The model is already in 4-bit NF4, which is ultra-low precision.
      # - Wrapping it in FP16/BF16 (autocast) doesn’t help because the weights are already quantized.
      #   The extra autocast adds overhead rather than speeding up computations.
      # 
      # GPU throughput penalty:
      # - Autocast tries to cast operations to FP16/BF16 on the fly.
      # - For 4-bit models, this is mostly unnecessary and can actually increase kernel launch overhead,
      #   slowing down generation instead of speeding it up.

      outputs = gen_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=len(seeds),   
            pad_token_id=tokenizer.eos_token_id
      )
    
    outs = [extract_answer(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]
    return outs


def get_keywords_from_gold(code: str, desc: str) -> set:
    # derive lightweight keywords from gold (still “model-ish”, but helps)
    # keep only identifiers / api tokens
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", (code or "") + " " + (desc or ""))
    # common CatBoost & sklearn words to bias towards
    canon = {"CatBoost", "CatBoostRegressor", "Pool", "fit", "predict",
             "iterations", "learning_rate", "depth", "RMSE", "mean_squared_error",
             "pandas", "read_csv"}
    toks = set(toks) | canon
    return set(t.lower() for t in toks)

def keyword_coverage(ans: str, kw: set) -> float:
    if not kw:
        return 0.0
    toks = set(t.lower() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", ans))
    hit = len(kw & toks)
    return hit / max(1, len(kw))

def grounding_score(ans: str, retrieved_chunks: list[str]) -> float:
    if not retrieved_chunks:
        return 0.0
    with torch.no_grad():
        a = embed.encode(ans, convert_to_tensor=True, normalize_embeddings=True)
        chunks = embed.encode(retrieved_chunks, convert_to_tensor=True, normalize_embeddings=True)
        # take the best-aligned chunk
        sims = util.cos_sim(a, chunks)[0]
        return float(sims.max().item())

def length_penalty(ans: str) -> float:
    # mild penalty for overly long rambles
    tokens = len(ans.split())
    return min(1.0, tokens / 400)  # cap

def score_answer(ans: str, retrieved_chunks: list[str], kw: set) -> float:
    s_ground = grounding_score(ans, retrieved_chunks)
    s_kw = keyword_coverage(ans, kw)
    s_len = length_penalty(ans)
    return (W_GROUND * s_ground) + (W_KEYWD * s_kw) - (W_LEN * s_len)

# ------------------------
# Main: build pairwise prefs
# ------------------------
df = pd.read_csv(DATA_CSV)
pairs = 0

with open(OUT_JSONL, "w", encoding="utf-8") as fout:
    #idx = 0 # Remove later -- FOR TEST ONLY!!!
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building prefs"):
        '''
        idx+=1
        if idx>1:
          break
        '''

        query = row["query"]
        gold_code = row.get("code", "")
        gold_desc = row.get("description", "")
  
        '''
        print("=" * 60)
        print("🔍 Query:")
        print(query)
        print("\n💻 Gold Code:")
        print(gold_code)
        print("\n📝 Gold Description:")
        print(gold_desc)
        print("=" * 60)
        '''

        # retrieve context
        docs = retriever.get_relevant_documents(query)
        retrieved = [d.page_content for d in docs]
        context = "\n\n".join(retrieved)
        prompt = make_prompt(context, query)

        # generate N candidates
        cands = generate_candidates(prompt, NUM_CANDIDATES)
        if not cands:
            continue

        # score with grounding-first heuristic
        kw = get_keywords_from_gold(gold_code, gold_desc)
        scored = [(cand, score_answer(cand, retrieved, kw)) for cand in cands]
        scored.sort(key=lambda x: x[1], reverse=True)

        chosen = scored[0][0]
        rejected = scored[-1][0]
        if chosen.strip() == rejected.strip():
            continue

        rec = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        fout.write(json.dumps(rec) + "\n")
        pairs += 1

        '''
        print("=" * 60)
        print("📝 Prompt:")
        print(rec["prompt"])
        print("\n✅ Chosen:")
        print(rec["chosen"])
        print("\n❌ Rejected:")
        print(rec["rejected"])
        print("=" * 60)
        '''


print(f"✅ Wrote {pairs} preference pairs to {OUT_JSONL}")

