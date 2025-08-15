import pandas as pd
import json
import os
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# === Step 1: Load vectorstore ===
print("🔹 Loading FAISS vectorstore...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "./data/vectordb/rag_vectorstore_db_v5",
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Vectorstore loaded.")

# === Step 2: Load Mistral model ===
print("🔹 Loading Mistral model...")
text_gen = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device=0,  # GPU
    torch_dtype="auto",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.3,
    top_p=0.9
)
llm = HuggingFacePipeline(
    pipeline=text_gen,
    model_kwargs={
        "max_new_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.9
    }
)
print("✅ Model loaded.")

# === Step 3: Prompt ===
prompt_template = PromptTemplate.from_template(
    """<s>[INST] Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question:
{question}
[/INST]"""
)

# === Step 4: QA Chain ===
print("🔹 Setting up QA chain...")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)
print("✅ QA chain ready.")

# === Step 5: Load gold dataset ===
print("🔹 Loading gold dataset: random_500.csv")
gold_df = pd.read_csv("data/raw/random_500.csv")

print(f"✅ Loaded {len(gold_df)} samples for evaluation.")

# === Step 6: Evaluation loop ===
results = []
output_path = "output/random500_eval_results.json"
os.makedirs("output", exist_ok=True)

print("🚀 Starting evaluation...")
for idx, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Processing queries"):
    query = row["query"]
    gold_code = row["code"]
    gold_desc = row["description"]

    # Debug progress every 10 queries
    if idx % 10 == 0 and idx > 0:
        print(f"🔹 Processed {idx}/{len(gold_df)} queries...")

    # Run pipeline
    try:
        result = qa_chain.invoke({"query": query})
    except Exception as e:
        print(f"⚠️ Error processing query {idx}: {e}")
        result = {"result": "", "source_documents": []}

    # Extract retrieved context
    retrieved_docs = result.get("source_documents", [])
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Answer from LLM
    answer = result.get("result", "")

    # Save to memory
    results.append({
        "query": query,
        "gold_code": gold_code,
        "gold_description": gold_desc,
        "retrieved_texts": retrieved_texts,
        "answer": answer
    })

    # Save progress every 50 queries
    if idx % 50 == 0 and idx > 0:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Progress saved at {idx} queries.")

# === Step 7: Final save ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Evaluation complete. Results saved to {output_path}")

