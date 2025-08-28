from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline
import re
import os
import json

# ---------- FastAPI setup ----------
app = FastAPI(title="Mistral-RAG QA API")

# ---------- Input schema ----------
class QueryRequest(BaseModel):
    query: str

# ---------- Load Vectorstore ----------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "../data/vectordb/rag_vectorstore_db_v5",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# ---------- Load Mistral model ----------
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

# ---------- Prompt template ----------
prompt_template = PromptTemplate.from_template(
    """<s>[INST] Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question:
{question}
[/INST]"""
)

# ---------- QA chain ----------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# ---------- API route ----------
@app.post("/query")
def ask_model(req: QueryRequest):
    result = qa_chain.invoke(req.query)

    # Extract context/question/answer
    text = result['result']
    context_match = re.search(r"Context:\n(.+?)\nQuestion:", text, re.DOTALL)
    question_match = re.search(r"Question:\n(.+?)\n\[/INST\]", text, re.DOTALL)
    answer_match = re.search(r"\[/INST\](.+)", text, re.DOTALL)

    context = context_match.group(1).strip() if context_match else ""
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # Optional: Save to JSON
    data_path = "../output/data_new1.json"
    if os.path.exists(data_path) and (os.path.getsize(data_path) != 0):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    new_id = max((entry["id"] for entry in data), default=0) + 1
    entry = {"id": new_id, "question": question, "context": context, "answer": answer}
    data.append(entry)
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {"id": new_id, "question": question, "context": context, "answer": answer}

