from langchain_community.vectorstores import FAISS
import sys
import re
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# Step 1: Load FAISS vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "rag_vectorstore_db_v2",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Step 2: Load Mistral-7B-Instruct model using HuggingFace pipeline
text_gen = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.1",
    device=0,  # GPU only
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

# Step 3: Define Mistral-style prompt
prompt_template = PromptTemplate.from_template(
    """<s>[INST] Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question:
{question}
[/INST]"""
)

# Step 4: Set up retrieval-based QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # Lowered k to avoid context overflow

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Step 5: Run query
#query = "How is the 'condition' column preprocessed before training CatBoost in Zillow samples?"
query = input("Input your query to Mistral about CatBoost usage.")
result = qa_chain.invoke(query)

#print("Answer:")
#print(result["result"])

# Debug
'''
print(type(result))
for key, value in result.items():
    print(f"{key}: {value}")
print(result)
'''

text = result['result']  # your raw string

context_match = re.search(r"Context:\n(.+?)\nQuestion:", text, re.DOTALL)
question_match = re.search(r"Question:\n(.+?)\n\[/INST\]", text, re.DOTALL)
answer_match = re.search(r"\[/INST\](.+)", text, re.DOTALL)

context = context_match.group(1).strip() if context_match else ""
question = question_match.group(1).strip() if question_match else ""
answer = answer_match.group(1).strip() if answer_match else ""

print("############ QUESTION ############\n", question)
print("\n############ CONTEXT RETRIEVED #############\n", context)
print("\n############ ANSWER ###########\n", answer)

# Load existing JSON or initialize new list
data_path = "output/data.json"
if os.path.exists(data_path) and (os.path.getsize(data_path) != 0):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    data = []

# Assign new unique ID
new_id = max((entry["id"] for entry in data), default=0) + 1

# Create new entry
entry = {
    "id": new_id,
    "question": question,
    "context": context,
    "answer": answer
}

# Append to dataset and save
data.append(entry)
with open(data_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"✅ Added result as id={new_id}")
