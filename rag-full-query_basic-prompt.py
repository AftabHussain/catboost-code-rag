from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Load embedding model and vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "rag_vectorstore_db_v1",
    embedding_model,
    allow_dangerous_deserialization=True
)

# GPT-2
'''
# Hugging Face pipeline
text_gen = pipeline(
    "text-generation",
    model="gpt2",
    pad_token_id=50256
)

# Wrap for LangChain with model_kwargs to avoid max_length issues
llm = HuggingFacePipeline(
    pipeline=text_gen,
    model_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95
    }
)
'''

# Flan T5 Base  

text_gen = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(
    pipeline=text_gen,
    model_kwargs={"temperature": 0.3}
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What can you tell me about CatBoost usage in Zillow dataset?"
result = qa_chain.invoke(query)

print("Answer:")
print(result)

