from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load the vectorstore with explicit allow_dangerous_deserialization=True
vectorstore = FAISS.load_local(
    "rag_vectorstore_db",
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True
)

query = "your search query here"
docs = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(docs):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print("------")

