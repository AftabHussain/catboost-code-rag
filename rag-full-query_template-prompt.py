from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# Load vectorstore and embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "rag_vectorstore_db_v1",
    embedding_model,
    allow_dangerous_deserialization=True
)

# Set up Flan-T5 model pipeline
text_gen = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=400
)

llm = HuggingFacePipeline(pipeline=text_gen)

# Create a custom prompt for Flan
prompt_template = PromptTemplate.from_template(
    """Use the following context to answer the question.
If you don't know the answer, just say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Use the prompt in the QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Run query
#query = "What can you tell me about CatBoost usage in Zillow dataset?"
query = "What preprocessing steps are applied before fitting a CatBoostRegressor with StandardScaler?"
result = qa_chain.invoke(query)

print("Answer:")
print(result)

