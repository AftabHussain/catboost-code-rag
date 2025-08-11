from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd

# LOAD CSV DATA
'''
df = pd.read_csv('samples/zillow_catboost_samples_16_to_25.csv')
df2 = pd.read_csv('samples/zillow_catboost_samples_26_to_100.csv')
df3 = pd.read_csv('samples/zillow_catboost_samples_101_to_103.csv')
df = pd.concat([df, df2], ignore_index=True)
df = pd.concat([df, df3], ignore_index=True)
'''

# Latest dataset (synthetically generated, template-driven dataset of diverse
# CatBoost code snippets and descriptions for housing data modeling) 
df = pd.read_csv('~/workspace/catboost-code-rag/data/raw/catboost_code_dataset_from_templates.csv')



# Prepare documents list by combining code and description columns
documents = [
            Document(page_content=f"{row['code']} {row['description']}")
                for _, row in df.iterrows()
                ]

# Initialize HuggingFace embedding model (ensure langchain-huggingface installed)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS vectorstore from documents and embeddings
vectorstore = FAISS.from_documents(documents, embedding_model)

# Save the vectorstore locally
vectorstore.save_local("rag_vectorstore_db_v4")

