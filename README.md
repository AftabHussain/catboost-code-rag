# Zillow-CodeRAG: A Retrieval-Augmented QA System for CatBoost Code Snippets
Retrieval-Augmented Generation (RAG) QA system on Zillow housing data and CatBoost code samples using FAISS, HuggingFace embeddings, and open-source language models. Includes a local pipeline for context-aware code search and question answering.

# Usage

## Query the Model

Run the retrieval-augmented generation pipeline with Mistral:

```bash
python rag_query_pipeline_mistral_prompt.py
```

This script performs a full RAG query using the Mistral-7B-Instruct model and saves the question, context, and answer to the dataset.

## View Results

Launch the Flask web app to browse the saved QA pairs:

```bash
python app.py
```
The web app provides an interactive interface to navigate through your question-answer samples, displaying the context, question, and model-generated answers in a clean format.

### Accessing the Web App Remotely

From your local machine, create an SSH tunnel to securely access the app running on your server:
```bash
ssh -L 5000:localhost:5000 user@server_address
```
Then open `http://localhost:5000` in your browser to interact with the app.

# Dependencies

Make sure you have the following packages installed:

- `langchain`
- `langchain-community`
- `langchain-huggingface`
- `transformers`
- `faiss-cpu` or `faiss-gpu`
- `flask`
- `sentence-transformers`
- `torch` (required by transformers models)
- `python-dotenv` (if you're using environment variables)

You can install them via pip:

```bash
pip install langchain langchain-community langchain-huggingface \
            transformers sentence-transformers faiss-cpu flask torch
```




