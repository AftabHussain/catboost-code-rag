# CodeContext Explorer: CatBoost Code Explanation with RAG on Zillow Data

In this exploratory work, a Retrieval-Augmented Generation (RAG) system, **CodeContext Explorer**, is built to **provide code explanations** aimed at demonstrating the **potential of combining RAG techniques with domain-specific code context**. In this work, housing-data-related code is picked, in particular CatBoost code snippets applied to housing datasets (e.g. Zillow). CodeContext Explorer enables users to query contextualized code examples and their descriptions, supporting better understanding and usage of CatBoost in real estate modeling tasks. 

The system relies upon synthetically created CatBoost code samples, used in the domain of housing data, with descriptive annotations. Using vector embeddings and a FAISS index, it retrieves the most relevant code-context pairs in response to user queries. These retrieved contexts are passed to the Mistral-7B-Instruct language model with custom prompts to generate explanations. 

Results are stored in JSON format and presented through an interactive Flask web interface, allowing easy browsing of questions, related code snippets, and explanations — facilitating learning and exploration for data scientists and ML practitioners.

See [here](https://github.com/AftabHussain/catboost-code-rag/blob/main/README.md#example) for an example.

## Usage

### Query the Model

Run the retrieval-augmented generation pipeline with Mistral:

```bash
python rag_query_pipeline_mistral_prompt.py
```

This script performs a full RAG query using the Mistral-7B-Instruct model and saves the question, context, and answer to the dataset.

### View Results

Launch the Flask web app to browse the saved QA pairs:

```bash
python app.py
```
The web app provides an interactive interface to navigate through your question-answer samples, displaying the context, question, and model-generated answers in a clean format.

#### Example

<img width="2366" height="928" alt="Screenshot from 2025-07-31 18-57-09" src="https://github.com/user-attachments/assets/c4be6ee3-2bd0-4a57-ac34-21c7226189df" />


#### Accessing the Web App Remotely

From your local machine, create an SSH tunnel to securely access the app running on your server:
```bash
ssh -L 5000:localhost:5000 user@server_address
```
Then open `http://localhost:5000` in your browser to interact with the app.

## About the Dataset

We name our dataset, CatBoostCH 1.0 (CatBoost Code for Housing Data). This
dataset consists of 1,000 synthetically generated Python code snippets
demonstrating diverse uses of the CatBoost library applied to housing datasets
similar to Zillow. The samples are created using multiple customizable
templates, covering a variety of common data processing, model training,
evaluation, and deployment scenarios. Each code snippet is paired with a
concise description explaining its purpose and context. Designed as a prototype
dataset, it offers a scalable foundation that can be expanded with additional
templates and real-world data for broader applicability in AI/ML research.
We have provided the scripts used to generate this dataset.

## Dependencies

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




