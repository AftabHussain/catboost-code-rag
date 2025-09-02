# CodeContext Explorer: CatBoost Code Explanation with RAG on Zillow Data

## Contents

- [Intro](#intro)

- [Quick Start / Using the Model](#quick-start--using-the-model)
  - [Query the Model on Server](#query-the-model-on-server)
  - [Query the Model from Client Using FastAPI](#query-the-model-from-client-using-fastapi)
  - [View Saved Results from Client Machine Using Flask](#view-saved-results-from-client-machine-using-flask)
    - [Example](#example)
    - [Accessing the Web App Remotely](#accessing-the-web-app-remotely)

- [Running the RAG Reward Model & PPO Pipeline](#running-the-rag-reward-model--ppo-pipeline)
  - [Step 1: Build Pairwise Preference Dataset](#step-1-build-pairwise-preference-dataset)
  - [Step 2: Train the Reward Model](#step-2-train-the-reward-model)
  - [Step 3: Train the Language Model with PPO](#step-3-train-the-language-model-with-ppo)

- [About the Dataset](#about-the-dataset)
- [Dependencies](#dependencies)
- [About Catboost](#about-catboost)


## Intro

In this exploratory work, a Retrieval-Augmented Generation (RAG) system, **CodeContext Explorer**, is built to **provide code explanations** aimed at demonstrating the **potential of combining RAG techniques with domain-specific code context**. In this work, housing-data-related code is picked, in particular CatBoost code snippets applied to housing datasets (e.g. Zillow). CodeContext Explorer enables users to query contextualized code examples and their descriptions, supporting better understanding and usage of CatBoost in real estate modeling tasks. 

The system relies upon synthetically created CatBoost code samples, used in the domain of housing data, with descriptive annotations. Using vector embeddings and a FAISS index, it retrieves the most relevant code-context pairs in response to user queries. These retrieved contexts are passed to the Mistral-7B-Instruct language model with custom prompts to generate explanations. 

Results are stored in JSON format and presented through an interactive Flask web interface, allowing easy browsing of questions, related code snippets, and explanations — facilitating learning and exploration for data scientists and ML practitioners.

See [here](https://github.com/AftabHussain/catboost-code-rag/blob/main/README.md#example) for an example.

## Quick Start / Using the Model

### Query the Model on Server

Run the retrieval-augmented generation pipeline with Mistral:

```bash
python rag_mistral_single_ip.py
```


### Query the Model from Client Using FastAPI

1. Additional prequisites for Dynamic RAG Viewer 

```
pip install fastapi uvicorn pydantic
```

2. Run the live viewer in the server on say port 8000 as follows:

```
$cd viewer && uvicorn live_rag_qa_viewer:app --host 0.0.0.0 --port 8000
```

You'll see the following on your screen, which indicates the FastAPI application is running:

```
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 55.04it/s]
Device set to use cuda:0
INFO:     Started server process [2657441]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

3. Then connect your client machine to the port of the server where the app is running. We use SSH tunnel here:

```
ssh -L 8000:127.0.0.1:8000 username@server_address
```

4. Then in another terminal window in your client machine, run the client app provided in the viewer folder as follows:

```
python3 rag_client.py
```
You are now able to remotely send queries to the model, and receive its responses.

_____

### View Saved Results from Client Machine Using Flask

Launch the Flask web app to browse the saved QA pairs:

```bash
python viewer/rag_qa_viewer.py
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

## Running the RAG Reward Model & PPO Pipeline

This document describes the full workflow for building a reward model and training a language model using reinforcement learning with the reward model.

### Step 1: Build Pairwise Preference Dataset

```
python3 RL_build_pairwise_prefs.py
```

Generates a dataset of pairwise preferences from model outputs. These pairwise comparisons are used to teach the reward model which outputs are better or preferred.

### Step 2: Train the Reward Model

```
python3 RL_train_reward_model_pairwise.py
```

Trains a reward model using the pairwise preference dataset created in Step 1. This reward model will later provide feedback during reinforcement learning to guide the main model’s behavior.

### Step 3: Train the Language Model with PPO

```
wandb login
python3 RL_ppo_train_with_reward.py
```

Uses Proximal Policy Optimization (PPO) to fine-tune the language model guided by the trained reward model. `wandb login` ensures your training logs and metrics are recorded on Weights & Biases for live monitoring. After starting the training, you can check the W&B dashboard to see live metrics, reward scores, and model performance.

✅ Following these three steps in order will allow you to:

1. Generate a preference dataset,
2. Train a reward model, and
3. Fine-tune your main model with reinforcement learning while monitoring progress live.


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

## About Catboost

CatBoost is a gradient boosting library developed by Yan-
dex [1] that is widely used for machine learning tasks such as
classification, regression, and ranking. It is particularly known
for handling categorical features efficiently without requiring
extensive preprocessing, which makes it very convenient in
real-world datasets across various domains including healthcare,
housing, environmental science, etc. [2], [3], [4], [5], [6], [7]

#### References

[1] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin,
“Catboost: unbiased boosting with categorical features,” in Proceedings
of the 32nd International Conference on Neural Information Processing
Systems, ser. NIPS’18. Red Hook, NY, USA: Curran Associates Inc.,
2018, p. 6639–6649.

[2] S. Shao, B. Zhao, X. Cui, Y. Dai, and B. Bao, “Housing rental information
management and prediction system based on catboost algorithm - a case
study of halifax region,” in Rough Sets: International Joint Conference,
IJCRS 2024, Halifax, NS, Canada, May 17–20, 2024, Proceedings, Part
II. Berlin, Heidelberg: Springer-Verlag, 2024, p. 230–246. [Online].
Available: https://doi.org/10.1007/978-3-031-65668-2_16

[3] C. Zou, “The house price prediction using machine learning algorithm: The
case of jinan, china,” Highlights in Science, Engineering and Technology,
vol. 39, pp. 327–333, 04 2023.

[4] J. T. Hancock and T. M. Khoshgoftaar, “Catboost for big data: an
interdisciplinary review,” Journal of Big Data, vol. 7, no. 1, p. 94, 2020.
[Online]. Available: https://doi.org/10.1186/s40537-020-00369-8

[5] X. Jin, W. Sun, Y. Li, Y. Su, L. Xu, and X. Zhu, “Use of catboost
algorithm to identify the need for surgery in infants with necrotizing
enterocolitis,” Frontiers in Pediatrics, vol. 13, p. 1465278, Feb. 2025.
[Online]. Available: https://doi.org/10.3389/fped.2025.1465278

[6] M. Hamid, F. Hajjej, A. S. Alluhaidan, and N. W. bin Mannie,
“Fine tuned catboost machine learning approach for early detection
of cardiovascular disease through predictive modeling,” Scientific
Reports, vol. 15, no. 1, p. 31199, Aug. 2025. [Online]. Available:
https://doi.org/10.1038/s41598-025-13790-x

[7] Z. Guo, X. Wang, and L. Ge, “Classification prediction model of
indoor pm2.5 concentration using catboost algorithm,” Frontiers in
Built Environment, vol. 9, p. 1207193, 2023. [Online]. Available:
https://doi.org/10.3389/fbuil.2023.1207193






