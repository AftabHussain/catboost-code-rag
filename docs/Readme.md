## Key Methodology Components

### Main Use Case: Using the RAG Pipeline

<p align="center">
<img src="figs/Screenshot%20from%202025-08-26%2023-42-58.png" alt="RAG pipeline workflow" width="800"/>
</p>

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering domain-specific questions using a large language model (LLM). It combines a precomputed FAISS vectorstore of embeddings with a Generator Model (i.e., the LLM) to provide accurate and context-aware responses. When a user submits a query, the pipeline first retrieves the most relevant context from the vectorstore using semantic similarity. This context is then inserted into a structured instruction-style prompt, which is fed to the LLM to generate an answer. The system parses the output into context, question, and answer components and logs each interaction in a JSON dataset for future reference. This approach allows efficient querying over large datasets.



### RAG LLM Optimization Phase 1: Pairwaise Pairs Dataset Generation for Training Reward Model

<p align="center">
<img src="figs/Screenshot from 2025-08-26 23-52-56.png" alt="RAG pipeline workflow" width="800"/>
</p>

This phase generates pairwise preference data to train a reward model for instruction-following or code-related tasks. For each query in the dataset, the process first retrieves relevant context from a precomputed FAISS vectorstore. A structured prompt is constructed combining the retrieved context and the query, which is then passed to a generative language model (Mistral-7B-Instruct) to produce multiple candidate answers. Each candidate is scored using a heuristic ranking system that combines: (1) similarity to the retrieved context (“grounding score”), (2) coverage of task-relevant keywords, and (3) a mild length penalty to discourage overly verbose answers. The top-scoring candidate is marked as “chosen” and the lowest-scoring candidate as “rejected,” forming a pair. These prompt–chosen–rejected triples are saved in a JSONL file (pairwise_prefs.jsonl) and provide training data for reward models that can later guide preference-aligned generation. This approach ensures that the reward model learns to prefer outputs that are both contextually grounded and relevant to the task.

### RAG LLM Optimization Phase 2: Training a Pairwise Reward Model

In this stage, a Reward Model (RM) is trained using the preference pairs generated earlier. The dataset consists of triplets: a prompt, a “chosen” answer (preferred), and a “rejected” answer (less preferred). A pretrained base encoder (e.g., bert-base-uncased) is fine-tuned to assign a scalar reward score to each answer. Training uses a pairwise loss function of the form `-log σ(r_chosen − r_rejected)`, which encourages the model to give higher scores to preferred answers compared to rejected ones. This setup aligns the model’s scoring function with human-like or heuristic preferences. The process includes splitting data into training and validation sets, optimizing with AdamW, and monitoring both loss and validation accuracy. After each epoch, checkpoints are saved, and a log file tracks progress. The trained reward model becomes a crucial evaluator for reinforcement learning or direct preference optimization steps that follow.

### Fine-tuning LLM with PPO and Feedback from Reward Model

<p align="center">
<img src="figs/Screenshot from 2025-08-26 23-52-56.png" alt="RAG pipeline workflow" width="800"/>
</p>

In this phase, the policy model (Mistral-7B-Instruct) is fine-tuned using Proximal Policy Optimization (PPO) with guidance from the Reward Model trained on synthetic preference data. Instead of relying on direct human annotations, the system uses heuristic-based rankings (context grounding, keyword coverage, and length penalty) to generate “chosen vs. rejected” pairs. These pairs allow the Reward Model to provide scalar rewards for policy outputs. During training, a frozen reference model is maintained to constrain policy updates and prevent instability. The pipeline samples prompts, generates candidate responses from the policy, scores them with the Reward Model, and updates the policy to maximize expected reward while staying close to the reference. This approach is an instance of Reinforcement Learning with AI Feedback (RLAIF), where synthetic preferences stand in for human judgments, enabling scalable alignment without manual labeling.
