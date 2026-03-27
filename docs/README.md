# Overall Approach

## Main Use Case: Using the RAG Pipeline

<p align="center">
<img src="figs/Screenshot%20from%202025-08-26%2023-42-58.png" alt="RAG pipeline workflow" width="800"/>
</p>

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering domain-specific questions using a large language model (LLM). It combines a precomputed FAISS vectorstore of embeddings with a Generator Model (i.e., the LLM) to provide accurate and context-aware responses. 

> [_FAISS vectorstore generation_](https://github.com/AftabHussain/catboost-code-rag/blob/99819f177eff99837373b949db9c1ba4dac2f086/data-gen/gen_db.py#L21-L34)
> 
> [_Loading the vectorstore and model in the RAG pipeline_](https://github.com/AftabHussain/catboost-code-rag/blob/99819f177eff99837373b949db9c1ba4dac2f086/rag_mistral_batch_ip.py#L11-L41)
> 
> [_Setting up the prompt template and QA chain using LangChain_](https://github.com/AftabHussain/catboost-code-rag/blob/32c42c7d4325e82126556f7b8024a359b33224ca/rag_mistral_batch_ip.py#L43-L66)

When a user submits a query, the pipeline first retrieves the most relevant context from the vectorstore using semantic similarity. 

>[_Retrieval of context using semantic similarity using LangChain's RetrievalQA_](https://github.com/AftabHussain/catboost-code-rag/blob/77d0b5e9de43ecb25a2ab248101ae6f0b4d95026/rag_mistral_batch_ip.py#L97)

This context is then inserted into a structured instruction-style prompt, which is fed to the LLM to generate an answer. The system parses the output into context, question, and answer components and logs each interaction in a JSON dataset for future reference. 

This approach allows efficient querying over large datasets.

> [_Query using a gold dataset, and save results_](https://github.com/AftabHussain/catboost-code-rag/blob/32c42c7d4325e82126556f7b8024a359b33224ca/rag_mistral_batch_ip.py#L70C38-L122C14)

## RAG LLM Optimization 

### Phase 1: Pairwaise Pairs Dataset Generation for Training Reward Model

<p align="center">
<img src="figs/Screenshot from 2025-08-26 23-52-56.png" alt="RAG pipeline workflow" width="800"/>
</p>

This phase generates pairwise preference data to train a reward model for instruction-following or code-related tasks. 

For each query in the dataset, the process first retrieves relevant context from a precomputed FAISS vectorstore. 

A structured prompt is constructed combining the retrieved context and the query, which is then passed to a generative language model (Mistral-7B-Instruct) to produce multiple candidate answers. 

> [_Generation of multiple candidates_](https://github.com/AftabHussain/catboost-code-rag/blob/77d0b5e9de43ecb25a2ab248101ae6f0b4d95026/RL_build_pairwise_prefs.py#L97)

Each candidate is scored using a heuristic ranking system that combines: (1) similarity to the retrieved context (“grounding score”), (2) coverage of task-relevant keywords, and (3) a mild length penalty to discourage overly verbose answers. 

> [_Candidate scoring function_](https://github.com/AftabHussain/catboost-code-rag/blob/77d0b5e9de43ecb25a2ab248101ae6f0b4d95026/RL_build_pairwise_prefs.py#L192)

The top-scoring candidate is marked as “chosen” and the lowest-scoring candidate as “rejected,” forming a pair. These prompt–chosen–rejected triples are saved in a JSONL file (pairwise_prefs.jsonl) and provide training data for reward models that can later guide preference-aligned generation. 

This approach ensures that the reward model learns to prefer outputs that are both contextually grounded and relevant to the task.

[**See more details of the implementation of this phase...**](./01-pairwise-pref-generation.pdf)

### Phase 2: Training a Pairwise Reward Model

In this stage, a Reward Model (RM) is trained using the preference pairs generated earlier. 

The dataset consists of triplets: a prompt, a “chosen” answer (preferred), and a “rejected” answer (less preferred). 

A pretrained base encoder (e.g., bert-base-uncased) is fine-tuned to assign a scalar reward score to each answer. 

> [_Model and data selection_](https://github.com/AftabHussain/catboost-code-rag/blob/5d921ef669bf5a1c125f95d01f998ce7ccacfda5/RL_train_reward_model_pairwise.py#L20-L21)

Training uses a pairwise loss function of the form `-log σ(r_chosen − r_rejected)`, which encourages the model to give higher scores to preferred answers compared to rejected ones. This setup aligns the model’s scoring function with human-like or heuristic preferences. 

> [_Training loop_](https://github.com/AftabHussain/catboost-code-rag/blob/5d921ef669bf5a1c125f95d01f998ce7ccacfda5/RL_train_reward_model_pairwise.py#L99-L119)

The process includes splitting data into training and validation sets, optimizing with AdamW, and monitoring both loss and validation accuracy. After each epoch, checkpoints are saved, and a log file tracks progress. 

> [_Validation_](https://github.com/AftabHussain/catboost-code-rag/blob/5d921ef669bf5a1c125f95d01f998ce7ccacfda5/RL_train_reward_model_pairwise.py#L124-L144)

The trained reward model becomes a crucial evaluator for reinforcement learning or direct preference optimization steps that follow.

[**See more details of the implementation of this phase...**](./02-pairwise-reward-model-training.pdf)

### Phase 3: Fine-tuning the RAG Generator Model (LLM) with PPO and Feedback from Reward Model

<p align="center">
<img src="figs/Screenshot from 2025-08-27 00-04-53.png" alt="RAG pipeline workflow" width="800"/>
</p>

In this phase, the policy model (Mistral-7B-Instruct), the main RAG model we want to optimize, is fine-tuned using Proximal Policy Optimization (PPO) with guidance from the reward model trained on synthetic preference data. 

In addition, during training, a frozen reference model is maintained to constrain policy updates and prevent instability. 

> [_Instantiation of the Policy Model (The RAG Model to be optimized)_](https://github.com/AftabHussain/catboost-code-rag/blob/7b0a07121535f8f52245944c1c44d062b389ad8a/RL_ppo_train_with_reward.py#L18-L35)
>
> [_Instantiation of the Reference Model (To be used as a frozen copy of the above model)_](https://github.com/AftabHussain/catboost-code-rag/blob/7b0a07121535f8f52245944c1c44d062b389ad8a/RL_ppo_train_with_reward.py#L46-L53)
>
> [_Instantiation of the Reward Model_](https://github.com/AftabHussain/catboost-code-rag/blob/7b0a07121535f8f52245944c1c44d062b389ad8a/RL_ppo_train_with_reward.py#L73-L78)

Instead of relying on direct human annotations, the system uses heuristic-based rankings (context grounding, keyword coverage, and length penalty) to generate “chosen vs. rejected” pairs. These pairs allow the reward model to provide scalar rewards for policy outputs. 

The pipeline samples prompts, generates candidate responses from the policy, scores them with the reward model, and updates the policy to maximize expected reward while staying close to the reference. 

In this approach, we utilize Reinforcement Learning with AI Feedback (RLAIF), where synthetic preferences stand in for human judgments, enabling scalable alignment without manual labeling.

[**See more details of the implementation of this phase...**](./03-ppo-training-with-reward-model.pdf)
