## Key Methodology Components

### The Base RAG Pipeline for Main Use Case Scenario

This project implements a Retrieval-Augmented Generation (RAG) pipeline for answering domain-specific questions using a large language model (LLM). It combines a precomputed FAISS vectorstore of embeddings with a Generator Model (i.e., the LLM) to provide accurate and context-aware responses. When a user submits a query, the pipeline first retrieves the most relevant context from the vectorstore using semantic similarity. This context is then inserted into a structured instruction-style prompt, which is fed to the LLM to generate an answer. The system parses the output into context, question, and answer components and logs each interaction in a JSON dataset for future reference. This approach allows efficient querying over large datasets.

<p align="center">
<img src="figs/Screenshot%20from%202025-08-26%2023-42-58.png" alt="RAG pipeline workflow" width="600"/>
</p>

### Pairwaise Pairs Dataset Generation for Training Reward Model

This phase generates pairwise preference data to train a reward model for instruction-following or code-related tasks. For each query in the dataset, the process first retrieves relevant context from a precomputed FAISS vectorstore. A structured prompt is constructed combining the retrieved context and the query, which is then passed to a generative language model (Mistral-7B-Instruct) to produce multiple candidate answers. Each candidate is scored using a heuristic ranking system that combines: (1) similarity to the retrieved context (“grounding score”), (2) coverage of task-relevant keywords, and (3) a mild length penalty to discourage overly verbose answers. The top-scoring candidate is marked as “chosen” and the lowest-scoring candidate as “rejected,” forming a pair. These prompt–chosen–rejected triples are saved in a JSONL file (pairwise_prefs.jsonl) and provide training data for reward models that can later guide preference-aligned generation. This approach ensures that the reward model learns to prefer outputs that are both contextually grounded and relevant to the task.



<p align="center">
<img src="figs/Screenshot from 2025-08-26 23-52-56.png" alt="RAG pipeline workflow" width="700"/>
</p>
