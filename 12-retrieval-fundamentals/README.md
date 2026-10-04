# 12 Retrieval Fundamentals

## Objective

Master dense vector retrieval, similarity score normalization, top-k candidate selection, and relevance score thresholding in ChromaDB collections.

## Why It Matters

Retrieving fixed top-$K$ chunks unconditionally injects irrelevant distractor nodes when queries have zero relevant documents in the corpus, causing LLM hallucinations. Score thresholding acts as a critical relevance gate.

## Core Concepts

- **Cosine Distance vs Similarity**: Converting distance metrics into intuitive similarity bounds $[0, 1]$.
- **Top-K Tuning**: Finding the sweet spot between retrieval recall and LLM context bloat.
- **Score Threshold Cutoffs**: Discarding low-confidence candidate chunks.

## Architecture

```
[Query] ──> [Vector Embedding] ──> [ChromaDB HNSW Top-K] ──> [Score Threshold Gate] ──> [Context Nodes]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [dense_retriever.py](implementation/dense_retriever.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Ultra-fast, intuitive, native vector database execution.
- **Weaknesses**: Vulnerable to out-of-vocabulary acronyms and exact keyword mismatches.

## Architectural Decision & Trade-off Questions

1. *How do you dynamically calibrate the score threshold across different domains without artificially dropping valid edge-case queries?*
2. *What is the relationship between retrieval Hit Rate@K and downstream generation hallucination rates?*
