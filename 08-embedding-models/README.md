# 08 Embedding Models

## Objective

Evaluate bi-encoder embedding models, asymmetric retrieval task prefixes, and Matryoshka dimension truncation for memory-efficient ChromaDB indexing.

## Why It Matters

Symmetric embedding treats questions and answers identically in vector space. Asymmetric models (e.g. Gemini, E5) use task instructions (`search_query:` vs `search_document:`) to align brief questions with expansive explanatory passages.

## Core Concepts

- **Asymmetric Prompt Prefixes**: Differentiates short interrogatives from descriptive paragraphs.
- **Matryoshka Representation Learning (MRL)**: Preserves up to 98% retrieval quality while slicing vector dimensions from 1536/768 down to 256 for radical RAM savings.
- **ChromaDB Vector Store Integration**: Indexing multi-task embeddings with metric consistency.

## Architecture

```
[Document Chunk] ──> ["search_document: "] ──> [Bi-Encoder] ──> [ChromaDB HNSW Index]
[User Query]     ──> ["search_query: "]    ──> [Bi-Encoder] ──> [Top-K Nearest Neighbor]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [embedding_providers.py](implementation/embedding_providers.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Drastic reduction in false positive matches across asymmetric text lengths.
- **Weaknesses**: Must strictly format prefixes consistently between ingestion and search.

## Architectural Decision & Trade-off Questions

1. *What is the architectural difference between a cross-encoder and a bi-encoder during real-time retrieval?*
2. *How does Matryoshka Representation Learning enable multi-stage vector search (fast coarse search on 128d, fine rerank on 768d)?*
