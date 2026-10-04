# 14 Reranking

## Objective

Implement a high-precision two-stage retrieval architecture combining fast bi-encoder retrieval (ChromaDB) with cross-encoder / fine-grained rerankers.

## Why It Matters

Bi-encoders generate embeddings for query and passage separately, missing fine-grained token-level cross-attention interactions. Cross-encoders evaluate full query-document attention simultaneously, yielding a 15–30% boost in NDCG@10.

## Core Concepts

- **Bi-Encoder vs Cross-Encoder**: Bi-encoders trade attention depth for offline pre-computation; cross-encoders provide maximum semantic precision on candidate subsets.
- **Two-Stage Retrieval Funnel**: Coarse fetch (top-50) followed by fine neural rerank (top-3).
- **Latency vs Precision Trade-off**: Allocating ~20ms rerank compute budget per query.

## Architecture

```
[Query] ──> [ChromaDB Bi-Encoder (Top-50)] ──> [Cross-Encoder Reranker] ──> [Top-3 Grounded Nodes]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [reranker.py](implementation/reranker.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Drastically lifts top-1 precision, suppressing false positives.
- **Weaknesses**: Adds a second compute hop with GPU/CPU inference overhead.

## Architectural Decision & Trade-off Questions

1. *Why can't we run cross-encoder scoring across all 10M documents in a database directly?*
2. *How does ColBERT's late-interaction mechanism bridge the latency gap between bi-encoders and full cross-encoders?*
