# 13 Advanced Retrieval

## Objective

Combine dense semantic vector search (ChromaDB) with lexical sparse indexing (BM25) using Reciprocal Rank Fusion (RRF) to eliminate both vocabulary mismatch and semantic drift.

## Why It Matters

Dense embeddings excel at conceptual semantic queries but fail miserably on exact alphanumeric part numbers, error codes (`ERR_504`), and proper nouns. Hybrid retrieval guarantees high recall across both query types.

## Core Concepts

- **Sparse vs Dense Complementarity**: BM25 captures exact token overlaps; dense embeddings capture synonymous semantics.
- **Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$ blends disparate score distributions without needing manual calibration.
- **Ensemble Retrieval**: Multi-branch querying executed concurrently.

## Architecture

```
[User Query] ──┬──> [BM25 Inverted Index] ─────┐
               └──> [ChromaDB Dense Index] ────┴──> [RRF Fusion Engine] ──> [Top-K Context]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [hybrid_retriever.py](implementation/hybrid_retriever.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Robust across both code/error identifiers and natural language conversational queries.
- **Weaknesses**: Requires maintaining two separate index stores (inverted text index + vector index).

## Architectural Decision & Trade-off Questions

1. *Why is raw score normalization (e.g. Min-Max) problematic when fusing BM25 scores (unbounded) with Cosine similarity ($[-1, 1]$)?*
2. *How does tuning the constant $k$ in RRF (typically $k=60$) govern the weight given to top-ranked outliers?*
