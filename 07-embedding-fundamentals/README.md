# 07 Embedding Fundamentals

## Objective

Master vector space geometry, high-dimensional dense representations, distance metrics (Cosine vs L2 vs Dot Product), and their native indexing in ChromaDB collections.

## Why It Matters

Mismatched distance metrics (e.g. indexing with L2 but querying assuming Cosine) produce inverted or erratic retrieval rankings, destroying RAG relevance.

## Core Concepts

- **High-Dimensional Geometry**: Dense float32/float16 embeddings map discrete semantic concepts into continuous manifold topologies.
- **Metric Selection**: Cosine similarity measures orientation independent of length; Dot product rewards magnitude when vectors are unnormalized.
- **ChromaDB HNSW Metric Configuration**: Specifying `hnsw:space: cosine` at collection creation.

## Architecture

```
[Text Node] ──> [Embedding Model (768d)] ──> [Vector Normalization] ──> [ChromaDB HNSW Index]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [vector_math.py](implementation/vector_math.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Cosine**: Invariant to document length variations.
- **Dot Product**: Sub-millisecond compute via SIMD hardware acceleration when normalized.

## Architectural Decision & Trade-off Questions

1. *Why does L2 distance equal $2(1 - \text{Cosine})$ when embeddings are unit-normalized ($L_2 = 1$)?*
2. *What is the curse of dimensionality and how does it degrade distance discrimination in $>1536$ dimensions?*
