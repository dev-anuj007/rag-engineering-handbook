# 28 Performance Optimization

## Objective

Maximize RAG throughput and minimize end-to-end latency via async query concurrency, batch embedding generation, and vector index tuning in ChromaDB.

## Why It Matters

Ingesting millions of documents sequentially takes days. Batching embedding requests into 64-item chunks with asyncio workers achieves up to a 20x speedup, and concurrent retrieval pipelines scale QPS linearly across read replicas.

## Core Concepts

- **Batch Ingestion**: Grouping raw texts to fully saturate embedding API quotas and SIMD matrix multipliers.
- **Asynchronous Concurrent Retrieval**: Non-blocking I/O dispatching multiple query branches in parallel.
- **Quantization**: Scalar quantization (SQ8) compressing 32-bit floats into 8-bit integers for a 4x RAM reduction.

## Architecture

```
[Document Stream] ──> [Batch Chunk Queue (Size=64)] ──> [Async Concurrent Ingestion] ──> [ChromaDB Index]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [async_batch_optimizer.py](implementation/async_batch_optimizer.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Drastic reduction in pipeline runtime and hosting costs.
- **Weaknesses**: Requires careful tuning of worker thread pool sizes to avoid API rate limiting.

## Architectural Decision & Trade-off Questions

1. *How do you size the async batch worker pool against external rate limits and upstream database thread connection pools?*
2. *What is the recall loss when applying Product Quantization (PQ) vs Scalar Quantization (SQ8) to 1536-dimensional embeddings?*
