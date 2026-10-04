# 10 Vector Databases

## Objective

Deeply understand vector database internals, HNSW approximate nearest neighbor graphs, parameter tuning ($M$, $efConstruction$, $efSearch$), and ChromaDB lifecycle management.

## Why It Matters

Linear vector scans scale at $O(N)$ with dataset size, causing catastrophic latency spikes beyond 50k vectors. HNSW graph indexing provides sub-linear $O(\log N)$ retrieval latency with $>95\%$ recall.

## Core Concepts

- **HNSW Skip-List Multilayer Graphs**: Multi-tier routing allows logarithmic convergence to nearest vector clusters.
- **Index Parameter Tuning**: Balancing $M$ and $ef$ for memory consumption vs query QPS.
- **ChromaDB CRUD & Metadata Indexing**: Real-time vector inserts, metadata-filtered searches, and targeted deletions.

## Architecture

```
[Document Vectors] ──> [HNSW Graph Builder] ──> [ChromaDB Layer 0..L Storage Engine]
                                                       │
[Query Vector] ──────> [Greedy Routing (efSearch)] ───► [Top-K Result Nodes]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [chroma_db_store.py](implementation/chroma_db_store.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: True sub-10ms nearest neighbor search over millions of records.
- **Weaknesses**: Significant memory overhead (approx. 1.5–2x raw vector size for graph edges).

## Architectural Decision & Trade-off Questions

1. *How does HNSW compare with IVFFlat (Inverted File Index) in terms of indexing throughput, memory, and recall?*
2. *How do you handle live deletion and tombstoning in HNSW graphs without triggering costly global re-indexing?*
