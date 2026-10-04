# 03 Document Ingestion

## Objective

Build a robust, deduplicated document ingestion subsystem that normalizes metadata, prevents duplicate embeddings in ChromaDB, and verifies ingestion accuracy against SQLite benchmarks.

## Why It Matters

Ingesting redundant or corrupt documents pollutes the vector index, skews similarity calculations, inflates hosting costs, and introduces duplicate context during LLM generation.

## Core Concepts

- **Cryptographic Content Hashing**: SHA-256 fingerprinting ensures idempotency across ingestion batches.
- **Deduplication Store**: Decoupled registry tracking ingested documents before vector computation.
- **ChromaDB Vector Store**: Structured vector insertion with rich metadata enrichment.

## Architecture

```
[Raw Sources] ──> [SHA-256 Fingerprint] ──> [Deduplication Check] ──> [ChromaDB Vector Store]
                                                   │ (If Duplicate)
                                                   └──> [Skipped]
```

See [architecture.md](architecture/architecture.md) for full design.

## Implementation

The ingestion service is implemented in [ingestion_manager.py](implementation/ingestion_manager.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Zero duplicate embedding generation, constant-time deduplication lookups.
- **Weaknesses**: Requires persistent hash state storage for enterprise scale.
- **Trade-offs**: Memory overhead of state store vs re-embedding compute cost.

## Failure Modes & Mitigations

- **Near-duplicate docs with minute formatting variations**: Mitigated by text normalization prior to hashing or locality-sensitive hashing (MinHash).

## Architectural Decision & Trade-off Questions

1. *How do you handle real-time vs batch ingestion when syncing millions of enterprise documents to a vector store?*
2. *What is the impact of document deletion and how do you achieve tombstoning and sync in ChromaDB?*
