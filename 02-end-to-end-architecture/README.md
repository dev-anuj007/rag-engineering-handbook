# 02 End-to-End Architecture

## Objective

Establish a clean, decoupled, production-grade End-to-End RAG architecture based on SOLID principles and clean architecture boundaries, separating data ingestion, vector indexing (ChromaDB), similarity retrieval, and grounded response synthesis.

## Why It Matters

Production RAG systems suffer from architectural coupling when retrieval mechanisms or model providers are hard-coded into ingestion loops. Decoupling ensures testability, zero-downtime model migration, and deterministic evaluation against golden datasets stored in SQLite.

## Core Concepts

- **Clean Decoupled Subsystems**: Independent Ingestion, Indexing, Vector Store, Retrieval, and Generation interfaces.
- **Persistent Vector Store (ChromaDB)**: Isolated index collections for zero data corruption and scalable vector search.
- **Golden Test Bench (SQLite)**: Verifiable query-answer pairs benchmarked against end-to-end pipeline outputs.

## Architecture

```
[Documents] ──> [Ingestion Engine] ──> [Indexing Engine] ──> [ChromaDB]
                                                                  │
[User Query] ──────────────────────────> [Retrieval Engine] <─────┘
                                                 │
                                         [Top-K Context]
                                                 │
                                                 ▼
[Synthesized Output] <────────────────── [Generation Engine]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

The complete implementation is structured under [pipeline.py](implementation/pipeline.py) using pure abstract protocols.

## Evaluation

Evaluation runs against SQLite golden test sets in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Pluggable vector databases, modular unit testing, clear boundary segregation.
- **Weaknesses**: Slight abstraction overhead compared to monolithic one-liner scripts.
- **Trade-offs**: Flexibility vs minimal line count.

## Failure Modes & Mitigations

- **Empty Retrieval**: Handled via explicit fallback verification before calling the LLM.
- **Context Overflows**: Chunk overlap and top-k bounding prevent unbounded context injection.

## Key Takeaways

1. Never leak vector database clients directly into user-facing controllers.
2. Maintain strict interface boundaries for ingestion, retrieval, and generation.
3. Keep golden evaluation decoupled from live query execution pipelines.

## Architectural Decision & Trade-off Questions

1. *How would you architect a RAG pipeline that can switch embedding models or vector databases without taking down live query traffic?*
2. *How do you isolate retrieval failures from generation failures in end-to-end latency degradation?*
