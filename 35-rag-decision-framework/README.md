# 35 RAG Decision Framework

## Objective

Provide a unified, rigorous decision framework for selecting vector databases, embedding models, chunking strategies, retrieval patterns, and synthesis engines for any enterprise workload.

## Why It Matters

Teams often default to hype-driven architecture (e.g. building an autonomous Graph Agent when a 256-token chunked dense vector search would suffice). This decision matrix matches workload traits to the minimum effective complexity.

## Core Concepts

- **Minimum Effective Complexity**: Starting with simple dense vector search and ascending to Hybrid, Reranking, CRAG, or Graph RAG only when empirical evaluation fails.
- **RAG vs Fine-Tuning**: RAG for dynamic external facts and provenance; fine-tuning for linguistic style, tone, and narrow classification format.
- **Decision Trees**: Deterministic pathways mapping business requirements to proven RAG patterns.

## Architecture

```
[Workload Profile Requirements] ──> [Architectural Decision Engine] ──> [Optimal Recommended Pattern]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [decision_engine.py](implementation/decision_engine.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Prevents over-engineering and keeps infrastructure costs lean.
- **Weaknesses**: Requires accurately profiling workload data characteristics upfront.

## Architectural Decision & Trade-off Questions

1. *How do you decide between Fine-Tuning, Prompt Engineering, and RAG for a customer service domain?*
2. *When does a system transition from Simple RAG to Graph RAG or Multi-Agent RAG?*
