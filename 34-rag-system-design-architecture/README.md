# 34 RAG System Design & Sizing Architecture

## Objective

Master the standardized 4-step framework for acing Staff/Principal level RAG system design architecture: Requirements & Sizing, High-Level Architecture, Deep-Dive Components, and Failure Bottlenecks.

## Why It Matters

Staff engineers are evaluated on their ability to perform back-of-the-envelope capacity estimations, calculate vector RAM requirements, justify architectural trade-offs, and handle edge-case failure modes under strict latency SLAs.

## Core Concepts

- **Capacity Estimation Math**: Calculating raw vector storage + HNSW graph overhead ($1.5 \times \text{dimension} \times 4 \text{ bytes}$).
- **Latency Budget Allocation**: Partitioning 500ms SLA across query embedding (30ms), ANN search (15ms), reranker (40ms), and LLM generation (350ms).
- **Trade-Off Rubrics**: Vector search recall vs latency vs index RAM cost.

## Architecture

```
[Workload Sizing & Specs] ──> [Capacity Calculations] ──> [High-Level Architecture] ──> [Component Deep-Dive]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [sizing_calculator.py](implementation/sizing_calculator.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Structured, deterministic sizing calculations covering both storage and LLM token costs.
- **Weaknesses**: Real-world caching dynamics will dynamically alter actual peak memory loads.

## Architectural Decision & Trade-off Questions

1. *Design an enterprise RAG system for 100M internal Confluence pages serving 10,000 peak QPS with a 300ms p95 latency budget.*
2. *How do you size the hardware cluster for 50M 1536-dimensional embeddings indexed via HNSW vs IVF-PQ?*
