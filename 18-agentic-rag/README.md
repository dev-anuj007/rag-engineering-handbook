# 18 Agentic RAG

## Objective

Build autonomous agentic RAG architectures using LlamaIndex `RouterQueryEngine`, ReAct reasoning agents, sub-question query engines, and multi-tool orchestration over segregated ChromaDB collections.

## Why It Matters

Real-world enterprise systems contain heterogeneous data stores (financial PDFs, tech specs, API docs, relational DBs). A single monolithic vector collection creates cross-domain noise. Agentic routers dynamically direct queries to the optimal specialized engine.

## Core Concepts

- **Router Query Engine**: Inspects query semantics and routes traffic to the single best tool or splits multi-part questions across multiple sub-engines.
- **Sub-Question Decomposition**: Breaks complex comparative questions ("Compare Q3 AWS spend vs GCP spend") into discrete sub-queries.
- **ReAct Tool-Calling Loops**: Iterative Thought -> Action -> Observation reasoning loops.

## Architecture

```
[Query] ──> [LLM Router Selector] ──┬── [Finance Tool] ──> [ChromaDB Finance Coll]
                                    └── [Tech Tool]    ──> [ChromaDB Tech Coll]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [agentic_router.py](implementation/agentic_router.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: True domain specialization, handles complex multi-hop queries.
- **Weaknesses**: Multi-step LLM routing calls increase latency.

## Architectural Decision & Trade-off Questions

1. *When should you use deterministic rule-based semantic routing vs LLM-based agent tool calling?*
2. *How do you prevent infinite execution loops in autonomous ReAct RAG agents?*
