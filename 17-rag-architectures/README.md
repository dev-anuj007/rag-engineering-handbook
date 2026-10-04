# 17 RAG Architectures

## Objective

Master advanced RAG architectural paradigms: Corrective RAG (CRAG), Self-RAG (self-reflection and hallucination grading), Modular RAG, and Adaptive RAG.

## Why It Matters

Standard linear RAG operates blindly: it executes retrieval and forces generation regardless of context quality. Corrective RAG dynamically self-grades retrieved nodes and triggers fallbacks or query rewriting when retrieval fails.

## Core Concepts

- **Corrective RAG (CRAG)**: Evaluates retrieval quality dynamically and branches between internal synthesis, ambiguous warning flags, or external web search.
- **Self-RAG**: Uses reflection critique tokens `[RETRIEVE]`, `[IS_REL]`, `[IS_SUP]` to self-correct during generation.
- **Modular RAG**: Decoupled, interchangeable routing, indexing, and synthesis micro-modules.

## Architecture

```
[Query] ──> [ChromaDB Retrieval] ──> [Grader] ──┬── [CORRECT]   ──> [LLM Synthesis]
                                                ├── [AMBIGUOUS] ──> [Cautious Synthesis]
                                                └── [INCORRECT] ──> [External Fallback]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [crag_pipeline.py](implementation/crag_pipeline.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Robust protection against hallucinations on out-of-domain queries.
- **Weaknesses**: Grader step introduces additional intermediate latency.

## Architectural Decision & Trade-off Questions

1. *How does Self-RAG differ from Corrective RAG in execution topology and model requirement?*
2. *How do you build an Adaptive RAG router that classifies query complexity to choose between No-RAG, Simple RAG, and Multi-Hop RAG?*
