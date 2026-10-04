# 19 Evaluation Fundamentals

## Objective

Establish rigorous, scientific evaluation methodology for RAG systems, isolating retrieval failures from generation failures via component-wise benchmarks stored in SQLite.

## Why It Matters

Evaluating only end-to-end user answers conceals whether an incorrect response stemmed from missing retrieved context (retrieval failure) or an unfaithful LLM output (hallucination).

## Core Concepts

- **Evaluation Triad**: Evaluating Context Relevance, Groundedness (Faithfulness), and Answer Relevance.
- **Component-Wise Isolation**: Benchmarking the retrieval engine independently from LLM synthesis.
- **SQLite Evaluation Bench**: Persistent regression suites tracking metric regressions across git commits.

## Architecture

```
[Golden Test Set] ──┬──> [Retrieval Metrics Engine] ──> [Hit Rate@K, MRR, NDCG]
                    └──> [Generation Metrics Engine] ──> [Faithfulness, Relevance]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [metrics_engine.py](implementation/metrics_engine.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Pinpoints root causes of system degradation with surgical precision.
- **Weaknesses**: Requires curated golden datasets with ground truth context spans.

## Architectural Decision & Trade-off Questions

1. *How do you build an automated CI/CD regression test suite for a production RAG system?*
2. *What is the difference between Reference-Based evaluation (requires golden answers) and Reference-Free evaluation (LLM-as-a-judge on source context)?*
