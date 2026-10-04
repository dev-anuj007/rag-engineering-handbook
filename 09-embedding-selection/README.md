# 09 Embedding Selection

## Objective

Formulate an objective decision framework for selecting embedding models based on MTEB retrieval scores, domain adaptation, latency budgets, and ChromaDB vector footprint constraints.

## Why It Matters

Using an oversized 4096-dimension general embedding model for low-latency code search wastes vector RAM and adds unnecessary network latency, whereas small models fail on nuanced legal or biomedical terminology.

## Core Concepts

- **MTEB Benchmark Suite**: Evaluating models specifically on Retrieval and Reranking tasks rather than classification averages.
- **Latency vs Dimension Trade-off**: 768d vs 1536d vectors scale storage RAM linearly.
- **Domain Specialization**: Code, legal, or multilingual fine-tuned models vs massive generalized bi-encoders.

## Architecture

```
[System Constraints (Latency, Dim, Domain)] ──> [Model Selector] ──> [ChromaDB Indexing Engine]
```

See [architecture.md](architecture/architecture.md) for full design.

## Implementation

Implemented in [model_selector.py](implementation/model_selector.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Deterministic, SLA-compliant model selection matching enterprise workload criteria.
- **Weaknesses**: Requires empirical benchmarking on internal proprietary corpora.

## Architectural Decision & Trade-off Questions

1. *How would you establish an empirical offline benchmark to compare Gemini Embeddings vs BGE-Large on internal company docs?*
2. *When does fine-tuning a small open-source embedding model beat using commercial frontier embedding APIs?*
