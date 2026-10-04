# 20 Retrieval Evaluation

## Objective

Master Information Retrieval (IR) metrics—Hit Rate@K, Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and NDCG@K—to objectively evaluate ChromaDB retrieval quality against SQLite golden sets.

## Why It Matters

Without formal IR metrics, teams tune chunk sizes and embedding models based on subjective "vibes." Tracking MRR and NDCG ensures that search upgrades quantifiably improve relevant document positioning.

## Core Concepts

- **Hit Rate@K**: Proportion of test queries where at least one ground-truth document is returned in the top $K$.
- **Mean Reciprocal Rank (MRR)**: Evaluates the exact ranking position of the first relevant result.
- **NDCG@K**: Accounts for graded relevance levels and penalizes relevant documents appearing deep in the ranked list.

## Architecture

```
[SQLite Golden Query Set] ──> [ChromaDB Search (Top-K)] ──> [IR Metrics Calculator] ──> [MRR, NDCG@K, HitRate]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [retrieval_metrics.py](implementation/retrieval_metrics.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Hit Rate**: Fast, intuitive, binary indicator.
- **NDCG@K**: Most comprehensive for multi-document graded relevance ranking.

## Architectural Decision & Trade-off Questions

1. *Why is MRR preferred for single-answer factoid QA while NDCG is preferred for multi-document synthesis tasks?*
2. *How do you automatically generate a high-quality retrieval evaluation dataset without manual human labeling?*
