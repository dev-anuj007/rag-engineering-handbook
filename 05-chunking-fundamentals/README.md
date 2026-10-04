# 05 Chunking Fundamentals

## Objective

Master foundational text chunking techniques, token vs character splitting geometries, overlap strategies, and their downstream impact on ChromaDB vector retrieval.

## Why It Matters

Arbitrary character slicing cuts sentences in half, severing semantic predicates and ruining embedding fidelity. Proper sentence-aware chunking preserves coherent propositions.

## Core Concepts

- **Token vs Character Splitting**: Token-based chunking aligns directly with LLM context windows, while character splitting is model-agnostic.
- **Overlap Math**: Retaining 10–20% token overlap prevents loss of context across split boundaries.
- **Sentence-Aware Boundaries**: Respecting punctuation stops prevents fragmented clauses.

## Architecture

```
[Raw Document] ──> [Sentence Tokenizer] ──> [Sliding Overlap Window] ──> [ChromaDB Vectors]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [chunker.py](implementation/chunker.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Small Chunks (128-256 tokens)**: High retrieval precision, low context noise, but risk losing broader thematic context.
- **Large Chunks (512-1024 tokens)**: Rich context, but introduce irrelevant noise into the retrieval ranking.

## Architectural Decision & Trade-off Questions

1. *How does chunk size inversely correlate with retrieval precision and generation groundedness?*
2. *How do you choose between sentence-level chunking and fixed-token sliding windows for code vs narrative text?*
