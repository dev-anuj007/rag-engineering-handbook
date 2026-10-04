# 06 Advanced Chunking

## Objective

Implement semantic boundary splitting, markdown header extraction, and multi-tier hierarchical parent-child chunking for high-precision retrieval with broad context synthesis.

## Why It Matters

Fixed-size chunking forces a trade-off: small chunks retrieve precisely but lack surrounding context for generation, while large chunks dilute semantic search vectors. Hierarchical chunking indexes small leaf chunks but resolves to larger parent context during synthesis.

## Core Concepts

- **Hierarchical Parent-Child Nodes**: Index leaf nodes in ChromaDB; pass parent nodes to the LLM.
- **Markdown & Structural Chunking**: Parse headers (#, ##, ###) into logically scoped sections.
- **Semantic Chunking**: Split when adjacent sentence embedding cosine similarity drops below a dynamic percentile threshold.

## Architecture

```
[Markdown / Document] ──> [Hierarchical Splitter] ──> [Leaves (Search)] ──> [ChromaDB]
                                                 └──> [Parents (Context LLM)]
```

See [architecture.md](architecture/architecture.md) for full design.

## Implementation

Implemented in [advanced_chunker.py](implementation/advanced_chunker.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Highest signal-to-noise ratio in search; zero context fragmentation.
- **Weaknesses**: Requires more complex storage topology (vector store + doc store lookup).

## Architectural Decision & Trade-off Questions

1. *How does Parent-Child retrieval resolve the 'Lost in the Middle' phenomenon during multi-chunk synthesis?*
2. *How do you compute dynamic threshold cutoffs in semantic embedding-based chunking?*
