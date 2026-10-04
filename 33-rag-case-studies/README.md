# 33 RAG Case Studies

## Objective

Analyze real-world domain adaptations of RAG systems across Legal discovery, Financial 10-K analysis, Medical guideline synthesis, and Software Engineering codebase QA.

## Why It Matters

One size does not fit all in RAG design. A chunking and embedding configuration optimized for narrative product FAQs will catastrophically fail when applied to dense contractual indemnity clauses or tabular financial balance sheets.

## Core Concepts

- **Legal Discovery RAG**: Requires exact verbatim clause retrieval, strict negation handling, and zero hallucination tolerance.
- **Financial 10-K RAG**: Requires table structure preservation and numerical grounding.
- **Codebase RAG**: Leverages Abstract Syntax Tree (AST) parsing and graph call-graph linking.

## Architecture

```
[Domain Inputs] ──> [Domain-Tailored Ingestion] ──> [ChromaDB Domain Index] ──> [Specialized Synthesizer]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [case_study_runner.py](implementation/case_study_runner.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Proven domain heuristics that double retrieval precision in vertical markets.
- **Weaknesses**: Requires domain-specific preprocessors and custom evaluation golden sets.

## Architectural Decision & Trade-off Questions

1. *How would you architect an SEC 10-K comparative financial analyst RAG system comparing 50 companies across 5 fiscal years?*
2. *What chunking strategy best preserves cross-file function definitions and class inheritance in a multi-repo code search system?*
