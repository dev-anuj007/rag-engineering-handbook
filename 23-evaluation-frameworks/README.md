# 23 Evaluation Frameworks

## Objective

Compare, integrate, and adapt prominent RAG evaluation frameworks (LlamaIndex Evaluators, Ragas, DeepEval, TruLens) behind a unified, SOLID-compliant adapter layer storing telemetry in SQLite.

## Why It Matters

Different evaluation tools define "faithfulness" and "context precision" with slight mathematical nuances. Encapsulating frameworks behind a common interface prevents vendor lock-in and facilitates cross-framework validation.

## Core Concepts

- **Framework Normalization**: Unifying metric schemas across LlamaIndex, Ragas, and DeepEval.
- **Pluggable Evaluation Adapters**: Runtime switching of metric evaluators.
- **Centralized Evaluation Telemetry**: Persisting cross-framework evaluation runs in SQLite.

## Architecture

```
[RAG Execution Payload] ──> [IEvalFrameworkAdapter] ──> [LlamaIndex / Ragas] ──> [SQLite Report Store]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [framework_adapter.py](implementation/framework_adapter.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Eliminates framework lock-in, unifies telemetry reporting.
- **Weaknesses**: Slight overhead in translating abstract data classes to framework formats.

## Architectural Decision & Trade-off Questions

1. *How do you choose between embedding-based similarity metrics (e.g. BERTScore) and LLM-based NLI evaluation in CI pipelines?*
2. *How do you integrate automated evaluation gates into GitHub Actions / GitLab CI to prevent regressions in production RAG systems?*
