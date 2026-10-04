# 21 Generation Evaluation

## Objective

Quantify the factual correctness, faithfulness (hallucination rate), and question relevance of LLM-generated synthesis against SQLite evaluation test suites.

## Why It Matters

An LLM can generate fluent, persuasive, but completely fabricated answers (hallucinations). Generation evaluation algorithms break answers into atomic claims and verify entailment against retrieved context.

## Core Concepts

- **Faithfulness (Groundedness)**: Proportion of claims in the generated response that can be mathematically or logically deduced from the provided context.
- **Answer Relevance**: Measure of whether the generated response directly answers the user's inquiry without extraneous digressions.
- **Hallucination Detection**: Automated alerting on ungrounded assertions.

## Architecture

```
[Answer] ──> [Atomic Claim Splitter] ──> [Context Entailment Verifier] ──> [Faithfulness Score %]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [generation_metrics.py](implementation/generation_metrics.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Detects subtle parametric hallucinations missed by string matching.
- **Weaknesses**: Claim extraction with LLMs incurs additional inference cost during test runs.

## Architectural Decision & Trade-off Questions

1. *How do you evaluate generation quality when no ground truth golden answers exist in production?*
2. *How do you isolate whether a hallucination was caused by conflicting retrieved context vs model parametric memory drift?*
