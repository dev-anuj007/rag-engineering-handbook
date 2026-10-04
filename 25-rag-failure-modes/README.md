# 25 RAG Failure Modes

## Objective

Catalogue, diagnose, and remediate the 7 canonical RAG failure modes (Missing Content, Missed Top-K, Not in Context, Not Extracted, Wrong Format, Incorrect Specificity, Incomplete Synthesis) with automated diagnostic decision trees.

## Why It Matters

When an end-user reports "The AI gave the wrong answer," engineering teams waste days blaming the LLM when 80% of issues stem from ingestion omissions, chunk truncation, or reranker cutoff errors.

## Core Concepts

- **The 7 Failure Modes**: A rigorous taxonomy isolating the exact pipeline stage where information was lost.
- **Diagnostic Triage**: Automated step-by-step verification isolating corpus presence, retrieval rank, prompt packing, and LLM reasoning.
- **Mitigation Playbook**: Tailored architectural interventions for each specific failure mode.

## Architecture

```
[Failed Response] ──> [Root Cause Diagnostic Tree] ──> [Identified Failure Mode] ──> [Targeted Remediation]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [failure_analyzer.py](implementation/failure_analyzer.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Drastically accelerates mean time to resolution (MTTR) for customer-reported bugs.
- **Weaknesses**: Requires telemetry tracing throughout all ingestion and retrieval intermediate layers.

## Architectural Decision & Trade-off Questions

1. *Walk me through your debugging methodology when a customer reports an incomplete answer on a complex multi-part policy query.*
2. *How do you prevent 'Incorrect Specificity' failures where the system retrieves high-level overviews when the user needed a specific config flag?*
