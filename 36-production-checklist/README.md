# 36 Production Checklist

## Objective

Provide the exhaustive, production-grade verification checklist and automated quality gates required before launching any RAG system to live production traffic.

## Why It Matters

Deploying an unvalidated RAG pipeline risks data leakage, unexpected cloud provider billing shocks, and catastrophic latency degradation during traffic spikes.

## Core Concepts

- **Mandatory Quality Gates**: Non-negotiable checks (Security, Reliability, Evaluation, Performance) that block releases if failed.
- **Automated Pre-Flight Audits**: Running programmatic checklist assertions against SQLite benchmark stores.
- **Continuous Post-Launch Monitoring**: Synthetic canary query monitoring on live production endpoints.

## Architecture

```
[CI/CD Release Candidate] ──> [Pre-Flight Checklist Verifier] ──> [Pass Gate] ──> [Live Traffic]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [checklist_runner.py](implementation/checklist_runner.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Zero-downtime, zero-surprise production deployments.
- **Weaknesses**: Requires discipline to maintain automated checks across all release cycles.

## Architectural Decision & Trade-off Questions

1. *What top 5 non-functional metrics must be green on your dashboard before cutting over DNS to a new RAG system?*
2. *How do you structure canary deployments (1% -> 10% -> 100%) with automated rollbacks for vector database index updates?*
