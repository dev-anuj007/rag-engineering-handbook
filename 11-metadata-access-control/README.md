# 11 Metadata & Access Control

## Objective

Enforce multi-tenant document isolation, Role-Based Access Control (RBAC), and Attribute-Based Access Control (ABAC) in ChromaDB retrieval pipelines with zero cross-tenant data leakage.

## Why It Matters

Allowing unauthorized context into RAG synthesis causes LLMs to leak confidential internal communications, salary data, or cross-tenant private records in generated answers.

## Core Concepts

- **Pre-Filtering**: Hard metadata partitioning during vector search (e.g. `tenant_id == user.tenant_id`) ensuring mathematical impossibility of cross-tenant leakage.
- **Hierarchical ABAC**: Dynamic clearance validation (Public -> Internal -> Confidential -> Restricted).
- **Zero-Leakage Assurance**: Verified via regression suites against SQLite golden permission datasets.

## Architecture

```
[Query + UserContext] ──> [Tenant Pre-Filter (ChromaDB)] ──> [ABAC Clearance Check] ──> [Safe LLM Prompt]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [access_control.py](implementation/access_control.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Strict mathematical tenant isolation and verifiable access control.
- **Weaknesses**: Pre-filtering can reduce effective vector graph connectivity if partitions are tiny.

## Architectural Decision & Trade-off Questions

1. *What are the failure modes of post-retrieval filtering vs pre-filtering in vector databases with low-selectivity metadata filters?*
2. *How do you implement document-level ACLs (Access Control Lists) with thousands of allowed user IDs in ChromaDB without exceeding metadata index limits?*
