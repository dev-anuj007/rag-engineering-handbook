# 32 Enterprise RAG Project

## Objective

Deliver a production-ready, multi-tenant enterprise RAG microservice implementing strict tenant data isolation, ChromaDB vector indexing, and automated SQLite evaluation test benches.

## Why It Matters

Moving from a prototype notebook to an enterprise B2B SaaS system requires rock-solid multi-tenancy, strict SLA enforcement, and verifiable regression suites.

## Core Concepts

- **Multi-Tenant Isolation**: Hard pre-filters on `tenant_id` preventing cross-customer data leakage.
- **Microservice DTO Contracts**: Typed request/response schemas with source document citation tags.
- **Automated SQLite Quality Gates**: Continuous verification of tenant boundary security and answer accuracy.

## Architecture

```
[Tenant API Request] ──> [Tenant Pre-Filter] ──> [ChromaDB Index] ──> [Grounded Answer + Citations]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [enterprise_service.py](implementation/enterprise_service.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: True enterprise security, pluggable database persistence, zero data leaks.
- **Weaknesses**: Requires tenant identifier propagation across all API gateway layers.

## Architectural Decision & Trade-off Questions

1. *How do you architect a multi-tenant RAG cluster serving 1,000 enterprise tenants with vastly different document volumes?*
2. *How do you implement GDPR 'Right to be Forgotten' document deletions across ChromaDB and downstream LLM caches?*
