# 31 LlamaIndex Deep Dive

## Objective

Master the core abstractions of the LlamaIndex framework (`BaseQueryEngine`, `BaseRetriever`, `BaseNodePostprocessor`, `CallbackManager`) to construct custom RAG engines with ChromaDB and SQLite telemetry.

## Why It Matters

High-level `index.as_query_engine()` abstractions conceal intermediate execution steps. Deeply mastering underlying base classes allows engineers to build customized enterprise logic without fighting framework constraints.

## Core Concepts

- **BaseQueryEngine Customization**: Overriding `_query()` and `_aquery()` for asynchronous enterprise execution.
- **BaseRetriever Subclassing**: Injecting custom ChromaDB multi-collection queries or business filters.
- **Event Bus & Callbacks**: Attaching hooks for distributed logging and metric collection.

## Architecture

```
[QueryBundle] ──> [CustomChromaRetriever] ──> [CustomSOLIDQueryEngine] ──> [LlamaIndex Response]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [custom_engine.py](implementation/custom_engine.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Full control over query lifecycle while maintaining compatibility with the LlamaIndex ecosystem.
- **Weaknesses**: Requires understanding framework interface contracts.

## Architectural Decision & Trade-off Questions

1. *How do you write a custom NodePostprocessor in LlamaIndex to filter out redundant near-duplicate chunks before generation?*
2. *How do you hook into the LlamaIndex Event Bus to stream token latency metrics directly to Prometheus / Datadog?*
