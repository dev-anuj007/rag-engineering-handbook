# 27 RAG Observability

## Objective

Implement full-lifecycle distributed tracing, latency waterfall breakdowns (Embedding, ChromaDB search, Rerank, TTFT, Generation), and token telemetry with SQLite persistence.

## Why It Matters

When end-to-end RAG latency spikes from 500ms to 4,000ms, distributed span metrics instantly identify whether the bottleneck was an upstream API rate limit, an un-indexed vector scan, or a network timeout.

## Core Concepts

- **Distributed Span Waterfalls**: Granular timing capturing every micro-hop from query arrival to final token streamed.
- **Time to First Token (TTFT)**: Key metric for user-perceived responsiveness during generation.
- **Trace Persistence**: Storing trace metadata and latency breakdowns in SQLite for SLA tracking.

## Architecture

```
[Query Execution] ──> [OpenTelemetry Span Context] ──> [Latency Breakdown Waterfalls] ──> [SQLite Trace Store]
```

See [architecture.md](architecture/architecture.md) for full details.

## Implementation

Implemented in [tracer.py](implementation/tracer.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Comprehensive visibility into tail latency (p95, p99) hotspots.
- **Weaknesses**: Negligible (<1ms) tracing memory overhead.

## Architectural Decision & Trade-off Questions

1. *How do you trace asynchronous tool-calling loops in agentic RAG across multiple microservices?*
2. *What sampling rate strategy should you use in high-throughput (10,000 QPS) RAG production environments?*
