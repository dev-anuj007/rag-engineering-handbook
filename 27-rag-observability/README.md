# 27 RAG Observability

## Objective

Implement full-lifecycle distributed tracing, latency waterfall breakdowns (Embedding, ChromaDB search, Rerank, TTFT, Generation), and token telemetry with SQLite persistence.

## Why It Matters

When end-to-end RAG latency spikes from 500ms to 4,000ms, distributed span metrics instantly identify whether the bottleneck was an upstream API rate limit, an un-indexed vector scan, or a network timeout.

## Core Concepts

- **Distributed Span Waterfalls**: Granular timing capturing every micro-hop from query arrival to final token streamed.
- **Time to First Token (TTFT)**: Key metric for user-perceived responsiveness during generation.
- **Trace Persistence**: Storing trace metadata and latency breakdowns in SQLite for SLA tracking.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of RAG Observability

Observability in RAG requires instrumenting the entire asynchronous query execution path with **distributed traces (Spans)**, **structured event logs**, and **real-time metrics (RED: Rate, Errors, Duration)** across all retrieval, reranking, and generation hops.

```
                                [ DISTRIBUTED TRACE SPAN HIERARCHY ]

Trace Root: GET /v1/query [Span ID: 001] (Total Duration: 420ms)
  ├── Span: guardrail.input_sanitizer (12ms)
  ├── Span: embedding.bi_encoder (28ms)
  ├── Span: vector_db.chroma_search [Span ID: 002] (18ms)
  │     └── Attributes: { collection: "kb_prod", top_k: 50, ef_search: 64 }
  ├── Span: reranker.cross_encoder (42ms)
  │     └── Attributes: { model: "bge-reranker-large", input_candidates: 50, output_k: 5 }
  ├── Span: context.compression_and_layout (5ms)
  └── Span: llm.synthesis [Span ID: 003] (315ms)
        └── Attributes: { model: "gemini-1.5-flash", prompt_tokens: 1840, completion_tokens: 120 }
```

---

## 2. Mathematical Formalization: Latency Attribution & Token Economics

### 1. Cumulative Pipeline Latency Decomposition:

$$T_{\text{total}} = T_{\text{embed}} + T_{\text{ANN}} + T_{\text{rerank}} + T_{\text{prompt\_prep}} + T_{\text{LLM\_TTFT}} + \frac{N_{\text{completion\_tokens}}}{\text{Throughput}_{\text{tokens/sec}}}$$

### 2. Token Cost Attribution Function:

$$\text{Cost}_{\text{query}} = (N_{\text{prompt\_tokens}} \times P_{\text{input\_rate}}) + (N_{\text{completion\_tokens}} \times P_{\text{output\_rate}})$$

Where $P_{\text{input\_rate}}, P_{\text{output\_rate}}$ are model provider pricing tiers per token.

---

## 3. Observability Architecture Trade-Offs

| Observability Engine | Performance Overhead | Trace Fidelity | Storage Cost |
|---|---|---|---|
| **Simple Standard Logging (JSON Stdout)** | Zero ($< 1\text{ ms}$) | Low (No visual span trees) | Minimal |
| **OpenTelemetry (OTel Collector -> Jaeger / OTLP)** | Very Low ($< 2\text{ ms}$) | High (Full distributed span graphs) | Moderate |
| **Specialized LLM APM (Arize Phoenix / Langfuse / OpenInference)** | Low ($< 5\text{ ms}$) | Very High (Token trees, evaluations, drift) | Moderate |

---

## 4. Failure Modes & Mitigations

1. **Observability-Induced Latency Degrades SLAs**:
   - *Failure*: Synchronously shipping traces to a remote collector blocks the request response thread.
   - *Mitigation*: Use async non-blocking batch processors (`BatchSpanProcessor`) with an internal queue.
2. **PII and Secret Data Leakage in Traces**:
   - *Failure*: Unsanitized user passwords or confidential contract text are logged to trace attributes.
   - *Mitigation*: Apply attribute redaction masks (`redact_pii()`) before exporting span attributes.

---

## 5. SOLID Principles in RAG Observability

- **Single Responsibility (SRP)**: `Tracer` creates spans; `MetricRecorder` tracks latency distributions; `SpanExporter` ships telemetry.
- **Open/Closed (OCP)**: New telemetry backends (OpenTelemetry, Datadog, Prometheus) implement `TelemetryProviderProtocol`.
- **Liskov Substitution (LSP)**: All tracers expose identical context managers `with tracer.start_span(name):`.
- **Dependency Inversion (DIP)**: Pipeline components depend on `TelemetryProviderProtocol`.

---

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
