# 28 Performance Optimization

## Objective

Maximize RAG throughput and minimize end-to-end latency via async query concurrency, batch embedding generation, and vector index tuning in ChromaDB.

## Why It Matters

Ingesting millions of documents sequentially takes days. Batching embedding requests into 64-item chunks with asyncio workers achieves up to a 20x speedup, and concurrent retrieval pipelines scale QPS linearly across read replicas.

## Core Concepts

- **Batch Ingestion**: Grouping raw texts to fully saturate embedding API quotas and SIMD matrix multipliers.
- **Asynchronous Concurrent Retrieval**: Non-blocking I/O dispatching multiple query branches in parallel.
- **Quantization**: Scalar quantization (SQ8) compressing 32-bit floats into 8-bit integers for a 4x RAM reduction.

## Architectural Deep Dive & Mental Model

## 1. First-Principles Mechanics of Semantic Caching

Semantic caching avoids re-executing expensive embedding, retrieval, and LLM synthesis pipelines by matching incoming queries against a vector cache of previously answered queries with high semantic similarity.

```
                    Incoming User Query (Q_new)
                                 │
                                 ▼
                    Dense Embedding: e(Q_new)
                                 │
                                 ▼
             ┌───────────────────────────────────────┐
             │ Semantic Cache Lookup (ChromaDB / RAM)│
             │   Compute: CosineSim(e(Q_new), e(Q_c))│
             └───────────────────┬───────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ Sim >= 0.95 (Cache Hit)       │ Sim < 0.95 (Cache Miss)
                 ▼                               ▼
     ┌───────────────────────┐       ┌───────────────────────┐
     │ Instant Cached Answer │       │ Full RAG Pipeline     │
     │ Latency: < 15ms       │       │ (Retrieve + LLM Synth)│
     │ Cost: $0.00           │       └───────────┬───────────┘
     └───────────────────────┘                   │
                                                 ▼
                                     Store (e(Q_new), Answer)
```

---

## 2. Mathematical Formalization: Cache Hit Threshold & TTFT Optimization

### 1. Semantic Cache Hit Condition:

$$\text{Hit}(q) = \begin{cases} \text{Return } \text{Answer}(d^*) & \text{if } \max_{d \in \mathcal{C}_{\text{cache}}} \cos(\mathbf{e}(q), \mathbf{e}(d)) \ge \tau_{\text{cache}} \\ \text{MISS (Execute Full RAG)} & \text{otherwise} \end{cases}$$

Where $\tau_{\text{cache}} \in [0.92, 0.98]$ (typically $0.95$).

### 2. Time-to-First-Token (TTFT) Optimization via Speculative RAG:
In parallel pipelines:
- Thread 1: Start LLM streaming immediately with top-1 cached/shallow context.
- Thread 2: Asynchronously retrieve and rerank deep context; if deep evidence contradicts shallow context, emit a correction stream.

---

## 3. Caching Paradigm Trade-Off Matrix

| Caching Strategy | Hit Rate | Latency on Hit | Storage Footprint | Risk of Stale Answers |
|---|---|---|---|---|
| **Exact Key Match (Redis SHA-256)** | Low ($10\% - 20\%$) | $< 2\text{ ms}$ | Minimal (RAM keys) | Zero (Deterministic) |
| **Semantic Cache (ChromaDB Vector)** | High ($40\% - 65\%$) | $< 15\text{ ms}$ | Moderate (Vector index) | Moderate (Requires TTL invalidation) |
| **KV Cache Prefill (vLLM / TensorRT)** | 100% for static prompt prefixes | Accelerates TTFT by $3\times$ | High VRAM | Zero |

---

## 4. Failure Modes & Mitigations

1. **Semantic Cache Inversion False Positives**:
   - *Failure*: Cache matches *"How to cancel subscription?"* with *"How to upgrade subscription?"* because general embeddings score $0.91$, serving the wrong guide.
   - *Mitigation*: Require high threshold ($\tau \ge 0.95$) and assert that negation keywords match exactly.
2. **Cache Poisoning via Ingestion Updates**:
   - *Failure*: An internal policy changes, but the semantic cache continues serving the old cached answer for weeks.
   - *Mitigation*: Invalidate or purge semantic cache entries whenever source documents in that domain are re-ingested.

---

## 5. SOLID Principles in Performance Optimization

- **Single Responsibility (SRP)**: `SemanticCache` performs vector lookups; `CacheInvalidator` manages TTLs; `PipelineOptimizer` manages concurrency.
- **Open/Closed (OCP)**: New caching engines (Redis, SQLite, In-Memory) implement `CacheStoreProtocol`.
- **Liskov Substitution (LSP)**: All caches expose `async def get(query_vector: list[float]) -> Optional[str]`.
- **Dependency Inversion (DIP)**: RAG pipeline wraps execution around `CacheStoreProtocol`.

---

## Implementation

Implemented in [async_batch_optimizer.py](implementation/async_batch_optimizer.py).

## Evaluation

Benchmarked via SQLite dataset in [evaluator.py](evaluation/evaluator.py).

## Strengths & Trade-offs

- **Strengths**: Drastic reduction in pipeline runtime and hosting costs.
- **Weaknesses**: Requires careful tuning of worker thread pool sizes to avoid API rate limiting.

## Architectural Decision & Trade-off Questions

1. *How do you size the async batch worker pool against external rate limits and upstream database thread connection pools?*
2. *What is the recall loss when applying Product Quantization (PQ) vs Scalar Quantization (SQ8) to 1536-dimensional embeddings?*
