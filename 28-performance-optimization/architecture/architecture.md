# Performance Optimization & Semantic Caching: Theoretical Deep Dive

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
