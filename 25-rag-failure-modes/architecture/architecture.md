# 7 Critical RAG Failure Modes: Theoretical Deep Dive

## 1. First-Principles Taxonomy of Failure Modes

Production RAG pipelines suffer from failure modes distributed across five distinct execution phases: **Ingestion**, **Retrieval**, **Reranking**, **Context Assembling**, and **Generation**.

```
[ THE 7 CRITICAL FAILURE MODES TAXONOMY ]

Phase 1: Ingestion
  └── 1. Missing / Corrupted Knowledge (Content absent from corpus or broken during PDF parsing)

Phase 2: Retrieval
  ├── 2. Retrieval Miss / Semantic Drift (Relevant document exists but not in Top-K)
  └── 3. Outdated / Stale Document Contamination (Old version retrieved over new version)

Phase 3: Context Assembly
  ├── 4. Extraction & Context Truncation (Evidence discarded during context compression)
  └── 5. Lost-in-the-Middle (Evidence positioned in center 50% of context window)

Phase 4: Generation
  ├── 6. Hallucination / Parametric Override (LLM generates answer contradicting evidence)
  └── 7. Format / Formatting Non-Compliance (LLM fails JSON schema or citation tags)
```

---

## 2. Theoretical Mechanics & Mathematical Failure Attributions

### Quantitative Failure Attribution Formula
To isolate root causes in production telemetry, compute failure attribution probabilities:

$$P(\text{System Failure}) = P(\text{Recall Miss}) + P(\text{Precision Noise} \mid \text{Hit}) + P(\text{Hallucination} \mid \text{Correct Context})$$

1. **Retrieval Recall Miss**: $\text{Ground Truth} \cap \text{Top-}K == \emptyset$. (Retriever / Embedding model is faulty).
2. **Context Dilution / Precision Noise**: $|\text{Top-}K \cap \text{Ground Truth}| = 1$, but $K=10$ (9 noise chunks confuse the generator).
3. **Parametric Hallucination**: $\text{Ground Truth} \in \text{Context}$, but generation $\not\models \text{Context}$. (Prompt or model temperature $T > 0$ is faulty).

---

## 3. Failure Mode Mitigation & Recovery Matrix

| Failure Mode | Root Cause | Architectural Mitigation | Automated Guardrail |
|---|---|---|---|
| **1. Missing Knowledge** | Document never ingested | Trigger web search fallback or gracefully return "UNKNOWN" | Minimum retrieval similarity score threshold $\tau \ge 0.65$ |
| **2. Retrieval Miss** | Embedding vocabulary mismatch | Hybrid Search (BM25 + ChromaDB Dense) with RRF | Reciprocal Rank Fusion ($k=60$) |
| **3. Stale Contamination** | No TTL / versioning in index | Soft delete tombstones + timestamp metadata filtering | Ingestion CDC with SHA-256 deduplication |
| **4. Context Truncation** | Rigid chunk token limit | Hierarchical Parent-Child chunking | Automatic parent chunk expansion |
| **5. Lost-in-the-Middle** | Positional attention bias | Peripheral sorting (Top rank at start and end) | Alternating document layout |
| **6. Hallucination** | Non-zero temperature / weak prompt | Greedy decoding ($T=0.0$) + strict grounding | NLI-based claim entailment assertion |
| **7. Schema Non-Compliance** | Free-form output | Pydantic constrained decoding / JSON Schema mode | Regex JSON parsing + automated retry |

---

## 4. SOLID Principles in Failure Prevention

- **Single Responsibility (SRP)**: `CircuitBreaker` manages failure thresholds; `FallbackRouter` steers degraded execution; `AnomalyLogger` records telemetry.
- **Open/Closed (OCP)**: New failure detectors implement `FailureGuardrailProtocol`.
- **Liskov Substitution (LSP)**: Guardrails return standardized `GuardrailResult(passed: bool, fallback_action: str)`.
- **Dependency Inversion (DIP)**: Production service wraps retrievers and synthesizers in `ResilienceMiddleware`.
