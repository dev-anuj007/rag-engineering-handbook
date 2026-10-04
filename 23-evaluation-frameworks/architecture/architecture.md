# Evaluation Frameworks (RAGAS & TruLens): Theoretical Deep Dive

## 1. First-Principles Framework Architecture

Evaluation frameworks provide standardized, automated adapters to compute multi-dimensional RAG health metrics across large-scale golden test datasets.

```
                    ┌────────────────────────────────────────┐
                    │ Evaluation Dataset (Queries, Answers)  │
                    └───────────────────┬────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Unified Framework Adapter              │
                    │   (Converts samples to Schema)         │
                    └───────────────────┬────────────────────┘
                                        │
          ┌─────────────────────────────┴─────────────────────────────┐
          ▼                                                           ▼
┌───────────────────────────────┐                   ┌───────────────────────────────┐
│ RAGAS Framework Engine        │                   │ TruLens Triad Engine          │
│ • Context Precision (MAP)     │                   │ • Context Relevance           │
│ • Context Recall              │                   │ • Groundedness (NLI)          │
│ • Faithfulness                │                   │ • Answer Relevance            │
│ • Answer Semantic Similarity  │                   │ • Toxicity & PII Feedback     │
└───────────────┬───────────────┘                   └───────────────┬───────────────┘
                │                                                   │
                └───────────────────────┬───────────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ SQLite Central Metrics Repository      │
                    │   - Historical Trend Logs              │
                    │   - CI/CD Release Quality Gate Assert  │
                    └────────────────────────────────────────┘
```

---

## 2. Mathematical Formalization: RAGAS Context Precision & Recall

### 1. RAGAS Context Precision (Weighted Mean Average Precision):
Evaluates whether the highest-ranked context chunks contain the true ground-truth answers:

$$\text{Context Precision}@K = \frac{\sum_{k=1}^K \left(\text{Precision}@k \times \mathbb{I}(\text{chunk}_k \text{ is relevant})\right)}{\text{Total Relevant Chunks in Top-}K}$$

Where:

$$\text{Precision}@k = \frac{\text{True Positives up to rank } k}{k}$$

### 2. RAGAS Context Recall:
Evaluates whether all reference ground-truth statements $G = \{g_1, \dots, g_n\}$ are present in the retrieved context chunks:

$$\text{Context Recall} = \frac{\sum_{i=1}^n \mathbb{I}(\text{Retrieved Context } \models g_i)}{n}$$

---

## 3. Evaluation Framework Feature Matrix

| Feature / Dimension | RAGAS | TruLens | Custom Lightweight Framework |
|---|---|---|---|
| **Core Abstraction** | Dataset-centric batch metrics | Execution-trace continuous feedback | In-memory protocol adapters |
| **Integration Hook** | Post-execution pandas/datasets | Async wrappers around LangChain/LlamaIndex | Native SOLID protocols |
| **Storage Engine** | In-memory / CSV / HuggingFace | SQLite / PostgreSQL / DuckDB | SQLite Embedded Repository |
| **Execution Overhead** | High (Multi-LLM calls per row) | High (Async feedback functions) | Minimal ($< 10\text{ ms}$ for non-LLM metrics) |

---

## 4. Failure Modes & Mitigations

1. **Framework-Induced Benchmark Flakiness**:
   - *Failure*: A test suite evaluating 50 samples fails intermittently due to 429 rate limits from the underlying LLM judge.
   - *Mitigation*: Implement local caching of LLM judge responses in SQLite keyed by `SHA-256(prompt + model)`.
2. **Metric Schema Drift Across Library Versions**:
   - *Failure*: Upgrading a framework breaks data structures due to internal metric renaming.
   - *Mitigation*: Isolate all third-party evaluation tools behind our decoupled `FrameworkAdapterProtocol`.

---

## 5. SOLID Principles in Evaluation Frameworks

- **Single Responsibility (SRP)**: `RAGASAdapter` formats RAGAS inputs; `TruLensAdapter` maps feedback records; `MetricsRepository` persists to SQLite.
- **Open/Closed (OCP)**: New third-party tools (DeepEval, Phoenix, Cleanlab) plug into `EvaluationFrameworkProtocol`.
- **Liskov Substitution (LSP)**: All adapters return standardized `BenchmarkReport(metrics: dict[str, float], passed: bool)`.
- **Dependency Inversion (DIP)**: Test harness depends on `EvaluationFrameworkProtocol`.
