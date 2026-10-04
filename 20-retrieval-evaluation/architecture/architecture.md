# Retrieval Evaluation Metrics: Theoretical Deep Dive

## 1. First-Principles Mechanics of Retrieval Metrics

Retrieval evaluation benchmarks the ability of an index and retriever to surface relevant ground-truth passages $\mathcal{G}_q \subset \mathcal{D}$ within the top-$K$ candidates returned $\mathcal{R}_q = [d_1, d_2, \dots, d_K]$.

```
Ground Truth Relevant Docs for Query Q: { doc_4, doc_9 }
Retriever Output List (K = 5):          [ doc_1, doc_4, doc_7, doc_9, doc_12 ]
                                                  │             │
                                                  ▼             ▼
                                               Rank 2        Rank 4
                                            (Hit = True)  (Hit = True)
```

---

## 2. Mathematical Formalization of IR Metrics

### 1. Hit Rate @ K (Binary Recall Indicator):

$$\text{Hit Rate}@K = \mathbb{I}(|\mathcal{R}_q[1:K] \cap \mathcal{G}_q| > 0)$$

### 2. Recall @ K:

$$\text{Recall}@K = \frac{|\mathcal{R}_q[1:K] \cap \mathcal{G}_q|}{|\mathcal{G}_q|}$$

### 3. Precision @ K:

$$\text{Precision}@K = \frac{|\mathcal{R}_q[1:K] \cap \mathcal{G}_q|}{K}$$

### 4. Mean Reciprocal Rank (MRR):
Evaluates the rank position of the **first** relevant item $r_1$:

$$\text{RR}(q) = \frac{1}{\text{rank}_1} \quad \implies \quad \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_1(q)}$$

### 5. Normalized Discounted Cumulative Gain (NDCG @ K):
Evaluates graded relevance with logarithmic rank position penalties:

$$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$

$$\text{IDCG}@K = \sum_{i=1}^{|\mathcal{G}_q|} \frac{2^{\text{rel}_i^*} - 1}{\log_2(i + 1)} \quad (\text{Ideal sorting})$$

$$\text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K} \in [0, 1]$$

---

## 3. Metric Applicability Matrix

| Metric | Focus / Priority | When to Use |
|---|---|---|
| **Hit Rate@K** | Binary coverage (Did we find at least 1 document?) | High-level sanity checks ($K \in [3, 10]$) |
| **MRR** | Pinpoint top-rank placement | Factoid question answering & search engines |
| **Recall@K** | Exhaustive candidate coverage | Upstream Stage 1 Bi-Encoder retrieval before Reranking |
| **NDCG@K** | Graded multi-document order | Multi-hop reasoning & enterprise document synthesis |

---

## 4. Failure Modes & Mitigations

1. **Evaluation Saturated by Artificially High $K$**:
   - *Failure*: Reporting Hit Rate@100 = 99%, while Hit Rate@3 = 40%. The LLM generator only accepts 3 chunks, causing real-world generation failures.
   - *Mitigation*: Always evaluate metrics at production context limits (e.g. Recall@3, Recall@5, MRR).
2. **Missing Negative Ground Truth**:
   - *Failure*: Evaluating only queries with known positive matches, ignoring adversarial queries that should return 0 results.
   - *Mitigation*: Include "Negative / Out-of-Domain" test queries in SQLite golden sets to benchmark false positive rates.

---

## 5. SOLID Principles in Retrieval Evaluation

- **Single Responsibility (SRP)**: Individual metric calculators (`MRRMetric`, `NDCGMetric`, `RecallMetric`) compute scalar scores.
- **Open/Closed (OCP)**: Custom evaluation metrics implement `RetrievalMetricProtocol`.
- **Liskov Substitution (LSP)**: All metrics accept `(retrieved_ids: list[str], golden_ids: set[str], k: int) -> float`.
- **Dependency Inversion (DIP)**: `RetrievalEvaluator` accepts `list[RetrievalMetricProtocol]`.
